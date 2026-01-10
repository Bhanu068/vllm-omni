from __future__ import annotations

import random
from collections import defaultdict
from time import time

import torch
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

logger = init_logger(__name__)


class OmniARScheduler(VLLMScheduler):
    """
    OmniARScheduler: Scheduler for vLLM-Omni multimodal processing.

    This scheduler extends vLLM's scheduler to support multimodal and
    non-autoregressive processing with additional fields and methods
    specific to vLLM-Omni.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set of request IDs that are waiting for upstream data
        self.reqs_waiting_for_upstream: set[str] = set()
        # Queue of pending upstream chunks per request (FIFO order)
        # req_id -> list of payload dicts to apply when request is ready
        self.pending_upstream_chunks: dict[str, list[dict]] = {}

    def _free_request(self, request: Request):
        """Override to clean up Omni-specific data when a request finishes.

        This ensures that:
        1. Request is removed from waiting_for_upstream set
        2. Any pending chunks are discarded
        3. Accumulated data in additional_information_cpu is cleared
        """
        req_id = request.request_id

        # Clean up our custom data structures
        self.reqs_waiting_for_upstream.discard(req_id)
        self.pending_upstream_chunks.pop(req_id, None)

        # Clear accumulated data from additional_information_cpu to free memory
        if hasattr(request, "additional_information_cpu"):
            request.additional_information_cpu = None

        logger.debug(f"[CLEANUP] Freed request {req_id}, cleared waiting/pending/additional_info")

        # Call parent's _free_request
        return super()._free_request(request)

    def has_unfinished_requests(self) -> bool:
        """Override to exclude paused requests from consideration.

        Paused requests (waiting for upstream data) should not count as
        "unfinished" for the purpose of deciding whether the EngineCore
        should block waiting for new work. This prevents the busy-wait loop
        when all active requests are paused.
        """
        # Check if there are any waiting requests
        if len(self.waiting) > 0:
            return True

        # Check if there are any running requests that are NOT paused
        for req in self.running:
            if req.request_id not in self.reqs_waiting_for_upstream:
                return True

        return False

    # Ensure scheduled_new_reqs carry omni-specific payloads
    # (e.g., additional_information)
    def schedule(self) -> SchedulerOutput:  # type: ignore[override]
        """Override to exclude paused requests from scheduling.

        Requests that are waiting for upstream data (tracked in _process_paused_upstream)
        are temporarily removed from self.running before calling the parent scheduler.
        This prevents them from being scheduled while they wait for more data.

        The requests remain in RUNNING state conceptually, they're just not executed
        until new data arrives via update_request().
        """

        # If there are paused requests, filter them out before scheduling
        original_running = None
        if self.reqs_waiting_for_upstream:
            original_running = self.running
            self.running = [r for r in original_running if r.request_id not in self.reqs_waiting_for_upstream]
            if len(self.running) < len(original_running):
                logger.debug(
                    "[schedule] Excluded %d paused requests from scheduling: %s",
                    len(original_running) - len(self.running),
                    list(self.reqs_waiting_for_upstream),
                )

        try:
            scheduler_output = super().schedule()
        finally:
            # Restore original running list (paused requests stay in "running" conceptually)
            if original_running is not None:
                self.running = original_running

        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData

            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_features=nr.mm_features,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=nr.block_ids,
                    num_computed_tokens=nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                    additional_information=(getattr(request, "additional_information", None) if request else None),
                )
                new_list.append(omni_nr)

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]
        except Exception:
            # If anything goes wrong, leave the original output unchanged
            init_logger(__name__).exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")

        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        # Check for requests that need to pause and wait for upstream data
        wait_for_upstream_req_ids = getattr(model_runner_output, "wait_for_upstream_reqs", set()) or set()

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(kv_connector_output.invalid_block_ids)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            # Skip output processing for requests that are now paused
            # They ran this step but will be excluded from next scheduling
            if req_id in wait_for_upstream_req_ids:
                if req_id not in self.reqs_waiting_for_upstream:
                    self.reqs_waiting_for_upstream.add(req_id)
                logger.debug("[update_from_output] Request %s will be paused after this step", req_id)

                # Try to apply any pending queued chunks
                # This allows the request to resume immediately if data is available
                if self.apply_pending_chunks(req_id):
                    logger.debug(f"[update_from_output] Applied pending chunk for {req_id}, request may resume")
                else:
                    logger.debug(f"[update_from_output] No pending chunks for {req_id}, will remain paused")
                # NOTE: Do NOT skip output processing! The model executed for this
                # request and generated tokens. We must process them normally.
                # The pausing only affects the NEXT schedule() call.

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                if request.output_token_ids:
                    stopped = check_stop(request, self.max_model_len, pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                struct_output_request.grammar.accept_tokens(req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _apply_update_to_request(self, request, payload: dict) -> bool:
        """Apply an update payload to a request. Used by both update_request and add_request.

        When new data arrives via payload, this method:
        1. Transforms thinker_result into thinker_reply_part queue for streaming decode
        2. Merges the payload into the request's additional_information_cpu (used by GPU model runner)
        3. Unpauses the request if it was paused waiting for upstream data

        The unpause happens by removing the request from reqs_waiting_for_upstream,
        which means it will be included in the next schedule() call.

        Note: We use additional_information_cpu because that's what the GPU model runner reads from
        in _gather_runtime_additional_information(). The additional_information field is for
        serialized payloads used during request creation/transfer.
        """
        try:
            upstream_finished = payload.get("upstream_finished", False)
            payload = {}
            n = random.randint(1, 10)
            thinker_result = torch.randn((n, 2048))
            payload["thinker_result"] = thinker_result
            payload["thinker_result_shape"] = list(thinker_result.shape)
            payload["upstream_finished"] = upstream_finished

            # Transform thinker_result into thinker_reply_part queue for decode consumption
            # Following the same pattern as thinker_to_talker_process (line 683 in qwen2_5_omni.py):
            # skip the first embedding and use the rest as the queue
            if isinstance(thinker_result, torch.Tensor) and thinker_result.ndim == 2 and thinker_result.shape[0] > 0:
                new_chunk = thinker_result[1:].detach().to("cpu").contiguous()
                logger.debug(
                    f"[UPDATE] Transformed thinker_result shape {thinker_result.shape} → thinker_reply_part chunk shape {new_chunk.shape}"
                )

                # Accumulate chunks in scheduler - needed because chunks may arrive faster than worker consumes
                # The sync mechanism will copy the accumulated queue to worker when worker's queue is empty
                if hasattr(request, "additional_information_cpu") and isinstance(
                    request.additional_information_cpu, dict
                ):
                    existing_queue = request.additional_information_cpu.get("thinker_reply_part")
                    if isinstance(existing_queue, torch.Tensor) and existing_queue.numel() > 0:
                        # Append new chunk to existing queue
                        merged_queue = torch.cat([existing_queue, new_chunk], dim=0)
                        payload["thinker_reply_part"] = merged_queue
                        logger.debug(
                            f"[UPDATE] Appended chunk to existing queue: {existing_queue.shape} + {new_chunk.shape} = {merged_queue.shape}"
                        )
                    else:
                        # No existing queue, initialize with new chunk
                        payload["thinker_reply_part"] = new_chunk
                        logger.debug(f"[UPDATE] Initialized thinker_reply_part queue with shape {new_chunk.shape}")
                else:
                    # No additional_information_cpu yet, initialize with new chunk
                    payload["thinker_reply_part"] = new_chunk
                    logger.debug(f"[UPDATE] Created thinker_reply_part queue with shape {new_chunk.shape}")

            # Update the request's additional_information_cpu with the new payload
            logger.debug(f"Applying update to request {request.request_id} with payload keys: {list(payload.keys())}")
            if not hasattr(request, "additional_information_cpu"):
                logger.debug(f"Request {request.request_id} has no additional_information_cpu, initializing empty dict")
                request.additional_information_cpu = {}

            # Merge the new payload into existing additional_information_cpu
            if isinstance(request.additional_information_cpu, dict):
                logger.debug(
                    f"Merging payload into existing additional_information_cpu dict with payload keys: {list(payload.keys())}"
                )
                request.additional_information_cpu.update(payload)
            else:
                # If it's not a dict, replace it entirely
                request.additional_information_cpu = payload

            # Log the actual tensor shapes being stored
            if "thinker_reply_part" in request.additional_information_cpu:
                trp = request.additional_information_cpu["thinker_reply_part"]
                if isinstance(trp, torch.Tensor):
                    logger.debug(f"[SCHEDULER] Stored thinker_reply_part shape: {trp.shape}, device: {trp.device}")

            # Unpause the request if it was paused waiting for upstream data
            # This removes it from the paused set so it will be scheduled next step
            if request.request_id in self.reqs_waiting_for_upstream:
                self.reqs_waiting_for_upstream.discard(request.request_id)
                logger.debug("Request %s received new upstream data, unpausing", request.request_id)
            return True

        except Exception as e:
            logger.exception("Failed to update request %s with payload: %s", request.request_id, e)
            return False

    def update_request(self, request_id: str, payload: dict) -> bool:
        """Update a running request with new streaming data.

        This method is called asynchronously (via UTILITY message from AsyncOmniLLM)
        to inject new data into a running request. Unlike _update_request_with_output
        which processes model outputs AFTER execution, this method updates request
        state BEFORE the next execution step.

        Typical use case: Incremental Thinker-to-Talker streaming where new embeddings
        arrive from the upstream Thinker stage while the Talker is still generating.

        Args:
            request_id: ID of the request to update.
            payload: Dictionary containing update data. Expected keys:
                - 'thinker_chunk': New embeddings/tokens to append to the queue
                - 'stream_finished': Boolean indicating if the stream is complete
                - Other model-specific data

        Returns:
            True if update was successful, False if request not found or error occurred.
        """
        # Try to get request from self.requests
        request = self.requests.get(request_id)

        if request is None:
            logger.warning(
                "[update_request] Request %s not found in scheduler. It may have finished or not been added yet.",
                request_id,
            )
            return False

        # ALWAYS apply the chunk to scheduler's state
        # This ensures _sync_update_to_worker syncs the actual new data
        success = self._apply_update_to_request(request, payload)

        if success:
            # Check if request was waiting for upstream data - if so, resume it
            if request_id in self.reqs_waiting_for_upstream:
                logger.debug(f"[update_request] Request {request_id} was waiting, now resumed after applying chunk")
                self.reqs_waiting_for_upstream.discard(request_id)
            else:
                logger.debug(f"[update_request] Applied chunk to running request {request_id}")

        return success

    def apply_pending_chunks(self, request_id: str) -> bool:
        """Apply pending queued chunks when request signals it's waiting for upstream.

        This is called when the model outputs wait_for_upstream_chunk=True.
        It applies the next pending chunk from the queue in FIFO order.

        Args:
            request_id: ID of the request that's waiting

        Returns:
            True if a chunk was applied, False if no pending chunks
        """
        if request_id not in self.pending_upstream_chunks:
            logger.debug(f"[apply_pending_chunks] No pending chunks for {request_id}")
            return False

        pending = self.pending_upstream_chunks[request_id]
        if not pending:
            logger.debug(f"[apply_pending_chunks] Pending chunk queue empty for {request_id}")
            return False

        # Get the next chunk in FIFO order
        payload = pending.pop(0)
        logger.debug(
            f"[apply_pending_chunks] Applying queued chunk for {request_id}. Remaining in queue: {len(pending)}"
        )

        request = self.requests.get(request_id)
        if request is None:
            logger.warning(f"[apply_pending_chunks] Request {request_id} not found")
            return False

        # Apply the chunk
        success = self._apply_update_to_request(request, payload)

        if success and request_id in self.reqs_waiting_for_upstream:
            self.reqs_waiting_for_upstream.discard(request_id)
            logger.debug(f"[apply_pending_chunks] Resumed request {request_id}, removed from waiting set")

        # Clean up empty queue
        if not pending:
            del self.pending_upstream_chunks[request_id]
            logger.debug(f"[apply_pending_chunks] Cleared empty queue for {request_id}")

        return success

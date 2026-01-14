from __future__ import annotations

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

    def _free_request(self, request: Request):
        """Override to clean up Omni-specific data when a request finishes.

        This ensures that:
        1. Pending upstream chunks are discarded (request finished, no longer needed)
        2. Accumulated data in additional_information_cpu is cleared
        3. Request is properly removed from running list
        """
        # Clear pending upstream chunks - request is done, discard any unprocessed data
        if hasattr(request, "pending_upstream_chunks"):
            request.pending_upstream_chunks = None

        # Clear accumulated data from additional_information_cpu to free memory
        if hasattr(request, "additional_information_cpu"):
            request.additional_information_cpu = None

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

        # Check if there are any running requests that are NOT waiting for chunks
        for req in self.running:
            if req.status != RequestStatus.WAITING_FOR_CHUNK:
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

        # If there are paused requests (waiting for chunks), filter them out before scheduling
        paused_requests = [r for r in self.running if r.status == RequestStatus.WAITING_FOR_CHUNK]
        if paused_requests:
            self.running = [r for r in self.running if r.status != RequestStatus.WAITING_FOR_CHUNK]

        try:
            scheduler_output = super().schedule()
        finally:
            still_valid_paused = [r for r in paused_requests if r.request_id in self.requests]
            if still_valid_paused:
                self.running.extend(still_valid_paused)

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
                if request.status != RequestStatus.WAITING_FOR_CHUNK:
                    request.status = RequestStatus.WAITING_FOR_CHUNK

                # Try to apply any pending queued chunks
                # This allows the request to resume immediately if data is available
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

    def _deserialize_tensor_payload(self, payload: dict) -> dict:
        """Deserialize bytes-based tensor payloads back to torch tensors.

        This reverses the serialization done by AsyncOmniLLM._serialize_tensor_payload.
        Detects entries with 'data', 'shape', 'dtype' keys and reconstructs tensors.
        """
        import numpy as np

        deserialized = {}
        for key, value in payload.items():
            if isinstance(value, dict) and "data" in value and "shape" in value and "dtype" in value:
                # Reconstruct tensor from serialized form
                dtype_str = value["dtype"]
                shape = value["shape"]
                data = value["data"]
                # Convert bytes to numpy array then to torch tensor
                np_dtype = getattr(np, dtype_str, np.float32)
                np_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
                tensor = torch.from_numpy(np_array.copy())  # Copy to own memory
                deserialized[key] = tensor
            else:
                deserialized[key] = value
        return deserialized

    def _apply_update_to_request(self, request, payload: dict) -> bool:
        """Apply an update payload to a request. Used by both update_request and add_request.

        When new data arrives via payload, this method:
        1. Deserializes bytes-based tensor payloads back to torch tensors
        2. Transforms thinker_result into thinker_reply_part queue for streaming decode
        3. Merges the payload into the request's additional_information_cpu (used by GPU model runner)

        Note: We use additional_information_cpu because that's what the GPU model runner reads from
        in _gather_runtime_additional_information(). The additional_information field is for
        serialized payloads used during request creation/transfer.
        """
        try:
            # Deserialize bytes-based tensor payloads
            payload = self._deserialize_tensor_payload(payload)

            if not hasattr(request, "pending_upstream_chunks"):
                request.pending_upstream_chunks = []

            request.pending_upstream_chunks.append(payload)
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

        return success

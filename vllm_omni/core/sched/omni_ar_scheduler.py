from __future__ import annotations

import importlib
from collections import defaultdict
from time import time

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.chunk_manager import AsyncChunkManagerForAR
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec


class OmniARScheduler(VLLMScheduler):
    """
    OmniARScheduler: Scheduler for vLLM-Omni multimodal processing.

    This scheduler extends vLLM's scheduler to support multimodal and
    non-autoregressive processing with additional fields and methods
    specific to vLLM-Omni.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_config = self.vllm_config.model_config
        scheduler_config = self.vllm_config.scheduler_config
        self.stage_id = getattr(self.vllm_config.model_config, "stage_id", None)
        self.model_name = None
        # TODO: I will refactor this to remove it out of Scheduler
        # and handle it in a much cleaner way perhaps using model_config
        if isinstance(model_config.hf_config, Qwen3OmniMoeConfig):
            self.model_name = "qwen3"
        if scheduler_config.async_chunk_stream:
            connector_specs = ConnectorSpec(
                name=scheduler_config.stage_connector_name, extra=scheduler_config.stage_connector_extra
            )
            self.omni_connector = OmniConnectorFactory.create_connector(connector_specs)

            if hasattr(self.vllm_config.model_config, "next_stage_chunk_process_input_func"):
                next_stage_chunk_process_input_func = self.vllm_config.model_config.next_stage_chunk_process_input_func
                if next_stage_chunk_process_input_func:
                    module_path, func_name = next_stage_chunk_process_input_func.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    self.next_stage_chunk_process_input_func = getattr(module, func_name)
                else:
                    self.next_stage_chunk_process_input_func = None
            else:
                self.next_stage_chunk_process_input_func = None

            self.async_chunk_handler = AsyncChunkManagerForAR(self.omni_connector, self.stage_id, self.model_name)

    # Ensure scheduled_new_reqs carry omni-specific payloads
    # (e.g., additional_information)
    def schedule(self) -> SchedulerOutput:  # type: ignore[override]
        # Filter out WAITING_FOR_CHUNK requests - they shouldn't be scheduled
        waiting_reqs = [request for request in self.running if request.status == RequestStatus.WAITING_FOR_CHUNK]
        self.running = [request for request in self.running if request.status != RequestStatus.WAITING_FOR_CHUNK]

        scheduler_output = super().schedule()
        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData

            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads including pending_chunk
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None

                # Get pending_chunk from Request object (if received)
                pending_chunk = getattr(request, "pending_chunk", None) if request else None
                # Use pending_chunk if available, otherwise fall back to additional_information
                additional_info = pending_chunk or (
                    getattr(request, "additional_information", None) if request else None
                )

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
                    additional_information=additional_info,
                )
                new_list.append(omni_nr)

                # Clear pending_chunk after copying to scheduled_new_reqs
                if request and pending_chunk:
                    request.pending_chunk = None

            # Re-add requests that are still WAITING_FOR_CHUNK to self.running
            # They will be checked for chunks again in next schedule() call
            self.running.extend(waiting_reqs)

            # Copy pending_chunk from Request objects to cached_reqs.additional_information
            # This makes chunks received in previous update_from_output available to workers
            cached_reqs = scheduler_output.scheduled_cached_reqs
            if not hasattr(cached_reqs, "additional_information"):
                cached_reqs.additional_information = {}

            for req_id in cached_reqs.req_ids:
                request = self.requests.get(req_id)
                if request:
                    pending_chunk = getattr(request, "pending_chunk", None)
                    if pending_chunk:
                        cached_reqs.additional_information[req_id] = pending_chunk
                        request.pending_chunk = None

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]
        except Exception:
            # If anything goes wrong, leave the original output unchanged
            init_logger(__name__).exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")

        return scheduler_output

    def get_request_objects(self, scheduled_cached_reqs: CachedRequestData):
        cached_requests = {}

        for req_id in scheduled_cached_reqs.req_ids:
            if req_id in self.requests:
                cached_requests[req_id] = self.requests[req_id]

        return cached_requests

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
                if hasattr(self, "async_chunk_handler"):
                    next_stage_chunk_process_input_func = self.next_stage_chunk_process_input_func

                    generation = False
                    # TODO: This is bad, I will refactor it before merge
                    func_path = (
                        f"{next_stage_chunk_process_input_func.__module__}."
                        f"{next_stage_chunk_process_input_func.__name__}"
                    )

                    if func_path.endswith(".talker2codewav_chunk"):
                        generation = True
                        self.async_chunk_handler.process_chunk(
                            new_token_ids, request, next_stage_chunk_process_input_func, generation
                        )
                    else:
                        self.async_chunk_handler.process_chunk(
                            pooler_output, request, next_stage_chunk_process_input_func, generation
                        )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Receive chunks for all active requests (RUNNING or WAITING_FOR_CHUNK)
        if hasattr(self, "async_chunk_handler") and self.stage_id == 1:
            active_requests = [
                req
                for req in self.requests.values()
                if req.status in (RequestStatus.RUNNING, RequestStatus.WAITING_FOR_CHUNK) and not req.is_finished()
            ]
            self.async_chunk_handler.receive_chunk(active_requests)

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

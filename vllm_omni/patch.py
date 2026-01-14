import sys

import torch
from aenum import extend_enum
from vllm.inputs.data import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest
from vllm.v1.request import RequestStatus

import vllm_omni.logger  # noqa: F401
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.mrope import MRotaryEmbedding
from vllm_omni.request import OmniRequest
from vllm_omni.utils import is_npu

for module_name, module in sys.modules.items():
    # only do patch on module of vllm, pass others
    if "vllm" not in module_name:
        continue
    if hasattr(module, "EngineCoreOutput") and module.EngineCoreOutput == _OriginalEngineCoreOutput:
        module.EngineCoreOutput = OmniEngineCoreOutput
    if hasattr(module, "EngineCoreOutputs") and module.EngineCoreOutputs == _OriginalEngineCoreOutputs:
        module.EngineCoreOutputs = OmniEngineCoreOutputs
    if hasattr(module, "TokensPrompt") and module.TokensPrompt == _OriginalTokensPrompt:
        module.TokensPrompt = OmniTokensPrompt
    if hasattr(module, "MRotaryEmbedding") and module.MRotaryEmbedding == _OriginalMRotaryEmbedding:
        module.MRotaryEmbedding = MRotaryEmbedding
    if hasattr(module, "Request") and module.Request == _OriginalRequest:
        module.Request = OmniRequest
    if hasattr(module, "EngineCoreRequest") and module.EngineCoreRequest == _OriginalEngineCoreRequest:
        module.EngineCoreRequest = OmniEngineCoreRequest


# Extend RequestStatus enum with omni-specific statuses
# CRITICAL: Value must be <= PREEMPTED (5) to NOT be treated as finished!
# We use a negative value to be safe and avoid conflicts with vLLM's existing statuses.
if not hasattr(RequestStatus, "WAITING_FOR_CHUNK"):
    extend_enum(RequestStatus, "WAITING_FOR_CHUNK", -1)


# Patch for vllm-ascend prefetch functions bug fix
# Issue: The original functions access forward_context attributes like
# prefetch_mlp_gate_up_proj, prefetch_mlp_down_proj, layer_idx without checking
# if they exist, which causes AttributeError when prefetch_mlp_enabled is not set.
# TODO: Remove this patch after upgrading to vllm-ascend v0.13.0 or later.
# This issue has been fixed in https://github.com/vllm-project/vllm-ascend/pull/5035
if is_npu():
    import torch
    import torch.nn as nn
    from vllm.model_executor.models.qwen2_5_omni_thinker import Qwen2_5_VLImageInputs, Qwen2_5_VLVideoInputs
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    class AscendQwen2_5OmniThinkerForConditionalGeneration(nn.Module):
        def _process_image_input(self, image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:
            if image_input["type"] == "image_embeds":
                return image_input["image_embeds"].type(self.visual.dtype)

            grid_thw = image_input["image_grid_thw"]
            assert grid_thw.ndim == 2

            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            with set_ascend_forward_context(None, self.vllm_config):
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
            # Split concatenated embeddings for each image item.
            merge_size = self.visual.spatial_merge_size
            sizes = grid_thw.prod(-1) // merge_size // merge_size

            return image_embeds.split(sizes.tolist())

        def _process_video_input(
            self,
            video_input: Qwen2_5_VLVideoInputs,
            video_hashes: list[str] | None = None,
            cached_video_embeds: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if video_input["type"] == "video_embeds":
                return video_input["video_embeds"].type(self.visual.dtype)

            grid_thw = video_input["video_grid_thw"]
            assert grid_thw.ndim == 2

            pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
            with set_ascend_forward_context(None, self.vllm_config):
                video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
            # Split concatenated embeddings for each video item.
            merge_size = self.visual.spatial_merge_size
            sizes = grid_thw.prod(-1) // merge_size // merge_size

            return video_embeds.split(sizes.tolist())

    Qwen2_5OmniThinkerForConditionalGeneration._process_image_input = (
        AscendQwen2_5OmniThinkerForConditionalGeneration._process_image_input
    )
    Qwen2_5OmniThinkerForConditionalGeneration._process_video_input = (
        AscendQwen2_5OmniThinkerForConditionalGeneration._process_video_input
    )


# Patch EngineCore to add update_request method for incremental streaming support
def _patch_engine_core_update_request():
    """Add update_request method to EngineCore for scheduler delegation.

    This enables the AsyncOmniLLM to call update_request via the UTILITY
    request type mechanism, which then delegates to scheduler.update_request().
    """
    from vllm.logger import init_logger
    from vllm.v1.engine.core import EngineCore

    logger = init_logger(__name__)

    def update_request(self, request_id: str, payload: dict) -> bool:
        if not hasattr(self.scheduler, "update_request"):
            logger.warning("Scheduler does not support update_request. Use OmniARScheduler for streaming support.")
            return False

        try:
            success = self.scheduler.update_request(request_id, payload)

            if success:
                # 2. Also update worker's CachedRequestState
                # This mirrors how add_request creates both scheduler Request and worker CachedRequestState
                self._sync_update_to_worker(request_id, payload)

            return success
        except Exception as e:
            logger.exception("Failed to update request %s: %s", request_id, e)
            return False

    def _sync_update_to_worker(self, request_id: str, payload: dict):
        try:
            scheduler_request = self.scheduler.requests.get(request_id)
            if scheduler_request is None:
                logger.warning(f"[SYNC] Request {request_id} not found in scheduler")
                return

            # Sync to worker
            if hasattr(self.model_executor, "driver_worker"):
                worker = self.model_executor.driver_worker
                if hasattr(worker, "model_runner"):
                    model_runner = worker.model_runner
                    req_state = model_runner.requests.get(request_id)

                    if req_state is not None:
                        scheduler_chunks = getattr(scheduler_request, "pending_upstream_chunks", None)
                        if scheduler_chunks:
                            if not hasattr(req_state, "pending_upstream_chunks"):
                                req_state.pending_upstream_chunks = []

                            req_state.pending_upstream_chunks.extend(scheduler_chunks)
                            scheduler_request.pending_upstream_chunks = []

                            if scheduler_request.status == RequestStatus.WAITING_FOR_CHUNK:
                                scheduler_request.status = RequestStatus.RUNNING

        except Exception as e:
            logger.warning(f"[SYNC] Failed to sync update to worker: {e}")

    # Only patch if not already patched
    if not hasattr(EngineCore, "update_request"):
        EngineCore.update_request = update_request
        EngineCore._sync_update_to_worker = _sync_update_to_worker


_patch_engine_core_update_request()

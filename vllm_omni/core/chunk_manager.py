"""
This module contains the AsyncChunkManager class, which is responsible for managing
the asynchronous transmission of chunks between different stages in the vLLM-Omni
distributed pipeline.
"""

import time
from collections import defaultdict
from collections.abc import Callable

from vllm.logger import init_logger
from vllm.v1.request import RequestStatus

from vllm_omni.request import OmniRequest

logger = init_logger(__name__)


class AsyncChunkManager:
    def __init__(self, connector, stage_id, model=None):
        self.connector = connector
        self.processed_chunks: dict[str, int] = defaultdict(int)
        self.received_chunks: dict[str, int] = defaultdict(int)
        self.stage_id = stage_id
        self.prev_stage_id = self.stage_id - 1
        self.next_stage_id = self.stage_id + 1
        self.first_chunk_after_prefill: dict[str, bool] = defaultdict(lambda: True)
        self.track_if_upstream_finished: dict[str, bool] = defaultdict(bool)
        self.model = model

    def _prepare_connector_key(self, chunk_id, stage_id, request_id):
        return f"{request_id}_{stage_id}_{chunk_id}"

    def process_chunk(self, pooler_output, request, next_stage_chunk_process_input_func):
        if not next_stage_chunk_process_input_func:
            return

        def _is_prefill() -> bool:
            # Qwen3 talker preprocess prefill requires atleast three output tokens from thinker
            if self.model == "qwen3":
                return len(request.output_token_ids) <= 3
            else:
                # If first dim is greater than 0, that means it in prefill mode
                # Decode mode outputs only one token at a time
                return pooler_output["hidden"].shape[0] > 1

        if _is_prefill():
            return
        # If it is first chunk after prefill, skip sending the chunk.
        # Because orchestrator should be sending it in async_omni.py
        elif self.first_chunk_after_prefill[request.request_id]:
            self.first_chunk_after_prefill[request.request_id] = False
            return
        else:
            self.send_chunk(
                pooler_output, request, next_stage_chunk_process_input_func, last_chunk=request.is_finished()
            )

    def send_chunk(self, pooler_output, request, next_stage_chunk_process_input_func, last_chunk=False) -> bool:
        """Send a chunk to shared memory. Raises exception on any failure."""

        request_id = request.request_id
        chunk_id = self.processed_chunks[request_id]

        try:
            chunk = next_stage_chunk_process_input_func(pooler_output, request)
        except Exception as e:
            raise RuntimeError(
                f"[send_chunk] stage={self.stage_id} req={request_id} chunk_id={chunk_id} "
                f"next_stage_chunk_process_input_func failed: {e}"
            ) from e

        if not chunk:
            raise RuntimeError(
                f"[send_chunk] stage={self.stage_id} req={request_id} chunk_id={chunk_id} "
                f"next_stage_chunk_process_input_func returned None/empty chunk"
            )

        connector_key = self._prepare_connector_key(chunk_id, self.stage_id, request_id)

        chunk["last_chunk"] = last_chunk
        success, size, _ = self.connector.put_chunk(
            from_stage=str(self.stage_id), to_stage=str(self.next_stage_id), put_key=connector_key, data=chunk
        )

        if not success:
            raise RuntimeError(
                f"[send_chunk] stage={self.stage_id} req={request_id} chunk_id={chunk_id} "
                f"put_chunk FAILED for key={connector_key}"
            )

        self.processed_chunks[request_id] += 1

        return True

    def _retrieve_chunk_from_connector(self, request_id, validate_chunk_func=None, retry_for_first_chunk=True):
        chunk_id = self.received_chunks[request_id]

        connector_key = self._prepare_connector_key(chunk_id, self.prev_stage_id, request_id)

        max_retries = 2
        sleep_sec = 0.1

        payload_data = None
        for attempt in range(max_retries):
            chunk = self.connector.get_chunk(
                from_stage=str(self.prev_stage_id),
                to_stage=str(self.stage_id),
                get_key=connector_key,
            )

            if chunk:
                payload_data, _ = chunk
                if payload_data:
                    if validate_chunk_func:
                        if not validate_chunk_func(payload_data):
                            logger.warning(
                                f"[Stage-{self.stage_id}] Received invalid data for request {request_id}. Waiting..."
                            )

                    self.received_chunks[request_id] += 1
                    return payload_data

            if attempt < max_retries - 1:
                time.sleep(sleep_sec)

        return None

    def receive_chunk(self, active_requests: list[OmniRequest], validate_chunk_func: Callable | None = None) -> None:
        """Handle chunks for active (RUNNING/WAITING_FOR_CHUNK) requests.

        Stores chunk data directly on the Request object's pending_chunk attribute.
        Called from update_from_output() method.

        Args:
            active_requests: List of Request objects that are currently active
            validate_chunk_func: Optional function to validate received chunks
        """
        if self.stage_id == 0:
            return

        for request in active_requests:
            req_id = request.request_id

            # This means it is in prefill-mode and it should only receive the first chunk from orchestrator.
            if len(request.prompt_token_ids) > request.num_computed_tokens:
                continue

            # Skip if upstream already finished for this request
            if self.track_if_upstream_finished.get(req_id, False):
                request.status = RequestStatus.RUNNING
                continue

            chunk = self._retrieve_chunk_from_connector(req_id, validate_chunk_func, retry_for_first_chunk=False)

            if chunk:
                # Store chunk on Request object for next scheduling step
                request.pending_chunk = chunk
                request.status = RequestStatus.RUNNING
                if chunk.get("last_chunk", False):
                    self.track_if_upstream_finished[req_id] = True
            else:
                request.status = RequestStatus.WAITING_FOR_CHUNK

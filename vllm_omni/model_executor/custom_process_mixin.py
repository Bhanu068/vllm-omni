from collections.abc import Callable
from typing import Any

import torch


class CustomProcessMixin:
    """
    Mixin class for all stages in the Omni model.
    """

    def set_custom_preprocess(self, preprocess_fn: Callable) -> None:
        """
        Set a preprocess function for the stage.
        Args:
            preprocess_fn: The preprocess function to register.
        """
        self.preprocess = preprocess_fn

    def set_custom_postprocess(self, postprocess_fn: Callable) -> None:
        """
        Set a postprocess function for the stage.
        Args:
            postprocess_fn: The postprocess function to register.
        """
        self.postprocess = postprocess_fn

    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **input_dict: object
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Process the input_ids and input_embeds for the given input_dict.
        Returns the processed input_ids, input_embeds, and the input_dict.
        If the stage don't applicable, return the original input_ids, input_embeds, and an empty dict.
        """
        raise NotImplementedError("Preprocess is not implemented for this stage.")

    def postprocess(self, model_output, **info_dict: object):
        """
        Postprocess the model output.
        Returns the postprocessed model output and the save dictionary.
        Args:
            model_output: The model output to postprocess.
        """
        raise NotImplementedError("Postprocess is not implemented for this stage.")


class StreamingChunkProcessorMixin:
    """Mixin for models that support processing streaming chunks from upstream stages.

    This allows model-specific chunk transformation logic (e.g., transforming
    thinker_result into internal queue format) to be encapsulated in the model
    rather than in the scheduler or model runner.
    """

    supports_streaming_chunks: bool = False

    def process_streaming_chunk(
        self, chunk_payload: dict[str, Any], existing_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Process an incoming streaming chunk and return updated state.

        This method is called by the model runner when pending chunks are available
        Models should implement this to transform raw chunk payloads (e.g., thinker_result)
        into their internal queue format (e.g., thinker_reply_part).

        Args:
            chunk_payload: Raw chunk from upstream stage. May contain keys like:
                - 'thinker_result': Tensor of embeddings from upstream
                - 'upstream_finished': Whether upstream has finished streaming
            existing_state: Current state dict (may contain accumulated queue).
                            None if this is the first chunk.

        Returns:
            Dictionary with updated state. Expected keys:
                - Keys to merge into additional_information_cpu (e.g., 'thinker_reply_part')
                - 'upstream_finished': Boolean indicating if upstream finished

        Raises:
            NotImplementedError: If model declares supports_streaming_chunks=True
                                 but doesn't implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} declares supports_streaming_chunks=True "
            "but doesn't implement process_streaming_chunk"
        )

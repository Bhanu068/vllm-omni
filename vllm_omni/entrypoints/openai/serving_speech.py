import asyncio
import base64
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Optional

from fastapi import Request
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.protocol import CreateSpeechRequest, OpenAIChatCompletionAudio
from vllm_omni.outputs import OmniRequestOutput

try:
    import soundfile
except ImportError:
    soundfile = None

logger = init_logger(__name__)


class OmniOpenAIServingSpeech(OpenAIServing):
    async def create_speech(
        self,
        request: CreateSpeechRequest,
        raw_request: Optional[Request] = None,
    ):
        """
        Create Speech API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createSpeech
        for the API specification. This API mimics the OpenAI
        Create Speech API.
        """

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = f"speech-{random_uuid()}"

        prompt = {"prompt": request.input}

        sampling_params_list = self.engine_client.default_sampling_params_list

        try:
            generator = self.engine_client.generate(
                prompt=prompt, request_id=request_id, sampling_params_list=sampling_params_list
            )

            try:
                final_output: Optional[OmniRequestOutput] = None
                async for res in generator:
                    final_output = res
            except asyncio.CancelledError:
                return self.create_error_response("Client disconnected")
            except ValueError as e:
                # TODO: Use a vllm-specific Validation Error
                return self.create_error_response(str(e))

            if final_output is None:
                return self.create_error_response("No output generated from the model.")

            if final_output.final_output_type != "audio":
                return self.create_error_response(f"Unexpected final output type: {final_output.final_output_type}")

            request_output = final_output.request_output

            completion = request_output.outputs[0]
            audio_obj = self._create_audio_choice(final_output, "assistant")
        except Exception as e:
            return self.create_error_response(str(e))

    def _create_audio_choice(self, omni_outputs: OmniRequestOutput, role: str):
        final_res = omni_outputs.request_output
        if not final_res.outputs:
            return self.create_error_response("Empty output from the model.")

        audio_tensor = final_res.multimodal_output["audio"].float().detach().cpu().numpy()

        # Convert numpy array to WAV bytes and encode as base64
        if soundfile is None:
            raise ImportError(
                "soundfile is required for audio generation. Please install it with: pip install soundfile"
            )

        # Default sample rate for TTS models (typically 24000 Hz)
        # You may need to adjust this based on your model's configuration
        sample_rate = 24000

        # Ensure audio is 1D (flatten if needed)
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.flatten()

        # Convert to WAV format and encode as base64
        with BytesIO() as buffer:
            soundfile.write(buffer, audio_tensor, sample_rate, format="WAV")
            wav_bytes = buffer.getvalue()

        audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")

        # Generate unique ID for the audio
        audio_id = f"audio-{uuid.uuid4().hex[:16]}"

        # Set expiration time (e.g., 24 hours from now) as Unix timestamp
        expires_at = int((datetime.now(timezone.utc) + timedelta(hours=24)).timestamp())

        # Create OpenAIChatCompletionAudio object with all required fields
        audio_obj = OpenAIChatCompletionAudio(
            id=audio_id,
            data=audio_base64,
            expires_at=expires_at,
            transcript="",  # Empty transcript if not available
        )

        return audio_obj

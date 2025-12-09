from typing import Literal

from pydantic import BaseModel, Field


class CreateSpeechRequest(BaseModel):
    input: str
    model: str | None = None
    voice: Literal["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
    instructions: str | None = None
    response_format: str
    speed: float | None = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
    )
    stream_format: Literal["sse", "audio"] | None = "audio"


class OpenAIChatCompletionAudio(BaseModel):
    id: str
    data: str
    expires_at: int
    transcript: str

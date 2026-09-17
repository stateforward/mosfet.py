"""Provider-neutral, time-bounded raw PCM voice segments."""

import math
import typing

import pydantic


class VoiceSegment(pydantic.BaseModel):
    """One clipped raw PCM voice segment with its source time span."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [
                {
                    "audio": "AAA=",
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 4,
                    "channels": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 0.25,
                }
            ],
        },
    )

    audio: bytes = pydantic.Field(
        min_length=1,
        description=(
            "Raw signed 16-bit PCM audio bytes clipped to this voice segment. "
            "JSON callers provide base64-encoded bytes."
        ),
        examples=[b"voice audio"],
    )
    media_type: typing.Literal["audio/pcm"] = pydantic.Field(
        description="Media type of the raw PCM bytes in this voice segment.",
        examples=["audio/pcm"],
    )
    sample_rate_hz: int = pydantic.Field(
        ge=1,
        description="Sample rate of the raw PCM bytes in this voice segment, in hertz.",
        examples=[48000],
    )
    channels: int = pydantic.Field(
        ge=1,
        description="Channel count of the interleaved raw PCM bytes in this voice segment.",
        examples=[1],
    )
    start_seconds: float = pydantic.Field(
        ge=0.0,
        description="Inclusive start time of this voice segment, measured from the beginning of the source audio.",
        examples=[0.0],
    )
    end_seconds: float = pydantic.Field(
        ge=0.0,
        description="Exclusive end time of this voice segment, measured from the beginning of the source audio.",
        examples=[0.25],
    )

    @pydantic.model_validator(mode="after")
    def validate_time_span(self) -> typing.Self:
        """Require each segment to cover a positive time span."""

        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds.")
        return self

    @pydantic.model_validator(mode="after")
    def validate_audio_duration(self) -> typing.Self:
        """Require clipped audio to contain PCM frames matching its time span."""

        frame_bytes = 2 * self.channels
        if len(self.audio) % frame_bytes:
            raise ValueError("audio must contain complete 16-bit PCM frames for its channel count.")

        frame_count = len(self.audio) // frame_bytes
        expected_frame_count = (self.end_seconds - self.start_seconds) * self.sample_rate_hz
        if not math.isclose(frame_count, expected_frame_count, abs_tol=1.0):
            raise ValueError("audio duration must agree with end_seconds-start_seconds within one sample frame.")
        return self

    @property
    def duration_seconds(self) -> float:
        """Duration represented by the clipped raw PCM bytes."""

        return len(self.audio) / float(2 * self.channels * self.sample_rate_hz)


__all__ = ["VoiceSegment"]

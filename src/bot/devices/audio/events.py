import collections.abc
import dataclasses
import typing

import hsm
import pydantic

TData = typing.TypeVar("TData")


class AudioFrameData(pydantic.BaseModel):
    """Generic chunk of audio carried between devices or device implementations."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [
                {
                    "audio": "YXVkaW8tY2h1bms=",
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 48000,
                    "channels": 1,
                },
                {
                    "audio": "b3B1cy1wYWNrZXQ=",
                    "media_type": "audio/opus",
                },
            ],
        },
    )

    audio: bytes = pydantic.Field(
        min_length=1,
        description=(
            "Raw or encoded audio bytes for a single device audio chunk. JSON callers must provide this field "
            "as base64-encoded bytes."
        ),
        examples=["YXVkaW8tY2h1bms="],
    )
    media_type: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Optional media type or codec label for the audio bytes, such as audio/pcm or audio/opus. "
            "Provider-specific transport details should stay in provider code."
        ),
        examples=["audio/pcm", "audio/opus"],
    )
    sample_rate_hz: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Optional sample rate for raw or decoded audio, measured in hertz.",
        examples=[48000],
    )
    channels: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Optional number of audio channels represented by the chunk.",
        examples=[1, 2],
    )


class AudioInputData(AudioFrameData):
    """Audio captured by an input peripheral and dispatched to another device."""


class AudioOutputData(AudioFrameData):
    """Audio requested for output and dispatched to an audio output implementation."""


InputEvent = hsm.Event[AudioInputData](
    name="devices.audio.input",
    schema=AudioInputData,
)
OutputEvent = hsm.Event[AudioOutputData](
    name="devices.audio.output",
    schema=AudioOutputData,
)


def _instance_event_endpoint(instance: hsm.Instance) -> str:
    return hsm.id(instance)


def routed_audio_event(
    event: hsm.Event[TData],
    data: TData,
    *,
    source: hsm.Instance,
    target: hsm.Instance,
    metadata: collections.abc.Mapping[str, object] | None = None,
) -> hsm.Event[TData]:
    """Return an immutable audio event addressed from one HSM instance to another."""

    return dataclasses.replace(
        event.with_data(data),
        source=_instance_event_endpoint(source),
        target=_instance_event_endpoint(target),
        metadata=dict(metadata) if metadata is not None else {},
    )

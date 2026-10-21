import collections.abc
import dataclasses
import typing

import hsm
import pydantic

TData = typing.TypeVar("TData")


class FrameData(pydantic.BaseModel):
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


class InputData(FrameData):
    """Audio captured by an input peripheral and dispatched to another device."""


class OutputData(FrameData):
    """Audio requested for output and dispatched to an audio output implementation."""

    mark: str | None = pydantic.Field(
        default=None,
        min_length=1,
        max_length=64,
        description=(
            "Optional opaque label the commanding controller puts on this chunk to learn when it has been "
            "played: the speaker answers devices.audio.played with the same mark once the chunk's last frame "
            "has finished playing, or lists it in devices.audio.flushed if it was dropped unplayed. It is "
            "never encoded on a wire."
        ),
        examples=["u1.3"],
    )


class FlushData(pydantic.BaseModel):
    """Controller command to a speaker: drop everything this controller commanded that has not played yet."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{}]}
    )


class PlayedData(pydantic.BaseModel):
    """Speaker report to the commanding controller: the chunk that carried ``mark`` has finished playing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{"mark": "u1.3"}]}
    )

    mark: str = pydantic.Field(
        min_length=1,
        max_length=64,
        description="The mark of the devices.audio.output chunk whose last frame has just finished playing.",
        examples=["u1.3"],
    )


class FlushedData(pydantic.BaseModel):
    """Speaker report to a controller: audio it commanded was dropped unplayed, by a flush or by overflow.

    A flush can cut a chunk short: ``played_seconds`` is how much of the first dropped chunk had played by then.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"dropped_seconds": 0.4, "marks": ["u1.4", "u1.5"], "played_seconds": 0.0},
                {"dropped_seconds": 0.3, "marks": ["u1.4"], "played_seconds": 0.2},
            ]
        },
    )

    dropped_seconds: float = pydantic.Field(
        ge=0.0,
        description="Playout time, in seconds, of this controller's audio that was dropped without playing.",
        examples=[0.4],
    )
    marks: tuple[str, ...] = pydantic.Field(
        description="Marks of the dropped chunks, in the order they would have played; empty when none carried one.",
        examples=[["u1.4", "u1.5"]],
    )
    played_seconds: float = pydantic.Field(
        default=0.0,
        ge=0.0,
        description=(
            "Playout time, in seconds, of the chunk whose mark is first in marks that played before it was cut "
            "short, counting the frame that was playing when the flush came, which finishes; 0.0 when none of it "
            "played (always for an overflow, which drops a chunk whole before it starts)."
        ),
        examples=[0.0, 0.2],
    )


InputEvent = hsm.Event[InputData](
    name="devices.audio.input",
    schema=InputData,
)
OutputEvent = hsm.Event[OutputData](
    name="devices.audio.output",
    schema=OutputData,
)
FlushEvent = hsm.Event[FlushData](
    name="devices.audio.flush",
    schema=FlushData,
)
PlayedEvent = hsm.Event[PlayedData](
    name="devices.audio.played",
    schema=PlayedData,
)
FlushedEvent = hsm.Event[FlushedData](
    name="devices.audio.flushed",
    schema=FlushedData,
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

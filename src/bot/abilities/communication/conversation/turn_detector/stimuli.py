"""Typed stimuli accepted by participant-owned turn abilities."""

from __future__ import annotations

import typing

import pydantic

from bot.abilities.identity import value


class AudioStimulus(pydantic.BaseModel):
    """Audio content produced by a participant."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "examples": [
                {
                    "kind": "audio",
                    "source_participant_ref": "caller",
                    "content": "YXVkaW8=",
                    "sample_rate_hz": 16_000,
                    "channels": 1,
                }
            ]
        },
    )

    kind: typing.Literal["audio"] = pydantic.Field(default="audio")
    source_participant_ref: value.IdentityRef = pydantic.Field(description="Participant that produced the audio.")
    content: bytes = pydantic.Field(description="Raw audio bytes (PCM or a self-describing container such as WAV).")
    sample_rate_hz: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Sample rate of raw PCM content in hertz. Required for correct STT packaging when content is not a WAV container.",
        examples=[16_000, 48_000],
    )
    channels: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Channel count of raw PCM content. Defaults to mono when omitted at the decoder.",
        examples=[1],
    )


class TextStimulus(pydantic.BaseModel):
    """Text content produced by a participant."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    kind: typing.Literal["text"] = pydantic.Field(default="text")
    source_participant_ref: value.IdentityRef = pydantic.Field(description="Participant that produced the text.")
    content: str = pydantic.Field(min_length=1, description="Readable text content.")


class ImageStimulus(pydantic.BaseModel):
    """Image content produced by a participant."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    kind: typing.Literal["image"] = pydantic.Field(default="image")
    source_participant_ref: value.IdentityRef = pydantic.Field(description="Participant that produced the image.")
    content: bytes = pydantic.Field(description="Image bytes.")


def _empty_payload() -> dict[str, object]:
    return {}


class EventStimulus(pydantic.BaseModel):
    """Structured event content produced by a participant or service."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    kind: typing.Literal["event"] = pydantic.Field(default="event")
    source_participant_ref: value.IdentityRef = pydantic.Field(description="Event source participant.")
    event: str = pydantic.Field(min_length=1, description="Stable modeled event name.")
    payload: dict[str, object] = pydantic.Field(default_factory=_empty_payload, description="Structured event payload.")


class ContentStimulus(pydantic.BaseModel):
    """Modality-neutral content carried through a participant turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    kind: typing.Literal["content"] = pydantic.Field(default="content")
    source_participant_ref: value.IdentityRef = pydantic.Field(description="Participant that produced the content.")
    content: object = pydantic.Field(
        description="Uninterpreted content payload.",
    )
    content_type: str = pydantic.Field(min_length=1, description="Content modality or media type.")


ParticipationStimulus: typing.TypeAlias = typing.Annotated[
    AudioStimulus | TextStimulus | ImageStimulus | EventStimulus | ContentStimulus,
    pydantic.Field(discriminator="kind"),
]


__all__ = [
    "AudioStimulus",
    "ContentStimulus",
    "EventStimulus",
    "ImageStimulus",
    "ParticipationStimulus",
    "TextStimulus",
]

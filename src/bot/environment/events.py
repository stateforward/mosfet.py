"""Environment-facing input event contracts (stimuli for bot input abilities).

These are bot-facing acoustic and visual stimuli, not device playout/capture
primitives (`devices.audio.*`). Speech and other interpretations are produced
by bot input abilities (hearing/listening/vision), not encoded in these events.
"""

from __future__ import annotations

import typing

import hsm
import pydantic


class SoundData(pydantic.BaseModel):
    """Acoustic energy available to a bot's input (hearing) abilities.

    Domain-specific elevation may use a :class:`SoundData` subclass with extra
    typed fields (for example phone ring elevation carries ``caller``) so models
    copy from ``event.data`` instead of inferring from ``event.id``.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": (
                "Universal sound stimulus. Not speech-specific: voice detection and decoding "
                "belong to hearing/listening abilities."
            ),
            "examples": [
                {
                    "audio": "YXVkaW8tY2h1bms=",
                    "media_type": "audio/pcm",
                    "sample_rate_hz": 48000,
                    "channels": 1,
                    "kind": "phone.call",
                }
            ],
        },
    )

    audio: bytes = pydantic.Field(
        min_length=1,
        description=(
            "Raw or encoded audio bytes for this acoustic chunk. JSON callers must provide this "
            "field as base64-encoded bytes."
        ),
        examples=["YXVkaW8tY2h1bms="],
    )
    media_type: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional media type or codec label, such as audio/pcm or audio/opus.",
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
    kind: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Optional low-cardinality acoustic provenance hint. Prefer domain-qualified labels "
            "when the source is clear (for example phone.ringing, phone.call, ambient, knock) so "
            "models do not confuse bare tokens like ring. Not a linguistic interpretation."
        ),
        examples=["phone.ringing", "phone.call", "ambient", "knock"],
    )
    amplitude_db: float | None = pydantic.Field(
        default=None,
        description=(
            "Optional loudness of this sound at its source, in dB SPL measured one metre away. A "
            "shout and a whisper leave the same mouth, so loudness belongs to the sound rather "
            "than to the thing that made it. How far it carries is the environment's to work out; "
            "omitting this means the sound reaches every listener regardless of distance."
        ),
        examples=[60.0, 25.0],
    )
    received_level_db: float | None = pydantic.Field(
        default=None,
        description=(
            "Optional loudness of this sound in dB SPL where the recipient of this event is, "
            "which is a different quantity from amplitude_db: the same shout is one level at the "
            "mouth and another at the far side of the room. The environment fills this in per "
            "recipient, so an emitter never sets it. None means the environment had nothing to "
            "measure — the sound carried no amplitude, or nobody said where it or the listener "
            "was — not that the sound was silent."
        ),
        examples=[54.0, 6.0],
    )


class VisualData(pydantic.BaseModel):
    """Visual energy available to a bot's input (vision) abilities."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": "Universal visual stimulus for bot input vision abilities.",
            "examples": [
                {
                    "image": "aW1hZ2UtYnl0ZXM=",
                    "media_type": "image/png",
                }
            ],
        },
    )

    image: bytes = pydantic.Field(
        min_length=1,
        description=(
            "Raw or encoded image bytes for this visual frame. JSON callers must provide this "
            "field as base64-encoded bytes."
        ),
        examples=["aW1hZ2UtYnl0ZXM="],
    )
    media_type: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional media type for the image bytes, such as image/png or image/jpeg.",
        examples=["image/png", "image/jpeg"],
    )
    kind: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional low-cardinality visual provenance hint (for example camera, screen).",
        examples=["camera", "screen"],
    )


SoundEvent = hsm.Event[SoundData](
    name="environment.sound",
    schema=SoundData,
)
VisualEvent = hsm.Event[VisualData](
    name="environment.visual",
    schema=VisualData,
)

__all__ = [
    "SoundData",
    "VisualData",
    "SoundEvent",
    "VisualEvent",
]

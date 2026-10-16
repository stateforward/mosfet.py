"""Own-voice enrollment: Speaking hands Listening a copy of the bot's own voice to learn it as its own.

This is the motor side of self-recognition. The mouth knows exactly what it said, so Speaking sends the PCM it
commanded, and the listening child that owns voice identification learns that voiceprint as the bot's own. It is
not a scripted calibration phrase: it is the first real utterance's signal. The copy names no voice id until the
bot has learned one: identification then matches the nearest own voice or mints one, and the enroller is told
which id its own voice has with ``EnrolledEvent``.
"""

from __future__ import annotations

import typing

import hsm
import pydantic

from mosfet.abilities import identification
from mosfet.abilities.hearing import voice


class EnrollData(pydantic.BaseModel):
    """A copy of the bot's own voice, as its mouth commanded it, with the voice id it has learned, if any."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": (
                "Speaking's motor-side copy of the bot's own voice: the PCM its mouth was commanded to play and, "
                "once learned, the voice id it has, so hearing can learn that voice as the bot's own."
            ),
        },
    )

    voice_id: identification.Id | None = pydantic.Field(
        default=None,
        description=(
            "The bot's own voice id once it has learned one; null until then, so identification matches the nearest "
            "own voice or mints one."
        ),
        examples=[None, "3f2a9c41d07e4b6a9c2f1e8d5b7a0c63"],
    )
    segment: voice.VoiceSegment = pydantic.Field(
        description="The commanded PCM of the bot's own voice, at least 1.5 seconds long.",
    )


class EnrolledData(pydantic.BaseModel):
    """Which id the bot's own voice has, now that hearing has learned it from the enroller's copy."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "The id identification gave the bot's own voice: the one declared, matched, or minted now.",
            "examples": [{"voice_id": "3f2a9c41d07e4b6a9c2f1e8d5b7a0c63"}],
        },
    )

    voice_id: identification.Id = pydantic.Field(
        description="The bot's own voice id; heard voices carry it as voice:<voice_id>.",
        examples=["3f2a9c41d07e4b6a9c2f1e8d5b7a0c63"],
    )


EnrollEvent = hsm.Event[EnrollData](
    name="bot.ability.listening.enroll",
    schema=EnrollData,
)
"""Speaking → Listening: learn this voice as the bot's own. A nerve, never a model-offerable selection.

Listening learns from it only where voice identification is composed to learn voices; elsewhere topology ignores it.
"""

EnrolledEvent = hsm.Event[EnrolledData](
    name="bot.ability.listening.enrolled",
    kind=hsm.CompletionEventKind,
    schema=EnrolledData,
)
"""Listening → the enroller: the copy was learned as the bot's own voice under this id.

Sent to the ``EnrollEvent``'s source under its id, once enrollment succeeded; a failed enrollment tells nobody.
"""


__all__ = ["EnrollData", "EnrollEvent", "EnrolledData", "EnrolledEvent"]

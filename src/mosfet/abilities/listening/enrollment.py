"""Own-voice enrollment: Speaking hands Listening a copy of the bot's own voice to learn it as its own.

This is the motor side of self-recognition. The mouth knows exactly what it said, so Speaking sends the PCM it
commanded, with the voice id it speaks as, and the listening child that owns voice identification learns that
voiceprint as the bot's own. It is not a scripted calibration phrase: it is the first real utterance's signal.
"""

from __future__ import annotations

import typing

import hsm
import pydantic

from mosfet.abilities import identification
from mosfet.abilities.hearing import voice


class EnrollData(pydantic.BaseModel):
    """A copy of the bot's own voice, as its mouth commanded it, with the voice id it speaks as."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": (
                "Speaking's motor-side copy of the bot's own voice: the PCM its mouth was commanded to play and the "
                "voice id it speaks as, so hearing can learn that voice as the bot's own."
            ),
        },
    )

    voice_id: identification.Id = pydantic.Field(
        description="The bot's own voice id (Speaking's voice_id), the id the learned voiceprint is enrolled under.",
        examples=["tts:warm"],
    )
    segment: voice.VoiceSegment = pydantic.Field(
        description="The commanded PCM of the bot's own voice, at least 1.5 seconds long.",
    )


EnrollEvent = hsm.Event[EnrollData](
    name="bot.ability.listening.enroll",
    schema=EnrollData,
)
"""Speaking → Listening: learn this voice as the bot's own. A nerve, never a model-offerable selection.

Listening learns from it only where voice identification is composed to learn voices; elsewhere topology ignores it.
"""


__all__ = ["EnrollData", "EnrollEvent"]

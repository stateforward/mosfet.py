"""Overlap: another voice heard while the bot's own mouth is producing.

This is the barge-in *fact*, never the barge-in decision. Listening states it once per overlapping
utterance and hands it to cognition like any other perception product; whether to stop, keep
talking or acknowledge is the bot's choice (``communication.stop`` is the capacity). Nothing in
topology stops speech because someone spoke.

A product is made only while the efference window is open (``sensitivity.OutputData.producing``)
and only for a sound above the product threshold, so the bot's own voice arriving at its predicted
level is never an overlap. A voice recognized as the bot's own makes no overlap fact either: that is
recognition of who spoke, not a choice.

Telemetry: ``bot.listening.overlap.count`` counts the facts made, by ``bot.listening.overlap.path``
(``local`` from Listening's own voice detection, ``stream`` from a streaming transcription session).
"""

from __future__ import annotations

import typing

import hsm
import mosfet
import pydantic

from ..hearing import voice

_MAX_VOICES = 8
_MAX_PARTICIPANTS = 32


class OverlapData(pydantic.BaseModel):
    """Another voice has been going while the bot's own mouth was producing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A perception fact: while the bot was speaking, another voice was heard and kept going. It says "
                "how long that voice had been going, the words heard so far when the hearing path has them, who "
                "the voice is when it was identified, and who produced the sound. It is not a request to stop; "
                "what to do about it is the bot's choice."
            ),
            "examples": [
                {
                    "voiced_seconds": 0.6,
                    "text": "wait, hold on",
                    "voices": [
                        {
                            "recognition": {"id": "3f2a9c", "recognized": True, "similarity": 0.82, "own": False},
                            "start_seconds": 0.0,
                            "end_seconds": 0.6,
                        }
                    ],
                    "participants": [{"ref": "voice:3f2a9c", "role": "source"}],
                },
                {"voiced_seconds": 0.5, "text": None, "voices": [], "participants": []},
            ],
        },
    )

    voiced_seconds: float = pydantic.Field(
        ge=0.0,
        description="How long, in seconds, the other voice had been going when this fact was made.",
        examples=[0.6],
    )
    text: str | None = pydantic.Field(
        default=None,
        description=(
            "The words heard from the other voice so far, when the hearing path transcribes as it listens; None "
            "when it does not (the words then arrive later, as speech)."
        ),
        examples=["wait, hold on", None],
    )
    voices: tuple[voice.HeardVoice, ...] = pydantic.Field(
        default=(),
        max_length=_MAX_VOICES,
        description=(
            "The overlapping voices that were identified: each voice id, whether it was recognized (heard before) "
            "or is new. Never the bot's own voice. Empty when no voice was identified."
        ),
    )
    participants: tuple[mosfet.Participant, ...] = pydantic.Field(
        default=(),
        max_length=_MAX_PARTICIPANTS,
        description=(
            "Who produced the overlapping sound: what the device stamped on it, then a voice:<id> source for every "
            "identified voice."
        ),
        examples=[[], [{"ref": "voice:3f2a9c", "role": "source"}]],
    )


OverlapEvent = hsm.Event[OverlapData](
    name="bot.ability.listening.overlap",
    schema=OverlapData,
)
"""Listening → cognition stimulus: another voice while the bot speaks. A perception fact, never a selection."""


__all__ = ["OverlapData", "OverlapEvent"]

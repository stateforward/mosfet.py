"""Typed turn-boundary events owned by a turn detector.

The events are the participant's ingress contract.  ``TurnDetector`` owns the
idle/open/paused lifecycle and emits ``TurnCompleteEvent`` when a turn ends.
"""

from __future__ import annotations

import typing

import hsm
import pydantic

from mosfet.abilities.identity import value
from . import stimuli

_Content = stimuli.ParticipationStimulus | None


def _validate_content_source(
    content: _Content,
    source_participant_ref: value.IdentityValue,
) -> None:
    if content is not None and content.source_participant_ref != source_participant_ref:
        raise ValueError("content source_participant_ref must match boundary source_participant_ref.")


class TurnStartData(pydantic.BaseModel):
    """A conversational turn opened by a source participant."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": "Turn boundary carrying an optional first text or audio chunk.",
            "examples": [
                {
                    "conversation_ref": "support-call",
                    "turn_ref": "turn-1",
                    "self_participant_ref": "bot",
                    "source_participant_ref": "caller",
                    "content": {"kind": "text", "source_participant_ref": "caller", "content": "hello"},
                }
            ],
        },
    )

    conversation_ref: str = pydantic.Field(default="conversation", min_length=1, examples=["support-call"])
    turn_ref: str = pydantic.Field(min_length=1, description="Stable reference for this active turn.")
    self_participant_ref: value.IdentityRef = pydantic.Field(default="bot", examples=["bot"])
    source_participant_ref: value.IdentityRef = pydantic.Field(default="caller", examples=["caller"])
    content: _Content = pydantic.Field(default=None, description="Optional first turn chunk.")

    @pydantic.model_validator(mode="after")
    def validate_content_source(self) -> typing.Self:
        _validate_content_source(self.content, self.source_participant_ref)
        return self


class TurnUpdateData(pydantic.BaseModel):
    """Additional text or audio for an open turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": "Incremental turn content accepted while a participant is open.",
            "examples": [
                {
                    "conversation_ref": "support-call",
                    "turn_ref": "turn-1",
                    "source_participant_ref": "caller",
                    "content": {"kind": "text", "source_participant_ref": "caller", "content": "world"},
                }
            ],
        },
    )

    conversation_ref: str = pydantic.Field(default="conversation", min_length=1, examples=["support-call"])
    turn_ref: str = pydantic.Field(min_length=1, description="Stable reference for the active turn.")
    source_participant_ref: value.IdentityRef = pydantic.Field(default="caller", examples=["caller"])
    content: stimuli.TextStimulus | stimuli.AudioStimulus = pydantic.Field(
        description="Incremental text or audio chunk."
    )

    @pydantic.model_validator(mode="after")
    def validate_content_source(self) -> typing.Self:
        _validate_content_source(self.content, self.source_participant_ref)
        return self


class TurnPauseData(pydantic.BaseModel):
    """A pause that keeps the current turn open."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Turn pause observation; it does not complete or discard the current turn.",
            "examples": [
                {
                    "conversation_ref": "support-call",
                    "turn_ref": "turn-1",
                    "source_participant_ref": "caller",
                    "duration_seconds": 0.2,
                }
            ],
        },
    )

    conversation_ref: str = pydantic.Field(default="conversation", min_length=1, examples=["support-call"])
    turn_ref: str = pydantic.Field(min_length=1, description="Stable reference for the active turn.")
    source_participant_ref: value.IdentityRef = pydantic.Field(default="caller", examples=["caller"])
    duration_seconds: float = pydantic.Field(default=0.0, ge=0.0, examples=[0.2])


class TurnEndData(pydantic.BaseModel):
    """A conversational turn closed by its source participant."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": "Turn boundary carrying an optional final text or audio chunk.",
            "examples": [
                {"conversation_ref": "support-call", "turn_ref": "turn-1", "source_participant_ref": "caller"}
            ],
        },
    )

    conversation_ref: str = pydantic.Field(default="conversation", min_length=1, examples=["support-call"])
    turn_ref: str = pydantic.Field(min_length=1, description="Stable reference for the active turn.")
    source_participant_ref: value.IdentityRef = pydantic.Field(default="caller", examples=["caller"])
    content: _Content = pydantic.Field(default=None, description="Optional final turn chunk.")

    @pydantic.model_validator(mode="after")
    def validate_content_source(self) -> typing.Self:
        _validate_content_source(self.content, self.source_participant_ref)
        return self


TurnStartEvent = hsm.Event[TurnStartData](
    name="bot.ability.turn_detector.turn.start",
    schema=TurnStartData,
)
TurnUpdateEvent = hsm.Event[TurnUpdateData](
    name="bot.ability.turn_detector.turn.update",
    schema=TurnUpdateData,
)
TurnPauseEvent = hsm.Event[TurnPauseData](
    name="bot.ability.turn_detector.turn.pause",
    schema=TurnPauseData,
)
TurnEndEvent = hsm.Event[TurnEndData](
    name="bot.ability.turn_detector.turn.end",
    schema=TurnEndData,
)

__all__ = [
    "TurnEndData",
    "TurnEndEvent",
    "TurnPauseData",
    "TurnPauseEvent",
    "TurnStartData",
    "TurnStartEvent",
    "TurnUpdateData",
    "TurnUpdateEvent",
]

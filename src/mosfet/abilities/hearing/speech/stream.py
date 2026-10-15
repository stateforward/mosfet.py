"""The provider-neutral streaming speech-to-text session contract.

A session is a provider HSM that is also an ``attachment.Attachment``: one controller attaches to it
(``attachment.attach``), feeds it audio from one sound source, and receives what it hears; detaching
closes it. While attached it keeps itself connected to whatever serves it (reconnecting with its own
backoff), so a controller never manages the connection.

In (controller → session):

* :data:`AudioEvent`: PCM heard from the source, in whatever format the source delivered it;
* :data:`ProducingEvent`: whether the bot's own mouth is producing now (sent on change), so the
  serving model can tell the bot's own speech from someone else's.

Out (session → its controller, ``source`` the session's id):

* :data:`ReadyEvent` once the session can take audio;
* :data:`VoiceStartedEvent` / :data:`VoiceEndedEvent` as the hearing model hears a voice start and stop;
* :data:`PartialEvent` with the words of the current utterance so far (cumulative);
* :data:`TranscriptEvent` once per finalized utterance;
* :data:`TurnEndedEvent`: the hearing model's own end-of-turn perception;
* :data:`OverlapEvent`: another voice has been sustained while the bot was producing;
* :data:`FailedEvent` when the session lost what serves it, in existing ``FAILURE_KINDS.md`` strings.

Every time is in seconds of the audio this session has received. Provider-specific reasons (why a
turn ended, word timings) are not part of the contract.
"""

from __future__ import annotations

import typing

import hsm
import pydantic

FailureKind: typing.TypeAlias = typing.Literal[
    "unavailable", "overloaded", "invalid_input", "invalid_response", "timeout"
]
"""Why a session lost what serves it: unreachable or dropped (``unavailable``), full (``overloaded``), refused the
stream's parameters (``invalid_input``), spoke something outside its protocol (``invalid_response``), or did not
answer in time (``timeout``)."""

_FROZEN: pydantic.ConfigDict = pydantic.ConfigDict(frozen=True, extra="forbid")


class AudioData(pydantic.BaseModel):
    """PCM heard from the session's sound source."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={"description": "Signed 16-bit little-endian PCM heard from the session's sound source."},
    )

    audio: bytes = pydantic.Field(description="Interleaved s16le PCM frames.")
    sample_rate_hz: int = pydantic.Field(ge=1, description="Sample rate of audio in hertz.", examples=[48_000])
    channels: int = pydantic.Field(ge=1, description="Channel count of audio.", examples=[1])


class ProducingData(pydantic.BaseModel):
    """Whether the bot's own mouth is producing sound now."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    producing: bool = pydantic.Field(
        description="True while the bot's own mouth is producing sound (its efference window is open).",
        examples=[True, False],
    )


class ReadyData(pydantic.BaseModel):
    """The session can take audio."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN


class VoiceStartedData(pydantic.BaseModel):
    """The hearing model heard a voice start."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    at_seconds: float = pydantic.Field(ge=0.0, description="When the voice started, in session seconds.")


class VoiceEndedData(pydantic.BaseModel):
    """The hearing model heard a voice stop."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    at_seconds: float = pydantic.Field(ge=0.0, description="When the voice stopped, in session seconds.")


class PartialData(pydantic.BaseModel):
    """The words of the utterance being heard, so far."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    text: str = pydantic.Field(description="The words heard so far in the current utterance (cumulative).")


class TranscriptData(pydantic.BaseModel):
    """One finalized utterance."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    text: str = pydantic.Field(description="The finalized words of the utterance.")
    confidence: float | None = pydantic.Field(
        default=None, ge=0.0, le=1.0, description="The hearing model's confidence in the words, when it gives one."
    )
    start_seconds: float = pydantic.Field(ge=0.0, description="Start of the utterance, in session seconds.")
    end_seconds: float = pydantic.Field(ge=0.0, description="End of the utterance, in session seconds.")


class TurnEndedData(pydantic.BaseModel):
    """The hearing model perceives that the speaker's turn has ended."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    at_seconds: float = pydantic.Field(ge=0.0, description="When the turn ended, in session seconds.")
    probability: float | None = pydantic.Field(
        default=None, ge=0.0, le=1.0, description="How sure the hearing model is the turn ended, when it says."
    )
    text: str | None = pydantic.Field(
        default=None, description="The words of the whole turn, when the hearing model gives them."
    )


class OverlapData(pydantic.BaseModel):
    """Another voice has been sustained while the bot was producing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    at_seconds: float = pydantic.Field(ge=0.0, description="When the overlap was perceived, in session seconds.")
    voiced_seconds: float = pydantic.Field(ge=0.0, description="How long the other voice had been going by then.")


class FailedData(pydantic.BaseModel):
    """The session lost what serves it; it recovers on its own while attached."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    kind: FailureKind = pydantic.Field(
        description="Why, in existing FAILURE_KINDS.md strings.", examples=["unavailable"]
    )


AudioEvent = hsm.Event[AudioData](name="bot.ability.hearing.speech.stream.audio", schema=AudioData)
ProducingEvent = hsm.Event[ProducingData](name="bot.ability.hearing.speech.stream.producing", schema=ProducingData)
ReadyEvent = hsm.Event[ReadyData](name="bot.ability.hearing.speech.stream.ready", schema=ReadyData)
VoiceStartedEvent = hsm.Event[VoiceStartedData](
    name="bot.ability.hearing.speech.stream.voice_started", schema=VoiceStartedData
)
VoiceEndedEvent = hsm.Event[VoiceEndedData](name="bot.ability.hearing.speech.stream.voice_ended", schema=VoiceEndedData)
PartialEvent = hsm.Event[PartialData](name="bot.ability.hearing.speech.stream.partial", schema=PartialData)
TranscriptEvent = hsm.Event[TranscriptData](name="bot.ability.hearing.speech.stream.transcript", schema=TranscriptData)
TurnEndedEvent = hsm.Event[TurnEndedData](name="bot.ability.hearing.speech.stream.turn_ended", schema=TurnEndedData)
OverlapEvent = hsm.Event[OverlapData](name="bot.ability.hearing.speech.stream.overlap", schema=OverlapData)
FailedEvent = hsm.Event[FailedData](name="bot.ability.hearing.speech.stream.failed", schema=FailedData)


__all__ = [
    "AudioData",
    "AudioEvent",
    "FailedData",
    "FailedEvent",
    "FailureKind",
    "OverlapData",
    "OverlapEvent",
    "PartialData",
    "PartialEvent",
    "ProducingData",
    "ProducingEvent",
    "ReadyData",
    "ReadyEvent",
    "TranscriptData",
    "TranscriptEvent",
    "TurnEndedData",
    "TurnEndedEvent",
    "VoiceEndedData",
    "VoiceEndedEvent",
    "VoiceStartedData",
    "VoiceStartedEvent",
]

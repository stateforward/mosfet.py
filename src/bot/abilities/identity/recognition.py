"""Hearing a name you already have — the counterpart of recognising someone else's voice.

``hearing.voice.identification`` answers "who is speaking". This answers "was that me". Both are
recognition against something already known: a voiceprint there, an adopted name here. Neither
interprets what was said.

This module is the single place where a bot checks heard audio for a name, and it is deliberately
narrow: it is given a name the bot has **already adopted** and answers yes or no. It never decides
that an utterance confers a name — that is cognition's, through ``identity.AdoptEvent``.
"""

from __future__ import annotations

from .. import classifying
from ..hearing import speech

import abc
import typing

import pydantic

from bot.environment import SoundData


class InputData(pydantic.BaseModel):
    """Heard sound paired with the name the bot already answers to."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra={
            "description": (
                "One acoustic chunk to check for a name the bot has already adopted. Recognition is a "
                "yes/no question about a known token, not an interpretation of what was said."
            ),
            "examples": [
                {
                    "heard": {"audio": "YXVkaW8tY2h1bms=", "media_type": "audio/pcm", "sample_rate_hz": 16000},
                    "name": "Bob",
                }
            ],
        },
    )

    heard: SoundData = pydantic.Field(
        description="Acoustic chunk the bot heard, exactly as it arrived from the environment.",
        examples=[{"audio": "YXVkaW8tY2h1bms=", "media_type": "audio/pcm", "sample_rate_hz": 16000}],
    )
    name: str = pydantic.Field(
        min_length=1,
        description="Name the bot has already adopted and is listening for. Never a name to be learned here.",
        examples=["Bob"],
    )


class OutputData(pydantic.BaseModel):
    """Whether the adopted name was present in the heard sound."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Recognition result for one acoustic chunk: was the adopted name in it?",
            "examples": [{"addressed": True, "confidence": 0.82}, {"addressed": False}],
        },
    )

    addressed: bool = pydantic.Field(
        description="True when the adopted name was present in the heard sound.",
        examples=[True, False],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional recognizer confidence for the decision, normalized from 0.0 to 1.0.",
        examples=[0.82],
    )


class NameRecognizer(classifying.Classifier[InputData, OutputData], abc.ABC):
    """Recognizer that answers whether an adopted name was present in heard sound.

    SEAM: this contract exists so name-spotting can stop being a text problem. The only
    implementation shipped here transcribes and then looks for the name as a token
    (:class:`SpeechNameRecognizer`); a keyword spotter matches the name acoustically and never
    produces text at all. ``kaistmm/Metric-UD-KWS`` (283K params, MIT) was measured for exactly
    this role — word AUC 0.93 with voice AUC 0.545, i.e. it separates *which word* was said from
    *who* said it. Such a provider implements this interface and replaces the implementation
    below without touching :class:`~bot.abilities.identity.identity.Identity`.
    """


def _tokens(text: str) -> tuple[str, ...]:
    """Split text into casefolded alphanumeric tokens, dropping punctuation and spacing."""

    tokens: list[str] = []
    for part in text.split():
        token = "".join(character for character in part if character.isalnum())
        if token:
            tokens.append(token.casefold())
    return tuple(tokens)


def hears_name(transcript: str, name: str) -> bool:
    """Return whether ``name`` appears as a whole token run inside ``transcript``.

    This is a lookup for a token the bot already has, not an interpretation of the utterance.
    It cannot learn a name, infer one, or decide that any phrase confers one: it is only ever
    called with a name the bot adopted, and it answers a yes/no question about that one string.
    Substring matching is deliberately avoided so a bot named "Al" is not addressed by "always".
    """

    heard = _tokens(transcript)
    wanted = _tokens(name)
    if not wanted or len(wanted) > len(heard):
        return False
    span = len(wanted)
    return any(heard[start : start + span] == wanted for start in range(len(heard) - span + 1))


class SpeechNameRecognizer(NameRecognizer):
    """Transcribe the sound, then check the transcript for the adopted name.

    Text is an implementation detail of this recognizer and of nothing else: no transcript
    leaves this class, and :class:`OutputData` carries no words. Replacing it with an acoustic
    keyword spotter changes nothing above it.
    """

    _decoder: speech.SpeechDecoder

    def __init__(self, *, decoder: speech.SpeechDecoder) -> None:
        self._decoder = decoder

    @typing.override
    async def classify(self, input: InputData) -> OutputData:
        decoded = await self._decoder.decode(input.heard.audio)
        try:
            transcript = decoded.decode("utf-8")
        except UnicodeDecodeError:
            # A decoder that yields acoustic bytes rather than text cannot be searched for a
            # token; that is a recognizer mismatch, not "the name was absent".
            raise ValueError("SpeechNameRecognizer requires a speech decoder that produces UTF-8 text.") from None
        return OutputData(addressed=hears_name(transcript, input.name))


__all__ = [
    "InputData",
    "NameRecognizer",
    "OutputData",
    "SpeechNameRecognizer",
    "hears_name",
]

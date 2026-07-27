"""Recognising a name the bot already has.

Nothing here decides that an utterance confers a name. Every case starts from a name the bot has
already adopted and asks the one question this module exists to answer: was it in what I heard?
"""

from __future__ import annotations

import asyncio
import typing
from typing import override

import pytest

from bot.abilities import identity
from bot.abilities.hearing import speech
from bot.environment import SoundData


class EchoSpeechDecoder(speech.SpeechDecoder):
    """Stands in for STT: whatever bytes were heard are the transcript."""

    @override
    async def decode(self, input: bytes) -> bytes:
        return input


class AcousticSpeechDecoder(speech.SpeechDecoder):
    """A decoder that yields audio rather than text, as an acoustic front end would."""

    @override
    async def decode(self, input: bytes) -> bytes:
        del input
        return b"\xff\xfe\x00sound"


def heard(transcript: str) -> SoundData:
    return SoundData(audio=transcript.encode("utf-8"), media_type="audio/pcm", sample_rate_hz=16_000, channels=1)


@pytest.mark.parametrize(
    ("transcript", "name", "expected"),
    [
        ("Hey Bob, are you there?", "Bob", True),
        ("hey bob are you there", "Bob", True),
        ("Bob.", "Bob", True),
        ("Are you there?", "Bob", False),
        # A bot named Al is not addressed by a word that merely contains its name.
        ("always answer promptly", "Al", False),
        ("Al, answer promptly", "Al", True),
        # Multi-word names match as a run of tokens, not as scattered words.
        ("please ask Ada Lovelace about it", "Ada Lovelace", True),
        ("Ada asked Lovelace about it", "Ada Lovelace", False),
        ("", "Bob", False),
        ("Bob", "", False),
    ],
)
def test_a_known_name_is_found_only_as_a_whole_token_run(transcript: str, name: str, expected: bool) -> None:
    """The check is a lookup for a token the bot already has, not an interpretation.

    It is given the name; it never produces one. Substring matching is deliberately excluded so a
    short name is not triggered by every longer word that happens to contain it.
    """

    assert identity.hears_name(transcript, name) is expected


def test_recognition_answers_only_about_the_name_it_was_given() -> None:
    """A recognizer takes an already-adopted name as input and returns yes or no.

    There is no code path by which a transcript can supply the name being looked for.
    """

    recognizer = identity.SpeechNameRecognizer(decoder=EchoSpeechDecoder())

    async def run() -> tuple[identity.recognition.OutputData, identity.recognition.OutputData]:
        addressed = await recognizer.classify(
            identity.recognition.InputData(heard=heard("your name is Bob"), name="Bob")
        )
        other = await recognizer.classify(identity.recognition.InputData(heard=heard("your name is Bob"), name="Ada"))
        return addressed, other

    addressed, other = asyncio.run(run())

    assert addressed.addressed is True
    assert other.addressed is False


def test_recognition_input_requires_a_name_to_look_for() -> None:
    """A nameless bot cannot form a recognition request at all."""

    with pytest.raises(Exception):
        _ = identity.recognition.InputData(heard=heard("anything"), name="")


def test_a_recognizer_that_cannot_read_its_decoder_output_fails_loudly() -> None:
    """Non-text decoder output is a recognizer mismatch, not "the name was absent".

    Silently answering "no" here would make a misconfigured bot permanently unaddressable with no
    signal, which is exactly the failure the ability's typed failure path exists to surface.
    """

    recognizer = identity.SpeechNameRecognizer(decoder=AcousticSpeechDecoder())

    async def run() -> identity.recognition.OutputData:
        return await recognizer.classify(identity.recognition.InputData(heard=heard("Bob"), name="Bob"))

    with pytest.raises(ValueError):
        _ = asyncio.run(run())


def test_the_recognizer_contract_is_the_seam_an_acoustic_matcher_replaces() -> None:
    """Any future keyword spotter satisfies this interface without touching Identity.

    ``SpeechNameRecognizer`` transcribes because that is all core can do today; a provider that
    matches the name acoustically implements the same ``NameRecognizer`` and produces no text.
    """

    assert issubclass(identity.SpeechNameRecognizer, identity.NameRecognizer)

    class AcousticNameRecognizer(identity.NameRecognizer):
        @override
        async def classify(self, input: identity.recognition.InputData) -> identity.recognition.OutputData:
            del input
            return identity.recognition.OutputData(addressed=True, confidence=0.93)

    async def run() -> identity.recognition.OutputData:
        return await AcousticNameRecognizer().classify(
            identity.recognition.InputData(heard=heard("anything at all"), name="Bob")
        )

    result = asyncio.run(run())

    assert result.addressed is True
    assert result.confidence == pytest.approx(0.93)


def test_recognition_output_carries_no_words() -> None:
    """The product of recognition is a decision, not a transcript.

    Text is an implementation detail of one recognizer. Nothing above it ever sees words, which is
    what keeps identity from becoming a second, weaker listening pipeline.
    """

    fields: dict[str, typing.Any] = dict(identity.recognition.OutputData.model_fields)

    assert set(fields) == {"addressed", "confidence"}

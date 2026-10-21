from __future__ import annotations

import asyncio
import base64
import collections.abc
import dataclasses

import pytest
from google.genai import errors as genai_errors

from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech
from mosfet.providers.gemini import ChatClient, SpeechEncoder
from mosfet.providers.gemini import client as gemini_client
from tests.bot.abilities.vocal.speech import contract


@dataclasses.dataclass
class InteractionCall:
    kwargs: dict[str, object]


@dataclasses.dataclass
class FakeContentClient:
    response: dict[str, object]
    calls: list[InteractionCall] = dataclasses.field(default_factory=list[InteractionCall])
    error: Exception | None = None

    def generate_content(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        raise AssertionError("The speech encoder must use create_interaction, not generate_content.")

    def create_interaction(
        self,
        *,
        model: str | None = None,
        input: object | None = None,
        response_format: object | None = None,
        generation_config: collections.abc.Mapping[str, object] | None = None,
        system_instruction: str | None = None,
        store: bool = False,
        extra: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            InteractionCall(
                kwargs={
                    "model": model,
                    "input": input,
                    "response_format": response_format,
                    "generation_config": dict(generation_config or {}),
                    "system_instruction": system_instruction,
                    "store": store,
                    "extra": dict(extra or {}),
                }
            )
        )
        if self.error is not None:
            raise self.error
        return self.response


@dataclasses.dataclass
class FailingResource:
    """SDK resource that fails the way the live API does."""

    error: Exception

    def create(self, **kwargs: object) -> object:
        del kwargs
        raise self.error

    def generate_content(self, **kwargs: object) -> object:
        del kwargs
        raise self.error


@dataclasses.dataclass
class FailingSdkClient:
    """google-genai client shape whose every call raises one SDK error."""

    interactions: gemini_client.GeminiInteractionsResource
    models: gemini_client.GeminiModelsResource


def _failing_sdk_client(error: Exception) -> FailingSdkClient:
    resource = FailingResource(error=error)
    return FailingSdkClient(interactions=resource, models=resource)


def _quota_error() -> genai_errors.ClientError:
    return genai_errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded for quota metric 'Generate requests per minute'.",
            }
        },
    )


def _tts_response(pcm: bytes, *, sample_rate: int = 24_000) -> dict[str, object]:
    return {
        "output_audio": {
            "type": "audio",
            "mime_type": "audio/l16",
            "sample_rate": sample_rate,
            "data": base64.b64encode(pcm).decode("ascii"),
        }
    }


async def _all(stream: collections.abc.AsyncIterator[speech.Audio]) -> list[bytes]:
    return [chunk.pcm async for chunk in stream]


def _say(encoder: SpeechEncoder, text: str = "hello") -> list[bytes]:
    return asyncio.run(_all(encoder.encode(contract.text(contract.words(text)))))


def test_encoder_uses_interactions_api_and_yields_one_pcm_chunk() -> None:
    pcm = b"\x00\x01" * 8
    client = FakeContentClient(response=_tts_response(pcm))
    encoder = SpeechEncoder(client=client, voice_name="Puck", model="gemini-3.1-flash-tts-preview")

    assert _say(encoder, "Hello there") == [pcm]
    assert encoder.format == speech.Format(sample_rate_hz=24_000, channels=1)
    call = client.calls[0].kwargs
    assert call["model"] == "gemini-3.1-flash-tts-preview"
    assert call["input"] == "Hello there"
    assert call["response_format"] == {"type": "audio"}
    assert call["store"] is False
    assert call["generation_config"] == {"speech_config": [{"voice": "Puck"}]}


def test_encoder_reads_the_text_to_its_end_and_aligns_its_one_chunk_to_the_whole_text() -> None:
    pcm = b"\x00\x01" * 8
    client = FakeContentClient(response=_tts_response(pcm))

    chunks = asyncio.run(
        contract.assert_speech_encoder(SpeechEncoder(client=client), contract.words("Hello ", "the", "re"))
    )

    assert chunks == [speech.Audio(pcm=pcm, text="Hello there")]
    # One request per non-blank stream (the whole text, then the one closed after its chunk); blank text sends none.
    assert [call.kwargs["input"] for call in client.calls] == ["Hello there", "Hello there"]


def test_every_part_is_one_whole_text_said_as_written_with_no_expression() -> None:
    client = FakeContentClient(response=_tts_response(b"\x00\x01"))
    pieces = [
        speech.Piece(part=0, words="Oh, ", expression=speech.Expression(sound="laugh")),
        speech.Piece(part=1, words="hi.", expression=speech.Expression(instructions="Softly.")),
    ]

    chunks = asyncio.run(contract.assert_speech_encoder(SpeechEncoder(client=client), pieces))

    assert chunks == [speech.Audio(pcm=b"\x00\x01", text="Oh, hi.")]
    assert client.calls[0].kwargs["input"] == "Oh, hi."
    assert SpeechEncoder(client=client).expressiveness == speech.Expressiveness()


def test_the_constructor_voice_names_the_voice() -> None:
    client = FakeContentClient(response=_tts_response(b"\x00\x01"))

    _ = _say(SpeechEncoder(client=client, voice_name="Kore"))

    assert client.calls[0].kwargs["generation_config"] == {"speech_config": [{"voice": "Kore"}]}


def test_encoder_rejects_blank_text_and_missing_or_ragged_audio() -> None:
    with pytest.raises(encoding.Error, match="non-blank") as blank:
        _ = _say(SpeechEncoder(client=FakeContentClient(response=_tts_response(b"xx"))), "   ")
    assert blank.value.kind == "invalid_input"

    with pytest.raises(encoding.Error, match="output_audio") as missing:
        _ = _say(SpeechEncoder(client=FakeContentClient(response={"output_text": "nope"})))
    assert missing.value.kind == "invalid_response"

    with pytest.raises(encoding.Error, match="mid-sample") as ragged:
        _ = _say(SpeechEncoder(client=FakeContentClient(response=_tts_response(b"abc"))))
    assert ragged.value.kind == "invalid_response"


def test_encoding_failure_names_the_underlying_sdk_error() -> None:
    """A refused request must say what refused it, in the message the caller can see."""

    client = FakeContentClient(response={}, error=_quota_error())

    with pytest.raises(encoding.Error) as failure:
        _ = _say(SpeechEncoder(client=client))

    message = str(failure.value)
    assert failure.value.kind == "encode_failed"
    assert "429" in message
    assert "RESOURCE_EXHAUSTED" in message
    assert "Quota exceeded" in message
    assert "ClientError" in message


def test_encoding_failure_names_the_cause_through_the_real_client() -> None:
    """The live path is speech encoder → ChatClient → SDK, and every layer must pass the reason up."""

    client = ChatClient(model="gemini-3.1-flash-tts-preview", client=_failing_sdk_client(_quota_error()))

    with pytest.raises(encoding.Error) as failure:
        _ = _say(SpeechEncoder(client=client))

    message = str(failure.value)
    assert "429" in message
    assert "RESOURCE_EXHAUSTED" in message
    # Each layer names itself once: a reason repeated per wrapper is noise an operator has to read past.
    assert message.count("Gemini interactions.create request failed") == 1
    assert message.count("Quota exceeded") == 1


def test_encoding_failure_names_a_transport_error_without_a_status() -> None:
    client = FakeContentClient(response={}, error=TimeoutError("request timed out after 30s"))

    with pytest.raises(encoding.Error) as failure:
        _ = _say(SpeechEncoder(client=client))

    assert "TimeoutError" in str(failure.value)
    assert "request timed out after 30s" in str(failure.value)


def test_encoding_failure_keeps_the_original_exception_chained() -> None:
    error = _quota_error()

    with pytest.raises(encoding.Error) as failure:
        _ = _say(SpeechEncoder(client=FakeContentClient(response={}, error=error)))

    assert failure.value.__cause__ is error


def test_pcm_at_a_rate_the_api_disagrees_with_is_refused() -> None:
    """Raw PCM has no header, so a rate disagreement is heard, not raised — unless we raise it."""

    client = FakeContentClient(response=_tts_response(b"\x01\x00\x02\x00", sample_rate=48_000))

    with pytest.raises(encoding.Error) as failure:
        _ = _say(SpeechEncoder(client=client, sample_rate_hz=24_000))

    assert failure.value.kind == "invalid_response"
    assert "48000" in str(failure.value)
    assert "24000" in str(failure.value)


def test_pcm_with_a_matching_or_unreported_rate_is_yielded() -> None:
    pcm = b"\x01\x00\x02\x00"
    matching = FakeContentClient(response=_tts_response(pcm, sample_rate=24_000))
    unreported = FakeContentClient(response={"output_audio": {"data": base64.b64encode(pcm).decode("ascii")}})

    assert _say(SpeechEncoder(client=matching, sample_rate_hz=24_000)) == [pcm]
    assert _say(SpeechEncoder(client=unreported, sample_rate_hz=24_000)) == [pcm]


def test_a_pause_splits_the_text_and_is_said_as_silence_in_its_place() -> None:
    client = FakeContentClient(response=_tts_response(b"\x00\x01"))

    _ = asyncio.run(contract.assert_pauses(SpeechEncoder(client=client)))

    assert [call.kwargs["input"] for call in client.calls] == ["Hi.", " there."]

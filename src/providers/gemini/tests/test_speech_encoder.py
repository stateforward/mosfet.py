from __future__ import annotations

import asyncio
import base64
import collections.abc
import dataclasses
import io
import wave

import pytest
from google.genai import errors as genai_errors

from mosfet.providers.gemini import ChatClient, SpeechEncoder, SpeechEncodingError
from mosfet.providers.gemini import client as gemini_client
from mosfet.providers.gemini.speech_encoder import pcm_to_wav


@dataclasses.dataclass
class InteractionCall:
    kwargs: dict[str, object]


@dataclasses.dataclass
class FakeContentClient:
    response: dict[str, object]
    calls: list[InteractionCall] = dataclasses.field(default_factory=list)
    error: Exception | None = None

    def generate_content(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        raise AssertionError("SpeechEncoder must use create_interaction, not generate_content.")

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


def test_speech_encoder_uses_interactions_api() -> None:
    pcm = b"\x00\x01" * 8
    client = FakeContentClient(response=_tts_response(pcm))
    encoder = SpeechEncoder(client=client, voice_name="Puck", model="gemini-3.1-flash-tts-preview")

    result = asyncio.run(encoder.encode(b"Hello there"))

    assert result == pcm
    assert len(client.calls) == 1
    call = client.calls[0].kwargs
    assert call["model"] == "gemini-3.1-flash-tts-preview"
    assert call["input"] == "Hello there"
    assert call["response_format"] == {"type": "audio"}
    assert call["store"] is False
    assert call["generation_config"] == {"speech_config": [{"voice": "Puck"}]}


def test_speech_encoder_wraps_wav_output() -> None:
    pcm = b"\x00\x01" * 16
    client = FakeContentClient(response=_tts_response(pcm, sample_rate=24_000))
    encoder = SpeechEncoder(client=client, output_format="wav")

    result = asyncio.run(encoder.encode(b"hi"))

    with wave.open(io.BytesIO(result), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == 24_000
        assert stream.readframes(stream.getnframes()) == pcm


def test_speech_encoder_is_awaitable() -> None:
    client = FakeContentClient(response=_tts_response(b"abc"))
    encoder = SpeechEncoder(client=client)

    encoded = encoder.encode(b"hello")
    assert isinstance(encoded, collections.abc.Coroutine)
    assert asyncio.run(encoded) == b"abc"


def test_speech_encoder_rejects_blank_text() -> None:
    client = FakeContentClient(response=_tts_response(b"x"))
    encoder = SpeechEncoder(client=client)
    try:
        _ = asyncio.run(encoder.encode(b"   "))
    except SpeechEncodingError as error:
        assert "non-blank" in str(error)
    else:
        raise AssertionError("Expected SpeechEncodingError.")


def test_speech_encoder_rejects_missing_audio() -> None:
    client = FakeContentClient(response={"output_text": "nope"})
    encoder = SpeechEncoder(client=client)
    try:
        _ = asyncio.run(encoder.encode(b"hello"))
    except SpeechEncodingError as error:
        assert "output_audio" in str(error)
    else:
        raise AssertionError("Expected SpeechEncodingError.")


def test_speech_encoding_failure_names_the_underlying_sdk_error() -> None:
    """A refused request must say what refused it, in the message the caller can see.

    Abilities turn a raised encoder error into typed failure data with ``str(error)``, so the
    exception message is the whole diagnostic budget for a live run: whatever it omits is gone
    by the time anything logs it. ``__cause__`` survives on the exception object and no boundary
    reads it. One generic line for a quota refusal, a timeout, and a malformed response is how a
    mute robot and a throttled one look identical afterwards.
    """

    client = FakeContentClient(response={}, error=_quota_error())
    encoder = SpeechEncoder(client=client)

    with pytest.raises(SpeechEncodingError) as failure:
        _ = asyncio.run(encoder.encode(b"hello"))

    message = str(failure.value)
    assert "429" in message
    assert "RESOURCE_EXHAUSTED" in message
    assert "Quota exceeded" in message
    assert "ClientError" in message


def test_speech_encoding_failure_names_the_cause_through_the_real_client() -> None:
    """The live path is encoder → ChatClient → SDK, and every layer must pass the reason up.

    ``ChatClient`` wraps SDK exceptions in ``RequestError`` before the encoder ever sees them, so
    a reason that only the encoder preserves is still lost in production.
    """

    client = ChatClient(model="gemini-3.1-flash-tts-preview", client=_failing_sdk_client(_quota_error()))
    encoder = SpeechEncoder(client=client)

    with pytest.raises(SpeechEncodingError) as failure:
        _ = asyncio.run(encoder.encode(b"hello"))

    message = str(failure.value)
    assert "429" in message
    assert "RESOURCE_EXHAUSTED" in message
    assert "Quota exceeded" in message
    # Each layer names itself once: a reason repeated per wrapper is noise an operator has to read past.
    assert message.count("Gemini interactions.create request failed") == 1
    assert message.count("Quota exceeded") == 1


def test_speech_encoding_failure_names_a_transport_error_without_a_status() -> None:
    """Timeouts carry no HTTP status, and the message still has to identify them."""

    client = FakeContentClient(response={}, error=TimeoutError("request timed out after 30s"))
    encoder = SpeechEncoder(client=client)

    with pytest.raises(SpeechEncodingError) as failure:
        _ = asyncio.run(encoder.encode(b"hello"))

    message = str(failure.value)
    assert "TimeoutError" in message
    assert "request timed out after 30s" in message


def test_speech_encoding_failure_keeps_the_original_exception_chained() -> None:
    """Naming the cause in the message must not cost the chain a debugger can walk."""

    error = _quota_error()
    client = FakeContentClient(response={}, error=error)
    encoder = SpeechEncoder(client=client)

    with pytest.raises(SpeechEncodingError) as failure:
        _ = asyncio.run(encoder.encode(b"hello"))

    assert failure.value.__cause__ is error


def test_pcm_to_wav_roundtrip() -> None:
    pcm = b"\x10\x00" * 4
    wav = pcm_to_wav(pcm, sample_rate_hz=16_000)
    with wave.open(io.BytesIO(wav), "rb") as stream:
        assert stream.getframerate() == 16_000
        assert stream.readframes(stream.getnframes()) == pcm


def test_pcm_output_rejects_a_sample_rate_the_api_disagrees_with() -> None:
    """Raw PCM has no header, so a rate disagreement is heard, not raised — unless we raise it.

    The label this encoder is configured with is what downstream plays the audio at. If Gemini
    returns a different rate there is nowhere to record the difference, so the voice comes out at
    the wrong speed and looks like a model problem. Fail with both numbers instead.
    """

    pcm = b"\x01\x00\x02\x00"
    client = FakeContentClient(response=_tts_response(pcm, sample_rate=48_000))
    encoder = SpeechEncoder(client=client, sample_rate_hz=24_000, output_format="pcm")

    with pytest.raises(SpeechEncodingError) as failure:
        _ = asyncio.run(encoder.encode(b"hello"))

    assert "48000" in str(failure.value)
    assert "24000" in str(failure.value)


def test_pcm_output_accepts_a_matching_reported_rate() -> None:
    pcm = b"\x01\x00\x02\x00"
    client = FakeContentClient(response=_tts_response(pcm, sample_rate=24_000))
    encoder = SpeechEncoder(client=client, sample_rate_hz=24_000, output_format="pcm")

    assert asyncio.run(encoder.encode(b"hello")) == pcm


def test_pcm_output_falls_back_to_the_configured_rate_when_none_is_reported() -> None:
    """An unreported rate is not an error: the configured value stands, as it always has."""

    pcm = b"\x01\x00\x02\x00"
    client = FakeContentClient(response={"output_audio": {"data": base64.b64encode(pcm).decode("ascii")}})
    encoder = SpeechEncoder(client=client, sample_rate_hz=24_000, output_format="pcm")

    assert asyncio.run(encoder.encode(b"hello")) == pcm


def test_wav_output_uses_the_reported_rate_over_the_configured_one() -> None:
    """The container can carry the difference, so it does — the mirror of the PCM branch."""

    pcm = b"\x01\x00\x02\x00"
    client = FakeContentClient(response=_tts_response(pcm, sample_rate=16_000))
    encoder = SpeechEncoder(client=client, sample_rate_hz=24_000, output_format="wav")

    assert asyncio.run(encoder.encode(b"hello")) == pcm_to_wav(pcm, sample_rate_hz=16_000)

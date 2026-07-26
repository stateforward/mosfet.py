from __future__ import annotations

import asyncio
import base64
import collections.abc
import dataclasses
import io
import wave

import pytest

from bot.providers.gemini import SpeechEncoder, SpeechEncodingError
from bot.providers.gemini.speech_encoder import pcm_to_wav


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

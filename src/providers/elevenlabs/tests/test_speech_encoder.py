import asyncio
import collections.abc
import dataclasses

import pytest

from bot.providers.elevenlabs import speech_encoder


@dataclasses.dataclass(frozen=True, kw_only=True)
class TextToSpeechCall:
    voice_id: str
    text: str
    model_id: str
    output_format: str


@dataclasses.dataclass
class FakeTextToSpeechClient:
    chunks: tuple[bytes, ...]
    calls: list[TextToSpeechCall] = dataclasses.field(default_factory=list)

    def convert(
        self,
        voice_id: str,
        *,
        text: str,
        model_id: str,
        output_format: str,
    ) -> collections.abc.Iterator[bytes]:
        self.calls.append(
            TextToSpeechCall(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                output_format=output_format,
            )
        )
        return iter(self.chunks)


@dataclasses.dataclass(frozen=True)
class FakeElevenLabsClient:
    text_to_speech: FakeTextToSpeechClient


def test_speech_encoder_uses_sdk_client_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[FakeElevenLabsClient] = []

    def fake_elevenlabs(*, api_key: str | None, base_url: str | None, timeout: float | None) -> FakeElevenLabsClient:
        client = FakeElevenLabsClient(text_to_speech=FakeTextToSpeechClient(chunks=(b"audio", b"-chunk")))
        clients.append(client)
        assert api_key == "test-api-key"
        assert base_url == "https://example.test"
        assert timeout == 10.0
        return client

    # Patch the construction site: SpeechEncoder._client imports from elevenlabs.client at call
    # time, so patching the re-exported name on speech_encoder would not intercept it.
    monkeypatch.setattr("elevenlabs.client.ElevenLabs", fake_elevenlabs)
    encoder = speech_encoder.SpeechEncoder(
        api_key="test-api-key",
        voice_id="voice-123",
        model_id="eleven_flash_v2_5",
        output_format="pcm_16000",
        base_url="https://example.test",
        timeout_seconds=10.0,
    )

    result = asyncio.run(encoder.encode(b"hello"))

    assert result == b"audio-chunk"
    assert clients[0].text_to_speech.calls == [
        TextToSpeechCall(
            voice_id="voice-123",
            text="hello",
            model_id="eleven_flash_v2_5",
            output_format="pcm_16000",
        )
    ]


def test_speech_encoder_is_awaitable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_elevenlabs(*, api_key: str | None, base_url: str | None, timeout: float | None) -> FakeElevenLabsClient:
        assert api_key == "test-api-key"
        assert base_url is None
        assert timeout == 240.0
        return FakeElevenLabsClient(text_to_speech=FakeTextToSpeechClient(chunks=(b"audio",)))

    # Patch the construction site: SpeechEncoder._client imports from elevenlabs.client at call
    # time, so patching the re-exported name on speech_encoder would not intercept it.
    monkeypatch.setattr("elevenlabs.client.ElevenLabs", fake_elevenlabs)
    encoder = speech_encoder.SpeechEncoder(api_key="test-api-key", voice_id="voice-123")

    encoded = encoder.encode(b"hello")

    assert isinstance(encoded, collections.abc.Coroutine)
    assert asyncio.run(encoded) == b"audio"

"""Unit tests for ``openai_compat.SpeechDecoder``: the multipart ``/v1/audio/transcriptions`` wire and failures."""

from __future__ import annotations

import asyncio
import collections.abc
import io
import typing
import wave

import hsm
import httpx
import openai
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mosfet.providers.openai_compat as openai_compat
from mosfet.abilities.hearing import speech
from mosfet.telemetry import span

_SPAN = "bot.provider.openai_compat.speech.decode"
_PCM = b"\x01\x00\x02\x00\x03\x00\x04\x00"

Handler = collections.abc.Callable[[httpx.Request], httpx.Response]


def _client(handle: Handler) -> openai_compat.AsyncOpenAIClient:
    sdk = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="http://speech.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    # The SDK's create takes its own keywords; the protocol is the narrow shape this provider calls.
    return typing.cast(openai_compat.AsyncOpenAIClient, typing.cast(object, sdk))


def _answer(requests: list[httpx.Request], text: object = "  hello there  ") -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"text": text})

    return handle


def _decoder(
    handle: Handler, *, sample_rate_hz: int = 16_000, channels: int = 1, language: str | None = None
) -> openai_compat.SpeechDecoder:
    return openai_compat.SpeechDecoder(
        model="nemotron",
        base_url="http://speech.test/v1",
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        language=language,
        client=_client(handle),
    )


def _wav(audio: bytes, *, rate: int, channels: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(audio)
    return output.getvalue()


def _sent_wav(request: httpx.Request) -> bytes:
    body = request.content
    start = body.index(b"RIFF")
    return body[start:]


def _failure(decoder: openai_compat.SpeechDecoder, audio: bytes = _PCM) -> openai_compat.SpeechDecodingError:
    with pytest.raises(openai_compat.SpeechDecodingError) as caught:
        _ = asyncio.run(decoder.decode(audio))
    return caught.value


def test_speech_decoder_is_a_core_speech_decoder() -> None:
    decoder = _decoder(_answer([]))

    assert isinstance(decoder, speech.SpeechDecoder)
    assert not isinstance(decoder, hsm.Instance)


def test_raw_pcm_is_sent_as_a_wav_file_at_its_rate_and_the_text_returned() -> None:
    requests: list[httpx.Request] = []
    decoder = _decoder(_answer(requests), sample_rate_hz=24_000, channels=2)

    transcript = asyncio.run(decoder.decode(_PCM))

    assert transcript == b"hello there"
    [request] = requests
    assert request.method == "POST"
    assert request.url.path == "/v1/audio/transcriptions"
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b'name="model"\r\n\r\nnemotron\r\n' in request.content
    assert b'name="response_format"\r\n\r\njson\r\n' in request.content
    assert b'name="language"' not in request.content
    assert b'name="file"; filename="speech.wav"' in request.content
    with wave.open(io.BytesIO(_sent_wav(request)), "rb") as sent:
        assert sent.getframerate() == 24_000
        assert sent.getnchannels() == 2
        assert sent.getsampwidth() == 2
        assert sent.readframes(sent.getnframes()) == _PCM


def test_the_default_input_is_mono_sixteen_kilohertz() -> None:
    requests: list[httpx.Request] = []

    _ = asyncio.run(_decoder(_answer(requests)).decode(_PCM))

    with wave.open(io.BytesIO(_sent_wav(requests[0])), "rb") as sent:
        assert (sent.getframerate(), sent.getnchannels()) == (16_000, 1)


def test_a_wav_input_is_sent_unchanged() -> None:
    requests: list[httpx.Request] = []
    audio = _wav(_PCM, rate=48_000, channels=1)

    _ = asyncio.run(_decoder(_answer(requests)).decode(audio))

    assert _sent_wav(requests[0]).startswith(audio)


def test_a_configured_language_is_sent() -> None:
    requests: list[httpx.Request] = []

    _ = asyncio.run(_decoder(_answer(requests), language="en").decode(_PCM))

    assert b'name="language"\r\n\r\nen\r\n' in requests[0].content


def test_empty_text_is_a_decoding_failure() -> None:
    failure = _failure(_decoder(_answer([], text="   ")))

    assert failure.kind == "decode_failed"
    assert failure.failure_kind == "decode_failed"


def test_a_response_without_text_is_an_invalid_response() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"words": []})

    assert _failure(_decoder(handle)).kind == "invalid_response"


@pytest.mark.parametrize("audio", [b"", b"\x01\x00\x02"])
def test_empty_or_unaligned_pcm_is_refused_without_a_request(audio: bytes) -> None:
    requests: list[httpx.Request] = []

    failure = _failure(_decoder(_answer(requests)), audio)

    assert failure.kind == "invalid_input"
    assert requests == []


@pytest.mark.parametrize(
    ("status", "kind"),
    [(400, "invalid_input"), (422, "invalid_input"), (500, "unavailable"), (503, "unavailable")],
)
def test_http_status_failures_map_to_decoding_kinds(status: int, kind: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(status, json={"error": {"message": "nope"}})

    assert _failure(_decoder(handle)).kind == kind


def test_an_unreachable_server_is_unavailable() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert _failure(_decoder(handle)).kind == "unavailable"


def test_a_server_that_does_not_answer_in_time_is_a_timeout() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    assert _failure(_decoder(handle)).kind == "timeout"


@pytest.fixture
def finished_spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)
    return exporter


def _spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [recorded for recorded in exporter.get_finished_spans() if recorded.name == _SPAN]


def test_the_decode_span_carries_counts_and_no_content(finished_spans: InMemorySpanExporter) -> None:
    _ = asyncio.run(_decoder(_answer([])).decode(_PCM))

    [recorded] = _spans(finished_spans)
    assert dict(recorded.attributes or {}) == {
        "bot.component.name": "openai_compat.speech",
        "bot.stage": "decode",
        "bot.audio.bytes": len(_PCM),
        "bot.transcript.chars": len("hello there"),
        "bot.http.status_class": "2xx",
        "bot.outcome": "ok",
    }


def test_the_decode_span_records_the_failure_kind(finished_spans: InMemorySpanExporter) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(422, json={"error": {"message": "too short"}})

    _ = _failure(_decoder(handle))

    [recorded] = _spans(finished_spans)
    attributes = dict(recorded.attributes or {})
    assert attributes["bot.outcome"] == "failed"
    assert attributes["bot.failure.kind"] == "invalid_input"
    assert attributes["bot.http.status_class"] == "4xx"

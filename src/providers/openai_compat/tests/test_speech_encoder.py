"""Unit tests for ``openai_compat.SpeechEncoder``: the streamed ``/v1/audio/speech`` PCM wire, framing, and failures."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import json
import typing

import hsm
import httpx
import openai
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mosfet.providers.openai_compat as openai_compat
from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech
from mosfet.telemetry import span
from tests.bot.abilities.vocal.speech import contract

_SPAN = "bot.provider.openai_compat.speech.encode"
_PCM_HEADERS = {"x-sample-rate": "24000", "x-sample-format": "s16le", "content-type": "application/octet-stream"}


@dataclasses.dataclass
class Body(httpx.AsyncByteStream):
    """A chunked response body that records how far it was read and whether the client closed it."""

    chunks: tuple[bytes, ...]
    sent: int = 0
    closed: bool = False

    @typing.override
    async def __aiter__(self) -> collections.abc.AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.sent += 1
            yield chunk

    @typing.override
    async def aclose(self) -> None:
        self.closed = True


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


def _answer(
    requests: list[httpx.Request], body: Body, headers: collections.abc.Mapping[str, str] = _PCM_HEADERS
) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers=dict(headers), stream=body)

    return handle


def _encoder(
    handle: Handler, *, voice: str | None = "warm", sample_rate_hz: int = 24_000, channels: int = 1
) -> openai_compat.SpeechEncoder:
    return openai_compat.SpeechEncoder(
        model="breeze",
        base_url="http://speech.test/v1",
        voice=voice,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        client=_client(handle),
    )


async def _all(stream: collections.abc.AsyncIterator[speech.Audio]) -> list[bytes]:
    return [chunk.pcm async for chunk in stream]


def _text(*pieces: str) -> speech.Text:
    text = speech.Text()
    for piece in pieces:
        text.write(piece)
    text.close()
    return text


def _failure(encoder: openai_compat.SpeechEncoder, *, said: str = "Hello there.") -> encoding.Error:
    with pytest.raises(encoding.Error) as caught:
        _ = asyncio.run(_all(encoder.encode(_text(said))))
    return caught.value


def test_the_encoder_is_a_core_speech_encoder_with_its_declared_format() -> None:
    encoder = _encoder(_answer([], Body(chunks=())), sample_rate_hz=16_000, channels=2)

    assert isinstance(encoder, speech.SpeechEncoder)
    assert not isinstance(encoder, hsm.Instance)
    assert encoder.format == speech.Format(sample_rate_hz=16_000, channels=2)


def test_the_encoder_reads_the_text_to_its_end_and_makes_one_request_with_whole_text_alignment() -> None:
    requests: list[httpx.Request] = []
    encoder = _encoder(_answer(requests, Body(chunks=(b"\x01\x00\x02\x00", b"\x03\x00"))))

    chunks = asyncio.run(contract.assert_speech_encoder(encoder, ["Hello", " there", "."]))

    # Every chunk carries "" but the last, which carries the whole text: it is said once its last audio plays.
    assert chunks == [
        speech.Audio(pcm=b"\x01\x00\x02\x00", text=""),
        speech.Audio(pcm=b"\x03\x00", text="Hello there."),
    ]
    # One request per non-blank stream (the whole text, then the one closed after its chunk); blank text sends none.
    assert len(requests) == 2
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/audio/speech"
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "breeze",
        "input": "Hello there.",
        "voice": "warm",
        "response_format": "pcm",
    }


def test_the_constructor_voice_names_the_voice() -> None:
    requests: list[httpx.Request] = []
    encoder = _encoder(_answer(requests, Body(chunks=(b"\x01\x00",))), voice="bright")

    _ = asyncio.run(_all(encoder.encode(_text("Hi."))))

    assert json.loads(requests[0].content)["voice"] == "bright"


def test_no_voice_leaves_the_provider_default() -> None:
    requests: list[httpx.Request] = []
    encoder = _encoder(_answer(requests, Body(chunks=(b"\x01\x00",))), voice=None)

    _ = asyncio.run(_all(encoder.encode(_text("Hi."))))

    assert "voice" not in json.loads(requests[0].content)


def test_closing_the_stream_before_the_text_ends_sends_no_request() -> None:
    requests: list[httpx.Request] = []
    encoder = _encoder(_answer(requests, Body(chunks=(b"\x01\x00",))))

    async def cancelled_while_written() -> None:
        text = speech.Text()
        text.write("Still being")
        stream = encoder.encode(text)
        pulling = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0.01)
        _ = pulling.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pulling

    asyncio.run(cancelled_while_written())

    assert requests == []


def test_blank_text_and_no_audio_are_refused() -> None:
    requests: list[httpx.Request] = []

    assert _failure(_encoder(_answer(requests, Body(chunks=(b"\x01\x00",)))), said="  ").kind == "invalid_input"
    assert requests == []
    assert _failure(_encoder(_answer([], Body(chunks=())))).kind == "invalid_response"


def test_chunked_pcm_is_reframed_to_whole_mono_samples() -> None:
    encoder = _encoder(_answer([], Body(chunks=(b"\x01", b"\x00\x02", b"\x00\x03\x00\x04", b"\x00"))))

    chunks = asyncio.run(_all(encoder.encode(_text("Hi."))))

    assert chunks == [b"\x01\x00", b"\x02\x00\x03\x00", b"\x04\x00"]
    assert all(len(chunk) % 2 == 0 for chunk in chunks)


def test_chunked_pcm_is_reframed_to_whole_stereo_frames() -> None:
    encoder = _encoder(
        _answer([], Body(chunks=(b"\x01\x00\x02", b"\x00\x03\x00\x04\x00"))),
        channels=2,
    )

    chunks = asyncio.run(_all(encoder.encode(_text("Hi."))))

    assert chunks == [b"\x01\x00\x02\x00\x03\x00\x04\x00"]


def test_pcm_that_ends_mid_frame_is_an_invalid_response() -> None:
    encoder = _encoder(_answer([], Body(chunks=(b"\x01\x00\x02",))))

    assert _failure(encoder).kind == "invalid_response"


def test_a_sample_rate_header_that_disagrees_with_the_format_is_an_invalid_response() -> None:
    body = Body(chunks=(b"\x01\x00",))
    encoder = _encoder(_answer([], body, {**_PCM_HEADERS, "x-sample-rate": "22050"}))

    failure = _failure(encoder)

    assert failure.kind == "invalid_response"
    assert failure.failure_kind == "invalid_response"
    assert body.sent == 0


def test_a_sample_format_header_other_than_s16le_is_an_invalid_response() -> None:
    encoder = _encoder(_answer([], Body(chunks=(b"\x01\x00",)), {**_PCM_HEADERS, "x-sample-format": "f32le"}))

    assert _failure(encoder).kind == "invalid_response"


def test_a_server_without_format_headers_is_trusted_to_serve_the_declared_format() -> None:
    encoder = _encoder(_answer([], Body(chunks=(b"\x01\x00",)), {"content-type": "application/octet-stream"}))

    assert asyncio.run(_all(encoder.encode(_text("Hi.")))) == [b"\x01\x00"]


def test_closing_the_stream_early_closes_the_response() -> None:
    body = Body(chunks=(b"\x01\x00", b"\x02\x00", b"\x03\x00", b"\x04\x00"))
    encoder = _encoder(_answer([], body))

    async def first_then_close() -> bytes:
        stream = typing.cast(collections.abc.AsyncGenerator[speech.Audio], encoder.encode(_text("A long reply.")))
        first = await anext(stream)
        await stream.aclose()
        return first.pcm

    assert asyncio.run(first_then_close()) == b"\x01\x00"
    assert body.closed is True
    assert body.sent < len(body.chunks)


@pytest.mark.parametrize(
    ("status", "kind"),
    [(400, "invalid_input"), (422, "invalid_input"), (500, "unavailable"), (503, "unavailable")],
)
def test_http_status_failures_map_to_encoding_kinds(status: int, kind: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(status, json={"error": {"message": "nope"}})

    assert _failure(_encoder(handle)).kind == kind


def test_an_unreachable_server_is_unavailable() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert _failure(_encoder(handle)).kind == "unavailable"


def test_a_server_that_does_not_answer_in_time_is_a_timeout() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    assert _failure(_encoder(handle)).kind == "timeout"


def test_a_stream_that_breaks_after_it_started_is_unavailable() -> None:
    class Broken(Body):
        @typing.override
        async def __aiter__(self) -> collections.abc.AsyncIterator[bytes]:
            yield b"\x01\x00"
            raise httpx.ReadError("connection reset")

    encoder = _encoder(_answer([], Broken(chunks=())))

    assert _failure(encoder).kind == "unavailable"


@pytest.fixture
def finished_spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)
    return exporter


def _spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [recorded for recorded in exporter.get_finished_spans() if recorded.name == _SPAN]


def test_the_encoding_span_carries_counts_and_no_content(finished_spans: InMemorySpanExporter) -> None:
    encoder = _encoder(_answer([], Body(chunks=(b"\x01\x00\x02\x00", b"\x03\x00"))))

    _ = asyncio.run(_all(encoder.encode(_text("Secret words."))))

    [recorded] = _spans(finished_spans)
    assert dict(recorded.attributes or {}) == {
        "bot.component.name": "openai_compat.speech",
        "bot.stage": "encode",
        "bot.text.chars": 13,
        "bot.audio.bytes": 6,
        "bot.http.status_class": "2xx",
        "bot.outcome": "ok",
    }


def test_the_encoding_span_records_the_failure_kind(finished_spans: InMemorySpanExporter) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, json={"error": {"message": "busy"}})

    _ = _failure(_encoder(handle))

    [recorded] = _spans(finished_spans)
    attributes = dict(recorded.attributes or {})
    assert attributes["bot.outcome"] == "failed"
    assert attributes["bot.failure.kind"] == "unavailable"
    assert attributes["bot.http.status_class"] == "5xx"


def test_a_stream_closed_early_is_recorded_as_cancelled(finished_spans: InMemorySpanExporter) -> None:
    encoder = _encoder(_answer([], Body(chunks=(b"\x01\x00", b"\x02\x00"))))

    async def first_then_close() -> None:
        stream = typing.cast(collections.abc.AsyncGenerator[speech.Audio], encoder.encode(_text("Hi.")))
        _ = await anext(stream)
        await stream.aclose()

    asyncio.run(first_then_close())

    [recorded] = _spans(finished_spans)
    assert dict(recorded.attributes or {})["bot.outcome"] == "cancelled"

"""Unit tests for ``openai_compat.SpeechEncoder``: the streamed ``/v1/audio/speech`` PCM wire, framing, failures, and
per-run expression (markup, instructions and the provider merge rule)."""

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


_LAB = speech.Expressiveness(
    cues=frozenset({"instructions", "emotion", "intensity", "emphasis", "style", "sound"}),
    styles=("whisper",),
    sounds=frozenset({"laugh", "giggle", "chuckle", "gasp", "sigh"}),
)
"""The lab engine's declaration: natural-language cues, the whisper style, and its five live sounds."""


def _encoder(
    handle: Handler,
    *,
    voice: str | None = "warm",
    sample_rate_hz: int = 24_000,
    channels: int = 1,
    voice_description: str | None = None,
    expressiveness: speech.Expressiveness = speech.Expressiveness(),
    markup: openai_compat.Markup = openai_compat.Markup(),
) -> openai_compat.SpeechEncoder:
    return openai_compat.SpeechEncoder(
        model="breeze",
        base_url="http://speech.test/v1",
        voice=voice,
        voice_description=voice_description,
        expressiveness=expressiveness,
        markup=markup,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        client=_client(handle),
    )


def _lab(handle: Handler, *, voice_description: str | None = None) -> openai_compat.SpeechEncoder:
    return _encoder(handle, voice_description=voice_description, expressiveness=_LAB, markup=openai_compat.ANGLE_TAGS)


async def _all(stream: collections.abc.AsyncIterator[speech.Audio]) -> list[bytes]:
    return [chunk.pcm async for chunk in stream]


def _text(*pieces: str) -> speech.Text:
    return contract.text(contract.words(*pieces))


def _parts(*parts: tuple[str, speech.Expression | None]) -> speech.Text:
    """Each part written whole, opening with its expression."""

    return contract.text(
        [
            speech.Piece(part=index, words=words, expression=expression)
            for index, (words, expression) in enumerate(parts)
        ]
    )


def _sent(requests: collections.abc.Sequence[httpx.Request]) -> list[dict[str, object]]:
    return [typing.cast(dict[str, object], json.loads(request.content)) for request in requests]


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

    chunks = asyncio.run(contract.assert_speech_encoder(encoder, contract.words("Hello", " there", ".")))

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
        text.write(speech.Piece(part=0, words="Still being"))
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
        "bot.speech.requests": 1,
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


# --- expression: markup and instructions from typed fields, and the provider merge rule (P-20) ---


def _pcm(requests: list[httpx.Request], *bodies: tuple[bytes, ...]) -> Handler:
    """Answer the n-th request with the n-th body, so each request's chunks can be told apart."""

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        chunks = bodies[min(len(requests), len(bodies)) - 1]
        return httpx.Response(200, headers=dict(_PCM_HEADERS), stream=Body(chunks=chunks))

    return handle


def test_the_lab_declaration_holds_by_its_markup_and_a_mismatch_is_refused() -> None:
    assert openai_compat.ANGLE_TAGS == openai_compat.Markup(
        sounds={"laugh": "<laugh>", "giggle": "<giggle>", "chuckle": "<chuckle>", "gasp": "<gasp>", "sigh": "<sigh>"},
        styles={"whisper": "<whisper>"},
    )
    assert _lab(_answer([], Body(chunks=()))).expressiveness == _LAB
    with pytest.raises(ValueError, match="markup"):
        _ = _encoder(_answer([], Body(chunks=())), expressiveness=_LAB)
    with pytest.raises(ValueError, match="markup"):
        _ = _encoder(_answer([], Body(chunks=())), markup=openai_compat.ANGLE_TAGS)
    for unapplicable in ("pitch", "volume"):
        declared = speech.Expressiveness.model_validate(
            {"cues": {unapplicable}, unapplicable: {"min": 0.5, "max": 1.5}}
        )
        with pytest.raises(ValueError, match=unapplicable):
            _ = _encoder(_answer([], Body(chunks=())), expressiveness=declared)


def test_a_sound_with_no_other_cue_merges_into_one_request_with_its_tag_beside_its_words() -> None:
    requests: list[httpx.Request] = []
    encoder = _lab(_pcm(requests, (b"\x01\x00", b"\x02\x00")))

    chunks = asyncio.run(
        _all_audio(encoder.encode(_parts(("Oh, ", speech.Expression(sound="laugh")), ("you got me.", None))))
    )

    assert _sent(requests) == [
        {"model": "breeze", "input": "<laugh>Oh, you got me.", "voice": "warm", "response_format": "pcm"}
    ]
    # The merged request's last chunk carries the run's joined words, never its tags; earlier chunks carry "".
    assert chunks == [speech.Audio(pcm=b"\x01\x00", text=""), speech.Audio(pcm=b"\x02\x00", text="Oh, you got me.")]


def test_a_sound_after_its_words_is_placed_after_them_inside_the_merged_input() -> None:
    requests: list[httpx.Request] = []
    encoder = _lab(_pcm(requests, (b"\x01\x00",)))

    _ = asyncio.run(
        _all_audio(
            encoder.encode(
                _parts(
                    ("Oh, ", speech.Expression(sound="laugh", sound_at="after")),
                    ("you got me. ", speech.Expression(sound="sigh")),
                    ("Again.", speech.Expression(sound="giggle", sound_at="after")),
                )
            )
        )
    )

    assert [sent["input"] for sent in _sent(requests)] == ["Oh, <laugh><sigh>you got me. Again.<giggle>"]


def test_a_part_that_is_only_a_sound_writes_its_tag_where_the_part_sits() -> None:
    requests: list[httpx.Request] = []
    encoder = _lab(_pcm(requests, (b"\x01\x00",), (b"\x02\x00",)))

    chunks = asyncio.run(
        _all_audio(
            encoder.encode(
                _parts(
                    ("Oh,", None),
                    ("", speech.Expression(sound="chuckle")),
                    (" you got me.", None),
                    ("", speech.Expression(sound="laugh", sound_at="after")),
                )
            )
        )
    )

    assert [sent["input"] for sent in _sent(requests)] == ["Oh,<chuckle> you got me.<laugh>"]
    assert "".join(chunk.text for chunk in chunks) == "Oh, you got me."


def test_every_declared_sound_maps_to_its_angle_tag_and_undeclared_sounds_change_nothing() -> None:
    for sound, tag in openai_compat.ANGLE_TAGS.sounds.items():
        requests: list[httpx.Request] = []
        _ = asyncio.run(
            _all_audio(_lab(_pcm(requests, (b"\x01\x00",))).encode(_parts(("Hi.", speech.Expression(sound=sound)))))
        )
        assert _sent(requests)[0]["input"] == f"{tag}Hi.", sound
    undeclared: tuple[speech.Sound, ...] = ("breath", "cough", "clear_throat")
    for sound in undeclared:
        requests = []
        chunks = asyncio.run(
            _all_audio(_lab(_pcm(requests, (b"\x01\x00",))).encode(_parts(("Hi.", speech.Expression(sound=sound)))))
        )
        assert _sent(requests) == [{"model": "breeze", "input": "Hi.", "voice": "warm", "response_format": "pcm"}]
        assert "".join(chunk.text for chunk in chunks) == "Hi."


def test_parts_with_another_direction_are_separate_requests_in_order_each_aligned_to_its_words() -> None:
    directions = (
        speech.Expression(emotion="amused"),
        speech.Expression(intensity=0.7),
        speech.Expression(style="whisper"),
        speech.Expression(emphasis=True),
    )
    for direction in directions:
        requests: list[httpx.Request] = []
        encoder = _lab(_pcm(requests, (b"\x01\x00", b"\x02\x00"), (b"\x03\x00",)))

        chunks = asyncio.run(_all_audio(encoder.encode(_parts(("Oh, ", None), ("you got me.", direction)))))

        assert [sent["input"] for sent in _sent(requests)] == [
            "Oh, ",
            "<whisper>you got me." if direction.style else "you got me.",
        ], direction
        assert "instructions" not in _sent(requests)[0]
        assert chunks == [
            speech.Audio(pcm=b"\x01\x00", text=""),
            speech.Audio(pcm=b"\x02\x00", text="Oh, "),
            speech.Audio(pcm=b"\x03\x00", text="you got me."),
        ], direction


def test_three_parts_in_two_runs_give_two_requests() -> None:
    requests: list[httpx.Request] = []
    amused = speech.Expression(emotion="amused", sound="chuckle")
    encoder = _lab(_pcm(requests, (b"\x01\x00",), (b"\x02\x00",)))

    chunks = asyncio.run(
        _all_audio(
            encoder.encode(
                _parts(("Well, ", amused), ("well. ", speech.Expression(emotion="amused")), ("Who's this?", None))
            )
        )
    )

    sent = _sent(requests)
    assert [request["input"] for request in sent] == ["<chuckle>Well, well. ", "Who's this?"]
    assert sent[0]["instructions"] == "Emotion: amused."
    assert "instructions" not in sent[1]
    assert [chunk.text for chunk in chunks] == ["Well, well. ", "Who's this?"]


def test_a_whisper_style_comes_first_and_instructions_follow_the_voice_description_only_when_directed() -> None:
    requests: list[httpx.Request] = []
    encoder = _lab(_pcm(requests, (b"\x01\x00",)), voice_description="A warm, low voice.")
    secret = speech.Expression(
        style="whisper", emotion="amused", instructions="Like sharing a secret.", intensity=0.5, emphasis=True
    )

    _ = asyncio.run(
        _all_audio(encoder.encode(_parts(("Listen. ", secret), ("Loud!", speech.Expression(sound="gasp")))))
    )

    sent = _sent(requests)
    assert sent[0]["input"] == "<whisper>Listen. "
    assert sent[0]["instructions"] == (
        "A warm, low voice. Like sharing a secret. Emotion: amused. Intensity: 0.5 out of 1. Stress these words."
    )
    assert sent[1]["input"] == "<gasp>Loud!"
    assert "instructions" not in sent[1]


def test_words_that_look_like_sounds_or_tags_are_sent_unchanged_and_said_as_words() -> None:
    requests: list[httpx.Request] = []
    encoder = _lab(_pcm(requests, (b"\x01\x00", b"\x02\x00")))
    pieces = [
        speech.Piece(part=0, words="<laugh> is a word, "),
        speech.Piece(part=0, words="and so is laugh [[laugh]]."),
        speech.Piece(part=1, words=" <whisper>", expression=speech.Expression(sound="sigh", sound_at="after")),
    ]

    chunks = asyncio.run(contract.assert_speech_encoder(encoder, pieces))

    assert _sent(requests)[0]["input"] == "<laugh> is a word, and so is laugh [[laugh]]. <whisper><sigh>"
    assert "".join(chunk.text for chunk in chunks) == "<laugh> is a word, and so is laugh [[laugh]]. <whisper>"


def test_undeclared_cues_never_change_a_request_and_a_declared_pace_is_clamped_to_its_range() -> None:
    requests: list[httpx.Request] = []
    unheard = speech.Expression(pace=1.3, pitch=3.0, volume=1.5, style="shout")
    _ = asyncio.run(_all_audio(_lab(_pcm(requests, (b"\x01\x00",))).encode(_parts(("Hi ", None), ("there.", unheard)))))
    assert _sent(requests) == [{"model": "breeze", "input": "Hi there.", "voice": "warm", "response_format": "pcm"}]

    paced = speech.Expressiveness(cues=frozenset({"pace"}), pace=speech.Range(min=0.75, max=1.25))
    for pace, speed in ((1.0, 1.0), (2.0, 1.25), (0.5, 0.75)):
        requests = []
        encoder = _encoder(_pcm(requests, (b"\x01\x00",)), expressiveness=paced)
        _ = asyncio.run(_all_audio(encoder.encode(_parts(("Hi.", speech.Expression(pace=pace, emotion="calm"))))))
        assert _sent(requests)[0]["speed"] == speed, pace
        assert "instructions" not in _sent(requests)[0]


def test_a_blank_part_says_nothing_of_its_own_and_its_words_stay_aligned() -> None:
    requests: list[httpx.Request] = []
    encoder = _lab(_pcm(requests, (b"\x01\x00",), (b"\x02\x00",)))
    pieces = [
        speech.Piece(part=0, words=" ", expression=speech.Expression(emphasis=True)),
        speech.Piece(part=1, words="Hi.", expression=speech.Expression(emotion="calm")),
        speech.Piece(part=2, words="\n", expression=speech.Expression(emotion="sad")),
    ]

    chunks = asyncio.run(_all_audio(encoder.encode(contract.text(pieces))))

    # The blank opening part takes the direction of the part after it, and the blank last one joins the run before.
    assert [sent["input"] for sent in _sent(requests)] == [" Hi."]
    assert chunks == [speech.Audio(pcm=b"\x01\x00", text=" Hi.\n")]


async def _all_audio(stream: collections.abc.AsyncIterator[speech.Audio]) -> list[speech.Audio]:
    return [chunk async for chunk in stream]


def test_a_pause_closes_the_run_and_is_said_as_silence_in_its_place() -> None:
    requests: list[httpx.Request] = []
    encoder = _lab(_pcm(requests, (b"\x01\x00",), (b"\x02\x00",)))

    _ = asyncio.run(contract.assert_pauses(encoder))

    # A pause is silence between requests: the run before it is sent whole, never merged across it.
    assert [sent["input"] for sent in _sent(requests)] == ["Hi.", " there."]

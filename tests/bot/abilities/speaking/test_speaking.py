"""Speaking: a reply's parts → one streaming encoding → the attached mouth, chunk by chunk, under aligned, played
marks; a voice given once, per-part expression reports, and the own voice id learned by hearing."""

from __future__ import annotations

import asyncio
import collections
import collections.abc
import dataclasses
import datetime
import itertools
import math
import pathlib
import re
import time
import typing

import hsm
import mosfet
import pydantic
import pytest
from mosfet import lifecycle
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mosfet.abilities import ability
from mosfet.abilities import encoding
from mosfet.abilities import listening
from mosfet.abilities import processing
from mosfet.abilities import speaking
from mosfet.abilities import vocal
from mosfet.abilities.communication import conversation
from mosfet.devices import audio
from mosfet.environment import Environment, SoundData, SoundEvent
from mosfet.protocols import attachment
from mosfet.telemetry import span
from tests import metric_reader
from tests.bot.abilities.vocal.speech import contract
from tests.hsm_instance_state import device_bots, start_ability_tree

RATE = 8_000
"""8 kHz mono s16le: 1 ms is 8 samples, 16 bytes."""
OWN_VOICE = "3f2a9c41d07e4b6a9c2f1e8d5b7a0c63"
"""The id hearing gives the bot's own voice in these tests."""
LOOKAHEAD = datetime.timedelta(seconds=1)
ENCODE_IDLE_TIMEOUT = datetime.timedelta(seconds=5)
ENROLL = datetime.timedelta(seconds=6)
STOP_TIMEOUT = datetime.timedelta(seconds=1)
DRAFT_TIMEOUT = datetime.timedelta(seconds=10)


def pcm(milliseconds: int) -> bytes:
    return b"\x01\x00" * (RATE * milliseconds // 1000)


def words(text: str) -> list[str]:
    """``text`` cut after the whitespace that follows each word: the pieces join back to exactly ``text``."""

    return re.findall(r"\s*\S+\s*", text) or ([text] if text else [])


class Script(vocal.speech.SpeechEncoder):
    """Says ``chunk_ms`` of PCM per word; records the pieces it read and its closes; refuses blank text at its end.

    Aligned finely by default (each chunk carries its word as it reads each piece), or, with ``whole``, the way a
    whole-text provider emulates alignment: it reads the text to its end, then every chunk carries ``""`` but the
    last, which carries the whole text.
    """

    def __init__(
        self,
        *,
        chunk_ms: int = 20,
        fail_on: str | None = None,
        fail_kind: encoding.FailureKind = "unavailable",
        stall_on: str | None = None,
        whole: bool = False,
        expressiveness: vocal.speech.Expressiveness | None = None,
    ) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=RATE, channels=1)
        self.expressiveness: vocal.speech.Expressiveness = expressiveness or vocal.speech.Expressiveness()
        self.chunk_ms: int = chunk_ms
        self.fail_on: str | None = fail_on
        self.fail_kind: encoding.FailureKind = fail_kind
        self.stall_on: str | None = stall_on
        self.whole: bool = whole
        self.reads: list[list[str]] = []
        self.pieces: list[list[vocal.speech.Piece]] = []
        self.texts: list[collections.abc.AsyncIterable[vocal.speech.Piece]] = []
        self.closed: list[str] = []
        self.pulled: int = 0

    async def _say(self, word: str, text: str) -> vocal.speech.Audio:
        if self.fail_on is not None and word.strip().startswith(self.fail_on):
            raise encoding.Error(self.fail_kind, "the speech service went away")
        if self.stall_on is not None and word.strip().startswith(self.stall_on):
            _ = await asyncio.Event().wait()
        self.pulled += 1
        return vocal.speech.Audio(pcm=pcm(self.chunk_ms), text=text)

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        read: list[str] = []
        pieces: list[vocal.speech.Piece] = []
        self.reads.append(read)
        self.pieces.append(pieces)
        self.texts.append(input)
        try:
            if self.whole:
                async for piece in input:
                    read.append(piece.words)
                    pieces.append(piece)
                if not "".join(read).strip():
                    raise encoding.Error("invalid_input", "blank text")
                said = words("".join(read))
                for index, word in enumerate(said):
                    yield await self._say(word, "".join(read) if index == len(said) - 1 else "")
                return
            async for piece in input:
                read.append(piece.words)
                pieces.append(piece)
                for word in words(piece.words):
                    yield await self._say(word, word)
            if not "".join(read).strip():
                raise encoding.Error("invalid_input", "blank text")
        finally:
            self.closed.append("".join(read))


class RecordingSpeaker(audio.Speaker):
    """A real speaker that also notes every frame it was commanded with."""

    def __init__(
        self,
        *,
        clock: collections.abc.Callable[[], float],
        max_buffered: datetime.timedelta | None = None,
    ) -> None:
        super().__init__(
            frame=datetime.timedelta(milliseconds=10),
            max_buffered=max_buffered or datetime.timedelta(seconds=30),
            clock=clock,
        )
        self.frames: list[hsm.Event[audio.OutputData]] = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        if isinstance(event.data, audio.OutputData):
            self.frames.append(event)
        return super().dispatch(ctx, event)


def instant() -> collections.abc.Callable[[], float]:
    """A speaker clock that is always far past the schedule, so playout reports arrive immediately."""

    ticks = itertools.count(0.0, 1_000.0)
    return lambda: next(ticks)


class Collector(hsm.Instance):
    """Records everything addressed to it: a requester, or a Listening peer."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[hsm.Event[typing.Any]] = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Collector", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)

    def named(self, event: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        return [received for received in self.events if received.name == event.name]

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Collector",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )


class SoundListener(hsm.Instance):
    """Environment participant that records the ``environment.sound`` a speaker transduces."""

    def __init__(self) -> None:
        super().__init__()
        self.sounds: list[SoundData] = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "SoundListener", event: hsm.Event[typing.Any]) -> None:
        del ctx
        if isinstance(event.data, SoundData):
            instance.sounds.append(event.data)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "SoundListener",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(SoundEvent), hsm.effect(_record))),
    )


class RecordingConversation(conversation.Conversation):
    def __init__(self) -> None:
        super().__init__()
        self.outputs: list[conversation.Messages] = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        if event.name == ability.TerminalOutputEvent.name and isinstance(event.data, hsm.Event):
            terminal = typing.cast(hsm.Event[typing.Any], event.data)
            if terminal.name == conversation.OutputEvent.name and isinstance(terminal.data, conversation.Messages):
                self.outputs.append(terminal.data)
        return super().dispatch(ctx, event)


def outputs(requester: Collector) -> list[speaking.OutputData]:
    return [event.data for event in requester.events if isinstance(event.data, speaking.OutputData)]


def frames(speaker: RecordingSpeaker) -> list[audio.OutputData]:
    return [frame.data for frame in speaker.frames if isinstance(frame.data, audio.OutputData)]


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition not met")
        await asyncio.sleep(0.001)


@dataclasses.dataclass
class Stage:
    environment: Environment
    speaking: speaking.Speaking
    speaker: RecordingSpeaker
    encoder: Script
    requester: Collector
    ear: Collector
    room: SoundListener

    async def say(self, *words: str | vocal.speech.Part, id: str = "utterance-1") -> None:
        """Say a reply of these parts: a plain string is a part with no expression."""

        parts = tuple(vocal.speech.Part(words=part) if isinstance(part, str) else part for part in words)
        _ = await hsm.dispatch(
            self.environment,
            self.speaking,
            dataclasses.replace(
                speaking.InputEvent.with_data(speaking.InputData(parts=parts)),
                id=id,
                source=hsm.id(self.requester),
                target=hsm.id(self.speaking),
            ),
        )

    async def draft(
        self,
        words: str,
        offset: int,
        *,
        part: int = 0,
        expression: vocal.speech.Expression | None = None,
        id: str = "utterance-1",
    ) -> None:
        _ = await hsm.dispatch(
            self.environment,
            self.speaking,
            dataclasses.replace(
                speaking.DraftEvent.with_data(
                    speaking.DraftData(part=part, words=words, offset=offset, expression=expression)
                ),
                id=id,
                source=hsm.id(self.requester),
                target=hsm.id(self.speaking),
            ),
        )

    async def stop(self, *, id: str = "utterance-1") -> None:
        _ = await hsm.dispatch(
            self.environment,
            self.speaking,
            dataclasses.replace(
                speaking.StopEvent.with_data(speaking.StopData()),
                id=id,
                source=hsm.id(self.requester),
                target=hsm.id(self.speaking),
            ),
        )

    def outputs(self) -> list[speaking.OutputData]:
        return outputs(self.requester)

    def progress(self) -> list[speaking.ProgressData]:
        return [event.data for event in self.requester.events if isinstance(event.data, speaking.ProgressData)]

    def copies(self) -> list[speaking.EfferenceData]:
        return [event.data for event in self.ear.events if isinstance(event.data, speaking.EfferenceData)]

    async def hear_own_voice(self, voice_id: str = OWN_VOICE, *, id: str = "utterance-1") -> None:
        """Hearing (the ear) tells Speaking which id the bot's own voice has, in reply to its copy ``id``."""

        _ = await hsm.dispatch(
            self.environment,
            self.speaking,
            dataclasses.replace(
                listening.EnrolledEvent.with_data(listening.EnrolledData(voice_id=voice_id)),
                id=id,
                source=hsm.id(self.ear),
                target=hsm.id(self.speaking),
            ),
        )


async def stage(
    encoder: Script | None = None,
    *,
    clock: collections.abc.Callable[[], float] | None = None,
    conversation_targets: collections.abc.Sequence[ability.Ability[typing.Any, typing.Any]] = (),
    lookahead: datetime.timedelta = LOOKAHEAD,
    encode_idle_timeout: datetime.timedelta = ENCODE_IDLE_TIMEOUT,
    enroll_seconds: datetime.timedelta = ENROLL,
    stop_timeout: datetime.timedelta = STOP_TIMEOUT,
    draft_timeout: datetime.timedelta = DRAFT_TIMEOUT,
    voicing_timeout: datetime.timedelta = datetime.timedelta(seconds=30),
    mouth: RecordingSpeaker | None = None,
    factory: collections.abc.Callable[[str, str | None], vocal.speech.SpeechEncoder] | None = None,
) -> Stage:
    environment = Environment()
    mouth = mouth or RecordingSpeaker(clock=clock or instant())
    script = encoder or Script()
    requester, ear, room = Collector(), Collector(), SoundListener()
    for actor in (requester, ear, room):
        _ = await mosfet.started(environment, actor, actor.model)
    environment.join(room)
    voice = speaking.Speaking(
        encoder=script if factory is None else factory,
        speaker=mouth,
        listening=(ear,),
        conversation=conversation_targets,
        lookahead=lookahead,
        encode_idle_timeout=encode_idle_timeout,
        enroll_seconds=enroll_seconds,
        stop_timeout=stop_timeout,
        draft_timeout=draft_timeout,
        voicing_timeout=voicing_timeout,
    )
    await start_ability_tree(environment, voice)
    await wait_until(
        lambda: (voice.state() or "").endswith("/wired/voiced/idle" if factory is None else "/unvoiced/holding")
    )
    return Stage(environment, voice, mouth, script, requester, ear, room)


@pytest.fixture
def finished_spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Route this module's spans to an in-memory exporter for the duration of one test."""

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)
    return exporter


def utterance_failure(exporter: InMemorySpanExporter) -> object:
    """The failure kind on the one terminal utterance span, or ``None`` when it did not fail."""

    (terminal,) = [recorded for recorded in exporter.get_finished_spans() if recorded.name == "bot.speaking.utterance"]
    return dict(terminal.attributes or {}).get("bot.failure.kind")


def test_speaking_requires_ability_conversation_targets_and_sane_bounds() -> None:
    mouth = audio.Speaker()
    with pytest.raises(TypeError):
        _ = speaking.Speaking(encoder=Script(), speaker=mouth, conversation=(typing.cast(typing.Any, hsm.Instance()),))
    with pytest.raises(ValueError, match="lookahead"):
        _ = speaking.Speaking(encoder=Script(), speaker=mouth, lookahead=datetime.timedelta(0))
    with pytest.raises(ValueError, match="enroll_seconds"):
        _ = speaking.Speaking(encoder=Script(), speaker=mouth, enroll_seconds=datetime.timedelta(seconds=1))
    with pytest.raises(ValueError, match="draft_timeout"):
        _ = speaking.Speaking(encoder=Script(), speaker=mouth, draft_timeout=datetime.timedelta(0))
    with pytest.raises(ValueError, match="stop_timeout"):
        _ = speaking.Speaking(encoder=Script(), speaker=mouth, stop_timeout=datetime.timedelta(0))
    with pytest.raises(ValueError, match="voicing_timeout"):
        _ = speaking.Speaking(encoder=Script(), speaker=mouth, voicing_timeout=datetime.timedelta(0))


def test_speaking_input_is_not_a_cognition_call_event() -> None:
    assert speaking.InputEvent.name == "bot.ability.speaking.input"
    assert speaking.InputEvent.kind == hsm.EventKind
    assert speaking.ProgressEvent.kind != processing.EventKind
    assert speaking.StopEvent.name == "bot.ability.speaking.stop"
    assert speaking.StopEvent.kind != processing.EventKind
    assert speaking.DraftEvent.name == "bot.ability.speaking.draft"
    assert speaking.DraftEvent.kind != processing.EventKind
    assert speaking.EfferenceEvent.kind != processing.EventKind
    assert speaking.VoiceEvent.name == "bot.ability.speaking.voice"
    assert speaking.VoiceEvent.kind != processing.EventKind
    assert speaking.VoicedEvent.name == "bot.ability.speaking.voiced"
    assert speaking.VoicedEvent.kind == hsm.CompletionEventKind


def _examples_in(schema: object, path: str = "") -> list[str]:
    """Every place the generated schema carries an example, named by where it sits."""

    found: list[str] = []
    if isinstance(schema, dict):
        fields = typing.cast(dict[str, object], schema)
        for key, value in fields.items():
            here = f"{path}.{key}" if path else key
            # A default that is not text, or one of a closed set, offers nothing to say.
            if key in ("examples", "example") or (key == "default" and isinstance(value, str) and "enum" not in fields):
                found.append(f"{here} = {value!r}")
            else:
                found.extend(_examples_in(value, here))
    elif isinstance(schema, list):
        for index, item in enumerate(typing.cast(list[object], schema)):
            found.extend(_examples_in(item, f"{path}[{index}]"))
    return found


def test_speak_input_schema_offers_no_example_utterance() -> None:
    """The schema a model reads in order to speak must not hand it something to say."""

    schema = processing.model_facing_event_json_schema(speaking.InputEvent)

    assert _examples_in(schema) == []


def test_the_scripted_encoder_keeps_the_alignment_contract() -> None:
    assert [
        chunk.text
        for chunk in asyncio.run(
            contract.assert_speech_encoder(Script(), contract.words("Of course, I can", " do that."))
        )
    ] == [
        "Of ",
        "course, ",
        "I ",
        "can",
        " do ",
        "that.",
    ]
    whole = asyncio.run(contract.assert_speech_encoder(Script(whole=True), contract.words("Of course,", " I can.")))
    assert [chunk.text for chunk in whole] == ["", "", "", "Of course, I can."]


def test_speaking_attaches_to_its_mouth_and_streams_each_chunk_under_its_mark() -> None:
    async def run() -> tuple[Stage, list[speaking.OutputData]]:
        voice = await stage()
        assert voice.speaking in device_bots(voice.speaker)
        await voice.say("Of course, I can do that. See you soon.")
        await wait_until(lambda: bool(voice.outputs()))
        # The room hears the mouth, never the ability: one sound per 10 ms slice of every 20 ms chunk.
        await wait_until(lambda: len(voice.room.sounds) == 2 * len(voice.speaker.frames))
        return voice, voice.outputs()

    voice, outputs = asyncio.run(run())

    # The whole text goes to the encoder as it is: core never cuts it.
    assert voice.encoder.reads == [["Of course, I can do that. See you soon."]]
    commanded = frames(voice.speaker)
    assert [frame.mark for frame in commanded] == [f"u1.{chunk}" for chunk in range(9)]
    assert {(frame.media_type, frame.sample_rate_hz, frame.channels) for frame in commanded} == {("audio/pcm", RATE, 1)}
    assert {frame.source for frame in voice.speaker.frames} == {hsm.id(voice.speaking)}
    assert outputs == [
        speaking.OutputData(
            text="Of course, I can do that. See you soon.",
            spoken="Of course, I can do that. See you soon.",
            unspoken="",
            outcome="completed",
            sample_rate_hz=RATE,
            channels=1,
        )
    ]
    terminal = voice.requester.named(speaking.OutputEvent)[0]
    assert (terminal.id, terminal.source) == ("utterance-1", hsm.id(voice.speaking))


def test_the_ear_is_told_the_playout_still_to_come_and_zero_when_the_mouth_is_done() -> None:
    async def run() -> list[speaking.EfferenceData]:
        voice = await stage(Script(chunk_ms=20))
        await voice.say("one two three")
        await wait_until(lambda: bool(voice.outputs()))
        return voice.copies()

    copies = asyncio.run(run())

    assert {copy.mouth for copy in copies} == {copies[0].mouth}
    assert math.isclose(max(copy.duration for copy in copies), 0.06, abs_tol=1e-9)
    assert copies[-1].duration == 0.0
    assert all(copy.duration >= 0.0 for copy in copies)
    assert (copies[0].media_type, copies[0].sample_rate_hz, copies[0].channels) == ("audio/pcm", RATE, 1)


def test_progress_reaches_the_requester_as_audio_starts_and_plays() -> None:
    async def run() -> list[speaking.ProgressData]:
        voice = await stage()
        await voice.say("First one. Second one.")
        await wait_until(lambda: bool(voice.outputs()))
        return [event.data for event in voice.requester.events if isinstance(event.data, speaking.ProgressData)]

    progress = asyncio.run(run())

    assert progress[0] == speaking.ProgressData(spoken_seconds=0.0)
    assert math.isclose(progress[-1].spoken_seconds, 0.08, abs_tol=1e-9)
    assert [item.spoken_seconds for item in progress] == sorted(item.spoken_seconds for item in progress)


def test_a_failed_encoding_still_plays_what_was_commanded_and_ends_failed() -> None:
    async def run() -> tuple[list[speaking.OutputData], list[str]]:
        voice = await stage(Script(fail_on="broken"))
        await voice.say("This part is fine. Then broken words. Never said.")
        await wait_until(lambda: bool(voice.outputs()))
        return voice.outputs(), voice.encoder.closed

    outputs, closed = asyncio.run(run())

    # What was said is exactly the aligned text of the chunks the mouth played.
    assert outputs[0].outcome == "failed"
    assert outputs[0].spoken == "This part is fine. Then "
    assert outputs[0].unspoken == "broken words. Never said."
    assert closed == ["This part is fine. Then broken words. Never said."]


@pytest.mark.parametrize("kind", ["invalid_input", "unavailable", "timeout", "invalid_response", "encode_failed"])
def test_every_encoding_error_kind_ends_the_utterance_failed_and_is_measured_as_failed(
    kind: encoding.FailureKind,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    finished_spans: InMemorySpanExporter,
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:

        def failed() -> int:
            return sum(
                point.count
                for point in reader.points("bot.speaking.encoding.duration")
                if point.attributes.get("outcome") == "failed"
            )

        before = failed()

        async def run() -> list[speaking.OutputData]:
            voice = await stage(Script(fail_on="broken", fail_kind=kind))
            await voice.say("Fine. Then broken.")
            await wait_until(lambda: bool(voice.outputs()))
            return voice.outputs()

        outputs = asyncio.run(run())
        after = failed()

    assert (outputs[0].outcome, outputs[0].spoken, outputs[0].unspoken) == ("failed", "Fine. Then ", "broken.")
    assert after == before + 1
    assert utterance_failure(finished_spans) == kind


def test_with_whole_text_alignment_nothing_is_said_until_the_last_chunk_plays() -> None:
    async def run() -> tuple[list[speaking.OutputData], list[speaking.OutputData]]:
        broken = await stage(Script(whole=True, fail_on="broken"))
        await broken.say("This part is fine. Then broken words.")
        await wait_until(lambda: bool(broken.outputs()))
        whole = await stage(Script(whole=True))
        await whole.say("All of this is said.")
        await wait_until(lambda: bool(whole.outputs()))
        return broken.outputs(), whole.outputs()

    broken, whole = asyncio.run(run())

    assert (broken[0].outcome, broken[0].spoken) == ("failed", "")
    assert broken[0].unspoken == "This part is fine. Then broken words."
    assert (whole[0].outcome, whole[0].spoken, whole[0].unspoken) == ("completed", "All of this is said.", "")


def test_an_encoder_that_goes_quiet_is_cancelled_at_the_provider_by_the_modeled_timeout() -> None:
    async def run() -> tuple[list[speaking.OutputData], Script]:
        voice = await stage(Script(stall_on="stuck"), encode_idle_timeout=datetime.timedelta(milliseconds=50))
        await voice.say("Said first. Then stuck here.")
        await wait_until(lambda: bool(voice.outputs()))
        return voice.outputs(), voice.encoder

    outputs, script = asyncio.run(run())

    assert outputs[0].outcome == "failed"
    assert outputs[0].spoken == "Said first. Then "
    assert script.closed == ["Said first. Then stuck here."]


class Chunks:
    """An async iterator that is not an async generator: closing it is its own ``aclose``."""

    def __init__(
        self, inner: collections.abc.AsyncGenerator[vocal.speech.Audio], closed: list[str], text: Script
    ) -> None:
        self._inner: collections.abc.AsyncGenerator[vocal.speech.Audio] = inner
        self._closed: list[str] = closed
        self._script: Script = text

    def __aiter__(self) -> Chunks:
        return self

    async def __anext__(self) -> vocal.speech.Audio:
        return await anext(self._inner)

    async def aclose(self) -> None:
        self._closed.append("".join(self._script.reads[-1]))
        await self._inner.aclose()


class IteratorScript(Script):
    """An encoder whose streams are plain async iterators with ``aclose``, not async generators."""

    def __init__(self, **options: typing.Any) -> None:
        super().__init__(**options)
        self.aclosed: list[str] = []

    @typing.override
    def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        inner = typing.cast(collections.abc.AsyncGenerator[vocal.speech.Audio], super().encode(input))
        return Chunks(inner, self.aclosed, self)


def test_the_stream_is_closed_when_the_utterance_stops_streaming() -> None:
    async def run() -> tuple[IteratorScript, IteratorScript]:
        stalled = IteratorScript(stall_on="stuck")
        voice = await stage(stalled, encode_idle_timeout=datetime.timedelta(milliseconds=50))
        await voice.say("Said first. Then stuck here.")
        await wait_until(lambda: bool(voice.outputs()))
        finished = IteratorScript()
        voice = await stage(finished)
        await voice.say("Said whole.")
        await wait_until(lambda: bool(voice.outputs()))
        return stalled, finished

    stalled, finished = asyncio.run(run())

    assert stalled.aclosed == ["Said first. Then stuck here."]
    assert finished.aclosed == ["Said whole."]


def test_the_next_chunk_is_pulled_only_once_the_mouth_has_played_down_to_the_lookahead() -> None:
    async def run() -> tuple[int, int, int]:
        # A real clock: 100 ms per chunk against a 50 ms lookahead.
        voice = await stage(Script(chunk_ms=100), clock=time.monotonic, lookahead=datetime.timedelta(milliseconds=50))
        await voice.say("one two three four")
        await wait_until(lambda: bool(voice.speaker.frames))
        await asyncio.sleep(0.03)
        pulled_mid_chunk = voice.encoder.pulled
        await wait_until(lambda: bool(voice.outputs()))
        return pulled_mid_chunk, voice.encoder.pulled, len(voice.speaker.frames)

    mid, total, commanded = asyncio.run(run())

    # Each 100 ms chunk goes to the mouth in four 25 ms pieces (half the lookahead), so the mouth always has the
    # next piece queued; the encoder is pulled again only once a chunk's last piece is commanded.
    assert (mid, total, commanded) == (1, 4, 16)


def test_a_long_whole_reply_is_written_whole_and_spoken_in_full(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W3-20: no whole-reply cap. The encoder reads every part whole, and none of it is an encoding failure."""

    parts = ["word " * 340] * 4 + ["word " * 339 + "word."]
    text = "".join(parts)
    assert len(text) == 8_500

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = collections.Counter(
            {str(point.attributes["outcome"]): point.count for point in reader.points("bot.speaking.encoding.duration")}
        )

        async def run() -> tuple[Stage, list[speaking.OutputData]]:
            voice = await stage()
            await voice.say(*parts)
            await wait_until(lambda: bool(voice.outputs()), timeout=20)
            return voice, voice.outputs()

        voice, outputs = asyncio.run(run())
        after = collections.Counter(
            {str(point.attributes["outcome"]): point.count for point in reader.points("bot.speaking.encoding.duration")}
        )

    assert voice.encoder.reads == [parts]
    assert (outputs[0].outcome, outputs[0].spoken, outputs[0].unspoken) == ("completed", text, "")
    assert after - before == collections.Counter({"completed": 1})


def test_one_chunk_longer_than_the_mouths_buffer_is_said_in_pieces_under_the_lookahead() -> None:
    """A whole-text provider yields 31 s chunks, past the mouth's 30 s buffer; Speaking, not the provider, cuts
    them into whole-frame pieces of at most half the lookahead, and the chunk's text rides on its last piece."""

    async def run() -> tuple[Stage, list[speaking.OutputData]]:
        voice = await stage(Script(chunk_ms=31_000, whole=True))
        await voice.say("A long answer.")
        await wait_until(lambda: bool(voice.outputs()), timeout=20)
        return voice, voice.outputs()

    voice, outputs = asyncio.run(run())

    commanded = frames(voice.speaker)
    assert (outputs[0].outcome, outputs[0].spoken, outputs[0].unspoken) == ("completed", "A long answer.", "")
    assert max(len(frame.audio) for frame in commanded) == len(pcm(500))
    assert sum(len(frame.audio) for frame in commanded) == 3 * len(pcm(31_000))
    assert len(commanded) == 3 * 62


def test_audio_the_mouth_drops_never_ends_the_utterance_completed(finished_spans: InMemorySpanExporter) -> None:
    """A mouth whose buffer cannot take a piece drops it unplayed: the utterance has failed overloaded, and nothing
    more is commanded after the drop."""

    async def run() -> tuple[Stage, list[speaking.OutputData]]:
        mouth = RecordingSpeaker(clock=instant(), max_buffered=datetime.timedelta(milliseconds=50))
        voice = await stage(Script(chunk_ms=100), mouth=mouth)
        await voice.say("Too big to buffer.")
        await wait_until(lambda: bool(voice.outputs()))
        return voice, voice.outputs()

    voice, outputs = asyncio.run(run())

    assert (outputs[0].outcome, outputs[0].spoken, outputs[0].unspoken) == ("failed", "", "Too big to buffer.")
    assert len(voice.speaker.frames) == 1
    assert voice.copies()[-1].duration == 0.0
    assert utterance_failure(finished_spans) == "overloaded"


def test_a_reply_is_one_to_sixty_four_parts_whose_words_are_not_blank() -> None:
    """R6-5: blank words never reach the encoder: the request that carries them cannot be made."""

    with pytest.raises(pydantic.ValidationError):
        _ = speaking.InputData(parts=())
    with pytest.raises(pydantic.ValidationError):
        _ = speaking.InputData(parts=tuple(vocal.speech.Part(words="a") for _ in range(65)))
    with pytest.raises(pydantic.ValidationError):
        _ = speaking.InputData.model_validate({"parts": [{"words": "   "}]})
    reply = speaking.InputData(parts=(vocal.speech.Part(words="Oh, "), vocal.speech.Part(words="you got me.")))
    assert vocal.speech.said(reply.parts) == "Oh, you got me."


def test_a_stream_whose_close_fails_is_still_measured(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Breaks(Chunks):
        @typing.override
        async def aclose(self) -> None:
            await super().aclose()
            raise RuntimeError("the provider broke while closing")

    class BreaksOnClose(IteratorScript):
        @typing.override
        def encode(
            self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
        ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
            inner = typing.cast(collections.abc.AsyncGenerator[vocal.speech.Audio], Script.encode(self, input))
            return Breaks(inner, self.aclosed, self)

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = sum(point.count for point in reader.points("bot.speaking.encoding.duration"))

        async def run() -> tuple[BreaksOnClose, list[speaking.OutputData]]:
            script = BreaksOnClose()
            voice = await stage(script)
            await voice.say("Closed badly.")
            await wait_until(lambda: bool(voice.outputs()))
            return script, voice.outputs()

        script, outputs = asyncio.run(run())
        after = sum(point.count for point in reader.points("bot.speaking.encoding.duration"))

    assert script.aclosed == ["Closed badly."]
    assert outputs[0].outcome == "completed"
    assert after - before == 1


def test_a_mouth_that_stops_reporting_ends_the_utterance_unspoken(
    monkeypatch: pytest.MonkeyPatch, finished_spans: InMemorySpanExporter
) -> None:
    from mosfet.abilities.speaking import speaking as speaking_source

    monkeypatch.setattr(speaking_source, "_PLAYOUT_GRACE", datetime.timedelta(milliseconds=20))

    class Mute(RecordingSpeaker):
        """Takes the frames and never reports them played."""

        @typing.override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
            if isinstance(event.data, audio.OutputData):
                self.frames.append(event)
                done: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
                done.set_result(True)
                return done
            return super().dispatch(ctx, event)

    async def run() -> list[speaking.OutputData]:
        environment = Environment()
        requester = Collector()
        _ = await mosfet.started(environment, requester, requester.model)
        voice = speaking.Speaking(encoder=Script(), speaker=Mute(clock=instant()))
        await start_ability_tree(environment, voice)
        await wait_until(lambda: (voice.state() or "").endswith("/wired/voiced/idle"))
        _ = await hsm.dispatch(
            environment,
            voice,
            dataclasses.replace(
                speaking.InputEvent.with_data(
                    speaking.InputData(parts=(vocal.speech.Part(words="Nobody hears this."),))
                ),
                id="unheard",
                source=hsm.id(requester),
            ),
        )
        await wait_until(lambda: bool(outputs(requester)))
        return outputs(requester)

    unheard = asyncio.run(run())

    assert unheard[0].outcome == "failed"
    assert (unheard[0].spoken, unheard[0].unspoken) == ("", "Nobody hears this.")
    assert utterance_failure(finished_spans) == "timeout"


def test_stop_flushes_the_mouth_and_ends_stopped_with_only_what_was_played() -> None:
    class Flushes(RecordingSpeaker):
        """A real speaker that also notes every flush it was commanded with."""

        def __init__(self) -> None:
            super().__init__(clock=time.monotonic)
            self.flushes: list[hsm.Event[typing.Any]] = []

        @typing.override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
            if isinstance(event.data, audio.FlushData):
                self.flushes.append(event)
            return super().dispatch(ctx, event)

    async def run() -> tuple[Stage, Flushes, list[speaking.OutputData], int]:
        # A real clock: 200 ms of audio per word, so the first word plays well before the rest.
        mouth = Flushes()
        voice = await stage(Script(chunk_ms=200), mouth=mouth)
        await voice.say("First. Then a few more words. And more.")
        await wait_until(lambda: any(item.spoken_seconds > 0.0 for item in voice.progress()))
        await voice.stop()
        await wait_until(lambda: bool(voice.outputs()))
        commanded = len(voice.speaker.frames)
        await asyncio.sleep(0.05)
        return voice, mouth, voice.outputs(), commanded

    voice, mouth, outputs, commanded = asyncio.run(run())

    assert [(event.source, event.target) for event in mouth.flushes] == [(hsm.id(voice.speaking), hsm.id(mouth))]
    assert outputs == [
        speaking.OutputData(
            text="First. Then a few more words. And more.",
            spoken="First. ",
            unspoken="Then a few more words. And more.",
            outcome="stopped",
            sample_rate_hz=RATE,
            channels=1,
        )
    ]
    terminal = voice.requester.named(speaking.OutputEvent)[0]
    assert terminal.id == "utterance-1"
    # Nothing more is commanded once stopped, and the ear is told the mouth has nothing left to say.
    assert len(voice.speaker.frames) == commanded
    assert voice.copies()[-1].duration == 0.0


def test_stop_closes_the_pipe_and_the_stream_in_flight_and_ignores_a_stop_for_another_utterance() -> None:
    async def run() -> tuple[str, list[speaking.OutputData], IteratorScript, str]:
        voice = await stage(IteratorScript(stall_on="stuck"))
        await voice.say("Said first. Then stuck here.")
        await wait_until(lambda: voice.encoder.pulled == 3)
        await wait_until(lambda: any(item.spoken_seconds >= 0.06 for item in voice.progress()))
        await voice.stop(id="some-other-utterance")
        await asyncio.sleep(0.02)
        still = voice.speaking.state() or ""
        await voice.stop()
        await wait_until(lambda: bool(voice.outputs()))
        await wait_until(lambda: (voice.speaking.state() or "").endswith("/wired/voiced/idle"))
        return still, voice.outputs(), typing.cast(IteratorScript, voice.encoder), voice.speaking.state() or ""

    still, outputs, script, state = asyncio.run(run())

    assert "/speaking/streaming/pulling/" in still
    assert outputs[0].outcome == "stopped"
    # Fine alignment: what was said is exactly the words whose audio played.
    assert (outputs[0].spoken, outputs[0].unspoken) == ("Said first. Then ", "stuck here.")
    # The provider saw the stream closed, and it had read the whole pipe to its end.
    assert script.aclosed == ["Said first. Then stuck here."]
    assert script.closed == ["Said first. Then stuck here."]
    assert state.endswith("/wired/voiced/idle")


def test_a_stop_with_whole_text_alignment_says_nothing_was_said() -> None:
    async def run() -> list[speaking.OutputData]:
        # A real clock: 200 ms per word, and the text is said only once its last chunk plays.
        voice = await stage(Script(chunk_ms=200, whole=True), mouth=RecordingSpeaker(clock=time.monotonic))
        await voice.say("First. Then more.")
        await wait_until(lambda: any(item.spoken_seconds > 0.0 for item in voice.progress()))
        await voice.stop()
        await wait_until(lambda: bool(voice.outputs()))
        return voice.outputs()

    outputs = asyncio.run(run())

    assert (outputs[0].outcome, outputs[0].spoken, outputs[0].unspoken) == ("stopped", "", "First. Then more.")


def test_a_stop_flushes_the_mouth_at_once_even_while_the_provider_takes_its_time_to_close() -> None:
    """Barge-in is prompt: the flush leaves on the stop, not after the provider's ``aclose`` returns."""

    class SlowClose(Script):
        """Says audio forever once the text ends; closing takes a second (a stalled WebSocket close)."""

        @typing.override
        async def encode(
            self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
        ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
            try:
                async for _ in input:
                    pass
                while True:
                    await asyncio.sleep(0.01)
                    yield vocal.speech.Audio(pcm=pcm(100), text="")
            finally:
                await asyncio.sleep(1.0)

    class Flushes(RecordingSpeaker):
        def __init__(self) -> None:
            super().__init__(clock=time.monotonic)
            self.flushed_at: list[float] = []

        @typing.override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
            if isinstance(event.data, audio.FlushData):
                self.flushed_at.append(time.monotonic())
            return super().dispatch(ctx, event)

    async def run() -> tuple[float, list[speaking.OutputData]]:
        mouth = Flushes()
        voice = await stage(SlowClose(), mouth=mouth)
        await voice.say("Hello there.")
        await wait_until(lambda: len(voice.speaker.frames) >= 3)
        stopped_at = time.monotonic()
        await voice.stop()
        await wait_until(lambda: bool(mouth.flushed_at))
        flush_delay = mouth.flushed_at[0] - stopped_at
        await wait_until(lambda: bool(voice.outputs()), timeout=5)
        return flush_delay, voice.outputs()

    flush_delay, outputs = asyncio.run(run())

    assert flush_delay < 0.5
    assert outputs[0].outcome == "stopped"


def test_a_stop_the_mouth_never_settles_ends_stopped_at_stop_timeout() -> None:
    class Mute(RecordingSpeaker):
        """Takes the frames and never reports them played, so a flush finds nothing of them to drop."""

        @typing.override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
            if isinstance(event.data, audio.OutputData):
                self.frames.append(event)
                done: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
                done.set_result(True)
                return done
            return super().dispatch(ctx, event)

    async def run() -> tuple[list[speaking.OutputData], float]:
        voice = await stage(mouth=Mute(clock=instant()), stop_timeout=datetime.timedelta(milliseconds=50))
        await voice.say("Nobody hears this.")
        await wait_until(lambda: bool(voice.speaker.frames))
        await voice.stop()
        await wait_until(lambda: bool(voice.outputs()))
        return voice.outputs(), voice.copies()[-1].duration

    outputs, last_copy = asyncio.run(run())

    assert outputs[0].outcome == "stopped"
    assert (outputs[0].spoken, outputs[0].unspoken) == ("", "Nobody hears this.")
    assert last_copy == 0.0


def _counts(reader: metric_reader.Reader, name: str) -> collections.Counter[str]:
    return collections.Counter({str(point.attributes["outcome"]): int(point.value) for point in reader.points(name)})


def test_drafted_text_is_spoken_while_it_is_written_and_the_final_text_is_not_said_twice(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = _counts(reader, "bot.speaking.drafted.count")

        async def run() -> tuple[Stage, list[str], list[speaking.OutputData]]:
            voice = await stage()
            await voice.draft("Sure. I can", 0)
            # A streaming-input encoder says what was written before the rest of the reply exists.
            await wait_until(lambda: any(item.spoken_seconds > 0.0 for item in voice.progress()))
            read_while_drafting = list(voice.encoder.reads[0])
            await voice.draft(" help with that", 11)
            await voice.say("Sure. I can help with that.")
            await wait_until(lambda: bool(voice.outputs()))
            return voice, read_while_drafting, voice.outputs()

        voice, read_while_drafting, outputs = asyncio.run(run())
        after = _counts(reader, "bot.speaking.drafted.count")

    assert read_while_drafting == ["Sure. I can"]
    # The deltas, then only the rest of the final text: nothing is cut, joined or said twice.
    assert voice.encoder.reads == [["Sure. I can", " help with that", "."]]
    assert outputs == [
        speaking.OutputData(
            text="Sure. I can help with that.",
            spoken="Sure. I can help with that.",
            unspoken="",
            outcome="completed",
            sample_rate_hz=RATE,
            channels=1,
        )
    ]
    assert after - before == collections.Counter({"used": 1})


def test_a_long_drafted_reply_is_written_and_spoken_in_full() -> None:
    """W3-20: no whole-reply cap, drafted or not; every delta and the rest of the final reach the encoder."""

    parts = ["word " * 400] * 4 + ["more " * 99 + "more."]
    text = "".join(parts)
    assert len(text) == 8_500

    async def run() -> tuple[Stage, list[speaking.OutputData]]:
        voice = await stage()
        for index, words in enumerate(parts[:-1]):
            await voice.draft(words, 0, part=index)
        await voice.draft(parts[-1][:-1], 0, part=len(parts) - 1)
        await voice.say(*parts)
        await wait_until(lambda: bool(voice.outputs()), timeout=20)
        return voice, voice.outputs()

    voice, outputs = asyncio.run(run())

    assert voice.encoder.reads == [[*parts[:-1], parts[-1][:-1], "."]]
    assert (outputs[0].outcome, outputs[0].spoken, outputs[0].unspoken) == ("completed", text, "")


def test_a_whole_text_encoder_reads_one_joined_text_and_speaks_only_after_the_final() -> None:
    async def run() -> tuple[int, Stage, list[speaking.OutputData]]:
        voice = await stage(Script(whole=True))
        await voice.draft("Sure. ", 0)
        await voice.draft("I can help.", 6)
        await asyncio.sleep(0.05)
        commanded_before_the_final = len(voice.speaker.frames)
        await voice.say("Sure. I can help.")
        await wait_until(lambda: bool(voice.outputs()))
        return commanded_before_the_final, voice, voice.outputs()

    commanded, voice, outputs = asyncio.run(run())

    assert commanded == 0
    assert voice.encoder.reads == [["Sure. ", "I can help."]]
    assert (outputs[0].spoken, outputs[0].outcome) == ("Sure. I can help.", "completed")


def test_a_final_that_diverges_from_what_was_drafted_keeps_what_was_said_and_splices_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = _counts(reader, "bot.speaking.drafted.count")

        async def run() -> tuple[Stage, list[speaking.OutputData]]:
            voice = await stage()
            await voice.draft("Yes. Then more", 0)
            await wait_until(lambda: any(item.spoken_seconds > 0.0 for item in voice.progress()))
            await voice.say("Nope. Then something else.")
            await wait_until(lambda: bool(voice.outputs()))
            return voice, voice.outputs()

        voice, outputs = asyncio.run(run())
        after = _counts(reader, "bot.speaking.drafted.count")

    assert voice.encoder.reads == [["Yes. Then more"]]
    assert outputs[0].text == "Nope. Then something else."
    assert (outputs[0].spoken, outputs[0].unspoken) == ("Yes. Then more", "")
    assert outputs[0].outcome == "completed"
    assert after - before == collections.Counter({"diverged": 1})


def test_a_repeated_delta_is_ignored_and_a_gap_is_counted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = _counts(reader, "bot.processing.draft.count")

        async def run() -> tuple[Stage, list[speaking.OutputData]]:
            voice = await stage()
            await voice.draft("Hel", 0)
            await voice.draft("Hel", 0)
            await voice.draft("lo", 3)
            await voice.draft("lo th", 3)
            await voice.draft("never written", 50)
            await voice.say("Hello there.")
            await wait_until(lambda: bool(voice.outputs()))
            return voice, voice.outputs()

        voice, outputs = asyncio.run(run())
        after = _counts(reader, "bot.processing.draft.count")

    assert voice.encoder.reads == [["Hel", "lo", " th", "ere."]]
    assert (outputs[0].text, outputs[0].spoken) == ("Hello there.", "Hello there.")
    assert after - before == collections.Counter({"gap": 1})


def test_drafts_that_stop_coming_end_abandoned_at_draft_timeout_with_what_was_said() -> None:
    async def run() -> list[speaking.OutputData]:
        voice = await stage(draft_timeout=datetime.timedelta(milliseconds=50))
        await voice.draft("Hello there. And then the", 0)
        await wait_until(lambda: bool(voice.outputs()))
        return voice.outputs()

    outputs = asyncio.run(run())

    # What was written is still said, the unfinished end too, and the utterance ends abandoned.
    assert outputs[0].outcome == "abandoned"
    assert outputs[0].text == "Hello there. And then the"
    assert outputs[0].spoken == "Hello there. And then the"


def test_a_stop_while_drafting_closes_the_pipe_the_encoder_reads() -> None:
    async def run() -> tuple[list[str], list[speaking.OutputData], Script]:
        voice = await stage(Script(whole=True))
        await voice.draft("Never finished", 0)
        await wait_until(lambda: bool(voice.encoder.reads and voice.encoder.reads[0]))
        await voice.stop()
        await wait_until(lambda: bool(voice.outputs()))
        # A pipe that is still open would never end this read.
        rest = await asyncio.wait_for(_read(voice.encoder.texts[0]), timeout=1.0)
        return rest, voice.outputs(), voice.encoder

    rest, outputs, script = asyncio.run(run())

    assert rest == ["Never finished"]
    assert script.closed == ["Never finished"]
    assert (outputs[0].outcome, outputs[0].text, outputs[0].spoken) == ("stopped", "Never finished", "")


async def _read(text: collections.abc.AsyncIterable[vocal.speech.Piece]) -> list[str]:
    return [piece.words async for piece in text]


def test_drafts_and_finals_for_a_finished_utterance_never_reopen_it() -> None:
    async def run() -> tuple[list[speaking.OutputData], list[list[str]], str]:
        voice = await stage()
        await voice.say("Done.", id="finished")
        await wait_until(lambda: bool(voice.outputs()))
        await wait_until(lambda: (voice.speaking.state() or "").endswith("/wired/voiced/idle"))
        await voice.draft("Done. And a late", 0, id="finished")
        await voice.say("Done.", id="finished")
        await asyncio.sleep(0.05)
        return voice.outputs(), voice.encoder.reads, voice.speaking.state() or ""

    outputs, reads, state = asyncio.run(run())

    assert len(outputs) == 1
    assert reads == [["Done."]]
    assert state.endswith("/wired/voiced/idle")


def test_the_next_utterance_waits_for_this_one() -> None:
    async def run() -> list[tuple[str, str]]:
        voice = await stage()
        await voice.draft("First thing", 0, id="first")
        await voice.draft("Second thing.", 0, id="second")
        await voice.say("First thing.", id="first")
        await voice.say("Second thing.", id="second")
        await wait_until(lambda: len(voice.outputs()) == 2)
        return [
            (event.id, event.data.spoken)
            for event in voice.requester.named(speaking.OutputEvent)
            if isinstance(event.data, speaking.OutputData)
        ]

    assert asyncio.run(run()) == [("first", "First thing."), ("second", "Second thing.")]


def test_one_utterance_at_a_time_the_next_waits_for_the_first_to_be_recorded() -> None:
    async def run() -> list[str]:
        voice = await stage()
        await voice.say("First thing.", id="first")
        await voice.say("Second thing.", id="second")
        await wait_until(lambda: len(voice.outputs()) == 2)
        return [event.id for event in voice.requester.named(speaking.OutputEvent)]

    assert asyncio.run(run()) == ["first", "second"]


def test_speaking_records_what_the_room_heard_as_one_trusted_outbound_message() -> None:
    async def run() -> tuple[list[conversation.Messages], str, str]:
        target = RecordingConversation()
        voice = await stage(Script(fail_on="broken"), conversation_targets=(target,))
        await start_ability_tree(voice.environment, target)
        await voice.say("Hello there. ", "broken off.")
        await wait_until(lambda: bool(target.outputs))
        return target.outputs, hsm.id(voice.speaking), hsm.id(target)

    histories, speaker_id, conversation_id = asyncio.run(run())

    assert all(len(history.messages) == 1 and history.messages[0].direction == "outbound" for history in histories)
    message = histories[0].messages[0]
    assert message.content == "Hello there. "
    assert message.provenance.event == conversation.AppendEvent.name
    assert message.provenance.source == speaker_id
    assert message.provenance.target == conversation_id
    # The bot has not yet heard itself, so it cannot say whose voice it was.
    assert message.participants == ()


def test_own_voice_is_enrolled_from_each_utterance_long_enough_until_hearing_tells_its_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P-12: no declared voice id. Speaking hands Listening a copy of its own voice with no id until hearing tells it
    which id that voice has; from then on the id is stamped on what it says, and no more copies are handed over."""

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = sum(int(point.value) for point in reader.points("bot.speaking.voice.learned.count"))

        async def run() -> tuple[Stage, list[conversation.Messages]]:
            target = RecordingConversation()
            voice = await stage(Script(chunk_ms=500), conversation_targets=(target,))
            await start_ability_tree(voice.environment, target)
            await voice.say("Short.", id="short")
            await wait_until(lambda: len(voice.outputs()) == 1)
            await voice.say("Long enough to learn from.", id="long")
            await wait_until(lambda: len(voice.outputs()) == 2)
            await voice.say("Still long enough to learn.", id="again")
            await wait_until(lambda: len(voice.outputs()) == 3)
            # A stranger claiming to be hearing teaches nothing; the ear Speaking was wired to does.
            stranger = Collector()
            _ = await mosfet.started(voice.environment, stranger, stranger.model)
            _ = await hsm.dispatch(
                voice.environment,
                voice.speaking,
                dataclasses.replace(
                    listening.EnrolledEvent.with_data(listening.EnrolledData(voice_id="not-mine")),
                    id="long",
                    source=hsm.id(stranger),
                ),
            )
            await voice.hear_own_voice(id="long")
            await voice.hear_own_voice("a-later-answer", id="again")
            await voice.say("Long enough, and known now.", id="known")
            await wait_until(lambda: len(voice.outputs()) == 4)
            await wait_until(lambda: len(target.outputs) == 4)
            return voice, target.outputs

        voice, histories = asyncio.run(run())
        learned = sum(int(point.value) for point in reader.points("bot.speaking.voice.learned.count")) - before

    enrollments = voice.ear.named(listening.EnrollEvent)
    assert [event.id for event in enrollments] == ["long", "again"]
    copy = enrollments[0].data
    assert isinstance(copy, listening.EnrollData)
    assert copy.voice_id is None
    assert (copy.segment.sample_rate_hz, copy.segment.channels, copy.segment.media_type) == (RATE, 1, "audio/pcm")
    assert math.isclose(copy.segment.duration_seconds, 2.5, abs_tol=1e-9)
    assert [output.voice_id for output in voice.outputs()] == [None, None, None, OWN_VOICE]
    assert histories[-1].messages[-1].participants == (
        mosfet.Participant(ref=f"voice:{OWN_VOICE}", role=mosfet.ParticipantRole.SOURCE),
    )
    assert learned == 1


def test_own_voice_enrollment_is_bounded_by_enroll_seconds() -> None:
    async def run() -> listening.EnrollData:
        voice = await stage(Script(chunk_ms=1_000), enroll_seconds=datetime.timedelta(seconds=2))
        await voice.say("Three whole seconds.")
        await wait_until(lambda: bool(voice.ear.named(listening.EnrollEvent)))
        return [event.data for event in voice.ear.events if isinstance(event.data, listening.EnrollData)][0]

    assert math.isclose(asyncio.run(run()).segment.duration_seconds, 2.0, abs_tol=1e-9)


def test_utterances_and_first_audio_are_measured(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def outcomes(points: tuple[metric_reader.Point, ...]) -> collections.Counter[str]:
        return collections.Counter({str(point.attributes["outcome"]): int(point.value) for point in points})

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        # Points are cumulative for the process: measure the difference this test makes.
        before = outcomes(reader.points("bot.speaking.utterance.count"))
        first_before = sum(point.count for point in reader.points("bot.speaking.first_audio.duration"))

        async def run() -> None:
            voice = await stage(Script(fail_on="broken", stall_on="stuck"))
            await voice.say("Fine.", id="fine")
            await wait_until(lambda: len(voice.outputs()) == 1)
            await voice.say("Then broken.", id="broken")
            await wait_until(lambda: len(voice.outputs()) == 2)
            await voice.say("Then stuck.", id="stuck")
            await wait_until(lambda: voice.encoder.pulled == 3 and len(voice.speaker.frames) == 3)
            await voice.stop(id="stuck")
            await wait_until(lambda: len(voice.outputs()) == 3)

        asyncio.run(run())
        after = outcomes(reader.points("bot.speaking.utterance.count"))
        first_audio = reader.points("bot.speaking.first_audio.duration")
        encoding = reader.points("bot.speaking.encoding.duration")

    assert after - before == collections.Counter({"completed": 1, "failed": 1, "stopped": 1})
    assert sum(point.count for point in first_audio) - first_before == 3
    assert {"completed", "failed", "cancelled"} <= {point.attributes["outcome"] for point in encoding}


class Deaf(audio.Speaker):
    """A mouth that never answers an attach."""

    @typing.override
    async def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> bool:
        del ctx, event
        return True


async def foreign_mouth() -> audio.Speaker:
    """A mouth already powered in another environment, which refuses a controller from this one."""

    mouth = audio.Speaker()
    _ = await mosfet.started(Environment(), mouth, typing.cast(hsm.Model, mouth.model))
    return mouth


@pytest.mark.parametrize("refusing", ["refuses", "never answers"])
def test_a_mouth_that_refuses_the_attach_fails_speakings_attach_and_its_owner_is_told(refusing: str) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]]]:
        environment = Environment()
        owner, requester = Collector(), Collector()
        for actor in (owner, requester):
            _ = await mosfet.started(environment, actor, actor.model)
        mouth = await foreign_mouth() if refusing == "refuses" else Deaf()
        voice = speaking.Speaking(encoder=Script(), speaker=mouth)
        _ = await voice.attach(
            environment,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner, timeout=datetime.timedelta(milliseconds=50)), "attach-speaking"
            ),
        )
        await wait_until(lambda: bool(owner.events))
        _ = await hsm.dispatch(
            environment,
            voice,
            dataclasses.replace(
                speaking.InputEvent.with_data(speaking.InputData(parts=(vocal.speech.Part(words="Anyone?"),))),
                id="refused",
                source=hsm.id(requester),
            ),
        )
        _ = await hsm.dispatch(
            environment,
            voice,
            dataclasses.replace(
                speaking.VoiceEvent.with_data(speaking.VoiceData(voice="v-3f2a9c41d07e")),
                id="voice",
                source=hsm.id(requester),
            ),
        )
        await wait_until(lambda: len(requester.named(speaking.Speaking.failed_event)) == 2)
        return (
            owner.named(attachment.AttachFailedEvent),
            owner.named(attachment.AttachCompleteEvent),
            requester.named(speaking.Speaking.failed_event),
        )

    refused, completed, failures = asyncio.run(run())

    assert completed == []
    assert [event.id for event in refused] == ["attach-speaking"]
    assert [event.id for event in failures] == ["refused", "voice"]
    assert {typing.cast(ability.FailureData, event.data).message for event in failures} == {"Speaking is not attached."}


def test_speaking_powers_an_unstarted_mouth_and_stops_only_the_mouth_it_started() -> None:
    async def run() -> tuple[bool, bool, bool, bool]:
        environment = Environment()
        owned, shared = audio.Speaker(), audio.Speaker()
        _ = await mosfet.started(environment, shared, typing.cast(hsm.Model, shared.model))
        powering = speaking.Speaking(encoder=Script(), speaker=owned)
        borrowing = speaking.Speaking(encoder=Script(), speaker=shared)
        await start_ability_tree(environment, powering)
        await start_ability_tree(environment, borrowing)
        powered = lifecycle.is_started(owned)
        await wait_until(lambda: borrowing in device_bots(shared))
        await powering.stop(environment)
        await borrowing.stop(environment)
        await wait_until(lambda: borrowing not in device_bots(shared))
        return powered, lifecycle.is_started(owned), lifecycle.is_started(shared), borrowing in device_bots(shared)

    powered, owned_after, shared_after, still_wired = asyncio.run(run())

    assert powered
    assert not owned_after
    assert shared_after
    assert not still_wired


def test_cognition_does_not_directly_select_speaking_output() -> None:
    """Body output wiring is absent from cognition's actor/tool inventory."""

    from mosfet.device import Device
    from tests.bot.test_bot import as_cognition

    class SpeakProcessor(processing.Processor):
        inputs: list[processing.InputData]

        def __init__(self) -> None:
            self.inputs = []

        @typing.override
        async def process(
            self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
        ) -> processing.Events:
            self.inputs.append(input)
            return ()

    async def run() -> tuple[list[list[str]], list[processing.InputData]]:
        script = Script()
        speaking_ability = speaking.Speaking(encoder=script, speaker=audio.Speaker())
        processor = SpeakProcessor()

        class Probe(mosfet.Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    output=(speaking_ability,),
                )

        probe = Probe()
        environment = Environment()
        _ = await probe.attach(environment)
        await wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await wait_until(lambda: (speaking_ability.state() or "").endswith("/wired/voiced/idle"))
        _ = await probe.dispatch(
            environment,
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=0)),
        )
        await wait_until(lambda: bool(processor.inputs))
        return script.reads, processor.inputs

    calls, inputs = asyncio.run(run())
    assert calls == []
    assert "bot.ability.speaking.input" not in {event.name for event in inputs[0].schemas}
    assert "speaking" not in inputs[0].actors


# ----- parts --------------------------------------------------------------------------------------------------------

LAUGH = vocal.speech.Expression(sound="laugh")
WHISPER = vocal.speech.Expression(style="whisper", emotion="sly")


def test_a_reply_of_parts_goes_to_the_encoder_part_by_part_each_opening_with_its_expression() -> None:
    async def run() -> tuple[Stage, list[speaking.OutputData]]:
        voice = await stage()
        await voice.say(vocal.speech.Part(words="Oh, ", expression=LAUGH), "you got me.")
        await wait_until(lambda: bool(voice.outputs()))
        return voice, voice.outputs()

    voice, outputs = asyncio.run(run())

    assert voice.encoder.pieces == [
        [
            vocal.speech.Piece(part=0, words="Oh, ", expression=LAUGH),
            vocal.speech.Piece(part=1, words="you got me."),
        ]
    ]
    # The text of record is the words joined verbatim; the sound was never words.
    assert (outputs[0].text, outputs[0].spoken, outputs[0].unspoken) == ("Oh, you got me.", "Oh, you got me.", "")


def test_drafts_write_each_part_as_it_is_written_and_a_draft_for_the_next_part_opens_it() -> None:
    async def run() -> tuple[Stage, list[vocal.speech.Piece], list[speaking.OutputData]]:
        voice = await stage()
        await voice.draft("Oh", 0, expression=LAUGH)
        await voice.draft(", ", 2)
        await voice.draft("you", 0, part=1, expression=WHISPER)
        await wait_until(lambda: len(voice.encoder.pieces[0]) == 3)
        drafted = list(voice.encoder.pieces[0])
        await voice.say(
            vocal.speech.Part(words="Oh, ", expression=LAUGH),
            vocal.speech.Part(words="you got me.", expression=WHISPER),
            vocal.speech.Part(words=" Again.", expression=LAUGH),
        )
        await wait_until(lambda: bool(voice.outputs()))
        return voice, drafted, voice.outputs()

    voice, drafted, outputs = asyncio.run(run())

    assert drafted == [
        vocal.speech.Piece(part=0, words="Oh", expression=LAUGH),
        vocal.speech.Piece(part=0, words=", "),
        vocal.speech.Piece(part=1, words="you", expression=WHISPER),
    ]
    # The final writes the rest of the part being drafted, then every part not drafted, whole and with its expression.
    assert voice.encoder.pieces[0][3:] == [
        vocal.speech.Piece(part=1, words=" got me."),
        vocal.speech.Piece(part=2, words=" Again.", expression=LAUGH),
    ]
    assert (outputs[0].text, outputs[0].spoken, outputs[0].outcome) == (
        "Oh, you got me. Again.",
        "Oh, you got me. Again.",
        "completed",
    )


def test_a_final_that_diverges_in_a_later_part_keeps_what_was_said(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = _counts(reader, "bot.speaking.drafted.count")

        async def run() -> tuple[Stage, list[speaking.OutputData]]:
            voice = await stage()
            await voice.draft("Oh, ", 0)
            await voice.draft("you got", 0, part=1)
            await wait_until(lambda: len(voice.encoder.pieces[0]) == 2)
            await voice.say("Oh, ", "they got me.", "Again.")
            await wait_until(lambda: bool(voice.outputs()))
            return voice, voice.outputs()

        voice, outputs = asyncio.run(run())
        after = _counts(reader, "bot.speaking.drafted.count")

    assert voice.encoder.reads == [["Oh, ", "you got"]]
    assert outputs[0].text == "Oh, they got me.Again."
    assert (outputs[0].spoken, outputs[0].unspoken) == ("Oh, you got", "")
    assert after - before == collections.Counter({"diverged": 1})


def test_a_draft_that_skips_a_part_or_adds_to_a_finished_one_is_a_gap(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = _counts(reader, "bot.processing.draft.count")

        async def run() -> Stage:
            voice = await stage()
            await voice.draft("One. ", 0)
            await voice.draft("Two", 0, part=1)
            await voice.draft("One. ", 0)  # a repeat of a finished part is harmless
            await voice.draft("And more. ", 5)  # but more of a finished part is a gap
            await voice.draft("Four.", 0, part=3)  # and so is a part that skips one
            await voice.draft(" more", 3, part=1, expression=LAUGH)  # an expression mid-part is not an opening
            await voice.say("One. ", "Two more. ", "Three. ", "Four.")
            await wait_until(lambda: bool(voice.outputs()))
            return voice

        voice = asyncio.run(run())
        after = _counts(reader, "bot.processing.draft.count")

    assert voice.encoder.pieces[0] == [
        vocal.speech.Piece(part=0, words="One. "),
        vocal.speech.Piece(part=1, words="Two"),
        vocal.speech.Piece(part=1, words=" more"),
        vocal.speech.Piece(part=1, words=". "),
        vocal.speech.Piece(part=2, words="Three. "),
        vocal.speech.Piece(part=3, words="Four."),
    ]
    assert voice.outputs()[0].spoken == "One. Two more. Three. Four."
    assert after - before == collections.Counter({"gap": 2})


EXPRESSIVE = vocal.speech.Expressiveness(
    cues=frozenset({"emotion", "pace", "sound", "style"}),
    emotions=("amused", "calm"),
    styles=("whisper",),
    pace=vocal.speech.Range(min=0.8, max=1.2),
    sounds=frozenset({"laugh"}),
)


def test_each_part_reports_what_its_voice_applied_clamped_and_could_not_carry(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:

        def expression_counts() -> collections.Counter[tuple[str, str]]:
            return collections.Counter(
                {
                    (str(point.attributes["cue"]), str(point.attributes["outcome"])): int(point.value)
                    for point in reader.points("bot.speaking.expression.count")
                }
            )

        before = expression_counts()

        async def run() -> list[speaking.OutputData]:
            voice = await stage(Script(expressiveness=EXPRESSIVE))
            await voice.say(
                vocal.speech.Part(words="Ha, ", expression=vocal.speech.Expression(sound="laugh", emotion="amused")),
                "plain words, ",
                vocal.speech.Part(
                    words="got you.",
                    expression=vocal.speech.Expression(pace=1.5, pitch=2.0, style="shout", emotion="smug"),
                ),
                vocal.speech.Part(words=" Phew.", expression=vocal.speech.Expression(sound="sigh", pace=1.0)),
                id="expressive",
            )
            await wait_until(lambda: len(voice.outputs()) == 1)
            await voice.say("Nothing to direct.", id="plain")
            await wait_until(lambda: len(voice.outputs()) == 2)
            return voice.outputs()

        outputs = asyncio.run(run())
        after = expression_counts()

    assert outputs[0].expression == (
        speaking.PartReport(applied=("emotion", "sound")),
        speaking.PartReport(),
        speaking.PartReport(applied=("pace",), clamped=("pace",), unsupported=("emotion", "pitch", "style")),
        speaking.PartReport(applied=("pace",), unsupported=("sound",)),
    )
    assert outputs[1].expression is None
    assert after - before == collections.Counter(
        {
            ("emotion", "applied"): 1,
            ("sound", "applied"): 1,
            ("pace", "applied"): 2,
            ("pace", "clamped"): 1,
            ("emotion", "unsupported"): 1,
            ("pitch", "unsupported"): 1,
            ("style", "unsupported"): 1,
            ("sound", "unsupported"): 1,
        }
    )


def test_a_cue_set_or_changed_only_on_the_final_is_late_for_a_part_already_begun() -> None:
    async def run() -> list[speaking.OutputData]:
        voice = await stage(Script(expressiveness=EXPRESSIVE))
        await voice.draft("Well, ", 0)
        await voice.draft("then", 0, part=1, expression=vocal.speech.Expression(emotion="calm"))
        await wait_until(lambda: len(voice.encoder.pieces[0]) == 2)
        await voice.say(
            vocal.speech.Part(words="Well, ", expression=vocal.speech.Expression(emotion="amused")),
            vocal.speech.Part(words="then.", expression=vocal.speech.Expression(emotion="amused", sound="laugh")),
            vocal.speech.Part(words=" Bye.", expression=WHISPER),
        )
        await wait_until(lambda: bool(voice.outputs()))
        return voice.outputs()

    outputs = asyncio.run(run())

    assert outputs[0].expression == (
        speaking.PartReport(late=("emotion",)),
        speaking.PartReport(applied=("emotion",), late=("emotion", "sound")),
        # Not drafted, so it opened with the final's expression: nothing about it is late.
        speaking.PartReport(applied=("style",), unsupported=("emotion",)),
    )


def test_with_whole_text_alignment_no_part_of_a_reply_is_said_until_its_last_chunk_plays() -> None:
    """R6-6a: an encoder that says a run of parts in one request aligns the run's words to its last chunk, so a stop
    before that chunk plays leaves every part of the run unspoken, and once it plays every part is said."""

    async def run() -> tuple[list[speaking.OutputData], list[speaking.OutputData]]:
        stopped = await stage(Script(chunk_ms=200, whole=True), mouth=RecordingSpeaker(clock=time.monotonic))
        await stopped.say(vocal.speech.Part(words="Oh, ", expression=LAUGH), "you got me. ", "Again.")
        await wait_until(lambda: any(item.spoken_seconds > 0.0 for item in stopped.progress()))
        await stopped.stop()
        await wait_until(lambda: bool(stopped.outputs()))
        said = await stage(Script(whole=True))
        await said.say(vocal.speech.Part(words="Oh, ", expression=LAUGH), "you got me.")
        await wait_until(lambda: bool(said.outputs()))
        return stopped.outputs(), said.outputs()

    stopped, said = asyncio.run(run())

    assert (stopped[0].outcome, stopped[0].spoken, stopped[0].unspoken) == ("stopped", "", "Oh, you got me. Again.")
    assert (said[0].outcome, said[0].spoken, said[0].unspoken) == ("completed", "Oh, you got me.", "")
    assert said[0].expression == (speaking.PartReport(unsupported=("sound",)), speaking.PartReport())


# ----- the voice ----------------------------------------------------------------------------------------------------


class Factory:
    """Constructs a ``Script`` voice from the key and description it is given, and remembers every call."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.made: list[Script] = []
        self.refuse: bool = refuse

    def __call__(self, key: str, description: str | None) -> vocal.speech.SpeechEncoder:
        self.calls.append((key, description))
        if self.refuse:
            raise ValueError(f"no voice {key!r} here")
        script = Script(expressiveness=EXPRESSIVE)
        self.made.append(script)
        return script


async def give_voice(voice: Stage, key: str, description: str | None = None, *, id: str = "voice-1") -> None:
    _ = await hsm.dispatch(
        voice.environment,
        voice.speaking,
        dataclasses.replace(
            speaking.VoiceEvent.with_data(speaking.VoiceData(voice=key, description=description)),
            id=id,
            source=hsm.id(voice.requester),
            target=hsm.id(voice.speaking),
        ),
    )


def _voicings(reader: metric_reader.Reader) -> collections.Counter[str]:
    return _counts(reader, "bot.speaking.voicing.count")


def test_built_with_a_factory_speaking_holds_speech_until_it_is_given_its_voice_and_makes_it_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = _voicings(reader)

        async def run() -> tuple[Stage, Factory, str, list[speaking.OutputData]]:
            factory = Factory()
            voice = await stage(factory=factory)
            await voice.draft("Hello ", 0)
            await voice.say("Hello there.")
            await asyncio.sleep(0.05)
            held = voice.speaking.state() or ""
            assert voice.outputs() == []
            await give_voice(voice, "v-3f2a9c41d07e", "A warm, unhurried voice.")
            await wait_until(lambda: bool(voice.outputs()))
            await give_voice(voice, "v-another", id="voice-2")
            await asyncio.sleep(0.02)
            return voice, factory, held, voice.outputs()

        voice, factory, held, outputs = asyncio.run(run())
        after = _voicings(reader)

    assert held.endswith("/wired/unvoiced/holding")
    # Exactly one encoder, constructed from the voice the bot was given; a second voice changes nothing.
    assert factory.calls == [("v-3f2a9c41d07e", "A warm, unhurried voice.")]
    assert factory.made[0].reads == [["Hello ", "there."]]
    assert (outputs[0].outcome, outputs[0].spoken) == ("completed", "Hello there.")
    (voiced,) = voice.requester.named(speaking.VoicedEvent)
    assert (voiced.id, voiced.source) == ("voice-1", hsm.id(voice.speaking))
    assert voiced.data == speaking.VoicedData(voice="v-3f2a9c41d07e", expressiveness=EXPRESSIVE)
    assert (voice.speaking.state() or "").endswith("/wired/voiced/idle")
    assert after - before == collections.Counter({"voiced": 1, "dropped": 1})


def test_speech_held_past_voicing_timeout_fails_unavailable_until_a_voice_comes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, finished_spans: InMemorySpanExporter
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = _voicings(reader)

        async def run() -> tuple[Stage, list[hsm.Event[typing.Any]], list[speaking.OutputData]]:
            voice = await stage(factory=Factory(), voicing_timeout=datetime.timedelta(milliseconds=50))
            await voice.draft("Too ", 0, id="early")
            await voice.draft("soon", 4, id="early")
            await voice.say("Too soon.", id="early")
            await wait_until(lambda: len(voice.requester.named(speaking.Speaking.failed_event)) == 2)
            await voice.say("Still nothing.", id="later")
            await wait_until(lambda: len(voice.requester.named(speaking.Speaking.failed_event)) == 3)
            await give_voice(voice, "v-3f2a9c41d07e")
            await wait_until(lambda: bool(voice.requester.named(speaking.VoicedEvent)))
            await voice.say("Now I can.", id="voiced")
            await wait_until(lambda: bool(voice.outputs()))
            return voice, voice.requester.named(speaking.Speaking.failed_event), voice.outputs()

        voice, failures, outputs = asyncio.run(run())
        after = _voicings(reader)

    # The held opening draft and the final each get their terminal; a continuing draft is not a request of its own.
    assert [event.id for event in failures] == ["early", "early", "later"]
    assert {typing.cast(ability.FailureData, event.data).message for event in failures} == {
        "Speaking has no voice yet."
    }
    kinds = {
        dict(recorded.attributes or {}).get("bot.failure.kind")
        for recorded in finished_spans.get_finished_spans()
        if recorded.name == "bot.speaking.terminal"
    }
    assert kinds == {"unavailable"}
    assert [(output.text, output.outcome) for output in outputs] == [("Now I can.", "completed")]
    assert after - before == collections.Counter({"timed_out": 1, "voiced": 1})
    assert voice.speaking.state() is not None


def test_a_voice_the_factory_cannot_construct_fails_and_leaves_speaking_unvoiced(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = _voicings(reader)

        async def run() -> tuple[Stage, Factory, list[hsm.Event[typing.Any]], str]:
            factory = Factory(refuse=True)
            voice = await stage(factory=factory)
            await give_voice(voice, "v-gone")
            await wait_until(lambda: bool(voice.requester.named(speaking.Speaking.failed_event)))
            await asyncio.sleep(0.02)
            return voice, factory, voice.requester.named(speaking.Speaking.failed_event), voice.speaking.state() or ""

        voice, factory, failures, state = asyncio.run(run())
        after = _voicings(reader)

    assert factory.calls == [("v-gone", None)]
    assert [event.id for event in failures] == ["voice-1"]
    assert "no voice 'v-gone' here" in typing.cast(ability.FailureData, failures[0].data).message
    assert voice.requester.named(speaking.VoicedEvent) == []
    assert state.endswith("/wired/unvoiced/holding")
    assert after - before == collections.Counter({"failed": 1})


def test_speaking_built_with_its_encoder_is_voiced_and_drops_a_voice_it_is_given(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = _voicings(reader)

        async def run() -> Stage:
            voice = await stage()
            await give_voice(voice, "v-3f2a9c41d07e")
            await voice.say("Same voice.")
            await wait_until(lambda: bool(voice.outputs()))
            return voice

        voice = asyncio.run(run())
        after = _voicings(reader)

    assert voice.requester.named(speaking.VoicedEvent) == []
    assert voice.outputs()[0].outcome == "completed"
    assert after - before == collections.Counter({"dropped": 1})

"""``companion_example.evaluate.turns``: taking turns with the companion like a person, and measuring each exchange.

The speaker's playback model and the measures are pure, on a manual clock. The turn-taking machine runs on the
real clock with its windows scaled down: a fake mouth stands in for the room's microphone, and the companion's voice is
played on the evaluator's own speaker, as the room plays it.
"""

import asyncio
import collections.abc
import dataclasses
import time

import hsm
import mosfet
import pytest
from companion_example.evaluate import turns

# A second of 24 kHz mono s16le, as the companion's voice reaches the room's speaker.
_SECOND_24K = bytes(48_000)


class _Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now: float = now

    def __call__(self) -> float:
        return self.now


def _chunk(start: float, end: float, *, arrived: float | None = None) -> turns.ChunkData:
    seconds = end - start
    return turns.ChunkData(
        arrived=start if arrived is None else arrived,
        start=start,
        end=end,
        pcm=bytes(2 * round(seconds * 24_000)),
        sample_rate_hz=24_000,
        channels=1,
    )


def _exchange(**changes: object) -> turns.ExchangeData:
    base: dict[str, object] = {
        "exchange": 0,
        "said_from": 10.0,
        "said_until": 12.0,
        "cut_in_from": None,
        "cut_in_until": None,
        "chunks": (),
        "outcome": "replied",
    }
    base.update(changes)
    return turns.ExchangeData.model_validate(base)


def test_the_speaker_plays_each_chunk_after_the_one_before_for_as_long_as_it_lasts() -> None:
    clock = _Clock(10.0)
    heard: list[turns.ChunkData] = []
    speaker = turns.Speaker(clock=clock, heard=heard.append)

    speaker.play(_SECOND_24K, sample_rate_hz=24_000, channels=1)
    clock.now = 10.2
    # The companion's voice reaches the room faster than it is spoken: this one plays once the first has finished.
    speaker.play(_SECOND_24K, sample_rate_hz=24_000, channels=1)
    clock.now = 20.0
    speaker.play(_SECOND_24K[:24_000], sample_rate_hz=24_000, channels=1)

    assert [(chunk.arrived, chunk.start, chunk.end) for chunk in heard] == [
        (10.0, 10.0, 11.0),
        (10.2, 11.0, 12.0),
        (20.0, 20.0, 20.5),
    ]
    assert heard[0].pcm == _SECOND_24K
    # Stopped, it is quiet from then on.
    clock.now = 20.1
    speaker.stop()
    speaker.play(_SECOND_24K, sample_rate_hz=24_000, channels=1)
    assert (heard[-1].start, heard[-1].end) == (20.1, 21.1)


def test_a_reply_is_measured_from_the_end_of_what_the_person_said_to_companions_first_sound() -> None:
    measured = turns.measure(_exchange(chunks=(_chunk(13.5, 15.0), _chunk(15.0, 16.0))))

    assert measured.replied
    assert measured.latency_seconds == pytest.approx(1.5)
    assert measured.reply_seconds == pytest.approx(2.5)
    assert not measured.talk_over
    assert measured.cut_in_seconds is None


def test_no_sound_from_companion_is_no_reply_and_has_no_latency() -> None:
    measured = turns.measure(_exchange(outcome="no_reply"))

    assert not measured.replied
    assert measured.latency_seconds is None
    assert measured.reply_seconds == 0.0
    assert not measured.talk_over


def test_companions_sound_starting_while_the_person_speaks_is_talk_over() -> None:
    measured = turns.measure(_exchange(chunks=(_chunk(11.0, 13.0),)))

    assert measured.talk_over
    assert measured.replied
    # Nothing of the companion's began after the person finished: there is no latency to speak of.
    assert measured.latency_seconds is None


def test_after_a_cut_in_it_is_measured_how_long_companion_kept_playing() -> None:
    measured = turns.measure(
        _exchange(chunks=(_chunk(13.0, 16.0), _chunk(16.0, 17.5)), cut_in_from=14.0, cut_in_until=15.0)
    )

    # The person talked over the companion; the companion did not talk over the person.
    assert not measured.talk_over
    assert measured.cut_in_seconds == pytest.approx(3.5)


@dataclasses.dataclass
class _Mouth:
    """The room's microphone as the turn-taking machine uses it: it says a line, taking as long as the line lasts.

    ``replies`` is what the companion plays after each line it hears, and how long after the line it starts; ``during``
    plays the companion's voice while the person is still speaking.
    """

    speaker: turns.Speaker
    replies: collections.abc.Sequence[tuple[float, bytes]] = ()
    during: bytes | None = None
    said: list[tuple[float, float, bytes]] = dataclasses.field(default_factory=list[tuple[float, float, bytes]])

    async def say(self, pcm: bytes) -> turns.SpokenData:
        start = time.monotonic()
        if self.during is not None:
            self.speaker.play(self.during, sample_rate_hz=24_000, channels=1)
            self.during = None
        await asyncio.sleep(len(pcm) / 32_000)
        end = time.monotonic()
        self.said.append((start, end, pcm))
        if len(self.said) == 1:
            loop = asyncio.get_running_loop()
            for delay, reply in self.replies:
                _ = loop.call_later(
                    delay, lambda reply=reply: self.speaker.play(reply, sample_rate_hz=24_000, channels=1)
                )
        return turns.SpokenData(start=start, end=end)


_TIMING = turns.Timing(echo_seconds=0.2, pause_seconds=0.3, reply_timeout_seconds=0.6, cut_in_after_seconds=0.2)
# A tenth of a second of the person, at the room's microphone rate, and a quarter second of the companion.
_LINE = bytes(3_200)
_REPLY = bytes(12_000)


async def _exchanged(
    say: turns.SayData,
    *,
    replies: collections.abc.Sequence[tuple[float, bytes]] = (),
    during: bytes | None = None,
    before: bytes | None = None,
) -> tuple[turns.ExchangeData, _Mouth, float]:
    """One exchange on a started machine; ``before`` is the companion still playing when the line is given."""

    done: asyncio.Future[turns.ExchangeData] = asyncio.get_running_loop().create_future()
    machine: turns.Turns
    speaker = turns.Speaker(
        clock=time.monotonic,
        heard=lambda chunk: machine.dispatch(machine.context(), turns.PlayedEvent.with_data(chunk)),
    )
    mouth = _Mouth(speaker=speaker, replies=replies, during=during)
    machine = turns.Turns(mouth=mouth, clock=time.monotonic, timing=_TIMING, on_exchange=done.set_result)
    _ = await mosfet.started(None, machine, turns.Turns.model)
    try:
        if before is not None:
            speaker.play(before, sample_rate_hz=24_000, channels=1)
        given = time.monotonic()
        _ = await machine.dispatch(machine.context(), turns.SayEvent.with_data(say))
        return await asyncio.wait_for(done, 5), mouth, given
    finally:
        await hsm.stop(machine)


def test_the_person_speaks_only_once_companions_voice_has_died_away_and_the_echo_window_passed() -> None:
    exchanged, mouth, given = asyncio.run(
        _exchanged(turns.SayData(exchange=0, pcm=_LINE), before=_SECOND_24K, replies=((0.1, _REPLY),))
    )

    ((start, _, said),) = mouth.said
    assert said == _LINE
    assert start >= given + 1.0 + _TIMING.echo_seconds
    assert exchanged.said_from == start
    assert exchanged.outcome == "replied"
    measured = turns.measure(exchanged)
    assert measured.latency_seconds is not None and 0.05 <= measured.latency_seconds < 0.5
    assert measured.reply_seconds == pytest.approx(0.25)
    assert not measured.talk_over


def test_a_break_holds_the_line_until_its_time_even_in_a_quiet_room() -> None:
    async def run() -> tuple[turns.ExchangeData, _Mouth, float]:
        return await _exchanged(turns.SayData(exchange=0, pcm=_LINE, not_before=time.monotonic() + 0.5))

    exchanged, mouth, given = asyncio.run(run())

    ((start, _, _),) = mouth.said
    assert start >= given + 0.45
    assert exchanged.outcome == "no_reply"


def test_companion_saying_nothing_within_the_reply_timeout_is_no_reply() -> None:
    exchanged, mouth, _ = asyncio.run(_exchanged(turns.SayData(exchange=0, pcm=_LINE)))

    ((_, end, _),) = mouth.said
    assert exchanged.outcome == "no_reply"
    assert exchanged.chunks == ()
    assert exchanged.said_until == end
    assert not turns.measure(exchanged).replied


def test_a_reply_is_heard_until_companion_pauses_however_many_chunks_it_takes() -> None:
    exchanged, _, _ = asyncio.run(
        _exchanged(turns.SayData(exchange=0, pcm=_LINE), replies=((0.1, _REPLY), (0.2, _REPLY), (0.55, _REPLY)))
    )

    # The third chunk arrives while the first two still play: one reply, three chunks.
    assert len(exchanged.chunks) == 3
    assert turns.measure(exchanged).reply_seconds == pytest.approx(0.75)


def test_companion_starting_while_the_person_speaks_is_heard_as_its_reply_and_marked_talk_over() -> None:
    exchanged, _, _ = asyncio.run(_exchanged(turns.SayData(exchange=0, pcm=_LINE), during=_REPLY))

    assert exchanged.outcome == "replied"
    assert turns.measure(exchanged).talk_over


def test_a_person_cuts_in_while_companion_is_still_answering() -> None:
    exchanged, mouth, _ = asyncio.run(
        _exchanged(turns.SayData(exchange=0, pcm=_LINE, cut_in=bytes(1_600)), replies=((0.1, _SECOND_24K),))
    )

    assert [said for _, _, said in mouth.said] == [_LINE, bytes(1_600)]
    first = exchanged.chunks[0]
    assert exchanged.cut_in_from is not None and exchanged.cut_in_until is not None
    assert exchanged.cut_in_from >= first.start + _TIMING.cut_in_after_seconds
    # The companion was still playing when the person cut in.
    assert exchanged.cut_in_from < first.end
    measured = turns.measure(exchanged)
    assert measured.cut_in_seconds is not None and measured.cut_in_seconds > 0
    assert not measured.talk_over


def test_a_cut_in_companion_finished_before_is_never_said() -> None:
    exchanged, mouth, _ = asyncio.run(
        _exchanged(
            turns.SayData(exchange=0, pcm=_LINE, cut_in=bytes(1_600)),
            replies=((0.05, bytes(2_400)),),
        )
    )

    # The companion's reply lasted 0.05 s, shorter than the person waits before cutting in.
    assert len(mouth.said) == 1
    assert exchanged.cut_in_from is None
    assert exchanged.outcome == "replied"


def test_the_microphone_streams_in_real_time_and_tells_when_the_line_began_and_ended() -> None:
    async def run() -> tuple[list[bytes], turns.SpokenData, float]:
        blocks: list[bytes] = []
        microphone = turns.Microphone(clock=time.monotonic, block_seconds=0.05)
        microphone.start(blocks.append)
        try:
            await asyncio.sleep(0.12)
            began = time.monotonic()
            spoken = await microphone.say(b"\x01\x00" * 4_000)
            await asyncio.sleep(0.1)
            return blocks, spoken, began
        finally:
            microphone.stop()

    blocks, spoken, began = asyncio.run(run())

    # Silence, then the quarter second said in blocks of 0.05 s, then silence again.
    assert all(len(block) == 1_600 for block in blocks)
    assert blocks[0] == bytes(1_600)
    voiced = [block for block in blocks if any(block)]
    assert b"".join(voiced) == b"\x01\x00" * 4_000
    assert blocks[-1] == bytes(1_600)
    assert began <= spoken.start < spoken.end
    assert spoken.end - spoken.start == pytest.approx(0.2, abs=0.08)

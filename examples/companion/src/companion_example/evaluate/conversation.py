"""One evaluation: join the companion's room as its far side, hold each scenario's conversation, grade it, and keep it
all.

:func:`run` checks every person's voice is a catalog voice of the speech engine (never a designed one: that may
be the companion's own), then connects as the room ``peer`` with a ``room.Room`` on the evaluator's devices
(``turns.Microphone`` and ``turns.Speaker``) and a ``turns.Turns`` taking the person's turns. For each scenario, on
each turn: the persona writes the next line (or ends the conversation), the speech engine says it in the person's
voice, the person says it once the companion's voice has died away, and the companion's reply (what the room's speaker
played until the companion paused) is transcribed by the lab's speech to text. A break waits out its seconds in a quiet
room first. After the last turn the judge grades the conversation.

Everything is written to ``out`` as it is known, each file readable by its owner alone (``0600``):
``transcript.jsonl`` (every utterance, as ``record.Utterance``), ``audio/<scenario>-<turn>.wav`` (each exchange
as the room heard it, 16 kHz mono), and at the end ``scores.json`` (``report.Scores``) and ``report.html``.

A failure is :class:`Failed`: the room's session ended before the evaluation did, or a service (speech, speech to
text, the chat model) failed. What was kept until then stays.

Telemetry (meter scope ``companion_example.evaluate``): ``bot.example.evaluate.scenarios.count{outcome}`` (``graded``,
``ungraded``) and ``bot.example.evaluate.latency`` (s), with no attributes. Nothing said, no scenario and no
person is an attribute.
"""

import asyncio
import collections.abc
import contextlib
import dataclasses
import datetime
import pathlib
import typing

import hsm
import mosfet
import pydantic
from mosfet.abilities import encoding, hearing, vocal
from mosfet.abilities.vocal import voice
from mosfet.providers import openai_compat
from opentelemetry import metrics

from companion_example import room
from companion_example.evaluate import minds, record, report, scenarios, sound, turns

_METER = metrics.get_meter(turns.SCOPE)
_SCENARIOS = _METER.create_counter(
    "bot.example.evaluate.scenarios.count",
    unit="{scenario}",
    description="Scenarios held with the companion, by whether the judge graded them: graded, or ungraded.",
)
_LATENCY = _METER.create_histogram(
    "bot.example.evaluate.latency",
    unit="s",
    description="From the end of a person's line to the companion's first sound after it.",
)
_LEAVE_SECONDS = 10.0
_CONNECT_TIMEOUT: typing.Final = datetime.timedelta(seconds=10)


class Failed(Exception):
    """The evaluation could not finish: the room's session ended, or a service failed. The message says which."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class Services:
    """What the evaluator talks to besides the companion, injected.

    ``catalog`` lists the speech engine's voices; ``speak`` makes a speech encoder in one of them, by its key;
    ``transcribe`` turns 16 kHz mono s16le into text; ``persona`` and ``judge`` are the evaluator's minds, and
    ``judge_info`` says which model judges.
    """

    catalog: voice.Catalog
    speak: collections.abc.Callable[[str], vocal.speech.SpeechEncoder]
    transcribe: hearing.speech.SpeechDecoder
    persona: minds.Persona
    judge: minds.Judge
    judge_info: record.Judge


async def _spoken(encoder: vocal.speech.SpeechEncoder, words: str) -> bytes:
    """``words`` said in the encoder's voice, as the room's microphone hears it (16 kHz mono)."""

    text = vocal.speech.Text()
    text.write(vocal.speech.Piece(part=0, words=words))
    text.close()
    pcm = b"".join([chunk.pcm async for chunk in encoder.encode(text)])
    return sound.resampled(pcm, sample_rate_hz=encoder.format.sample_rate_hz, channels=encoder.format.channels)


async def _heard(transcribe: hearing.speech.SpeechDecoder, pcm: bytes) -> str:
    """What speech to text heard in ``pcm``; empty when it heard no words."""

    if not pcm:
        return ""
    try:
        return (await transcribe.decode(pcm)).decode("utf-8")
    except openai_compat.SpeechDecodingError as error:
        if error.kind == "decode_failed":
            return ""
        raise


@dataclasses.dataclass
class _Room:
    """The evaluator in the companion's room: its far side, its turn-taking, and what each tells."""

    far_side: room.Room
    taking: turns.Turns
    microphone: turns.Microphone
    closed: asyncio.Future[room.ClosedData]
    exchanged: asyncio.Queue[turns.ExchangeData]

    async def until[T](self, awaitable: collections.abc.Awaitable[T]) -> T:
        """``awaitable``'s result, unless the room's session ends first.

        Raises:
            Failed: the session ended.
        """

        waiting = asyncio.ensure_future(awaitable)
        done, _ = await asyncio.wait({waiting, self.closed}, return_when=asyncio.FIRST_COMPLETED)
        if waiting not in done:
            _ = waiting.cancel()
            ended = self.closed.result()
            detail = "" if ended.code is None else f" ({ended.code})"
            raise Failed(f"the room's session ended: {ended.reason}{detail}")
        return waiting.result()


@contextlib.asynccontextmanager
async def _joined(
    *,
    url: str,
    peer: str,
    token: pydantic.SecretStr,
    clock: turns.Clock,
    timing: turns.Timing,
    connect: room.Connect,
    connect_timeout: datetime.timedelta,
) -> collections.abc.AsyncGenerator[_Room]:
    """Join the companion's room ``peer`` as its far side, and leave it after."""

    loop = asyncio.get_running_loop()
    closed: asyncio.Future[room.ClosedData] = loop.create_future()
    exchanged: asyncio.Queue[turns.ExchangeData] = asyncio.Queue()
    microphone = turns.Microphone(clock=clock)
    taking = turns.Turns(mouth=microphone, clock=clock, timing=timing, on_exchange=exchanged.put_nowait)
    speaker = turns.Speaker(
        clock=clock, heard=lambda chunk: taking.dispatch(taking.context(), turns.PlayedEvent.with_data(chunk))
    )
    far_side = room.Room(
        url=url,
        peer=peer,
        token=token,
        microphone=microphone,
        speaker=speaker,
        on_closed=closed.set_result,
        connect=connect,
        connect_timeout=connect_timeout,
        open_timeout=connect_timeout,
    )
    _ = await mosfet.started(None, taking, turns.Turns.model)
    _ = await mosfet.started(None, far_side, room.Room.model)
    joined = _Room(far_side=far_side, taking=taking, microphone=microphone, closed=closed, exchanged=exchanged)
    try:
        _ = await joined.until(microphone.streaming.wait())
        yield joined
    finally:
        if not closed.done():
            _ = await far_side.dispatch(far_side.context(), room.LeaveEvent.with_data(room.LeaveData()))
            with contextlib.suppress(TimeoutError):
                _ = await asyncio.wait_for(asyncio.shield(closed), _LEAVE_SECONDS)
        await hsm.stop(far_side)
        await hsm.stop(taking)


def _companion_note(
    measures: turns.Measures, exchange: turns.ExchangeData, person: str, timing: turns.Timing
) -> str | None:
    notes: list[str] = []
    if not measures.replied:
        notes.append(f"{scenarios.COMPANION_NAME} said nothing within {timing.reply_timeout_seconds:g} s")
    if measures.talk_over:
        notes.append(f"{scenarios.COMPANION_NAME} began while {person} was still speaking")
    if measures.cut_in_seconds is not None and exchange.cut_in_from is not None:
        notes.append(f"{scenarios.COMPANION_NAME} kept talking {measures.cut_in_seconds:.1f} s after {person} cut in")
    return "; ".join(notes) or None


async def run(
    chosen: collections.abc.Sequence[scenarios.Scenario],
    *,
    url: str,
    peer: str,
    token: pydantic.SecretStr,
    services: Services,
    out: pathlib.Path,
    timing: turns.Timing,
    clock: turns.Clock,
    connect: room.Connect = room.connect_websocket,
    connect_timeout: datetime.timedelta = _CONNECT_TIMEOUT,
) -> report.Scores:
    """Hold, grade and keep every scenario in ``chosen`` with the companion in room ``peer``; return the scores.

    ``out`` must exist; what is kept goes in it.

    Raises:
        Failed: a voice is not a catalog voice, the room's session ended, or a service failed.
    """

    try:
        scenarios.check_voices(chosen, await services.catalog.voices())
    except voice.Error as error:
        raise Failed(f"the speech engine's voices could not be listed: {error.kind}") from error
    except ValueError as error:
        raise Failed(str(error)) from error
    audio = out / "audio"
    audio.mkdir(mode=0o700, exist_ok=True)
    began = clock()
    kept: list[str] = []
    results: list[record.ScenarioResult] = []
    serial = 0

    def keep(utterance: record.Utterance) -> None:
        kept.append(utterance.model_dump_json())
        report.write_private(out / "transcript.jsonl", "".join(f"{line}\n" for line in kept).encode())

    async with _joined(
        url=url,
        peer=peer,
        token=token,
        clock=clock,
        timing=timing,
        connect=connect,
        connect_timeout=connect_timeout,
    ) as joined:
        for scenario in chosen:
            people = {person.id: person for person in scenario.people}
            breaks = {pause.after_turn: pause for pause in scenario.breaks}
            said: list[record.Utterance] = []
            exchanges: list[record.Exchange] = []
            ended: record.Ended = "turns_used"
            for turn in range(scenario.max_turns):
                back_from = breaks.get(turn)
                try:
                    line = await services.persona.next(
                        scenario, said, turn=turn, cut_in=turn in scenario.interrupt_turns, back_from=back_from
                    )
                    if line is None:
                        ended = "done"
                        break
                    person = people[line.speaker]
                    encoder = services.speak(person.voice)
                    pcm = await _spoken(encoder, line.say)
                    cut_in = None if line.cut_in is None else await _spoken(encoder, line.cut_in)
                except minds.Unanswered:
                    ended = "persona_unanswered"
                    break
                except (openai_compat.RequestError, encoding.Error) as error:
                    raise Failed(f"a service failed: {type(error).__name__}") from error
                say = turns.SayData(
                    exchange=serial,
                    pcm=pcm,
                    cut_in=cut_in,
                    not_before=0.0 if back_from is None else clock() + back_from.seconds,
                )
                serial += 1
                _ = await joined.taking.dispatch(joined.taking.context(), turns.SayEvent.with_data(say))
                exchange = await joined.until(joined.exchanged.get())
                measures = turns.measure(exchange)
                if measures.latency_seconds is not None:
                    _LATENCY.record(measures.latency_seconds)
                try:
                    reply = await _heard(services.transcribe, sound.reply(exchange.chunks))
                except openai_compat.SpeechDecodingError as error:
                    raise Failed(f"speech to text failed: {error.kind}") from error
                name = f"{scenario.id}-{turn}.wav"
                report.write_private(audio / name, sound.wav(sound.mixed(exchange, line=pcm, cut_in=cut_in)))
                exchanges.append(
                    record.Exchange(
                        scenario=scenario.id, exchange=turn, speaker=person.id, measures=measures, audio=f"audio/{name}"
                    )
                )
                spoken = [
                    record.Utterance(
                        scenario=scenario.id,
                        exchange=turn,
                        speaker=person.id,
                        name=person.name,
                        text=line.say,
                        started_s=exchange.said_from - began,
                        ended_s=exchange.said_until - began,
                    ),
                    record.Utterance(
                        scenario=scenario.id,
                        exchange=turn,
                        speaker=record.COMPANION,
                        name=scenarios.COMPANION_NAME,
                        text=reply,
                        note=_companion_note(measures, exchange, person.name, timing),
                        started_s=exchange.chunks[0].start - began if exchange.chunks else None,
                        ended_s=exchange.chunks[-1].end - began if exchange.chunks else None,
                    ),
                ]
                if line.cut_in is not None and exchange.cut_in_from is not None and exchange.cut_in_until is not None:
                    spoken.append(
                        record.Utterance(
                            scenario=scenario.id,
                            exchange=turn,
                            speaker=person.id,
                            name=person.name,
                            text=line.cut_in,
                            note=f"cut in while {scenarios.COMPANION_NAME} was answering",
                            started_s=exchange.cut_in_from - began,
                            ended_s=exchange.cut_in_until - began,
                        )
                    )
                for utterance in spoken:
                    said.append(utterance)
                    keep(utterance)
            try:
                graded = (
                    await services.judge.grade(scenario, said)
                    if exchanges
                    else record.Ungraded(reason="nothing was said")
                )
            except openai_compat.RequestError as error:
                raise Failed(f"the judge's model failed: {type(error).__name__}") from error
            _SCENARIOS.add(1, {"outcome": "graded" if isinstance(graded, record.Verdict) else "ungraded"})
            results.append(
                record.ScenarioResult(
                    id=scenario.id,
                    title=scenario.title,
                    criteria=scenario.criteria,
                    people=scenario.people,
                    ended=ended,
                    utterances=tuple(said),
                    exchanges=tuple(exchanges),
                    graded=graded,
                )
            )
    evaluated = record.Results(judge=services.judge_info, scenarios=tuple(results))
    scored = report.scores(evaluated)
    report.write_private(out / "scores.json", scored.model_dump_json(indent=2).encode())
    report.write_private(out / "report.html", report.html(evaluated, scored).encode())
    return scored


__all__ = ["Failed", "Services", "run"]

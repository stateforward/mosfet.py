#!/usr/bin/env -S uv run --project examples/phone_bot python
"""Live Learning proof: experience a real ring, hear a spoken lesson, answer the next ring.

Three live phases against a real ``Phone`` device. Nothing about the ring is synthesized:
the phone firmware produces the stimulus and this script only observes typed events.

1. **Experience** — an injected :class:`~bot.devices.phone.PhoneEventRecorder` delivers a
   provider incoming call. Firmware rings and elevates that ring to a real ``environment.sound``
   stimulus (``PhoneSoundData`` with the packaged ring WAV plus ``caller``). Intuition
   deliberates that turn live against the phone's offered call events, and the turn is
   stored in memory as a cognitive episode.
2. **Lesson** — macOS ``say`` renders *When the phone rings make sure you answer it* to WAV.
   Learning decodes the lesson, grounds its runtime input in the remembered turn from
   phase 1 (memory is Learning's only source of live-input knowledge), and Revision authors
   and installs the behavior in that same memory.
3. **Proof** — a second real ring on a fresh phone runs through Autonomy with the installed
   behavior loaded. Pass requires a ``phone.answer_call`` selection produced by the authored
   behavior from the real stimulus; the phone's own answer request is reported as
   confirmation.

Only public contracts are used: no ``tests`` imports, no private state access, no ``state()``
gating (progression waits on typed events), and no back-filled stimulus payload. If the
authored behavior does not fire on the real ring, that is reported as a failure instead of
reshaping the stimulus to fit.

Credentials (loaded from repo / phone_bot ``.env`` without printing secrets):

- ``BOT_OPENAI_API_KEY`` / ``OPENAI_API_KEY``
- optional ``BOT_REFLECTION_MODEL`` (default ``gpt-5.6-terra``)
"""

from __future__ import annotations

import asyncio
import collections.abc
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import typing
import uuid

import hsm
import bot
from sqlalchemy.sql import Executable

from bot import abilities
from bot import behavior
from bot.abilities import cognition
from bot.abilities import decoding
from bot.abilities import learning
from bot.abilities import memory
from bot.behavior import storage as behavior_storage
from bot.devices import phone as phone_device
from bot.protocols import attachment
from bot.providers.openai_compat import ChatClient as OpenAIChatClient
from bot.providers.openai_compat import Processor as OpenAIProcessor
from bot.environment import SoundEvent, Environment

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LESSON = "When the phone rings make sure you answer it"
_EXPERIENCE_CALL_ID = "learning-say:first-caller"
_PROOF_CALL_ID = "learning-say:second-caller"
_DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_SAMPLE_RATE_HZ = 22_050
_LEARNING_TIMEOUT_S = 120.0
_TURN_TIMEOUT_S = 90.0
_ATTACH_TIMEOUT_S = 10.0
_DEVICE_TIMEOUT_S = 10.0
_CONFIRMATION_TIMEOUT_S = 5.0

_TAwaited = typing.TypeVar("_TAwaited")


def _load_dotenv_file(path: pathlib.Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            _ = os.environ.setdefault(key, value)


def _load_dotenv() -> None:
    for path in (_REPO_ROOT / ".env", _REPO_ROOT / "examples" / "phone_bot" / ".env"):
        _load_dotenv_file(path)


def _openai_api_key() -> str | None:
    _load_dotenv()
    for name in ("BOT_OPENAI_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _openai_model() -> str:
    return os.environ.get("BOT_REFLECTION_MODEL") or _DEFAULT_OPENAI_MODEL


def _openai_base_url() -> str:
    return os.environ.get("BOT_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or _DEFAULT_OPENAI_BASE_URL


def _processor() -> OpenAIProcessor:
    api_key = _openai_api_key()
    if api_key is None:
        raise SystemExit("Set BOT_OPENAI_API_KEY or OPENAI_API_KEY for live Learning.")
    client = OpenAIChatClient(model=_openai_model(), api_key=api_key, base_url=_openai_base_url())
    return OpenAIProcessor(client=client, provider="learning_say_live")


def _require_macos_say() -> None:
    missing = [name for name in ("say", "afconvert") if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"Need macOS speech tools: missing {', '.join(missing)}")


def _say_to_wav(text: str, destination: pathlib.Path) -> bytes:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="learning-say-") as tmp:
        aiff = pathlib.Path(tmp) / "lesson.aiff"
        wav = pathlib.Path(tmp) / "lesson.wav"
        _ = subprocess.run(["say", "-o", str(aiff), text], check=True, capture_output=True)
        _ = subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{_SAMPLE_RATE_HZ}", str(aiff), str(wav)],
            check=True,
            capture_output=True,
        )
        data = wav.read_bytes()
    _ = destination.write_bytes(data)
    return data


class LessonTextDecoder(decoding.Decoder[learning.InputData, learning.DecodedData]):
    """Decode Learning input: prefer explicit text, else UTF-8 from content bytes."""

    @typing.override
    async def decode(self, input: learning.InputData) -> learning.DecodedData:
        if isinstance(input.content, str) and input.content.strip():
            return learning.DecodedData(text=input.content.strip(), kind="instruction")
        if isinstance(input.content, bytes):
            # Audio bytes are not transcribed here — companion transcript is the Learning text.
            raise TypeError("LessonTextDecoder expects text content; pass the spoken lesson transcript.")
        raise TypeError("Learning input content must be non-empty text.")


_EventPredicate: typing.TypeAlias = collections.abc.Callable[[hsm.Event[typing.Any]], bool]


class EventWaiters:
    """Futures parked on typed events so callers await deliveries instead of probing state."""

    _waiters: list[tuple[_EventPredicate, asyncio.Future[hsm.Event[typing.Any]]]]

    def __init__(self) -> None:
        self._waiters = []

    def expect(
        self,
        predicate: _EventPredicate,
        *,
        seen: collections.abc.Sequence[hsm.Event[typing.Any]] = (),
    ) -> asyncio.Future[hsm.Event[typing.Any]]:
        """Future for the first matching event, resolved immediately when already seen."""

        future = asyncio.get_running_loop().create_future()
        for event in seen:
            if predicate(event):
                future.set_result(event)
                return future
        self._waiters.append((predicate, future))
        return future

    def resolve(self, event: hsm.Event[typing.Any]) -> None:
        """Complete every waiter this event satisfies."""

        pending: list[tuple[_EventPredicate, asyncio.Future[hsm.Event[typing.Any]]]] = []
        for predicate, future in self._waiters:
            if future.done():
                continue
            if predicate(event):
                future.set_result(event)
                continue
            pending.append((predicate, future))
        self._waiters = pending


async def awaited(future: asyncio.Future[_TAwaited], *, timeout: float, what: str) -> _TAwaited:
    """Await one parked future, naming what never arrived when it times out."""

    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except TimeoutError as error:
        raise TimeoutError(f"{what} did not arrive within {timeout:g}s") from error


class RingWatcher(phone_device.PhoneEventRecorder):
    """Injected phone service: provider ingress plus typed waits on committed phone events.

    Adds no capability to the phone contract — ``attach`` / ``receive`` / ``publish`` /
    ``events`` are the public :class:`~bot.devices.phone.PhoneEventRecorder` surface. The
    parked futures let this proof gate on ``phone.ringing`` and on service requests instead
    of polling firmware ``state()``.
    """

    _attached_target: hsm.Instance | None
    _attach_waiters: list[asyncio.Future[None]]
    _published: EventWaiters

    def __init__(self) -> None:
        super().__init__()
        self._attached_target = None
        self._attach_waiters = []
        self._published = EventWaiters()

    def attached(self) -> asyncio.Future[None]:
        """Future that resolves once phone firmware has attached this service."""

        future = asyncio.get_running_loop().create_future()
        if self._attached_target is not None:
            future.set_result(None)
            return future
        self._attach_waiters.append(future)
        return future

    def expect(self, name: str) -> asyncio.Future[hsm.Event[typing.Any]]:
        """Future for the next (or already published) phone event with this name."""

        return self._published.expect(lambda event: event.name == name, seen=self.events)

    @typing.override
    async def attach(self, environment: Environment, target: hsm.Instance) -> None:
        await super().attach(environment, target)
        self._attached_target = target
        for future in self._attach_waiters:
            if not future.done():
                future.set_result(None)
        self._attach_waiters = []

    @typing.override
    async def detach(self, environment: Environment, target: hsm.Instance) -> None:
        await super().detach(environment, target)
        if self._attached_target is target:
            self._attached_target = None

    @typing.override
    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        super().publish(ctx, event)
        self._published.resolve(event)


class TerminalCollector(hsm.Instance):
    """Attachment owner and environment listener for one phase of the proof.

    Abilities dispatch their terminal events to their attachment owner, and a started
    instance in a :class:`~bot.environment.Environment` receives environment broadcasts. Collecting both is
    what lets the script await typed attach, ability, and stimulus events.
    """

    _received: list[hsm.Event[typing.Any]]
    _waiters: EventWaiters

    def __init__(self) -> None:
        super().__init__()
        self._received = []
        self._waiters = EventWaiters()

    def expect(self, predicate: _EventPredicate) -> asyncio.Future[hsm.Event[typing.Any]]:
        """Future for the first delivered event matching ``predicate``."""

        return self._waiters.expect(predicate, seen=tuple(self._received))

    @staticmethod
    def _collect(ctx: hsm.Context, instance: "TerminalCollector", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._received.append(event)
        instance._waiters.resolve(event)

    model: typing.ClassVar[hsm.Model | None] = bot.define(
        "TerminalCollector",
        hsm.initial(hsm.target("collecting")),
        hsm.state("collecting", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_collect))),
    )


async def started_collector(environment: Environment) -> TerminalCollector:
    collector = TerminalCollector()
    _ = await bot.started(environment, collector, typing.cast(hsm.Model, TerminalCollector.model))
    environment.join(collector)
    return collector


async def attach_ability(
    environment: Environment,
    collector: TerminalCollector,
    ability: abilities.Ability[typing.Any, typing.Any],
) -> None:
    """Attach one ability to the collector and wait for its typed attach terminal."""

    operation_id = uuid.uuid4().hex
    terminal = collector.expect(
        lambda event: event.id == operation_id
        and event.name in (attachment.AttachCompleteEvent.name, attachment.AttachFailedEvent.name)
    )
    _ = await ability.attach(
        environment,
        attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=collector), operation_id),
    )
    event = await awaited(terminal, timeout=_ATTACH_TIMEOUT_S, what=f"{type(ability).__name__} attach terminal")
    if event.name == attachment.AttachFailedEvent.name:
        data = event.data
        message = data.message if isinstance(data, attachment.FailedData) else str(data)
        raise RuntimeError(f"{type(ability).__name__} attach failed: {message}")


async def ability_output(
    environment: Environment,
    collector: TerminalCollector,
    ability: abilities.Ability[typing.Any, typing.Any],
    input: object,
    *,
    operation_id: str,
    timeout: float,
) -> object:
    """Dispatch one ability input and await its typed output (or raise its typed failure)."""

    output_name = ability.output_event.name
    failed_name = ability.failed_event.name
    terminal = collector.expect(lambda event: event.id == operation_id and event.name in (output_name, failed_name))
    _ = await hsm.dispatch(environment, ability, ability.input_event.with_data_and_id(input, operation_id))
    event = await awaited(terminal, timeout=timeout, what=f"{type(ability).__name__} terminal")
    if event.name == failed_name:
        data = event.data
        message = data.message if isinstance(data, abilities.FailureData) else str(data)
        raise RuntimeError(f"{type(ability).__name__} failed: {message}")
    return event.data


async def ringing_phone(
    environment: Environment,
    collector: TerminalCollector,
    *,
    call_id: str,
) -> tuple[phone_device.Phone, RingWatcher, hsm.Event[typing.Any]]:
    """Start a phone, ring it through its service, and return the elevated environment stimulus."""

    watcher = RingWatcher()
    phone = phone_device.Phone(service=watcher)
    _ = await bot.started(environment, phone, typing.cast(hsm.Model, phone_device.Phone.model))
    await awaited(watcher.attached(), timeout=_DEVICE_TIMEOUT_S, what="phone service attach")
    ringing = watcher.expect(phone_device.RingingEvent.name)
    elevated = collector.expect(
        lambda event: event.name == SoundEvent.name
        and isinstance(event.data, phone_device.PhoneSoundData)
        and event.data.caller == call_id
    )
    await watcher.receive(
        phone.context(),
        phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id=call_id, caller=call_id)),
    )
    _ = await awaited(ringing, timeout=_DEVICE_TIMEOUT_S, what=f"{phone_device.RingingEvent.name} for {call_id}")
    stimulus = await awaited(
        elevated, timeout=_DEVICE_TIMEOUT_S, what=f"environment.sound ring elevation for {call_id}"
    )
    return phone, watcher, stimulus


def ring_turn(
    phone: phone_device.Phone,
    stimulus: hsm.Event[typing.Any],
    *,
    operation_id: str,
) -> cognition.types.TurnData:
    """One cognition turn over the real ring stimulus with the phone as the only actor."""

    return cognition.types.TurnData(
        input=cognition.input.InputData(
            stimulus=stimulus,
            abilities=(),
            actors={"phone": phone},
            focus="phone",
            focus_candidates=("phone",),
        ),
        operation_id=operation_id,
        generation=f"{operation_id}:generation",
    )


def selection_events(completion: object) -> tuple[cognition.types.EventData, ...]:
    if not isinstance(completion, cognition.types.CompletionData):
        return ()
    return completion.output or ()


def describe_stimulus(stimulus: hsm.Event[typing.Any]) -> str:
    data = stimulus.data
    if not isinstance(data, phone_device.PhoneSoundData):
        return f"{stimulus.name} data={type(data).__name__}"
    packaged = data.audio == phone_device.RING_SOUND_WAV
    return (
        f"{stimulus.name} kind={data.kind!r} caller={data.caller!r} "
        f"audio_bytes={len(data.audio)} media_type={data.media_type!r} "
        f"sample_rate_hz={data.sample_rate_hz} channels={data.channels} device_ring_wav={packaged}"
    )


def installed_behavior(store: memory.Memory, name: str) -> behavior.Instance | None:
    """Read one installed behavior row back from memory (public inventory contract)."""

    clauses = behavior_storage.select_behavior_by_name_clauses(name)
    output = store.execute(
        memory.InputData(statements=memory.compile_statements(*(typing.cast(Executable, clause) for clause in clauses)))
    )
    if len(output.results) < 2:
        return None
    instances = behavior_storage.instances_from_behavior_results(
        tuple(row.as_mapping() for row in output.results[0].rows),
        tuple(row.as_mapping() for row in output.results[1].rows),
    )
    return instances[0] if instances else None


async def experience_ring(environment: Environment, store: memory.Memory) -> cognition.episodes.CognitiveEpisode:
    """Phase 1: deliberate one real ring live and remember that turn in memory."""

    collector = await started_collector(environment)
    phone, watcher, stimulus = await ringing_phone(environment, collector, call_id=_EXPERIENCE_CALL_ID)
    print(f"ring #1 stimulus: {describe_stimulus(stimulus)}")
    intuition = cognition.Intuition(processor=_processor())
    await attach_ability(environment, collector, intuition)
    operation_id = uuid.uuid4().hex
    turn = ring_turn(phone, stimulus, operation_id=operation_id)
    processing_input = cognition.input.build_processing_input(turn.input)
    offered = tuple(event.name for event in processing_input.schemas)
    print(f"ring #1 offered events: {list(offered)}")
    completion = await ability_output(
        environment,
        collector,
        intuition,
        cognition.intuition.InputData(turn=turn, processing_input=processing_input),
        operation_id=operation_id,
        timeout=_TURN_TIMEOUT_S,
    )
    selections = selection_events(completion)
    print(f"ring #1 intuition selected: {[item.event for item in selections]}")
    if not selections:
        print("WARNING: intuition left ring #1 unhandled; the remembered turn carries no selection.")
    else:
        requested = watcher.expect(phone_device.ServiceAnswerRequestedEvent.name)
        try:
            _ = await awaited(requested, timeout=_CONFIRMATION_TIMEOUT_S, what="phone answer request")
            print("ring #1 firmware requested answer from the phone service.")
        except TimeoutError as error:
            print(f"NOTE: {error} (selection was not an answer, or firmware refused it).")
    episode = cognition.episodes.CognitiveEpisode(
        focus=turn.input.focus,
        focus_candidates=turn.input.focus_candidates,
        stimulus_name=cognition.episodes.stimulus_name(turn.input.stimulus),
        output=selections,
    )
    _ = store.execute(cognition.episodes.episode_insert_input(episode, context_ref=turn.input.focus))
    await hsm.stop(phone, environment)
    await hsm.stop(collector, environment)
    return episode


async def learn_lesson(environment: Environment, store: memory.Memory, lesson: str) -> learning.OutputData:
    """Phase 2: Learning decodes the spoken lesson and Revision authors the behavior."""

    collector = await started_collector(environment)
    ability = learning.Learning(decoder=LessonTextDecoder(), processor=_processor(), memory=store)
    await attach_ability(environment, collector, ability)
    output = await ability_output(
        environment,
        collector,
        ability,
        learning.InputData(content=lesson, media_type="text/plain"),
        operation_id=uuid.uuid4().hex,
        timeout=_LEARNING_TIMEOUT_S,
    )
    if not isinstance(output, learning.OutputData):
        raise RuntimeError(f"Learning returned {type(output).__name__}, expected OutputData.")
    return output


async def prove_ring_answers(environment: Environment, store: memory.Memory) -> tuple[tuple[str, ...], bool]:
    """Phase 3: ring a fresh phone and run Autonomy with the installed behavior loaded."""

    collector = await started_collector(environment)
    autonomy = cognition.Autonomy(memory=store)
    await attach_ability(environment, collector, autonomy)
    phone, watcher, stimulus = await ringing_phone(environment, collector, call_id=_PROOF_CALL_ID)
    print(f"ring #2 stimulus: {describe_stimulus(stimulus)}")
    operation_id = uuid.uuid4().hex
    turn = ring_turn(phone, stimulus, operation_id=operation_id)
    requested = watcher.expect(phone_device.ServiceAnswerRequestedEvent.name)
    completion = await ability_output(
        environment,
        collector,
        autonomy,
        turn,
        operation_id=operation_id,
        timeout=_TURN_TIMEOUT_S,
    )
    events = tuple(item.event for item in selection_events(completion))
    answered = False
    if phone_device.AnswerCallEvent.name in events:
        try:
            _ = await awaited(requested, timeout=_CONFIRMATION_TIMEOUT_S, what="phone answer request")
            answered = True
        except TimeoutError as error:
            print(f"NOTE: {error}")
    await hsm.stop(phone, environment)
    await hsm.stop(collector, environment)
    return events, answered


async def prove(lesson: str) -> int:
    store = memory.Memory()
    environment = Environment()

    print("--- phase 1: experience a real ring ---")
    episode = await experience_ring(environment, store)
    print(
        f"remembered turn: stimulus_name={episode.stimulus_name!r} focus={episode.focus!r} "
        f"output={[item.event for item in episode.output]}"
    )

    print("--- phase 2: learn the spoken lesson ---")
    output = await learn_lesson(environment, store, lesson)
    print(f"decoded.text={output.decoded.text!r} kind={output.decoded.kind!r}")
    print(
        f"runtime_input.stimulus_name={output.runtime_input.stimulus_name!r} "
        f"payload_keys={sorted(output.runtime_input.payload)} "
        f"expected_event={output.runtime_input.expected_event!r}"
    )
    print(
        f"behavior.name={output.behavior.name!r} triggers={output.behavior.triggers!r} "
        f"source_len={len(output.behavior.source or '')}"
    )
    if not output.behavior.source:
        print("FAIL: no Starlark source on the applied behavior", file=sys.stderr)
        return 1
    print("--- authored source ---")
    print(output.behavior.source)
    print("---")
    installed = installed_behavior(store, output.behavior.name)
    if installed is None:
        print(f"FAIL: behavior {output.behavior.name!r} is not in memory inventory", file=sys.stderr)
        return 1
    print(f"installed: status={installed.status} triggers={installed.triggers} reason={installed.status_reason!r}")

    print("--- phase 3: ring again and let Autonomy run the behavior ---")
    events, answered = await prove_ring_answers(environment, store)
    print(f"ring #2 autonomy selected: {list(events)} answer_requested={answered}")
    if phone_device.AnswerCallEvent.name not in events:
        print(
            f"FAIL: the authored behavior did not select {phone_device.AnswerCallEvent.name} "
            f"on the real ring; got {list(events)}",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS: Learning authored {output.behavior.name!r} from the spoken lesson and Autonomy "
        f"selected {list(events)} on a real device ring (answer_requested={answered})."
    )
    return 0


def main() -> int:
    _require_macos_say()
    if _openai_api_key() is None:
        print("FAIL: OpenAI API key not set (BOT_OPENAI_API_KEY / OPENAI_API_KEY)", file=sys.stderr)
        return 2

    assets = _REPO_ROOT / "examples" / "assets"
    lesson_wav = assets / "learning_say_lesson.wav"
    print(f"say → {lesson_wav} phrase={_LESSON!r}")
    wav_bytes = _say_to_wav(_LESSON, lesson_wav)
    print(f"wav_bytes={len(wav_bytes)} model={_openai_model()!r} base_url={_openai_base_url()!r}")
    if shutil.which("afplay") is not None:
        _ = subprocess.run(["afplay", str(lesson_wav)], check=False)

    try:
        return asyncio.run(prove(_LESSON))
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

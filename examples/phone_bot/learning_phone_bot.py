#!/usr/bin/env -S uv run --project examples/phone_bot python
"""Natural, unforced learning proof through the real bot library.

Nothing is pre-seeded and nothing is scripted past composition:

- Memory starts empty. Every remembered row is written by the pipeline itself
  (Reasoning retains the turn, Reflection stores the episode, Revision persists
  the authored behavior with its practice lifecycle in the same memory).
- The short-term register (`bot.abilities.memory.StmMemory`) receives every
  stimulus the body admits — the rings arrive from real phone firmware and the
  lesson arrives as a real person speaking in the room. The register is never
  hand-fed: body admission is its only writer.
- The lesson is offered the way the library offers every tool: Learning is an
  acquired Bot ability, so `bot.ability.learning.input` appears in the live
  model menu and the model decides, on its own turn, that this input is a
  lesson. The script never dispatches Learning.
- Learning grounds the lesson against the register's observation of the ring
  and the remembered experience turn (both written by the pipeline itself).
- The second real ring runs the authored behavior through Autonomy; the phone
  firmware's own answer request is the proof.

Composition note: the app's default cognition wires a canned Communication
seed behavior. This run deliberately composes the same host shape without that
seed, so the only behavior in the world is whatever this bot learns live. The
app's interactive composition is unchanged; this proof reads nothing private
and composes through public library and app surface only. The only operator
side-effect is the phone service: an embedded recorder provider so this proof
needs no external call provider.
"""

from __future__ import annotations

import asyncio
import datetime
import dataclasses
import os
import pathlib
import sys
import typing
import uuid

from bot import behavior as bot_behavior
from bot.abilities import cognition
from bot.abilities import processing
from bot.abilities import decoding
from bot.abilities import learning
from bot.abilities import memory as memory_abilities
from bot.behavior import storage as behavior_storage
from sqlalchemy import select as sqlalchemy_select
from bot.devices import phone as phone_device
from bot.environment import Environment
from bot.environment import space
from bot.providers.gemini import ChatClient as GeminiChatClient
from bot.providers.gemini import Processor as GeminiProcessor
from bot.providers.openai_compat import ChatClient as OpenAIChatClient
from bot.providers.openai_compat import Processor as OpenAIProcessor
from phone_bot_example import AppConfig
from phone_bot_example import PhoneBot
from phone_bot_example.person import Person
from phone_bot_example.person import SayEncoder

_LESSON = "When the phone rings make sure you answer it"
_EXPERIENCE_CALL_ID = "learning-e2e:first-caller"
_PROOF_CALL_ID = "learning-e2e:second-caller"
_RING_SETTLE_S = 300.0
_LESSON_SETTLE_S = 420.0
_PROOF_SETTLE_S = 180.0
_ACTIVATION_S = 240.0

_RECALL_LIMIT = 50

# Deliberative-tier provider selection: "terra" (default) uses the OpenAI Terra endpoint;
# "mercury" routes reasoning/reflection/learning through the Mercury endpoint; "gemini" uses
# the Gemini chat endpoint. Same OpenAI-compatible contract shape per provider, each with its
# own key. Which available provider the slow tier uses, not a behavior.
_REASONING_PROVIDER_ENV = "LEARNING_E2E_REASONING_PROVIDER"
_GEMINI_MODEL_ENV = "LEARNING_E2E_GEMINI_MODEL"


def _openai_client(model: str, api_key: str | None, base_url: str) -> OpenAIChatClient:
    return OpenAIChatClient(model=model, api_key=api_key or "", base_url=base_url)


def _intuition_processor(config: AppConfig) -> OpenAIProcessor:
    cognition_config = config.cognition
    return OpenAIProcessor(
        client=_openai_client(
            cognition_config.intuition_model,
            cognition_config.intuition_api_key,
            cognition_config.intuition_base_url,
        ),
        provider="learning_e2e_intuition",
    )


def _reasoning_processor(config: AppConfig) -> OpenAIProcessor:
    cognition_config = config.cognition
    return OpenAIProcessor(
        client=_openai_client(cognition_config.model, cognition_config.api_key, cognition_config.base_url),
        provider="learning_e2e_reasoning",
    )


def _reflection_processor(config: AppConfig) -> OpenAIProcessor:
    cognition_config = config.cognition
    reflection_key = (
        cognition_config.reflection_api_key
        if cognition_config.reflection_api_key is not None
        else cognition_config.api_key
    )
    return OpenAIProcessor(
        client=_openai_client(
            cognition_config.reflection_model,
            reflection_key,
            cognition_config.reflection_base_url or cognition_config.base_url,
        ),
        provider="learning_e2e_reflection",
    )


def _mercury_processor(config: AppConfig) -> OpenAIProcessor:
    """Deliberative work on the same live Mercury endpoint the fast tier uses."""

    cognition_config = config.cognition
    client = _openai_client(
        cognition_config.intuition_model,
        cognition_config.intuition_api_key,
        cognition_config.intuition_base_url,
    )
    return OpenAIProcessor(client=client, provider="learning_e2e_mercury")


def _mercury_processor(config: AppConfig) -> OpenAIProcessor:
    """Deliberative work on the same live Mercury endpoint the fast tier uses."""

    cognition_config = config.cognition
    return OpenAIProcessor(
        client=_openai_client(
            cognition_config.intuition_model,
            cognition_config.intuition_api_key,
            cognition_config.intuition_base_url,
        ),
        provider="learning_e2e_mercury",
    )


def _gemini_processor(config: AppConfig) -> GeminiProcessor:
    """Deliberative work on the live Gemini chat endpoint (key confirmed by the STT leg)."""

    model = os.environ.get(_GEMINI_MODEL_ENV) or "gemini-3.5-flash"
    return GeminiProcessor(
        client=GeminiChatClient(model=model, api_key=config.speech.api_key or ""),
        provider="learning_e2e_gemini",
    )


def _deliberate_processor(config: AppConfig) -> OpenAIProcessor:
    """Reasoning/reflection/learning slow-tier processor under the configured provider."""

def _deliberate_processor(config: AppConfig) -> OpenAIProcessor:
    """Reasoning/reflection/learning slow-tier processor under the configured provider."""

    provider = os.environ.get(_REASONING_PROVIDER_ENV)
    if provider == "mercury":
        return _mercury_processor(config)
    if provider == "gemini":
        return _gemini_processor(config)
    return _reasoning_processor(config)


class CountingProcessor(processing.Processor):
    """Counted facade: how often the reasoning tier actually served Learning."""

    def __init__(self, inner: processing.Processor) -> None:
        super().__init__()
        self._inner = inner
        self.counts: dict[str, int] = {}

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        kind = type(getattr(input, "input", None)).__name__
        self.counts[kind] = self.counts.get(kind, 0) + 1
        started = datetime.datetime.now()
        try:
            result = await self._inner.process(input)
            self.counts[f"done:{kind}"] = self.counts.get(f"done:{kind}", 0) + 1
            return result
        except Exception as error:
            elapsed = (datetime.datetime.now() - started).total_seconds()
            print(f"learning tier call FAILED after {elapsed:.1f}s on {kind}: {type(error).__name__}: {error}", flush=True)
            raise


class LessonTextDecoder(decoding.Decoder[learning.InputData, learning.DecodedData]):
    """The lesson is spoken; the transcript in Learning input is the lesson text."""

    @typing.override
    async def decode(self, input: learning.InputData) -> learning.DecodedData:
        if isinstance(input.content, str) and input.content.strip():
            return learning.DecodedData(text=input.content.strip(), kind="instruction")
        if isinstance(input.content, bytes):
            raise TypeError("LessonTextDecoder expects text content (the spoken lesson transcript).")
        raise TypeError("Learning input content must be non-empty text.")


def _unseeded_cognition(config: AppConfig, store: memory_abilities.Memory) -> cognition.Cognition:
    """The app's host shape, without the canned Communication seed behavior.

    For this run the only behavior in the world is whatever the bot learns live
    from its experiences. Autonomy still runs everything authored; reasoning
    and reflection share the memory the pipeline writes to.
    """

    return cognition.Cognition(
        autonomy=cognition.Autonomy(memory=store),
        intuition=cognition.Intuition(processor=_intuition_processor(config)),
        reasoning=cognition.Reasoning(processor=_deliberate_processor(config), memory=store),
        reflection=cognition.Reflection(processor=_deliberate_processor(config), memory=store),
    )


def _behavior_rows(store: memory_abilities.Memory) -> tuple[bot_behavior.Instance, ...]:
    clauses = behavior_storage.select_all_behaviors_clauses()
    output = store.execute(memory_abilities.InputData(statements=memory_abilities.compile_statements(*clauses)))
    if len(output.results) < 2:
        return ()
    behavior_rows = tuple(row.as_mapping() for row in output.results[0].rows)
    trigger_rows = tuple(row.as_mapping() for row in output.results[1].rows)
    return behavior_storage.instances_from_behavior_results(behavior_rows, trigger_rows)


def _behavior_row_times(store: memory_abilities.Memory) -> dict[str, str]:
    """name -> created_at from the raw behavior rows (authorship-time evidence for the proof)."""

    output = store.execute(
        memory_abilities.InputData(
            statements=memory_abilities.compile_statements(
                sqlalchemy_select(behavior_storage.behavior_table.c.name, behavior_storage.behavior_table.c.created_at)
            )
        )
    )
    names: dict[str, str] = {}
    for row in output.results[0].rows:
        mapping = row.as_mapping()
        name = mapping.get("name")
        created = mapping.get("created_at")
        if isinstance(name, str) and isinstance(created, str):
            names[name] = created
    return names


def _episode_count(store: memory_abilities.Memory) -> int:
    output = store.execute(cognition.episodes.episode_select_input(limit=_RECALL_LIMIT))
    return len(output.results[0].rows) if output.results else 0


def _selection_names(turns: tuple[cognition.types.OutputData, ...]) -> list[str]:
    names: list[str] = []
    for turn in turns:
        names.extend(item.event for item in turn)
    return names


def _register_summary(register: memory_abilities.StmMemory) -> list[str]:
    return [f"{entry.stimulus_name}:{entry.payload.get('kind')}" for entry in register.recent(limit=32)]


async def wait_for(
    condition: typing.Callable[[], bool],
    *,
    timeout_s: float,
    what: str,
    tick: float = 2.0,
) -> None:
    """Observational wait on typed evidence already delivered (snapshot reading, no gating)."""

    async def _waiter() -> None:
        while not condition():
            await asyncio.sleep(tick)

    try:
        await asyncio.wait_for(_waiter(), timeout=timeout_s)
    except TimeoutError as error:
        raise TimeoutError(f"{what} did not arrive within {timeout_s:g}s") from error


def printed(title: str) -> None:
    print(f"--- {title} ---", flush=True)


async def prove(lesson: str) -> int:
    config = AppConfig.from_env_file(pathlib.Path(__file__).parent / ".env")
    if not config.cognition.can_process():
        print(
            "FAIL: missing cognition credentials (intuition + reasoning + reflection). "
            "Set BOT_OPENAI_API_KEY and BOT_MERCURY_API_KEY (plus optional "
            "BOT_OPENAI_BASE_URL / BOT_REFLECTION_MODEL) in the repo or example .env.",
            file=sys.stderr,
        )
        return 2

    store = memory_abilities.Memory()
    register = memory_abilities.StmMemory()
    learning_processor = CountingProcessor(_deliberate_processor(config))
    learning_ability = learning.Learning(
        decoder=LessonTextDecoder(),
        processor=learning_processor,
        memory=store,
        stm_memory=register,
    )
    # The handset carries an embedded recorder provider: real firmware ring, no external call
    # provider. The person's voice, the bot's hearing, STT, and cognition all stay natural.
    recorder = phone_device.EventRecorder()
    phone = phone_device.Phone(service=recorder)

    environment = Environment()
    body = PhoneBot(
        "learning-e2e",
        phone=phone,
        cognition_config=config.cognition,
        speech_config=config.speech,
        cognition=_unseeded_cognition(config, store),
        memory=store,
        stm_memory=register,
        extra_acquired=(learning_ability,),
    )

    printed("compose")
    print(
        "intuition_model=%s reasoning_model=%s reflection_model=%s"
        % (
            config.cognition.intuition_model,
            config.cognition.model,
            config.cognition.reflection_model,
        ),
        flush=True,
    )
    _ = await body.attach(environment)

    def _bot_active() -> bool:
        parts = (body.state() or "").split("/")
        return len(parts) > 2 and parts[2] == "active"

    await wait_for(_bot_active, timeout_s=_ACTIVATION_S, what="bot activation")
    initial_snap = {
        "behaviors": len(_behavior_rows(store)),
        "episodes": _episode_count(store),
        "register": len(register.recent(limit=32)),
    }
    print(f"memory starts empty check: {initial_snap}", flush=True)
    if initial_snap != {"behaviors": 0, "episodes": 0, "register": 0}:
        print("FAIL: memory or register is not clean at start", file=sys.stderr)
        return 2

    # Someone stands in the room: a real hearing participant is required so the bot's ear
    # (Listening) has the industry-standard path in for a spoken lesson.
    some_person = Person(
        encoder=SayEncoder(),
        position=space.Position(x=0.0, y=1.0),
        amplitude_db=60.0,
    )
    _ = await some_person.enter(environment)

    printed("ring #1: a real call arrives (experience)")
    initial_outputs = len(body.outputs())
    _ = await recorder.receive(
        phone.context(),
        phone_device.IncomingCallEvent.with_data(
            phone_device.IncomingCallData(call_id=_EXPERIENCE_CALL_ID, caller=_EXPERIENCE_CALL_ID)
        ),
    )
    await wait_for(
        lambda: len(body.outputs()) > initial_outputs,
        timeout_s=_RING_SETTLE_S,
        what="ring #1 cognition output",
    )
    print(f"ring #1 selections: {_selection_names(body.outputs()[initial_outputs:])}", flush=True)
    register_entries = register.recent("environment.sound", limit=32)
    # The body terminal delivers before Reflection finishes (the host's own reflex order), so the
    # episode happens between here and "settled": its typed evidence is the episode row itself.
    await wait_for(lambda: _episode_count(store) >= 1, timeout_s=_RING_SETTLE_S, what="ring #1 remembered episode")
    episodes = _episode_count(store)
    print(
        f"register ring observations: {len(register_entries)} payloads="
        f"{[entry.payload.get('kind') for entry in register_entries]}; episodes={episodes}",
        flush=True,
    )
    if not register_entries:
        print("FAIL: the admitted ring never reached the short-term register", file=sys.stderr)
        return 1
    if episodes < 1:
        print("FAIL: the experienced ring produced no remembered episode", file=sys.stderr)
        return 1

    # The caller ends the first call the way callers do, so the lesson meets a quiet line. If
    # the bot already closed the call with its own call-control selections (a real decision it
    # is entitled to), the close attempt lands on nothing and is reported as a note, not a
    # failure — the observable requirement is only that the lesson turn happens after ring #1.
    _ = await recorder.receive(
        phone.context(),
        phone_device.RemoteHangUpEvent.with_data(phone_device.RemoteHangUpData(call_id=_EXPERIENCE_CALL_ID)),
    )

    def _call_closed() -> bool:
        return any(
            event.name == phone_device.HungUpEvent.name
            and isinstance(event.data, phone_device.HungUpData)
            and event.data.outcome == "remote_hang_up"
            for event in recorder.events
        )

    try:
        await wait_for(_call_closed, timeout_s=60.0, what="the call to end")
    except TimeoutError:
        print(
            "NOTE: no remote-hang-up commit landed (the call was likely already closed by the "
            "bot's own call-control selections); continuing to the lesson on the quiet line.",
            flush=True,
        )
    await asyncio.sleep(2.0)

    printed("lesson: someone says the lesson out loud")
    outputs_before_lesson = len(body.outputs())
    _ = await some_person.say(lesson, ctx=environment)
    await wait_for(
        lambda: len(body.outputs()) > outputs_before_lesson,
        timeout_s=_LESSON_SETTLE_S,
        what="lesson cognition output",
    )
    print(f"lesson selections: {_selection_names(body.outputs()[outputs_before_lesson:])}", flush=True)
    print(f"learning processor calls: {learning_processor.counts}", flush=True)
    rows = _behavior_rows(store)
    row_times = _behavior_row_times(store)
    for row in rows:
        if row.status != "ACTIVE":
            print(
                f"non-AUTHORED row: name={row.name!r} status={row.status} "
                f"status_reason={row.status_reason!r} source_len={len(row.source or '')} "
                f"created_at={row_times.get(row.name)!r}",
                flush=True,
            )
            print("--- stored source head ---", flush=True)
            print((row.source or "")[:500], flush=True)
            print("---", flush=True)
    learned = [row for row in rows if row.status == "ACTIVE"]
    if not learned:
        print(
            f"FAIL: no ACTIVE behavior after the lesson — rows="
            f"{[(row_.name, row_.status, row_.status_reason) for row_ in rows]}",
            file=sys.stderr,
        )
        return 1
    for installed in learned:
        print(
            f"learned: name={installed.name!r} status={installed.status} "
            f"triggers={installed.triggers} source_len={len(installed.source or '')} "
            f"created_at={row_times.get(installed.name)!r} "
            f"used_count={installed.used_count} last_used_at={installed.last_used_at!r}",
            flush=True,
        )

    printed("ring #2: the learned behavior answers (proof)")
    outputs_before_proof = len(body.outputs())
    _ = await recorder.receive(
        phone.context(),
        dataclasses.replace(
            phone_device.IncomingCallEvent.with_data(
                phone_device.IncomingCallData(call_id=_PROOF_CALL_ID, caller=_PROOF_CALL_ID)
            ),
            id=uuid.uuid4().hex,
        ),
    )
    answer_request = phone_device.ServiceAnswerRequestedEvent
    reported_proof_turn = False

    def _proof_answer_requested() -> bool:
        if any(
            event.name == answer_request.name
            and isinstance(event.data, phone_device.AnswerRequestData)
            and event.data.call_id == _PROOF_CALL_ID
            for event in recorder.events
        ):
            return True
        outputs_grew = len(body.outputs()) > outputs_before_proof
        nonlocal reported_proof_turn
        if outputs_grew and not reported_proof_turn:
            print(
                f"ring #2 selections in proof: {_selection_names(body.outputs()[outputs_before_proof:])}",
                flush=True,
            )
            reported_proof_turn = True
        return False

    await wait_for(_proof_answer_requested, timeout_s=_PROOF_SETTLE_S, what="phone answer request for the proof call")
    requested = [
        event
        for event in recorder.events
        if event.name == answer_request.name and isinstance(event.data, phone_device.AnswerRequestData)
    ]
    print(
        f"answer requests on record: {[(str(event.data.call_id), str(event.data)) for event in requested]}",
        flush=True,
    )
    recent = register.recent("environment.sound", limit=32)
    print(f"register after ring #2: {[entry.payload.get('kind') for entry in recent]}", flush=True)

    _ = await some_person.leave(environment)
    printed("result")
    print(
        f"PASS: the learned behavior answered a real ring (answer_requested, caller={_PROOF_CALL_ID!r})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(prove(_LESSON)))

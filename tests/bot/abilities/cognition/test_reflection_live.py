"""Live e2e Reflection tests against real model providers.

Primary Reflection model: OpenAI **gpt-5.6-terra** (override with ``BOT_REFLECTION_MODEL``).

Provider keys (skipped unless available):

- OpenAI: ``BOT_OPENAI_API_KEY`` / ``OPENAI_API_KEY``

Run::

    uv run pytest tests/bot/abilities/cognition/test_reflection_live.py -m live -v -s
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import os
import typing
from pathlib import Path

import hsm
import pytest

import bot
from bot import behavior as behavior_events
from bot.abilities import cognition
from bot.abilities import memory
from bot.abilities import processing
from bot.devices import phone as phone_device
from bot.behavior import storage as behavior_storage
from bot.environment import SoundData, SoundEvent, Environment
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test
from tests.hsm_instance_state import phone_firmware

# Optional provider packages — skip collection when not installed in this env.
OpenAIChatClient = pytest.importorskip(
    "bot.providers.openai_compat", reason="bot-provider-openai-compat not installed"
).ChatClient
OpenAIProcessor = pytest.importorskip(
    "bot.providers.openai_compat", reason="bot-provider-openai-compat not installed"
).Processor

# Live calls are slower than unit tests.
_LIVE_TIMEOUT_S = 90.0
_DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_ProcessorFactory = collections.abc.Callable[..., processing.Processor]


def _openai_model() -> str:
    # Prefer explicit reflection override; default Luna (not a global chat-completions model).
    return os.environ.get("BOT_REFLECTION_MODEL") or _DEFAULT_OPENAI_MODEL


def _openai_base_url() -> str:
    return os.environ.get("BOT_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or _DEFAULT_OPENAI_BASE_URL


def _openai_reflection_processor() -> processing.Processor:
    api_key = _openai_api_key()
    assert api_key is not None
    model = _openai_model()
    client = OpenAIChatClient(model=model, api_key=api_key, base_url=_openai_base_url())
    return OpenAIProcessor(client=client, provider="openai_terra_reflection_live")


def _insert_content(
    *,
    content: str,
    scope: str = "memory",
    context_ref: str | None = None,
    subject_ref: str | None = None,
    kind: str | None = "task",
    query_tags: str | None = None,
    content_format: str = "text/plain",
    memory_id: str | None = None,
) -> memory.Statement:
    import uuid
    from sqlalchemy import insert

    table = memory.memory_table
    clause = insert(table).values(
        memory_id=memory_id or uuid.uuid4().hex,
        scope=scope,
        context_ref=context_ref,
        subject_ref=subject_ref,
        kind=kind,
        sensitivity="standard",
        retention="retain",
        content=content,
        content_format=content_format,
        query_tags=query_tags,
    )
    return memory.compile_statement(clause)


def _select_by_query_tags(*, query_tags: str, context_ref: str | None = None, limit: int = 50) -> memory.Statement:
    from sqlalchemy import or_, select

    table = memory.memory_table
    clause = select(table).where(table.c.query_tags == query_tags)
    if context_ref is not None:
        clause = clause.where(or_(table.c.context_ref.is_(None), table.c.context_ref == context_ref))
    clause = clause.order_by(table.c.created_at).limit(limit)
    return memory.compile_statement(clause)


def _load_dotenv_file(path: Path) -> None:
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
            os.environ.setdefault(key, value)


def _load_dotenv() -> None:
    """Load known local env files without overriding existing values."""

    root = Path(__file__).resolve().parents[4]
    candidates = (
        root / ".env",
        root / "examples" / "phone_bot" / ".env",
        root.parent / "bot.py.bak.py" / ".env",
        Path.home() / "VSCode" / "stateforward" / "agent" / "bot.py.bak.py" / ".env",
        Path.home() / "VSCode" / "podium-spacex-agent" / ".env",
        Path.home() / "VSCode" / "cortext-companion" / ".env",
    )
    for path in candidates:
        _load_dotenv_file(path)


def _openai_api_key() -> str | None:
    _load_dotenv()
    for name in ("BOT_OPENAI_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


requires_openai = pytest.mark.skipif(
    _openai_api_key() is None,
    reason="OpenAI API key not set (BOT_OPENAI_API_KEY / OPENAI_API_KEY)",
)


def _focus_output(device: str, reason: str) -> cognition.types.OutputData:
    return (
        cognition.types.EventData(
            event=bot.FocusDeviceEvent.name,
            data=bot.FocusDeviceEventData(device=device).model_dump(),
            reason=reason,
        ),
    )


def _answer_call_output(call_id: str, reason: str) -> cognition.types.OutputData:
    return (
        cognition.types.EventData(
            event="phone.answer_call",
            target="phone",
            data={"call_id": call_id},
            reason=reason,
        ),
    )


def _cognition_input(*, focus: str | None = "phone") -> cognition.InputData:
    return cognition.InputData(
        stimulus=bot.InputEventData(target_device=focus or "phone", priority=0),
        abilities=(),
        focus=focus,
        focus_candidates=(focus,) if focus else (),
    )


async def _seed_ring_answer_episodes(store: memory.Memory, *, count: int = 3) -> None:
    """Store repeated ring→answer cognition episodes so reflection has a clear pattern."""

    for index in range(count):
        episode = cognition.episodes.CognitiveEpisode(
            focus="phone",
            focus_candidates=("phone",),
            stimulus_name="environment.sound",
            output=_answer_call_output(f"call-{index}", f"answered incoming ring episode {index}"),
            behavior=None,
        )
        insert = cognition.episodes.episode_insert_input(episode, context_ref="phone")
        _ = store.execute(insert)


async def _behavior_contents(store: memory.Memory) -> tuple[str, ...]:
    from bot.behavior import storage as behavior_storage

    select = memory.InputData(statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses()))
    out = store.execute(select)
    if not out.results:
        return ()
    sources: list[str] = []
    for row in out.results[0].rows:
        source = row.as_mapping().get("source")
        if isinstance(source, str):
            sources.append(source)
    return tuple(sources)


async def _episodes(store: memory.Memory) -> tuple[cognition.episodes.CognitiveEpisode, ...]:
    select = cognition.episodes.episode_select_input(context_ref="phone")
    out = store.execute(select)
    return cognition.episodes.episodes_from_output(out)


async def _start_reflection(reflection: cognition.Reflection) -> hsm.Context:
    ctx = shared_hsm_context()
    await start_abilities_for_test(ctx, reflection)
    return ctx


@requires_openai
@pytest.mark.live
def test_live_reflection_writes_behavior_from_repeated_ring_pattern() -> None:
    """Real OpenAI Terra: repeated ring→answer episodes → Reflection creates executable Starlark behavior."""

    api_key = _openai_api_key()
    assert api_key is not None
    print(f"\nReflection model={_openai_model()!r} base_url={_openai_base_url()!r}")

    async def run() -> tuple[None, tuple[str, ...], tuple[cognition.episodes.CognitiveEpisode, ...]]:
        store = memory.Memory()
        await _seed_ring_answer_episodes(store, count=3)
        reflection = cognition.Reflection(processor=_openai_reflection_processor(), memory=store)
        ctx = await _start_reflection(reflection)
        input = cognition.reflection.InputData(
            cognition_input=_cognition_input(focus="phone"),
            cognition_output=_answer_call_output("call-live", "answered another incoming ring"),
        )
        result = await asyncio.wait_for(
            dispatch_ability_for_test(reflection, ctx, input, timeout=_LIVE_TIMEOUT_S),
            timeout=_LIVE_TIMEOUT_S + 5.0,
        )
        assert result is None
        return result, await _behavior_contents(store), await _episodes(store)

    _result, stored_behaviors, episodes = asyncio.run(run())

    assert len(episodes) >= 1, "Reflection should store this turn as a cognitive episode"
    latest = episodes[-1]
    assert latest.focus == "phone"
    # Behavior create is the point of this e2e: inventory must gain an executable behavior row.
    assert len(stored_behaviors) >= 1, f"expected behavior inventory row(s), got {stored_behaviors!r}"
    joined = "\n".join(stored_behaviors)
    assert any(
        key in joined.lower()
        for key in ("answer", "ring", "phone", "sound", "call", "triggers", "description", "hsm.define", "behavior")
    ), f"behavior contents look empty or malformed: {stored_behaviors!r}"
    # Episode should carry the behavior payload when create succeeded.
    assert latest.behavior is not None, f"latest episode missing behavior payload: {latest!r}"
    assert isinstance(latest.behavior, behavior_events.CreateData)
    assert latest.behavior.name
    assert latest.behavior.event == behavior_events.CreateEvent.name
    assert latest.behavior.source, f"create must install starlark source, got {latest.behavior!r}"
    # Source must parse/compile as real executable behavior.
    behavior = behavior_events.start(latest.behavior.source, name=latest.behavior.name)
    compiled = behavior_events.build(behavior.source)
    assert compiled.input_event.name
    assert "hsm.define" in latest.behavior.source or "behavior_program" in latest.behavior.source


@requires_openai
@pytest.mark.live
def test_live_reflection_empty_when_no_repeated_pattern() -> None:
    """Real OpenAI Terra: one-off turn with no priors should not invent a behavior."""

    api_key = _openai_api_key()
    assert api_key is not None
    print(f"\nReflection model={_openai_model()!r} base_url={_openai_base_url()!r}")

    async def run() -> tuple[tuple[str, ...], tuple[cognition.episodes.CognitiveEpisode, ...]]:
        store = memory.Memory()
        # No prior episodes seeded.
        reflection = cognition.Reflection(processor=_openai_reflection_processor(), memory=store)
        ctx = await _start_reflection(reflection)
        input = cognition.reflection.InputData(
            cognition_input=_cognition_input(focus="phone"),
            cognition_output=_focus_output("phone", "one-off focus, no pattern"),
        )
        _ = await asyncio.wait_for(
            dispatch_ability_for_test(reflection, ctx, input, timeout=_LIVE_TIMEOUT_S),
            timeout=_LIVE_TIMEOUT_S + 5.0,
        )
        return await _behavior_contents(store), await _episodes(store)

    behaviors, episodes = asyncio.run(run())

    assert len(episodes) == 1
    # Prefer empty behavior inventory; if the model still creates one, fail loudly so we can tighten prompts.
    assert behaviors == (), f"expected no behavior for one-off turn, got {behaviors!r}"
    assert episodes[0].behavior is None


def _live_behavior_answers_phone_call(
    *,
    label: str,
    reflection_processor: _ProcessorFactory,
) -> None:
    """Generate behavior via Reflection processor, run Autonomy, assert phone answers."""

    async def generate_behavior() -> behavior_events.Instance:
        errors: list[str] = []
        # Dry-run install uses the same event shape Autonomy sees at runtime
        # (call_id on event id, not metadata).
        ring_stimulus = dataclasses.replace(
            SoundEvent.with_data(
                SoundData(
                    audio=phone_device.RING_SOUND_WAV,
                    media_type="audio/wav",
                    sample_rate_hz=16_000,
                    channels=1,
                    kind="phone.ringing",
                )
            ),
            id="call-live",
        )
        reflection_turn_input = cognition.InputData(
            stimulus=ring_stimulus,
            abilities=(),
            focus="phone",
            focus_candidates=("phone",),
        )
        for attempt in range(3):
            store = memory.Memory()
            await _seed_ring_answer_episodes(store, count=3)
            reflection = cognition.Reflection(processor=reflection_processor, memory=store)
            ctx = await _start_reflection(reflection)
            input = cognition.reflection.InputData(
                cognition_input=reflection_turn_input,
                cognition_output=_answer_call_output(
                    "call-live",
                    "answered another incoming ring",
                ),
            )
            try:
                _ = await asyncio.wait_for(
                    dispatch_ability_for_test(reflection, ctx, input, timeout=_LIVE_TIMEOUT_S),
                    timeout=_LIVE_TIMEOUT_S + 5.0,
                )
            except Exception as error:  # noqa: BLE001 — live model flakiness; retry
                errors.append(f"attempt {attempt + 1}: {error}")
                continue
            behaviors = await _behavior_contents(store)
            if not behaviors:
                errors.append(f"attempt {attempt + 1}: no behavior inventory row")
                continue
            episodes = await _episodes(store)
            latest = episodes[-1]
            if latest.behavior is None or not isinstance(latest.behavior, behavior_events.CreateData):
                errors.append(f"attempt {attempt + 1}: episode missing create payload")
                continue
            if not latest.behavior.source:
                errors.append(f"attempt {attempt + 1}: create missing source")
                continue
            try:
                return behavior_events.start(
                    latest.behavior.source,
                    name=latest.behavior.name,
                    triggers=latest.behavior.triggers or None,
                    description=latest.behavior.description,
                )
            except Exception as error:  # noqa: BLE001
                errors.append(f"attempt {attempt + 1}: start failed: {error}")
                continue
        raise AssertionError(f"{label} failed to install a valid behavior after retries:\n" + "\n".join(errors))

    installed = asyncio.run(generate_behavior())
    print(f"\n--- {label} behavior {installed.name!r} triggers={installed.triggers!r} ---\n{installed.source}\n---")

    assert installed.triggers, f"behavior must declare triggers from the observed pattern, got {installed.triggers!r}"
    compiled = behavior_events.build(installed.source)
    assert compiled.input_event.name
    assert compiled.output_event.name

    async def run_behavior_and_answer() -> tuple[object, tuple[str, ...], str]:
        call_id = "livekit:caller"
        store = memory.Memory()
        _ = store.execute(
            memory.InputData(statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(installed)))
        )

        environment = Environment()
        phone = phone_device.Phone()
        autonomy = cognition.Autonomy(memory=store)
        shared = shared_hsm_context(environment)
        await start_abilities_for_test(shared, autonomy)
        _ = await bot.started(shared, phone, typing.cast(hsm.Model, phone.model))

        firmware = phone_firmware(phone)
        assert firmware is not None
        await firmware.event_recorder().receive(
            phone.context(),
            phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id=call_id, caller="caller")),
        )
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if "ringing" in firmware.state():
                break
            await asyncio.sleep(0.01)
        assert "ringing" in firmware.state(), f"phone did not enter ringing: {firmware.state()}"

        # Elevation contract: call_id is the ring event id (not metadata / SoundData).
        ring = dataclasses.replace(
            SoundEvent.with_data(
                SoundData(
                    audio=phone_device.RING_SOUND_WAV,
                    media_type="audio/wav",
                    sample_rate_hz=16_000,
                    channels=1,
                    kind="phone.ringing",
                )
            ),
            id=call_id,
            source=hsm.id(phone),
        )
        cognition_input = cognition.InputData(
            stimulus=ring,
            abilities=(),
            actors={"phone": phone},
            focus="phone",
            focus_candidates=("phone",),
        )
        turn = cognition.types.TurnData(
            input=cognition_input,
            operation_id="live-behavior-answer-turn",
            generation="operation-token",
        )
        output = await asyncio.wait_for(
            dispatch_ability_for_test(autonomy, shared, turn, timeout=_LIVE_TIMEOUT_S),
            timeout=_LIVE_TIMEOUT_S + 5.0,
        )
        await asyncio.sleep(0.3)
        published = tuple(event.name for event in firmware.event_recorder().events)
        return output, published, firmware.state()

    output, published, phone_state = asyncio.run(run_behavior_and_answer())
    print(f"autonomy output={output!r}")
    print(f"phone published={published!r}")
    print(f"phone_state={phone_state!r}")

    assert isinstance(output, cognition.types.CompletionData), f"expected CompletionData, got {output!r}"
    selections = processing.coerce_event_selections(output.output)
    assert selections is not None and selections, f"behavior/autonomy produced no selections: {output!r}"
    answer_selections = [item for item in selections if item.event == "phone.answer_call"]
    assert answer_selections, f"expected phone.answer_call selection, got {selections!r}"
    answer_data = answer_selections[0].data
    assert isinstance(answer_data, dict)
    assert answer_data.get("call_id") == "livekit:caller", (
        f"behavior must use the ringing call_id from event['id'], got {answer_data!r}"
    )
    assert phone_device.ServiceAnswerRequestedEvent.name in published, (
        f"phone never requested answer; published={published!r} state={phone_state!r}"
    )
    assert "answering" in phone_state or "answered" in phone_state or "media" in phone_state, (
        f"phone should leave ringing after answer, state={phone_state!r}"
    )


@requires_openai
@pytest.mark.live
def test_live_openai_terra_behavior_answers_phone_call() -> None:
    """Real OpenAI Terra: generate behavior from episodes, Autonomy answers phone."""

    api_key = _openai_api_key()
    assert api_key is not None
    model = _openai_model()
    print(f"\nUsing OpenAI Terra reflection model={model!r} base_url={_openai_base_url()!r}")
    _live_behavior_answers_phone_call(
        label=f"OpenAI/{model}",
        reflection_processor=_openai_reflection_processor,
    )


@requires_openai
@pytest.mark.live
def test_live_behavior_forms_after_n_natural_calls_without_preload() -> None:
    """No pre-seeded episodes: run N real ring→answer cognition turns until Reflection installs a behavior.

    Deliberative reasoning answers every call. Reflection runs after each turn on shared Memory.
    Reports the first N where ACTIVE behavior inventory appears.
    """

    api_key = _openai_api_key()
    assert api_key is not None
    model = _openai_model()
    print(f"\nNatural behavior formation model={model!r} base_url={_openai_base_url()!r}")

    max_calls = int(os.environ.get("BOT_BEHAVIOR_NATURAL_MAX_CALLS", "10"))

    class _EscalateIntuition(processing.Processor):
        @typing.override
        async def process(self, input: processing.InputData) -> processing.Events:
            del input
            # Unhandled → Cognition cascades to reasoning.
            return typing.cast(processing.Events, typing.cast(object, processing.Result[processing.Events].unhandled()))

    class _AnswerRingReasoning(processing.Processor):
        answered: list[str]

        def __init__(self) -> None:
            self.answered = []

        @typing.override
        async def process(self, input: processing.InputData) -> processing.Events:
            # Reasoning wraps the deliberative host input; the live ring is an hsm.Event.
            event = input.input
            host = getattr(event, "host_input", None)
            if host is not None:
                event = getattr(host, "input", event)
            call_id: str
            if isinstance(event, hsm.Event) and isinstance(event.id, str) and event.id:
                call_id = event.id
            else:
                call_id = "missing-call-id"
            self.answered.append(call_id)
            return (
                processing.SelectedEvent(
                    event="phone.answer_call",
                    target="phone",
                    data={"call_id": call_id},
                    reason=f"deliberative answer for call {call_id}",
                ),
            )

    async def wait_idle(ability: hsm.Instance, *, timeout: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            state = ability.state() or ""
            if state.endswith("/idle") or state.endswith("/behavior/idle"):
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(f"{type(ability).__name__} not idle: {ability.state()!r}")

    async def run() -> tuple[int | None, tuple[str, ...], list[str]]:
        store = memory.Memory()
        reasoning = _AnswerRingReasoning()
        reflection = cognition.Reflection(processor=_openai_reflection_processor(), memory=store)
        ability = cognition.Cognition(
            autonomy=cognition.Autonomy(memory=store),
            intuition=cognition.Intuition(processor=_EscalateIntuition()),
            reasoning=cognition.Reasoning(processor=reasoning, memory=store),
            reflection=reflection,
        )
        environment = Environment()
        shared = shared_hsm_context(environment)
        phone = phone_device.Phone()
        await start_abilities_for_test(shared, ability)
        _ = await bot.started(shared, phone, typing.cast(hsm.Model, phone.model))
        await wait_idle(ability, timeout=30.0)
        await wait_idle(reflection, timeout=30.0)

        formed_at: int | None = None
        for n in range(1, max_calls + 1):
            call_id = f"call-{n}"
            # Fresh ringing call each turn (phone may still be answering previous).
            firmware = phone_firmware(phone)
            assert firmware is not None
            # Prefer a clean phone when possible; ignore if still mid-call from prior answer.
            try:
                await firmware.event_recorder().receive(
                    phone.context(),
                    phone_device.IncomingCallEvent.with_data(
                        phone_device.IncomingCallData(call_id=call_id, caller="caller")
                    ),
                )
            except Exception as error:  # noqa: BLE001 — phone may reject while busy
                print(f"call {n}: phone incoming skipped ({error!r}); still running cognition turn")

            ring = dataclasses.replace(
                SoundEvent.with_data(
                    SoundData(
                        audio=phone_device.RING_SOUND_WAV,
                        media_type="audio/wav",
                        sample_rate_hz=16_000,
                        channels=1,
                        kind="phone.ringing",
                    )
                ),
                id=call_id,
                source=hsm.id(phone),
            )
            cognition_input = cognition.InputData(
                stimulus=ring,
                abilities=(),
                actors={"phone": phone},
                focus="phone",
                focus_candidates=("phone",),
            )
            print(f"\n--- natural call {n}/{max_calls} id={call_id!r} ---")
            result = await asyncio.wait_for(
                dispatch_ability_for_test(ability, shared, cognition_input, timeout=_LIVE_TIMEOUT_S),
                timeout=_LIVE_TIMEOUT_S + 30.0,
            )
            print(f"cognition output={result!r}")
            # Reflection is fire-and-forget after cognition terminal; wait until it settles.
            await wait_idle(reflection, timeout=_LIVE_TIMEOUT_S + 30.0)
            await wait_idle(ability, timeout=30.0)

            behaviors = await _behavior_contents(store)
            episodes = await _episodes(store)
            print(f"after call {n}: behaviors={len(behaviors)} episodes={len(episodes)}")
            if behaviors and formed_at is None:
                formed_at = n
                print(f"\n*** behavior formed after {n} natural call(s) ***\n{behaviors[0][:800]}\n")
                break

        final_behaviors = await _behavior_contents(store)
        return formed_at, final_behaviors, list(reasoning.answered)

    formed_at, behaviors, answered = asyncio.run(run())
    print(f"\nformed_at={formed_at} answered={answered} behavior_count={len(behaviors)}")
    assert answered, "reasoning never answered a call"
    assert formed_at is not None, (
        f"no behavior after {max_calls} natural ring→answer turns (no preload); "
        f"answered={answered!r} behaviors={behaviors!r}"
    )
    assert formed_at >= 1
    assert len(behaviors) >= 1
    assert "hsm.define" in behaviors[0] or "behavior" in behaviors[0].lower()

"""Live e2e Reflection tests against real model providers.

Providers (skipped unless keys are available):

- Gemini: ``BOT_GEMINI_API_KEY`` / ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``
- OpenAI-compatible: ``BOT_OPENAI_API_KEY`` / ``OPENAI_API_KEY``

Run::

    uv run pytest tests/bot/abilities/cognition/test_reflection_live.py -m live -v -s
"""

from __future__ import annotations

import bot
from bot import habit as habit_events
from bot.abilities import cognition
from bot.abilities import memory
from bot.abilities import processing
import asyncio
import collections.abc
import dataclasses
import os
import typing
from pathlib import Path

import hsm
import pytest

# Optional provider packages — skip collection when not installed in this env.
GeminiChatClient = pytest.importorskip("bot.providers.gemini", reason="bot-provider-gemini not installed").ChatClient
GeminiProcessor = pytest.importorskip("bot.providers.gemini", reason="bot-provider-gemini not installed").Processor
OpenAIChatClient = pytest.importorskip(
    "bot.providers.openai_compat", reason="bot-provider-openai-compat not installed"
).ChatClient
OpenAIProcessor = pytest.importorskip(
    "bot.providers.openai_compat", reason="bot-provider-openai-compat not installed"
).Processor

from bot.habit import storage as habit_storage
from bot.devices import phone as phone_device
from bot.world import SoundData, SoundEvent, World
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test
from tests.hsm_instance_state import phone_firmware

# Live calls are slower than unit tests.
_LIVE_TIMEOUT_S = 90.0
_DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
_DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
_ProcessorFactory = collections.abc.Callable[..., processing.Processor]




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


def _gemini_api_key() -> str | None:
    _load_dotenv()
    for name in ("BOT_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "VA_GEMINI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _openai_api_key() -> str | None:
    _load_dotenv()
    for name in ("BOT_OPENAI_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


requires_gemini = pytest.mark.skipif(
    _gemini_api_key() is None,
    reason="Gemini API key not set (BOT_GEMINI_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY)",
)

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
            stimulus_name="world.sound",
            output=_answer_call_output(f"call-{index}", f"answered incoming ring episode {index}"),
            habit=None,
        )
        insert = cognition.episodes.episode_insert_input(episode, context_ref="phone")
        _ = store.execute(insert)


async def _habit_contents(store: memory.Memory) -> tuple[str, ...]:
    from bot.habit import storage as habit_storage

    select = memory.InputData(
        statements=memory.compile_statements(*habit_storage.select_all_habits_clauses())
    )
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


async def _start_reflection(reflection: cognition.Reflection) -> object:
    ctx = shared_hsm_context()
    await start_abilities_for_test(ctx, reflection)
    return ctx


@requires_gemini
@pytest.mark.live
def test_live_reflection_writes_habit_from_repeated_ring_pattern() -> None:
    """Real Gemini: repeated ring→answer episodes → Reflection creates executable Starlark habit."""

    api_key = _gemini_api_key()
    assert api_key is not None
    model = os.environ.get("BOT_GEMINI_MODEL") or _DEFAULT_GEMINI_MODEL
    client = GeminiChatClient(model=model, api_key=api_key)

    def reflection_processor() -> processing.Processor:
        return GeminiProcessor(
            client=client,
            provider="gemini_reflection_live",
        )

    async def run() -> tuple[None, tuple[str, ...], tuple[cognition.episodes.CognitiveEpisode, ...]]:
        store = memory.Memory()
        await _seed_ring_answer_episodes(store, count=3)
        reflection = cognition.Reflection(processor=reflection_processor, memory=store)
        ctx = await _start_reflection(reflection)
        input = processing.InputData(
            input=cognition.reflection.InputData(
                cognition_input=_cognition_input(focus="phone"),
                cognition_output=_answer_call_output("call-live", "answered another incoming ring"),
            )
        )
        result = await asyncio.wait_for(
            dispatch_ability_for_test(reflection, ctx, input, timeout=_LIVE_TIMEOUT_S),
            timeout=_LIVE_TIMEOUT_S + 5.0,
        )
        assert result is None
        return result, await _habit_contents(store), await _episodes(store)

    _result, stored_habits, episodes = asyncio.run(run())

    assert len(episodes) >= 1, "Reflection should store this turn as a cognitive episode"
    latest = episodes[-1]
    assert latest.focus == "phone"
    # Habit create is the point of this e2e: inventory must gain an executable habit row.
    assert len(stored_habits) >= 1, f"expected habit inventory row(s), got {stored_habits!r}"
    joined = "\n".join(stored_habits)
    assert any(
        key in joined.lower()
        for key in ("answer", "ring", "phone", "sound", "call", "triggers", "description", "hsm.define", "habit")
    ), f"habit contents look empty or malformed: {stored_habits!r}"
    # Episode should carry the habit payload when create succeeded.
    assert latest.habit is not None, f"latest episode missing habit payload: {latest!r}"
    assert isinstance(latest.habit, habit_events.CreateData)
    assert latest.habit.name
    assert latest.habit.event == habit_events.CreateEvent.name
    assert latest.habit.source, f"create must install starlark source, got {latest.habit!r}"
    # Source must parse/compile as real executable habit behavior.
    habit = habit_events.start(latest.habit.source, name=latest.habit.name)
    compiled = habit_events.build(habit.source)
    assert compiled.input_event.name
    assert "hsm.define" in latest.habit.source or "habit_behavior" in latest.habit.source


@requires_gemini
@pytest.mark.live
def test_live_reflection_empty_when_no_repeated_pattern() -> None:
    """Real Gemini: one-off turn with no priors should not invent a habit."""

    api_key = _gemini_api_key()
    assert api_key is not None
    model = os.environ.get("BOT_GEMINI_MODEL") or _DEFAULT_GEMINI_MODEL
    client = GeminiChatClient(model=model, api_key=api_key)

    def reflection_processor() -> processing.Processor:
        return GeminiProcessor(
            client=client,
            provider="gemini_reflection_live",
        )

    async def run() -> tuple[tuple[str, ...], tuple[cognition.episodes.CognitiveEpisode, ...]]:
        store = memory.Memory()
        # No prior episodes seeded.
        reflection = cognition.Reflection(processor=reflection_processor, memory=store)
        ctx = await _start_reflection(reflection)
        input = processing.InputData(
            input=cognition.reflection.InputData(
                cognition_input=_cognition_input(focus="phone"),
                cognition_output=_focus_output("phone", "one-off focus, no pattern"),
            )
        )
        _ = await asyncio.wait_for(
            dispatch_ability_for_test(reflection, ctx, input, timeout=_LIVE_TIMEOUT_S),
            timeout=_LIVE_TIMEOUT_S + 5.0,
        )
        return await _habit_contents(store), await _episodes(store)

    habits, episodes = asyncio.run(run())

    assert len(episodes) == 1
    # Prefer empty habit inventory; if the model still creates one, fail loudly so we can tighten prompts.
    assert habits == (), f"expected no habit for one-off turn, got {habits!r}"
    assert episodes[0].habit is None


def _live_habit_answers_phone_call(
    *,
    label: str,
    reflection_processor: _ProcessorFactory,
) -> None:
    """Generate habit via Reflection processor, run Autonomy, assert phone answers."""

    async def generate_habit() -> habit_events.Instance:
        errors: list[str] = []
        # Dry-run install uses this same cognition_input as Autonomy will at runtime.
        ring_stimulus = dataclasses.replace(
            SoundEvent.with_data(
                SoundData(
                    audio=phone_device.RING_SOUND_WAV,
                    media_type="audio/wav",
                    sample_rate_hz=16_000,
                    channels=1,
                    kind="ring",
                )
            ),
            metadata={"bot.phone.call_id": "call-live"},
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
            input = processing.InputData(
                input=cognition.reflection.InputData(
                    cognition_input=reflection_turn_input,
                    cognition_output=_answer_call_output(
                        "call-live",
                        "answered another incoming ring",
                    ),
                )
            )
            try:
                _ = await asyncio.wait_for(
                    dispatch_ability_for_test(reflection, ctx, input, timeout=_LIVE_TIMEOUT_S),
                    timeout=_LIVE_TIMEOUT_S + 5.0,
                )
            except Exception as error:  # noqa: BLE001 — live model flakiness; retry
                errors.append(f"attempt {attempt + 1}: {error}")
                continue
            habits = await _habit_contents(store)
            if not habits:
                errors.append(f"attempt {attempt + 1}: no habit inventory row")
                continue
            episodes = await _episodes(store)
            latest = episodes[-1]
            if latest.habit is None or not isinstance(latest.habit, habit_events.CreateData):
                errors.append(f"attempt {attempt + 1}: episode missing create payload")
                continue
            if not latest.habit.source:
                errors.append(f"attempt {attempt + 1}: create missing source")
                continue
            try:
                return habit_events.start(
                    latest.habit.source,
                    name=latest.habit.name,
                    triggers=latest.habit.triggers or None,
                    description=latest.habit.description,
                )
            except Exception as error:  # noqa: BLE001
                errors.append(f"attempt {attempt + 1}: start failed: {error}")
                continue
        raise AssertionError(f"{label} failed to install a valid habit after retries:\n" + "\n".join(errors))

    installed = asyncio.run(generate_habit())
    print(f"\n--- {label} habit {installed.name!r} triggers={installed.triggers!r} ---\n{installed.source}\n---")

    assert installed.triggers, f"habit must declare triggers from the observed pattern, got {installed.triggers!r}"
    compiled = habit_events.build(installed.source)
    assert compiled.input_event.name
    assert compiled.output_event.name

    async def run_habit_and_answer() -> tuple[object, tuple[str, ...], str]:
        call_id = "livekit:caller"
        store = memory.Memory()
        _ = store.execute(
            memory.InputData(
                statements=memory.compile_statements(*habit_storage.insert_habit_clauses(installed))
            )
        )

        world = World()
        phone = phone_device.Phone()
        autonomy = cognition.Autonomy(memory=store)
        shared = shared_hsm_context(world.context)
        await start_abilities_for_test(shared, autonomy)
        _ = await hsm.started(shared, phone, typing.cast(hsm.Model, phone.model))

        firmware = phone_firmware(phone)
        assert firmware is not None
        await firmware.event_recorder().receive(
            phone.context(),
            phone_device.IncomingCallEvent.with_data(
                phone_device.IncomingCallData(call_id=call_id, display_hint="caller")
            ),
        )
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if "ringing" in firmware.state():
                break
            await asyncio.sleep(0.01)
        assert "ringing" in firmware.state(), f"phone did not enter ringing: {firmware.state()}"

        ring = dataclasses.replace(
            SoundEvent.with_data(
                SoundData(
                    audio=phone_device.RING_SOUND_WAV,
                    media_type="audio/wav",
                    sample_rate_hz=16_000,
                    channels=1,
                    kind="ring",
                )
            ),
            source=hsm.id(phone),
            metadata={"bot.phone.call_id": call_id},
        )
        cognition_input = cognition.InputData(
            stimulus=ring,
            abilities=(),
            actors={"phone": phone},
            focus="phone",
            focus_candidates=("phone",),
        )
        output = await asyncio.wait_for(
            dispatch_ability_for_test(autonomy, shared, cognition_input, timeout=_LIVE_TIMEOUT_S),
            timeout=_LIVE_TIMEOUT_S + 5.0,
        )
        await asyncio.sleep(0.3)
        published = tuple(event.name for event in firmware.event_recorder().events)
        return output, published, firmware.state()

    output, published, phone_state = asyncio.run(run_habit_and_answer())
    print(f"autonomy output={output!r}")
    print(f"phone published={published!r}")
    print(f"phone_state={phone_state!r}")

    selections = processing.coerce_event_selections(output)
    assert selections is not None and selections, f"habit/autonomy produced no selections: {output!r}"
    answer_selections = [item for item in selections if item.event == "phone.answer_call"]
    assert answer_selections, f"expected phone.answer_call selection, got {selections!r}"
    assert answer_selections[0].data is not None
    assert answer_selections[0].data.get("call_id") == "livekit:caller", (
        f"habit must use the ringing call_id, got {answer_selections[0].data!r}"
    )
    assert phone_device.ServiceAnswerRequestedEvent.name in published, (
        f"phone never requested answer; published={published!r} state={phone_state!r}"
    )
    assert "answering" in phone_state or "answered" in phone_state or "media" in phone_state, (
        f"phone should leave ringing after answer, state={phone_state!r}"
    )


@requires_gemini
@pytest.mark.live
def test_live_gemini_habit_answers_phone_call() -> None:
    """Real Gemini: generate habit from episodes, run via Autonomy, phone receives answer_call."""

    api_key = _gemini_api_key()
    assert api_key is not None
    model = os.environ.get("BOT_GEMINI_MODEL") or _DEFAULT_GEMINI_MODEL
    client = GeminiChatClient(model=model, api_key=api_key)

    def reflection_processor() -> processing.Processor:
        return GeminiProcessor(
            client=client,
            provider="gemini_reflection_live",
        )

    _live_habit_answers_phone_call(label="Gemini", reflection_processor=reflection_processor)


@requires_openai
@pytest.mark.live
def test_live_openai_gpt54mini_habit_answers_phone_call() -> None:
    """Real OpenAI gpt-5.4-mini: generate habit from episodes, Autonomy answers phone."""

    api_key = _openai_api_key()
    assert api_key is not None
    model = (
        os.environ.get("BOT_OPENAI_MODEL")
        or os.environ.get("CHAT_COMPLETIONS_MODEL")
        or _DEFAULT_OPENAI_MODEL
    )
    base_url = (
        os.environ.get("BOT_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    client = OpenAIChatClient(model=model, api_key=api_key, base_url=base_url)

    def reflection_processor() -> processing.Processor:
        return OpenAIProcessor(
            client=client,
            provider="openai_gpt54mini_reflection_live",
        )

    print(f"\nUsing OpenAI-compatible model={model!r} base_url={base_url!r}")
    _live_habit_answers_phone_call(label=f"OpenAI/{model}", reflection_processor=reflection_processor)

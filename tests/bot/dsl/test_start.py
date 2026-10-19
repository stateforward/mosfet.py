"""``bot.start`` and ``bot.stop``: one body built from a model's parts, B0101..B0103, and start telemetry.

DSL-7 and DSL-9 of the wave 7 bot DSL design (W7-30; OQ-55).
"""

import asyncio
import collections.abc
import datetime
import gc
import pathlib
import typing
import warnings

import hsm
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mosfet
from mosfet import bot, lifecycle
from mosfet import abilities
from mosfet.abilities import cognition, communication, listening, memory, persona, speaking, vocal
from mosfet.abilities.communication import conversation
from mosfet.abilities.communication.conversation import turn_detector
from mosfet.abilities.hearing import voice
from mosfet.behavior import schedule
from mosfet.bot import body, diagnostics, elements, testing
from mosfet.device import Device
from mosfet.devices import audio, phone
from mosfet.environment import Environment
from mosfet.telemetry import span
from tests import metric_reader
from tests.bot.dsl import support


@pytest.fixture
def here(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


Built = list[tuple[str, dict[str, object]]]
"""Each faculty, persona and store ``bot.start`` built, by class, with the arguments it was given."""


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> Built:
    """Record what ``bot.start`` builds. Define a model before using it: the recorded classes are subclasses."""

    found: Built = []
    for owner, cls in (
        (cognition, cognition.Reflection),
        (cognition, cognition.Autonomy),
        (cognition, cognition.Routines),
        (memory, memory.Associative),
        (memory, memory.Consolidation),
        (memory, memory.Memory),
        (persona, persona.Persona),
        (vocal.voice, vocal.voice.Inception),
    ):
        monkeypatch.setattr(owner, cls.__name__, _recorded(cls, found))
    return found


def _recorded(cls: type[typing.Any], found: Built) -> type[typing.Any]:
    def __init__(self: object, **given: object) -> None:
        found.append((cls.__name__, given))
        cls.__init__(self, **given)

    return type(cls.__name__, (cls,), {"__init__": __init__})


class Recording(body.Bot):
    """The body, recording what ``bot.start`` gives it."""

    given: typing.ClassVar[list[tuple[tuple[Device, ...], dict[str, object]]]] = []
    data: typing.ClassVar[list[object]] = []

    def __init__(
        self,
        devices: collections.abc.Sequence[Device],
        *,
        cognition: abilities.Ability[cognition.InputData, typing.Any],
        input: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        output: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        acquired_abilities: tuple[abilities.Ability[typing.Any, typing.Any], ...] = (),
        processing_timeout: datetime.timedelta | None = None,
        deactivation_timeout: datetime.timedelta | None = None,
        attachment_timeout: datetime.timedelta | None = None,
        stm_memory: memory.StmMemory | None = None,
        firmware: hsm.Instance | None = None,
        name: str | None = None,
        observe: collections.abc.Sequence[hsm.ObservationElement[typing.Any]] = (),
    ) -> None:
        parts: dict[str, object] = {
            "cognition": cognition,
            "input": input,
            "output": output,
            "acquired_abilities": acquired_abilities,
            "processing_timeout": processing_timeout,
            "deactivation_timeout": deactivation_timeout,
            "attachment_timeout": attachment_timeout,
            "stm_memory": stm_memory,
            "firmware": firmware,
            "name": name,
            "observe": observe,
        }
        Recording.given.append((tuple(devices), parts))
        super().__init__(
            devices,
            cognition=cognition,
            input=input,
            output=output,
            acquired_abilities=acquired_abilities,
            processing_timeout=processing_timeout,
            deactivation_timeout=deactivation_timeout,
            attachment_timeout=attachment_timeout,
            stm_memory=stm_memory,
            firmware=firmware,
            name=name,
            observe=observe,
        )

    @typing.override
    def start(self, ctx: hsm.Context, data: object = None) -> collections.abc.Awaitable[typing.Self]:
        Recording.data.append(data)
        return super().start(ctx, data)


@pytest.fixture
def recording(monkeypatch: pytest.MonkeyPatch) -> type[Recording]:
    Recording.given.clear()
    Recording.data.clear()
    monkeypatch.setattr(body, "Bot", Recording)
    return Recording


async def _started(model: bot.Model, data: object = None) -> tuple[mosfet.Bot, Environment]:
    environment = Environment()
    return await bot.start(environment, model, data), environment


def _clock() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


# Building and activating


def test_start_returns_the_activated_body_holding_the_models_devices_by_id_in_order() -> None:
    store = memory.Memory()
    model = bot.define(
        "greeter",
        bot.cognition(support.Silent(), intuition=support.Undecided(), memory=store),
        bot.attach(phone.Phone(id="desk")),
        bot.attach(audio.Speaker()),
    )

    async def run() -> tuple[str, list[str], bool, bool]:
        started, environment = await _started(model)
        try:
            owned = typing.cast(dict[str, str], (started.take_snapshot().Attributes or {})["/greeter/owned_devices"])
            return (
                started.state(),
                list(owned),
                lifecycle.is_started(store),
                environment.contains(started),
            )
        finally:
            await bot.stop(started)

    state, owned, store_started, in_environment = asyncio.run(run())

    assert state.startswith("/greeter/active")
    assert owned == ["desk", "speaker"]
    assert store_started
    assert in_environment


def test_the_body_takes_each_part_in_the_role_it_was_given(recording: type[Recording]) -> None:
    desk, speaker = phone.Phone(id="desk"), audio.Speaker()
    bench, senses, voice, register = (
        memory.Memory(id="bench"),
        memory.Memory(id="senses"),
        memory.Memory(),
        memory.StmMemory(),
    )
    model = bot.define(
        "greeter",
        bot.cognition(support.Silent(), intuition=support.Undecided()),
        bot.attach(desk),
        bot.attach(bench),
        bot.input(senses),
        bot.output(voice),
        bot.attach(speaker),
        bot.register(register),
        bot.timeouts(processing=datetime.timedelta(seconds=30), attachment=datetime.timedelta(seconds=5)),
    )

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())

    ((devices, parts),) = recording.given
    assert devices == (desk, speaker)
    assert parts["acquired_abilities"] == (bench,)
    assert parts["input"] == (senses,)
    assert parts["output"] == (voice,)
    assert parts["stm_memory"] is register
    assert parts["processing_timeout"] == datetime.timedelta(seconds=30)
    assert parts["deactivation_timeout"] is None
    assert parts["attachment_timeout"] == datetime.timedelta(seconds=5)
    assert isinstance(parts["cognition"], cognition.Cognition)
    assert parts["name"] == "greeter"


@pytest.mark.parametrize(
    ("timeouts", "step"),
    [
        ((bot.timeouts(step=datetime.timedelta(minutes=3)),), datetime.timedelta(minutes=3)),
        ((bot.timeouts(processing=datetime.timedelta(seconds=30)),), None),
        ((), None),
    ],
)
def test_the_step_timeout_bounds_each_step_of_the_bots_cognition(
    timeouts: tuple[bot.Element[hsm.Element], ...], step: datetime.timedelta | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-R-3, D-R-17: ``bot.timeouts(step=)`` is the step timeout of cognition and of its reflection.

    Without it each keeps cognition's default. Live (builder check round 3, H4): reflection's revision child
    had a fixed 30 s bound, so a slow model's revision timed out and failed its turn 124 times.
    """

    found: Built = []
    monkeypatch.setattr(cognition, "Cognition", _recorded(cognition.Cognition, found))
    monkeypatch.setattr(cognition, "Reflection", _recorded(cognition.Reflection, found))
    model = bot.define(
        "greeter", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(phone.Phone()), *timeouts
    )

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())

    assert sorted((name, given["child_operation_timeout"]) for name, given in found) == [
        ("Cognition", step),
        ("Reflection", step),
    ]


def test_one_ability_given_as_input_and_acquired_is_given_to_the_body_in_both_roles(
    recording: type[Recording],
) -> None:
    senses = memory.Memory(id="senses")
    model = bot.define(
        "greeter", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(senses), bot.input(senses)
    )

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())

    ((_, parts),) = recording.given
    assert parts["acquired_abilities"] == (senses,)
    assert parts["input"] == (senses,)


def test_start_gives_data_to_the_bodys_start(recording: type[Recording]) -> None:
    model = bot.define(
        "greeter", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(phone.Phone())
    )
    data = object()

    async def run() -> None:
        started, _ = await _started(model, data)
        await bot.stop(started)

    asyncio.run(run())

    assert data in recording.data


Flags = tuple[bool, bool, bool, schedule.WallClock | None]
"""``bot.cognition``'s ``associative``, ``consolidation``, ``autonomy`` and ``routines``."""


@pytest.mark.parametrize(
    ("flags", "faculties"),
    [
        ((False, False, False, None), []),
        ((True, False, False, None), ["Associative"]),
        ((False, True, False, None), ["Consolidation"]),
        ((False, False, True, None), ["Autonomy"]),
        ((False, False, False, _clock), ["Routines"]),
        ((True, True, True, _clock), ["Associative", "Consolidation", "Autonomy", "Routines"]),
    ],
)
def test_each_cognition_flag_builds_exactly_its_faculty_on_the_one_store(
    flags: Flags, faculties: list[str], built: Built
) -> None:
    store = memory.Memory()
    associative, consolidation, autonomy, routines = flags
    model = bot.define(
        "greeter",
        bot.cognition(
            support.Silent(),
            intuition=support.Undecided(),
            memory=store,
            associative=associative,
            consolidation=consolidation,
            autonomy=autonomy,
            routines=routines,
        ),
        bot.attach(phone.Phone()),
    )
    built.clear()

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())

    names = [name for name, _ in built]
    assert sorted(names) == sorted(["Reflection", *faculties])
    assert all(given["memory"] is store for _, given in built)


def test_intuition_decides_on_its_decision_processor_and_every_other_faculty_thinks_on_the_cognitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-6: intuition is never a chat model; reasoning, reflection and consolidation keep cognition's processor."""

    thinking, deciding = support.Silent(), support.Undecided()
    model = bot.define(
        "greeter", bot.cognition(thinking, intuition=deciding, consolidation=True), bot.attach(phone.Phone())
    )
    found: Built = []
    for owner, cls in (
        (cognition, cognition.Intuition),
        (cognition, cognition.Reasoning),
        (cognition, cognition.Reflection),
        (memory, memory.Consolidation),
    ):
        monkeypatch.setattr(owner, cls.__name__, _recorded(cls, found))

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())

    assert {name: given["processor"] for name, given in found} == {
        "Intuition": deciding,
        "Reasoning": thinking,
        "Reflection": thinking,
        "Consolidation": thinking,
    }


def test_routines_run_on_the_wall_clock_cognition_was_given(built: Built) -> None:
    model = bot.define(
        "greeter",
        bot.cognition(support.Silent(), intuition=support.Undecided(), routines=_clock),
        bot.attach(phone.Phone()),
    )
    built.clear()

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())

    ((_, given),) = [entry for entry in built if entry[0] == "Routines"]
    assert given["now"] is _clock


def test_without_memory_the_bot_gets_a_fresh_in_memory_store(built: Built) -> None:
    model = bot.define(
        "greeter", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(phone.Phone())
    )
    built.clear()

    async def run() -> bool:
        started, _ = await _started(model)
        try:
            ((_, reflection),) = [entry for entry in built if entry[0] == "Reflection"]
            store = reflection["memory"]
            return isinstance(store, memory.Memory) and lifecycle.is_started(store)
        finally:
            await bot.stop(started)

    fresh = asyncio.run(run())

    assert [given for name, given in built if name == "Memory"] == [{}]
    assert fresh


def test_with_traits_the_persona_carries_them_to_reflection_and_shares_the_store(built: Built) -> None:
    store = memory.Memory()
    model = bot.define(
        "greeter",
        bot.traits("curious", "patient"),
        bot.cognition(support.Silent(), intuition=support.Undecided(), memory=store),
        bot.attach(phone.Phone()),
    )
    built.clear()

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())

    ((_, given),) = [entry for entry in built if entry[0] == "Persona"]
    assert given["traits"] == persona.Traits(traits=("curious", "patient"))
    assert given["memory"] is store
    ((_, reflection),) = [entry for entry in built if entry[0] == "Reflection"]
    assert isinstance(reflection["persona"], persona.Persona)


def test_without_traits_there_is_no_persona(built: Built) -> None:
    model = bot.define(
        "greeter", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(phone.Phone())
    )
    built.clear()

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())

    assert [name for name, _ in built if name == "Persona"] == []
    ((_, reflection),) = [entry for entry in built if entry[0] == "Reflection"]
    assert reflection["persona"] is None


# Seeds and the voice given at inception (W8-10)


async def _until(done: collections.abc.Callable[[], bool], *, seconds: float = 5.0) -> None:
    """Yield to the bot until ``done()``; the bound is the test's, for a wait that never ends."""

    async with asyncio.timeout(seconds):
        while not done():
            await asyncio.sleep(0.01)


def test_the_seeds_reach_autonomy_as_given(built: Built) -> None:
    heard = communication.speech_heard_seed()
    model = bot.define(
        "greeter",
        bot.cognition(support.Silent(), intuition=support.Undecided(), autonomy=True, seeds=(heard,)),
        bot.attach(phone.Phone()),
    )
    built.clear()

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())

    ((_, given),) = [entry for entry in built if entry[0] == "Autonomy"]
    seeded = given["seeded_behaviors"]
    assert isinstance(seeded, tuple) and seeded == (heard,)
    assert typing.cast(tuple[object, ...], seeded)[0] is heard


def test_without_seeds_autonomy_runs_only_what_it_learns(built: Built) -> None:
    model = bot.define(
        "greeter",
        bot.cognition(support.Silent(), intuition=support.Undecided(), autonomy=True),
        bot.attach(phone.Phone()),
    )
    built.clear()

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())

    ((_, given),) = [entry for entry in built if entry[0] == "Autonomy"]
    assert given["seeded_behaviors"] == ()


@pytest.mark.parametrize("seeded", [True, False])
def test_the_speech_seed_takes_a_heard_turn_into_the_conversation(seeded: bool) -> None:
    """With ``communication.speech_heard_seed()``, heard speech goes to the conversation before the model is
    asked about it; without it, the turn is the model's to deliberate. Neither pins what the bot says."""

    model = testing.Scripted()
    talk = conversation.Conversation(
        turn_detector=turn_detector.TurnDetector(participant_ref="bot", conversation_ref="ambient")
    )
    mouth = speaking.Speaking(encoder=support.Mute(), conversation=(talk,))
    greeter = bot.define(
        "greeter",
        bot.cognition(
            model,
            intuition=testing.Decider(),
            autonomy=seeded,
            seeds=(communication.speech_heard_seed(),) if seeded else (),
        ),
        bot.output(mouth),
        bot.attach(communication.Communication(conversations=(talk,), speaking=mouth)),
        bot.attach(phone.Phone()),
    )
    speech = listening.SpeechData(
        content=bytes([0, 1]) * 160,
        voice_detection=voice.detection.ApplyData(
            segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=0.02, confidence=0.9),)
        ),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({"voice:3f2a9c"}),
    )

    async def run() -> str:
        async with testing.running(greeter) as running:
            # What Listening hands the body: a heard product, as a cognition input.
            heard = cognition.InputEvent.with_data(
                cognition.InputData(stimulus=listening.SpeechEvent.with_data(speech))
            )
            _ = await hsm.dispatch(running.environment, running.bot, heard)
            if seeded:
                await _until(lambda: "/behavior/active" in (talk.state() or ""))
            else:
                await _until(lambda: bool(model.woken_by(listening.SpeechEvent.name)))
            return talk.state()

    state = asyncio.run(run())

    if seeded:
        assert "/behavior/active" in state
        assert model.woken_by(listening.SpeechEvent.name) == ()
    else:
        assert "/behavior/inactive" in state


def test_a_voice_is_given_once_at_inception_from_the_traits_and_the_bot_is_born(built: Built) -> None:
    store, shelf, voices, conceiving = memory.Memory(), support.Shelf(), support.Voices(), support.Conceiving()
    mouth = speaking.Speaking(encoder=voices)
    model = bot.define(
        "greeter",
        bot.traits("warm", "patient"),
        bot.voice(shelf, languages=("en",), allowed=("warm",)),
        bot.cognition(conceiving, intuition=support.Undecided(), memory=store),
        bot.output(mouth),
        bot.attach(phone.Phone()),
    )
    built.clear()

    async def run() -> None:
        started, _ = await _started(model)
        try:
            await _until(lambda: bool(voices.calls))
        finally:
            await bot.stop(started)

    asyncio.run(run())

    ((_, given),) = [entry for entry in built if entry[0] == "Inception"]
    assert given == {
        "traits": persona.Traits(traits=("warm", "patient")),
        "languages": ("en",),
        "speaking": mouth,
        "processor": conceiving,
        "catalog": shelf,
        "designer": None,
        "memory": store,
        # The voices of the catalog its owner allowed it to be born as.
        "allowed": ("warm",),
    }
    assert given["speaking"] is mouth and given["memory"] is store
    # Born: the one conception made, and Speaking told the voice it was given, once.
    assert conceiving.conceived == 1
    assert voices.calls == [("warm", support.WARM.description)]


def test_without_a_voice_there_is_no_inception(built: Built) -> None:
    model = bot.define(
        "greeter",
        bot.traits("warm"),
        bot.cognition(support.Silent(), intuition=support.Undecided()),
        bot.output(speaking.Speaking(encoder=support.Voices())),
        bot.attach(phone.Phone()),
    )
    built.clear()

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())

    assert [name for name, _ in built if name == "Inception"] == []


# Boundary observers (DSL-26)


def _watching(seen: list[str]) -> bot.Observer:
    """An observer that records the name of each body request it sees, and does nothing else."""

    def watch(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        seen.append(typing.cast(dict[str, hsm.Event[typing.Any]], event.data)["event"].name)

    return bot.observe(mosfet.ActivateEvent, mosfet.DeactivateEvent, watch)


def test_bot_observe_is_hsms_observe() -> None:
    assert bot.observe is hsm.observe


def test_start_runs_its_observers_on_this_bots_body_and_only_for_this_start() -> None:
    """The observers watch the body ``bot.start`` started, under the bot's name; a later start has none."""

    model = bot.define(
        "greeter", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(phone.Phone(id="desk"))
    )
    seen: list[str] = []

    async def run() -> list[str]:
        environment = Environment()
        watched = await bot.start(environment, model, observe=(_watching(seen),))
        states = [watched.state()]
        await bot.stop(watched)
        again = await bot.start(environment, model)
        states.append(again.state())
        await bot.stop(again)
        return states

    states = asyncio.run(run())

    assert [state.split("/")[1:3] for state in states] == [["greeter", "active"], ["greeter", "active"]]
    assert seen == ["bot.activate", "bot.deactivate"]


def test_start_gives_its_observers_to_the_body(recording: type[Recording]) -> None:
    model = bot.define(
        "greeter", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(phone.Phone())
    )
    observer = _watching([])

    async def run() -> None:
        started = await bot.start(Environment(), model, observe=(observer,))
        await bot.stop(started)

    asyncio.run(run())

    ((_, parts),) = recording.given
    assert parts["observe"] == (observer,)


def test_start_observes_with_observers_only() -> None:
    model = bot.define(
        "greeter", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(phone.Phone())
    )

    async def run() -> None:
        _ = await bot.start(
            Environment(), model, observe=typing.cast(collections.abc.Sequence[bot.Observer], (bot.state("extra"),))
        )

    with pytest.raises(TypeError, match="observe takes hsm observers"):
        asyncio.run(run())


def test_start_takes_a_bot_model() -> None:
    async def run() -> None:
        _ = await bot.start(Environment(), typing.cast(bot.Model, object()))

    with pytest.raises(
        TypeError, match=r"^bot.start takes a bot.Model, not object: pass the model `bot.define` returned$"
    ):
        asyncio.run(run())


# Stopping


def test_stop_stops_the_body_and_every_part_it_started() -> None:
    """D-W7-9: the body detaches its parts, then stops them and itself: devices, cognition, its memory, the register."""

    store, desk, register = memory.Memory(), phone.Phone(), memory.StmMemory()
    model = bot.define(
        "greeter",
        bot.cognition(support.Silent(), intuition=support.Undecided(), memory=store),
        bot.attach(desk),
        bot.register(register),
    )

    async def run() -> list[bool]:
        started, _ = await _started(model)
        await bot.stop(started)
        return [lifecycle.is_started(part) for part in (started, store, desk, register)]

    assert asyncio.run(run()) == [False, False, False, False]


def test_a_stopped_model_starts_again_and_leaves_nothing_open() -> None:
    """D-W7-9: the same model starts, stops and starts again, and no database it opened is left unclosed."""

    def twice() -> list[str]:
        model = bot.define(
            "greeter",
            bot.cognition(
                support.Silent(), intuition=support.Undecided(), associative=True, consolidation=True, autonomy=True
            ),
            bot.attach(phone.Phone(id="desk")),
            bot.register(memory.StmMemory()),
        )

        async def run() -> list[str]:
            states: list[str] = []
            for _ in range(2):
                started, _ = await _started(model)
                states.append(started.state())
                await bot.stop(started)
            return states

        return asyncio.run(run())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        states = twice()
        _ = gc.collect()

    assert [state.split("/")[2] for state in states] == ["active", "active"]
    assert [str(warning.message) for warning in caught if issubclass(warning.category, ResourceWarning)] == []


def test_a_stopped_model_redefines_and_its_stopped_parts_define_another_bot() -> None:
    """D-W7-9, W7-19: once bot.stop has stopped its parts, the model redefines like hsm's, and its parts are free."""

    desk = phone.Phone(id="desk")
    model = bot.define("greeter", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(desk))

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)

    asyncio.run(run())
    again = bot.redefine(model)
    other = bot.define("other", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(desk))

    assert list(again.members) == list(model.members)
    assert list(other.members) == ["/other", "/other/.cognition", "/other/desk"]


def test_stopping_a_stopped_bot_does_nothing() -> None:
    model = bot.define(
        "greeter", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(phone.Phone())
    )

    async def run() -> bool:
        started, _ = await _started(model)
        await bot.stop(started)
        await bot.stop(started)
        return lifecycle.is_started(started)

    assert not asyncio.run(run())


# B0101..B0103


def _error(path: pathlib.Path, *arguments: object) -> bot.StartError:
    module = support.run(path)
    main = typing.cast(collections.abc.Callable[..., collections.abc.Awaitable[None]], module.main)

    async def run() -> None:
        await main(*arguments)

    with pytest.raises(bot.StartError) as caught:
        asyncio.run(run())
    return caught.value


def _raising_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    def __init__(self: object, **given: object) -> None:
        del self, given
        raise ValueError("this processor cannot reason")

    monkeypatch.setattr(cognition.Reasoning, "__init__", __init__)


def _rendered(error: bot.StartError) -> str:
    return diagnostics.report(error.diagnostics, action="start", name="greeter")


def test_a_model_started_again_is_b0101_with_the_part_that_runs(here: pathlib.Path) -> None:
    error = _error(support.copy("B0101", here), Environment())

    (found,) = error.diagnostics
    assert found.code == "B0101"
    assert _rendered(error) == support.golden("B0101")
    assert str(error) == f"{here / 'greeter.py'}:{found.primary.lineno}: {support.golden('B0101')}"


def test_a_part_shared_with_a_running_bot_is_b0101() -> None:
    store = memory.Memory()
    first = bot.define(
        "greeter",
        bot.cognition(support.Silent(), intuition=support.Undecided(), memory=store),
        bot.attach(phone.Phone()),
    )
    second = bot.define(
        "helper",
        bot.cognition(support.Silent(), intuition=support.Undecided(), memory=store),
        bot.attach(phone.Phone()),
    )

    async def run() -> bot.StartError:
        started, environment = await _started(first)
        try:
            with pytest.raises(bot.StartError) as caught:
                _ = await bot.start(environment, second)
            return caught.value
        finally:
            await bot.stop(started)

    (found,) = asyncio.run(run()).diagnostics
    assert found.code == "B0101"
    assert [span.label for span in found.secondary] == ["this memory is already running"]


def test_two_parts_with_one_id_are_never_both_the_firmwares_peer() -> None:
    """DSL-18: the firmware sends by id, so a model edited past define's B0003 is B0102, not a shadowed part."""

    model = bot.define(
        "greeter",
        bot.cognition(support.Silent(), intuition=support.Undecided()),
        bot.attach(phone.Phone(id="ear")),
        bot.initial(bot.target("idle")),
        bot.state("idle"),
    )
    model.members["/greeter/voice"] = elements.AttachmentElement(
        qualified_name="/greeter/voice", instance=memory.Memory(id="ear"), roles=["output"], location=model.location
    )

    with pytest.raises(bot.StartError) as caught:
        _ = asyncio.run(_started(model))

    (found,) = caught.value.diagnostics
    assert found.code == "B0102"
    assert isinstance(caught.value.__cause__, ValueError)


def test_a_part_that_raises_while_the_bot_is_built_is_b0102_chained_to_its_cause(
    here: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = support.copy("B0102", here)
    _raising_reasoning(monkeypatch)

    error = _error(path, Environment())

    (found,) = error.diagnostics
    assert found.code == "B0102"
    assert isinstance(error.__cause__, ValueError)
    assert _rendered(error) == support.golden("B0102")


def test_a_bot_started_outside_an_environment_is_b0103(here: pathlib.Path) -> None:
    error = _error(support.copy("B0103", here))

    (found,) = error.diagnostics
    assert found.code == "B0103"
    assert _rendered(error) == support.golden("B0103")


def test_a_start_error_is_a_bot_error_rendered_after_hsms_file_and_line(here: pathlib.Path) -> None:
    error = _error(support.copy("B0103", here))

    assert isinstance(error, bot.Error)
    assert isinstance(error, hsm.ErrorValidatingModel)
    assert str(error).startswith(f"{here / 'greeter.py'}:13: error[B0103]:")
    assert str(error).endswith("error: could not start bot `greeter`: 1 error")


_START_CODES = ("B0101", "B0102", "B0103")
_START_EXPLANATIONS = [explanation for explanation in diagnostics.EXPLANATIONS if explanation.code in _START_CODES]


@pytest.mark.parametrize(
    "explanation", _START_EXPLANATIONS, ids=[explanation.code for explanation in _START_EXPLANATIONS]
)
def test_each_start_explanation_shows_its_mistake_and_the_fix(
    explanation: diagnostics.Explanation, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patched:
        if explanation.code == "B0102":
            # No part a bot file can give raises while it is built today; the example's part is made to.
            _raising_reasoning(patched)
        with pytest.raises(bot.StartError) as caught:
            asyncio.run(support.execute(explanation.wrong))
    assert explanation.code in [diagnostic.code for diagnostic in caught.value.diagnostics]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        asyncio.run(support.execute(explanation.right))


@pytest.mark.parametrize("code", _START_CODES)
def test_every_start_code_has_an_explanation(code: str) -> None:
    assert bot.explain(code).startswith(f"error[{code}]: ")


# Telemetry (DSL-9)


def _increase(
    before: tuple[metric_reader.Point, ...], after: tuple[metric_reader.Point, ...], *keys: str
) -> dict[str, float]:
    def totals(points: tuple[metric_reader.Point, ...]) -> dict[str, float]:
        found: dict[str, float] = {}
        for point in points:
            label = "/".join(str(point.attributes.get(key)) for key in keys)
            found[label] = found.get(label, 0.0) + point.value
        return found

    prior, later = totals(before), totals(after)
    return {label: later[label] - prior.get(label, 0.0) for label in later if later[label] != prior.get(label, 0.0)}


def test_each_start_is_one_span_and_one_count_on_closed_attributes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = bot.define(
        "greeter", bot.cognition(support.Silent(), intuition=support.Undecided()), bot.attach(phone.Phone(id="desk"))
    )

    async def run() -> None:
        started, _ = await _started(model)
        await bot.stop(started)
        with pytest.raises(bot.StartError):
            _ = await bot.start(typing.cast(Environment, hsm.Context()), model)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points("bot.model.start.count")
        asyncio.run(run())
        after = reader.points("bot.model.start.count")

    assert _increase(before, after, "bot.outcome", "bot.failure.kind") == {
        "ok/none": 1,
        "failed/not_an_environment": 1,
    }
    assert {key for point in after for key in point.attributes} == {"bot.outcome", "bot.failure.kind"}
    starts = [finished for finished in exporter.get_finished_spans() if finished.name == "bot.model.start"]
    assert len(starts) == 2
    assert {finished.instrumentation_scope.name for finished in starts if finished.instrumentation_scope} == {
        "bot.model"
    }
    keys = {key for finished in starts for key in (finished.attributes or {})}
    assert keys <= {"bot.component.name", "bot.stage", "bot.outcome", "bot.failure.kind"}
    values = {str(value) for finished in starts for value in (finished.attributes or {}).values()}
    assert not any("greeter" in value or "desk" in value for value in values)

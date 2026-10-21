"""Intuition: the fast tier between reflex and reasoning, a chat model with Reasoning's chart (D-R-52)."""

from mosfet.abilities import ability
from mosfet.abilities import cognition
from mosfet.abilities import processing
from mosfet.abilities.cognition import intuition as intuition_module
from mosfet.environment import SoundData, SoundEvent
from mosfet.telemetry import span

import asyncio
import typing

import hsm
import mosfet
import pydantic
import pytest
from mosfet import event
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.bot.abilities.cognition.metadata_contract import assert_metadata_is_not_coordination
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test
from tests.hsm_model import choice_transitions, transition_map


class _SaidData(pydantic.BaseModel):
    text: str


_SayEvent = hsm.Event[_SaidData](name="tests.intuition.say", kind=event.EventKind, schema=_SaidData)


class _ThinkData(pydantic.BaseModel):
    value: str


_ThinkEvent = hsm.Event[_ThinkData](name="tests.intuition.think", kind=event.EventKind, schema=_ThinkData)


class _Mouth(hsm.Instance):
    """An M-effort actor within intuition's ceiling that keeps what it was told to say."""

    effort: typing.ClassVar[ability.Effort] = ability.Effort.M
    said: list[str]

    @staticmethod
    def _say(ctx: hsm.Context, instance: "_Mouth", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _SaidData)
        instance.said.append(data.text)

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "IntuitionMouth",
        hsm.initial(hsm.target("/IntuitionMouth/active")),
        hsm.state("active", hsm.transition(hsm.on(_SayEvent), hsm.effect(_say))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.said = []


def _accept(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
    del ctx, instance, event


class _Bench(hsm.Instance):
    """An L-effort actor: above intuition's ceiling, held by reasoning."""

    effort: typing.ClassVar[ability.Effort] = ability.Effort.L
    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "IntuitionBench",
        hsm.initial(hsm.target("/IntuitionBench/active")),
        hsm.state("active", hsm.transition(hsm.on(_ThinkEvent), hsm.effect(_accept))),
    )


class _Host(hsm.Instance):
    """The cognition host's call surface as a frame offers it: the deliberative handoff."""

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "IntuitionHost",
        hsm.initial(hsm.target("/IntuitionHost/active")),
        hsm.state("active", hsm.transition(hsm.on(cognition.reasoning.InputEvent), hsm.effect(_accept))),
    )


class _Answering(processing.Processor):
    """A chat model that answers every call with ``answer`` and keeps each frame it was given."""

    calls: list[processing.InputData]
    answer: processing.Events

    def __init__(self, answer: processing.Events = ()) -> None:
        self.calls = []
        self.answer = answer

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del drafts
        self.calls.append(input)
        return self.answer


class _Deciding(processing.DecisionProcessor):
    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input, drafts
        return ()


def _intuit(
    processor: processing.Processor,
) -> tuple[cognition.types.CompletionData, processing.InputData, _Mouth]:
    async def run() -> tuple[cognition.types.CompletionData, _Mouth]:
        ctx = shared_hsm_context()
        mouth, bench, host = _Mouth(), _Bench(), _Host()
        for actor in (mouth, bench, host):
            assert actor.model is not None
            _ = await mosfet.started(ctx, actor, actor.model)
        stimulus = SoundEvent.with_data(SoundData(audio=b"hello", kind="speech"))
        turn_input = cognition.InputData(
            stimulus=stimulus, actors={"mouth": mouth, "sandbox": bench, "cognition": host}
        )
        intuition = cognition.Intuition(processor=processor)
        await start_abilities_for_test(ctx, intuition)
        output = await dispatch_ability_for_test(
            intuition,
            ctx,
            cognition.intuition.InputData(
                turn=cognition.types.TurnData(input=turn_input, operation_id="intuition-turn", generation="token"),
                processing_input=cognition.input.build_processing_input(turn_input),
            ),
            timeout=1.0,
        )
        return typing.cast(cognition.types.CompletionData, typing.cast(object, output)), mouth

    processing_answer = asyncio.run(run())
    (frame,) = typing.cast(_Answering, processor).calls
    return processing_answer[0], frame, processing_answer[1]


def test_intuition_never_uses_metadata_for_coordination() -> None:
    assert_metadata_is_not_coordination(intuition_module)


def test_intuition_thinks_on_a_chat_model_never_a_decision_model() -> None:
    """D-R-52: intuition writes words, which a decision model cannot."""

    with pytest.raises(TypeError, match="DecisionProcessor"):
        _ = cognition.Intuition(processor=_Deciding())


def test_intuition_shares_reasonings_chart_under_its_own_events() -> None:
    model = typing.cast(hsm.Model, intuition_module.Intuition.submodel)

    for state in ("initializing", "idle", "applying", "routing_rejection"):
        assert f"/Intuition/{state}" in model.members
    assert transition_map(model)["/Intuition/applying"]["bot.ability.intuition.selection.rejected"]
    assert len(choice_transitions(model, "/Intuition/routing_rejection")) == 3
    assert cognition.Intuition.input_event.name == "bot.ability.intuition.input"
    assert cognition.Intuition.output_event.name == "bot.ability.intuition.output"


def test_intuition_is_offered_its_ceiling_and_the_handoff_and_told_what_reasoning_holds() -> None:
    """Its default ceiling is M; the handoff stays offered, and ``<held>`` names what is above M."""

    _, frame, _ = _intuit(_Answering())

    host = frame.input
    assert isinstance(host, cognition.reasoning.ProcessorInput)
    offered = {schema.name for schema in frame.schemas}
    assert _SayEvent.name in offered
    assert cognition.reasoning.InputEvent.name in offered
    assert _ThinkEvent.name not in offered
    instructions = frame.instructions or ""
    assert f'<held reached_by="{cognition.reasoning.InputEvent.name}">' in instructions
    assert f'<capacity event="{_ThinkEvent.name}"' in instructions


def test_intuition_that_selects_without_the_handoff_settles_the_turn() -> None:
    say = processing.SelectedEvent(event=_SayEvent.name, target="mouth", data={"text": "Hello."})

    completion, _, mouth = _intuit(_Answering((say,)))

    assert mouth.said == ["Hello."]
    assert completion.output is not None
    assert [item.event for item in completion.output] == [_SayEvent.name]
    assert completion.turn.acted is None


def test_selecting_the_handoff_runs_the_other_selections_first_then_hands_the_turn_on() -> None:
    """Words before a long think are the bot's own choice; what it said rides the turn's ``acted`` on."""

    say = processing.SelectedEvent(event=_SayEvent.name, target="mouth", data={"text": "Let me think."})
    handoff = processing.SelectedEvent(event=cognition.reasoning.InputEvent.name, target="cognition", data={})

    completion, _, mouth = _intuit(_Answering((say, handoff)))

    assert mouth.said == ["Let me think."]
    assert completion.output is None
    acted = completion.turn.acted
    assert acted is not None
    assert acted.percepts == (0,)
    assert [(item.event, item.data) for item in acted.acts] == [(_SayEvent.name, {"text": "Let me think."})]


def test_a_handoff_with_no_actor_to_dispatch_to_hands_on_having_done_nothing() -> None:
    """Only what was dispatched rides the turn's ``acted``: with no actors on the frame, nothing was."""

    say = processing.SelectedEvent(event=_SayEvent.name, target="mouth", data={"text": "Let me think."})
    handoff = processing.SelectedEvent(event=cognition.reasoning.InputEvent.name, target="cognition", data={})

    async def run() -> cognition.types.CompletionData:
        ctx = shared_hsm_context()
        stimulus = SoundEvent.with_data(SoundData(audio=b"hello", kind="speech"))
        turn_input = cognition.InputData(stimulus=stimulus, actors={})
        intuition = cognition.Intuition(processor=_Answering((say, handoff)))
        await start_abilities_for_test(ctx, intuition)
        output = await dispatch_ability_for_test(
            intuition,
            ctx,
            cognition.intuition.InputData(
                turn=cognition.types.TurnData(input=turn_input, operation_id="bare-turn", generation="token"),
                processing_input=cognition.input.build_processing_input(turn_input),
            ),
            timeout=1.0,
        )
        return typing.cast(cognition.types.CompletionData, typing.cast(object, output))

    completion = asyncio.run(run())

    assert completion.output is None
    assert completion.turn.acted is None


def test_intuition_reports_its_steps_under_its_own_span(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)

    _ = _intuit(_Answering())

    stages = {
        str((recorded.attributes or {}).get("bot.stage"))
        for recorded in exporter.get_finished_spans()
        if recorded.name == "bot.intuition.stage"
    }
    assert stages == {"intuition_initialize", "intuition_reason"}
    (reasoned,) = [
        recorded
        for recorded in exporter.get_finished_spans()
        if (recorded.attributes or {}).get("bot.stage") == "intuition_reason"
    ]
    assert (reasoned.attributes or {}).get("bot.intuition.held.count") == 1
    assert (reasoned.attributes or {}).get("bot.cognition.escalated") is False

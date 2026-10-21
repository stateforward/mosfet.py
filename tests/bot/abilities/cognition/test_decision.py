from mosfet.abilities.cognition import decision
from mosfet.abilities.cognition import types
from mosfet.abilities.cognition import input as cognition_input
from mosfet.abilities import ability as ability_module
from mosfet.abilities import processing
from mosfet.abilities import sandbox
import mosfet
from mosfet import event

import asyncio
import collections.abc
import datetime
import inspect
import typing

import hsm
import pydantic
import pytest

from tests.bot.abilities.cognition.metadata_contract import assert_metadata_is_not_coordination
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context
from tests.hsm_model import choice_transitions, transition_map
from tests.hsm_instance_state import start_ability_tree


def _accept_selection(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
    del ctx, instance, event


class _SelectionData(pydantic.BaseModel):
    value: str


_SelectionEvent = hsm.Event[_SelectionData](
    name="tests.decision.selection",
    kind=event.EventKind,
    schema=_SelectionData,
)


class _SelectionActor(hsm.Instance):
    """Minimal recipient used to exercise the real typed dispatch boundary."""

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "DecisionSelectionActor",
        hsm.initial(hsm.target("/DecisionSelectionActor/active")),
        hsm.state(
            "active",
            hsm.transition(hsm.on(_SelectionEvent), hsm.effect(_accept_selection)),
            hsm.transition(hsm.on(mosfet.FocusDeviceEvent), hsm.effect(_accept_selection)),
        ),
    )


class _SequenceProcessor(processing.DecisionProcessor):
    calls: list[processing.InputData]
    outputs: tuple[object, ...]

    def __init__(self, *outputs: object) -> None:
        self.calls = []
        self.outputs = outputs

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        return typing.cast(processing.Events, self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)])


class _CyclingProcessor(processing.DecisionProcessor):
    calls: list[processing.InputData]
    outputs: tuple[object, ...]

    def __init__(self, *outputs: object) -> None:
        self.calls = []
        self.outputs = outputs

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        return typing.cast(processing.Events, self.outputs[(len(self.calls) - 1) % len(self.outputs)])


def _selection_input(actor: hsm.Instance, *, operation_id: str) -> decision.InputData:
    stimulus = mosfet.InputEventData(target_device="phone", priority=0)
    actors: collections.abc.Mapping[str, hsm.Instance] = {"target": actor}
    turn = types.TurnData(
        input=cognition_input.InputData(
            stimulus=stimulus,
            actors=actors,
            focus=None,
            focus_candidates=(),
        ),
        operation_id=operation_id,
        generation="selection-generation",
    )
    return decision.InputData(
        turn=turn,
        processing_input=processing.InputData(
            input=stimulus,
            schemas=(_SelectionEvent,),
            actors=actors,
        ),
    )


def _selection(*, value: object) -> processing.Events:
    return (
        processing.SelectedEvent(
            event=_SelectionEvent.name,
            target="target",
            data={"value": value},
            confidence=99,
        ),
    )


def _focus_selection(*, device: object) -> processing.Events:
    return (
        processing.SelectedEvent(
            event=mosfet.FocusDeviceEvent.name,
            data={"device": device},
            confidence=99,
        ),
    )


def _run_decision(
    processor: processing.DecisionProcessor,
    *,
    operation_id: str,
) -> tuple[types.CompletionData, processing.DecisionProcessor]:
    async def run() -> types.CompletionData:
        ctx = shared_hsm_context()
        actor = _SelectionActor()
        assert actor.model is not None
        _ = await mosfet.started(ctx, actor, actor.model)
        ability = decision.Decision(processor=processor)
        return typing.cast(
            types.CompletionData,
            typing.cast(
                object,
                await dispatch_ability_for_test(
                    ability,
                    ctx,
                    _selection_input(actor, operation_id=operation_id),
                ),
            ),
        )

    return asyncio.run(run()), processor


class _Chatting(processing.Processor):
    """A chat model's processor: it writes and calls tools, so it is no decision processor."""

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input, drafts
        return ()


def test_decision_decides_only_on_a_decision_processor() -> None:
    # A caller the type checker cannot see: what is given is checked, not promised.
    chatting = typing.cast(processing.DecisionProcessor, typing.cast(object, _Chatting()))

    with pytest.raises(TypeError, match="processing.DecisionProcessor"):
        _ = decision.Decision(processor=chatting)

    assert not hasattr(decision, "DecisionProcessor")


def test_decision_never_uses_metadata_for_coordination() -> None:
    assert_metadata_is_not_coordination(decision)


def test_decision_returns_directed_terminal_to_one_shot_operation() -> None:
    async def run() -> hsm.Event[typing.Any]:
        ctx = shared_hsm_context()
        actor = _SelectionActor()
        assert actor.model is not None
        _ = await mosfet.started(ctx, actor, actor.model)
        child = decision.Decision(processor=_SequenceProcessor(()))
        await start_ability_tree(ctx, child)
        operation_id = "decision-terminal-operation"
        return await ability_module.run_terminal_operation(
            ctx,
            child=child,
            request=child.input_event.with_data_and_id(
                _selection_input(actor, operation_id=operation_id),
                operation_id,
            ),
            terminals=(child.output_event, child.failed_event),
            timeout=datetime.timedelta(seconds=1),
        )

    terminal = asyncio.run(run())

    assert terminal.name == decision.OutputEvent.name
    assert terminal.id == "decision-terminal-operation"
    assert terminal.source
    assert terminal.target


def test_decision_classification_updates_tuner_only_in_rtc() -> None:
    """The asynchronous classifier must not read or mutate the owned tuner."""

    source = inspect.getsource(getattr(decision.Decision, "_classify_activity"))
    assert "_confidence_tuner" not in source


def test_decision_models_processing_as_explicit_typed_phases() -> None:
    """Invocation, classification, dispatch, rejection routing, and publication are graph-visible."""

    model = typing.cast(hsm.Model, decision.Decision.submodel)
    transitions = transition_map(model)
    for path in (
        "/Decision/workflow/invoking",
        "/Decision/workflow/classifying",
        "/Decision/workflow/dispatching",
        "/Decision/workflow/routing_rejection",
        "/Decision/workflow/publishing",
    ):
        assert path in model.members
    assert transitions["/Decision/workflow/invoking"]["bot.ability.decision.invocation.completed"]
    assert transitions["/Decision/workflow/classifying"]["bot.ability.decision.dispatch.planned"]
    assert transitions["/Decision/workflow/dispatching"]["bot.ability.decision.selection.rejected"]
    # A processor's own refusal of a selection reaches the same repair.
    assert transitions["/Decision/workflow/invoking"]["bot.ability.decision.selection.rejected"]
    rejection_routes = choice_transitions(model, "/Decision/workflow/routing_rejection")
    assert len(rejection_routes) == 3
    assert rejection_routes[-1].guard is None


def test_decision_retries_selection_rejection_with_pydantic_feedback() -> None:
    processor = _SequenceProcessor(_selection(value=0), _selection(value="phone"))

    output, _ = _run_decision(processor, operation_id="repair-selection")

    assert len(processor.calls) == 2
    feedback = processor.calls[1].instructions or ""
    assert "The previous event selection was rejected during dispatch." in feedback
    assert "Input should be a valid string" in feedback
    assert "input_value" not in feedback
    assert output.output == (
        types.EventData(
            event=_SelectionEvent.name,
            target="target",
            data={"value": "phone"},
        ),
    )


def test_decision_retries_malformed_focus_selection_with_full_pydantic_feedback() -> None:
    processor = _SequenceProcessor(_focus_selection(device=0), _selection(value="phone"))

    output, _ = _run_decision(processor, operation_id="repair-focus-selection")

    assert len(processor.calls) == 2
    feedback = processor.calls[1].instructions or ""
    assert "not an instruction" in feedback
    assert "BEGIN UNTRUSTED VALIDATOR DIAGNOSTIC" in feedback
    assert "Input should be a valid string" in feedback
    assert "input_value" not in feedback
    assert "END UNTRUSTED VALIDATOR DIAGNOSTIC" in feedback
    assert output.output == (
        types.EventData(
            event=_SelectionEvent.name,
            target="target",
            data={"value": "phone"},
        ),
    )


class _HeldThenOfferedProcessor(processing.DecisionProcessor):
    """First selects an event the turn does not offer, the way the live model picked a held sandbox write.

    Selections are parsed against the offered menu the way a provider's dispatch tool parses them.
    """

    calls: list[processing.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        if len(self.calls) == 1:
            return processing.events_from_dispatch_args(
                {"events": [{"event": sandbox.WriteEvent.name, "data": {"path": "fib.py", "content": ""}}]},
                offered=input.schemas,
            )
        return _selection(value="phone")


def test_decision_asks_again_when_the_processor_selects_an_event_the_turn_does_not_offer() -> None:
    """Live check round 1 (D2): selecting a held event failed the whole turn, with no repair and nothing shown.

    The provider's "unavailable event" error surfaced as apply.failed, so decision's bounded selection repair
    never ran. It is now a selection rejection: decision asks again with the diagnostic.
    """

    processor = _HeldThenOfferedProcessor()

    output, _ = _run_decision(processor, operation_id="unoffered-selection")

    assert len(processor.calls) == 2
    feedback = processor.calls[1].instructions or ""
    assert f"Processing selected unavailable event: {sandbox.WriteEvent.name}." in feedback
    assert output.output == (types.EventData(event=_SelectionEvent.name, target="target", data={"value": "phone"}),)


def test_decision_stops_on_same_normalized_selection_rejection() -> None:
    processor = _SequenceProcessor(_selection(value=0), _selection(value=1), _selection(value="phone"))

    with pytest.raises(RuntimeError, match="selection") as failure:
        _ = _run_decision(processor, operation_id="repeated-selection-rejection")

    assert len(processor.calls) == 2
    assert "input_value" not in str(failure.value)


def test_decision_retries_again_after_different_selection_rejection() -> None:
    processor = _SequenceProcessor(
        _selection(value=0),
        processing.SelectedEvent(
            event=_SelectionEvent.name,
            target="target",
            data={},
            confidence=99,
        ),
        _selection(value="phone"),
    )

    output, _ = _run_decision(processor, operation_id="different-selection-rejection")

    assert len(processor.calls) == 3
    first_feedback = processor.calls[1].instructions or ""
    second_feedback = processor.calls[2].instructions or ""
    assert "input_value" not in first_feedback
    assert "Field required" in second_feedback
    assert output.output is not None


def test_decision_bounds_alternating_selection_rejections() -> None:
    processor = _CyclingProcessor(
        _selection(value=0),
        processing.SelectedEvent(
            event=_SelectionEvent.name,
            target="target",
            data={},
            confidence=99,
        ),
        _selection(value=1),
    )

    with pytest.raises(RuntimeError, match="selection"):
        _ = _run_decision(processor, operation_id="alternating-selection-rejection")

    assert len(processor.calls) == 3
    assert all("input_value" not in (call.instructions or "") for call in processor.calls)


def test_confidence_gate_drops_low_confidence_and_cascades() -> None:
    """A composition confidence gate drops below-floor per-event selections.

    All selections dropped: the turn is an explicit unhandled cascade (None output),
    never a handled-empty selection. High-confidence survivors still dispatch.
    """

    async def run() -> object:
        ctx = shared_hsm_context()
        actor = _SelectionActor()
        assert actor.model is not None
        _ = await mosfet.started(ctx, actor, actor.model)
        ability = decision.Decision(
            processor=_SequenceProcessor(_selection_gate(value="low", confidence=20)),
            confidence_gate=70,
        )
        return await dispatch_ability_for_test(
            ability,
            ctx,
            _selection_input(actor, operation_id="gate-cascade"),
        )

    completed = asyncio.run(run())
    assert isinstance(completed, types.CompletionData)
    assert completed.output is None


def test_confidence_gate_passes_high_confidence_survivors() -> None:
    async def run() -> object:
        ctx = shared_hsm_context()
        actor = _SelectionActor()
        _ = await mosfet.started(ctx, actor, actor.model)
        ability = decision.Decision(
            processor=_SequenceProcessor(_selection_gate(value="solid", confidence=99)),
            confidence_gate=95,
        )
        return await dispatch_ability_for_test(
            ability,
            ctx,
            _selection_input(actor, operation_id="gate-pass"),
        )

    completed = asyncio.run(run())
    assert isinstance(completed, types.CompletionData)
    assert completed.output == (
        types.EventData(
            event=_SelectionEvent.name,
            target="target",
            data={"value": "solid"},
            reason=None,
        ),
    )


def _selection_gate(*, value: object, confidence: int) -> processing.Events:
    return (
        processing.SelectedEvent(
            event=_SelectionEvent.name,
            target="target",
            data={"value": value},
            confidence=confidence,
        ),
    )


class _DeliberateActor(hsm.Instance):
    """An M-effort actor: above decision's default ceiling, so decision must never offer it."""

    effort: typing.ClassVar[ability_module.Effort] = ability_module.Effort.M
    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "DecisionDeliberateActor",
        hsm.initial(hsm.target("/DecisionDeliberateActor/active")),
        hsm.state("active", hsm.transition(hsm.on(_SelectionEvent), hsm.effect(_accept_selection))),
    )


def test_decision_offers_only_what_its_own_effort_ceiling_admits() -> None:
    """The host sends the whole frame; decision applies its own ceiling to what it offers and dispatches."""

    async def run() -> processing.InputData:
        ctx = shared_hsm_context()
        deliberate = _DeliberateActor()
        assert deliberate.model is not None
        _ = await mosfet.started(ctx, deliberate, deliberate.model)
        stimulus = mosfet.InputEventData(target_device="phone", priority=0)
        turn_input = cognition_input.InputData(stimulus=stimulus, actors={"deliberate": deliberate})
        processor = _SequenceProcessor(())
        _ = await dispatch_ability_for_test(
            decision.Decision(processor=processor, max_effort=ability_module.Effort.S),
            ctx,
            decision.InputData(
                turn=types.TurnData(input=turn_input, operation_id="ceiling-turn", generation="ceiling-generation"),
                processing_input=cognition_input.build_processing_input(turn_input),
            ),
        )
        (frame,) = processor.calls
        return frame

    frame = asyncio.run(run())

    assert "deliberate" not in frame.actors
    assert _SelectionEvent.name not in {schema.name for schema in frame.schemas}
    assert _SelectionEvent.name not in frame.actor_events


class _BenchActor(hsm.Instance):
    """An M-effort actor enabling the real sandbox run command, above the decision layer's ceiling."""

    effort: typing.ClassVar[ability_module.Effort] = ability_module.Effort.M
    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "DecisionBenchActor",
        hsm.initial(hsm.target("/DecisionBenchActor/active")),
        hsm.state("active", hsm.transition(hsm.on(sandbox.RunEvent), hsm.effect(_accept_selection))),
    )


def test_decision_is_told_what_a_later_tier_holds() -> None:
    """Live check P1: the bench (sandbox.run, effort M) is above the decision layer's ceiling (S).

    The decision layer never offered it and nothing told the model it existed, so a confident model made
    up test output instead of passing the turn on. Every decision turn now names each capacity held above
    its ceiling in a ``<held>`` block, with what it does, so passing the turn on is the model's informed
    choice; the capacity itself stays off the decision layer's dispatch tool.
    """

    async def run() -> processing.InputData:
        ctx = shared_hsm_context()
        bench = _BenchActor()
        nearby = _SelectionActor()
        assert bench.model is not None and nearby.model is not None
        _ = await mosfet.started(ctx, bench, bench.model)
        _ = await mosfet.started(ctx, nearby, nearby.model)
        stimulus = mosfet.InputEventData(target_device="desk", priority=0)
        turn_input = cognition_input.InputData(stimulus=stimulus, actors={"sandbox": bench, "nearby": nearby})
        processor = _SequenceProcessor(())
        _ = await dispatch_ability_for_test(
            decision.Decision(processor=processor, max_effort=ability_module.Effort.S),
            ctx,
            decision.InputData(
                turn=types.TurnData(input=turn_input, operation_id="held-turn", generation="held-generation"),
                processing_input=cognition_input.build_processing_input(turn_input),
            ),
        )
        (frame,) = processor.calls
        return frame

    frame = asyncio.run(run())

    offered = {schema.name for schema in frame.schemas}
    assert sandbox.RunEvent.name not in offered
    assert _SelectionEvent.name in offered
    instructions = frame.instructions or ""
    assert sandbox.RunEvent.name in instructions
    assert "Run one command in the sandbox and wait for it to finish." in instructions
    # What decision itself offers is on its tool, not repeated as something a later tier holds.
    assert _SelectionEvent.name not in instructions
    assert '<held reached_by="passing this turn on to a later tier of thinking">' in instructions
    assert "<reasoning_only" not in instructions

from bot.abilities.cognition import intuition
from bot.abilities.cognition import types
from bot.abilities.cognition import input as cognition_input
from bot.abilities import ability as ability_module
from bot.abilities import processing
import bot
from bot import event

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
    name="tests.intuition.selection",
    kind=event.EventKind,
    schema=_SelectionData,
)


class _SelectionActor(hsm.Instance):
    """Minimal recipient used to exercise the real typed dispatch boundary."""

    model: typing.ClassVar[hsm.Model | None] = bot.define(
        "IntuitionSelectionActor",
        hsm.initial(hsm.target("/IntuitionSelectionActor/active")),
        hsm.state(
            "active",
            hsm.transition(hsm.on(_SelectionEvent), hsm.effect(_accept_selection)),
            hsm.transition(hsm.on(bot.FocusDeviceEvent), hsm.effect(_accept_selection)),
        ),
    )


class _SequenceProcessor(processing.Processor):
    calls: list[processing.InputData]
    outputs: tuple[object, ...]

    def __init__(self, *outputs: object) -> None:
        self.calls = []
        self.outputs = outputs

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        return typing.cast(processing.Events, self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)])


class _CyclingProcessor(processing.Processor):
    calls: list[processing.InputData]
    outputs: tuple[object, ...]

    def __init__(self, *outputs: object) -> None:
        self.calls = []
        self.outputs = outputs

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        return typing.cast(processing.Events, self.outputs[(len(self.calls) - 1) % len(self.outputs)])


def _selection_input(actor: hsm.Instance, *, operation_id: str) -> intuition.InputData:
    stimulus = bot.InputEventData(target_device="phone", priority=0)
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
    return intuition.InputData(
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
            event=bot.FocusDeviceEvent.name,
            data={"device": device},
            confidence=99,
        ),
    )


def _run_intuition(
    processor: processing.Processor,
    *,
    operation_id: str,
) -> tuple[types.CompletionData, processing.Processor]:
    async def run() -> types.CompletionData:
        ctx = shared_hsm_context()
        actor = _SelectionActor()
        assert actor.model is not None
        _ = await bot.started(ctx, actor, actor.model)
        ability = intuition.Intuition(processor=processor)
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


def test_intuition_never_uses_metadata_for_coordination() -> None:
    assert_metadata_is_not_coordination(intuition)


def test_intuition_returns_directed_terminal_to_one_shot_operation() -> None:
    async def run() -> hsm.Event[typing.Any]:
        ctx = shared_hsm_context()
        actor = _SelectionActor()
        assert actor.model is not None
        _ = await bot.started(ctx, actor, actor.model)
        child = intuition.Intuition(processor=_SequenceProcessor(()))
        await start_ability_tree(ctx, child)
        operation_id = "intuition-terminal-operation"
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

    assert terminal.name == intuition.OutputEvent.name
    assert terminal.id == "intuition-terminal-operation"
    assert terminal.source
    assert terminal.target


def test_intuition_classification_updates_tuner_only_in_rtc() -> None:
    """The asynchronous classifier must not read or mutate the owned tuner."""

    source = inspect.getsource(getattr(intuition.Intuition, "_classify_activity"))
    assert "_confidence_tuner" not in source


def test_intuition_models_processing_as_explicit_typed_phases() -> None:
    """Invocation, classification, dispatch, rejection routing, and publication are graph-visible."""

    model = typing.cast(hsm.Model, intuition.Intuition.submodel)
    transitions = transition_map(model)
    for path in (
        "/Intuition/workflow/invoking",
        "/Intuition/workflow/classifying",
        "/Intuition/workflow/dispatching",
        "/Intuition/workflow/routing_rejection",
        "/Intuition/workflow/publishing",
    ):
        assert path in model.members
    assert transitions["/Intuition/workflow/invoking"]["bot.ability.intuition.invocation.completed"]
    assert transitions["/Intuition/workflow/classifying"]["bot.ability.intuition.dispatch.planned"]
    assert transitions["/Intuition/workflow/dispatching"]["bot.ability.intuition.selection.rejected"]
    rejection_routes = choice_transitions(model, "/Intuition/workflow/routing_rejection")
    assert len(rejection_routes) == 3
    assert rejection_routes[-1].guard is None


def test_intuition_retries_selection_rejection_with_pydantic_feedback() -> None:
    processor = _SequenceProcessor(_selection(value=0), _selection(value="phone"))

    output, _ = _run_intuition(processor, operation_id="repair-selection")

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


def test_intuition_retries_malformed_focus_selection_with_full_pydantic_feedback() -> None:
    processor = _SequenceProcessor(_focus_selection(device=0), _selection(value="phone"))

    output, _ = _run_intuition(processor, operation_id="repair-focus-selection")

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


def test_intuition_stops_on_same_normalized_selection_rejection() -> None:
    processor = _SequenceProcessor(_selection(value=0), _selection(value=1), _selection(value="phone"))

    with pytest.raises(RuntimeError, match="selection") as failure:
        _ = _run_intuition(processor, operation_id="repeated-selection-rejection")

    assert len(processor.calls) == 2
    assert "input_value" not in str(failure.value)


def test_intuition_retries_again_after_different_selection_rejection() -> None:
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

    output, _ = _run_intuition(processor, operation_id="different-selection-rejection")

    assert len(processor.calls) == 3
    first_feedback = processor.calls[1].instructions or ""
    second_feedback = processor.calls[2].instructions or ""
    assert "input_value" not in first_feedback
    assert "Field required" in second_feedback
    assert output.output is not None


def test_intuition_bounds_alternating_selection_rejections() -> None:
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
        _ = _run_intuition(processor, operation_id="alternating-selection-rejection")

    assert len(processor.calls) == 3
    assert all("input_value" not in (call.instructions or "") for call in processor.calls)

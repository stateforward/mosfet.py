from mosfet.abilities import cognition
from mosfet.abilities import ability
from mosfet.abilities import memory
from mosfet.abilities import processing
from mosfet.abilities.cognition import reasoning as reasoning_module
from mosfet.environment import SoundData, SoundEvent

import asyncio
import dataclasses
import datetime
import inspect
import typing

import hsm
import mosfet
import pydantic
from mosfet import event

from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test
from tests.bot.abilities.cognition.metadata_contract import assert_metadata_is_not_coordination


def test_reasoning_never_uses_metadata_for_coordination() -> None:
    assert_metadata_is_not_coordination(reasoning_module)


def test_reasoning_returns_directed_terminal_to_one_shot_operation() -> None:
    async def run() -> hsm.Event[typing.Any]:
        child = cognition.Reasoning(processor=EmptyProcessor())
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, child)
        operation_id = "reasoning-terminal-operation"
        request = reasoning_input(
            SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing")),
            operation_id=operation_id,
        )
        return await ability.run_terminal_operation(
            ctx,
            child=child,
            request=child.input_event.with_data_and_id(request, operation_id),
            terminals=(child.output_event, child.failed_event),
            timeout=datetime.timedelta(seconds=1),
        )

    terminal = asyncio.run(run())

    assert terminal.name == cognition.reasoning.OutputEvent.name
    assert terminal.id == "reasoning-terminal-operation"
    assert terminal.source
    assert terminal.target


class EmptyProcessor(processing.Processor):
    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        return ()


class HeldProcessor(processing.Processor):
    def __init__(self) -> None:
        self.called = asyncio.Event()
        self.release = asyncio.Event()

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        self.called.set()
        await self.release.wait()
        return ()


def reasoning_input(
    stimulus: hsm.Event[typing.Any],
    *,
    operation_id: str = "reasoning-turn",
) -> cognition.reasoning.InputData:
    cognition_input = cognition.InputData(
        stimulus=stimulus,
        abilities=(),
        actors={},
        focus=None,
        focus_candidates=(),
    )
    return cognition.reasoning.InputData(
        turn=cognition.types.TurnData(
            input=cognition_input,
            operation_id=operation_id,
            generation="operation-token",
        ),
        processing_input=processing.InputData(input=stimulus),
    )


def test_reasoning_applies_from_idle_and_takes_no_memory() -> None:
    """Reasoning neither recalls nor retains: initializing → idle → applying → idle, no memory input."""

    async def run() -> tuple[processing.CompletionData, str]:
        reasoning = cognition.Reasoning(processor=EmptyProcessor())
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, reasoning)

        stimulus = SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing"))
        output = await dispatch_ability_for_test(reasoning, ctx, reasoning_input(stimulus), timeout=0.2)
        return output, reasoning.state()

    output, state = asyncio.run(run())

    assert output.output == ()
    assert state == "/ReasoningLifecycle/attached/behavior/idle"
    assert "memory" not in inspect.signature(cognition.Reasoning).parameters
    assert set(reasoning_module.ProcessorInput.model_fields) == {"host_input"}
    members = reasoning_module.Reasoning.model.members if reasoning_module.Reasoning.model is not None else {}
    behavior = "/ReasoningLifecycle/attached/behavior"
    assert f"{behavior}/applying" in members
    assert f"{behavior}/recalling" not in members
    assert f"{behavior}/retaining" not in members


def test_reasoning_ignores_forged_public_terminals_for_unknown_operations() -> None:
    """Terminals stamped with an unknown operation id never disturb the live turn.

    Behavioral negative through public seams only (no stage-private events or
    capabilities): while one reasoning turn is held inside the processor, forged
    output/failure terminals for a stale operation are ignored — the live turn
    still completes normally once the processor releases it.
    """

    async def run() -> tuple[processing.CompletionData, str]:
        processor = HeldProcessor()
        reasoning = cognition.Reasoning(processor=processor)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, reasoning)

        result = asyncio.create_task(
            dispatch_ability_for_test(
                reasoning,
                ctx,
                reasoning_input(
                    SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing")),
                    operation_id="live-operation",
                ),
                timeout=0.3,
            )
        )
        await asyncio.wait_for(processor.called.wait(), timeout=0.2)
        forged_turn = reasoning_input(
            SoundEvent.with_data(SoundData(audio=b"forged", kind="phone.ringing")),
            operation_id="stale-operation",
        ).turn
        forged_output = dataclasses.replace(
            reasoning.output_event.with_data_and_id(
                cognition.types.CompletionData(turn=forged_turn, output=()),
                "stale-operation",
            ),
            source="forged-actor",
            target=hsm.id(reasoning),
        )
        _ = await hsm.dispatch(ctx, reasoning, forged_output)
        forged_failure = dataclasses.replace(
            reasoning.failed_event.with_data_and_id(
                cognition.types.FailureData(turn=forged_turn, message="forged stage failure"),
                "stale-operation",
            ),
            source="forged-actor",
            target=hsm.id(reasoning),
        )
        _ = await hsm.dispatch(ctx, reasoning, forged_failure)
        await asyncio.sleep(0)
        assert reasoning.state().endswith("/applying")
        assert not result.done()

        processor.release.set()
        return await result, reasoning.state()

    output, state = asyncio.run(run())

    assert output.output == ()
    assert state.endswith("/idle")


class CapturingProcessor(processing.Processor):
    inputs: list[processing.InputData]

    def __init__(self) -> None:
        self.inputs = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.inputs.append(input)
        return ()


def test_reasoning_passes_what_memory_surfaced_to_its_processor() -> None:
    """The host frame's recalled associations reach the processor input unchanged."""

    surfaced = memory.AssociationsData(
        items=(
            memory.Association(
                ref="5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f",
                kind=memory.NodeKind.EPISODE,
                text="Ada asked to be called back at noon.",
                at=datetime.datetime(2026, 10, 14, 12, 0, tzinfo=datetime.UTC),
                lanes=(memory.Lane.RECENT,),
            ),
        )
    )

    async def run() -> processing.InputData:
        processor = CapturingProcessor()
        reasoning = cognition.Reasoning(processor=processor)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, reasoning)

        stimulus = SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing"))
        request = reasoning_input(stimulus)
        request = request.model_copy(
            update={"processing_input": request.processing_input.model_copy(update={"recalled": surfaced})}
        )
        _ = await dispatch_ability_for_test(reasoning, ctx, request, timeout=0.2)
        assert len(processor.inputs) == 1
        return processor.inputs[0]

    processed = asyncio.run(run())

    assert processed.recalled == surfaced
    assert isinstance(processed.input, reasoning_module.ProcessorInput)


def _accept(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
    del ctx, instance, event


class _DeliberateData(pydantic.BaseModel):
    value: str


_DeliberateEvent = hsm.Event[_DeliberateData](
    name="tests.reasoning.deliberate", kind=event.EventKind, schema=_DeliberateData
)


class _DeliberateActor(hsm.Instance):
    """An XL-effort actor: above a reasoning ceiling of M."""

    effort: typing.ClassVar[ability.Effort] = ability.Effort.XL
    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "ReasoningDeliberateActor",
        hsm.initial(hsm.target("/ReasoningDeliberateActor/active")),
        hsm.state("active", hsm.transition(hsm.on(_DeliberateEvent), hsm.effect(_accept))),
    )


class RecordingProcessor(processing.Processor):
    calls: list[processing.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        return ()


def test_reasoning_offers_only_what_its_own_effort_ceiling_admits() -> None:
    """The host sends the whole frame; reasoning applies its own ceiling to what it offers and dispatches."""

    async def run() -> processing.InputData:
        ctx = shared_hsm_context()
        deliberate = _DeliberateActor()
        assert deliberate.model is not None
        _ = await mosfet.started(ctx, deliberate, deliberate.model)
        stimulus = SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing"))
        turn_input = cognition.InputData(stimulus=stimulus, actors={"deliberate": deliberate})
        processor = RecordingProcessor()
        reasoning = cognition.Reasoning(processor=processor, max_effort=ability.Effort.M)
        await start_abilities_for_test(ctx, reasoning)
        _ = await dispatch_ability_for_test(
            reasoning,
            ctx,
            cognition.reasoning.InputData(
                turn=cognition.types.TurnData(input=turn_input, operation_id="ceiling-turn", generation="token"),
                processing_input=cognition.input.build_processing_input(turn_input),
            ),
            timeout=0.5,
        )
        (frame,) = processor.calls
        return frame

    frame = asyncio.run(run())

    host = frame.input
    assert isinstance(host, cognition.reasoning.ProcessorInput)
    for offered in (frame, host.host_input):
        assert "deliberate" not in offered.actors
        assert _DeliberateEvent.name not in {schema.name for schema in offered.schemas}

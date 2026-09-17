from mosfet.abilities import cognition
from mosfet.abilities import ability
from mosfet.abilities import memory
from mosfet.abilities import processing
from mosfet.abilities.cognition import reasoning as reasoning_module
from mosfet.environment import SoundData, SoundEvent

import asyncio
import dataclasses
import datetime
import typing

import hsm

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


def test_reasoning_retains_environment_event_stimulus_without_stranding() -> None:
    async def run() -> tuple[processing.CompletionData, tuple[cognition.episodes.CognitiveEpisode, ...], str]:
        store = memory.Memory()
        reasoning = cognition.Reasoning(processor=EmptyProcessor(), memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, reasoning)

        stimulus = SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing"))
        output = await dispatch_ability_for_test(
            reasoning,
            ctx,
            reasoning_input(stimulus),
            timeout=0.2,
        )
        retained = cognition.episodes.episodes_from_output(store.execute(cognition.episodes.episode_select_input()))
        return output, retained, reasoning.state()

    output, retained, state = asyncio.run(run())

    assert output.output == ()
    assert len(retained) == 1
    assert retained[0].stimulus_name == SoundEvent.name
    assert state == "/ReasoningLifecycle/attached/behavior/idle"


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


def _reasoning_frame(processor: CapturingProcessor) -> reasoning_module.ProcessorInput:
    assert len(processor.inputs) == 1
    frame = processor.inputs[0].input
    assert isinstance(frame, reasoning_module.ProcessorInput)
    return frame


def test_reasoning_recalls_standing_directives_alongside_prior_episodes() -> None:
    """One apply, one transaction, two statements: the turn never sees half a memory."""

    async def run() -> reasoning_module.ProcessorInput:
        store = memory.Memory()
        _ = store.execute(
            cognition.episodes.episode_insert_input(
                cognition.episodes.CognitiveEpisode(stimulus_name="bot.input", output=()),
                context_ref=None,
            )
        )
        _ = store.execute(
            cognition.directives.directive_insert_input(
                cognition.directives.Directive(text="Call Bob when you get a chance."),
                context_ref=None,
            )
        )
        processor = CapturingProcessor()
        reasoning = cognition.Reasoning(processor=processor, memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, reasoning)

        stimulus = SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing"))
        _ = await dispatch_ability_for_test(reasoning, ctx, reasoning_input(stimulus), timeout=0.2)
        return _reasoning_frame(processor)

    frame = asyncio.run(run())

    assert [directive.text for directive in frame.standing_directives] == ["Call Bob when you get a chance."]
    assert [episode.stimulus_name for episode in frame.prior_episodes] == ["bot.input"]


def test_reasoning_without_memory_recalls_no_standing_directives() -> None:
    async def run() -> reasoning_module.ProcessorInput:
        processor = CapturingProcessor()
        reasoning = cognition.Reasoning(processor=processor)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, reasoning)

        stimulus = SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing"))
        _ = await dispatch_ability_for_test(reasoning, ctx, reasoning_input(stimulus), timeout=0.2)
        return _reasoning_frame(processor)

    frame = asyncio.run(run())

    assert frame.standing_directives == ()
    assert frame.prior_episodes == ()


def test_reasoning_recall_decodes_both_statements_from_one_transaction() -> None:
    """The two SELECTs are positional inside a single committed transaction."""

    store = memory.Memory()
    _ = store.execute(
        cognition.episodes.episode_insert_input(
            cognition.episodes.CognitiveEpisode(stimulus_name="bot.input", output=()),
            context_ref=None,
        )
    )
    _ = store.execute(
        cognition.directives.directive_insert_input(
            cognition.directives.Directive(text="Water the plants."),
            context_ref=None,
        )
    )

    recalled = store.execute(
        memory.InputData(
            statements=(
                *cognition.episodes.episode_select_input().statements,
                *cognition.directives.directive_select_input().statements,
            )
        )
    )

    assert len(recalled.results) == 2
    assert [
        episode.stimulus_name for episode in cognition.episodes.episodes_from_output(recalled, statement_index=0)
    ] == ["bot.input"]
    assert [
        directive.text for directive in cognition.directives.directives_from_output(recalled, statement_index=1)
    ] == ["Water the plants."]
    # Each statement decodes only its own rows: episodes are not directives and never leak across.
    assert cognition.episodes.episodes_from_output(recalled, statement_index=1) == ()
    assert cognition.directives.directives_from_output(recalled, statement_index=0) == ()

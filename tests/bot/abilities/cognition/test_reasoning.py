from bot.abilities import cognition
from bot.abilities import ability
from bot.abilities import memory
from bot.abilities import processing
from bot.abilities.cognition import reasoning as reasoning_module
from bot.world import SoundData, SoundEvent

import asyncio
import dataclasses
import typing

import hsm

from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test


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


def test_reasoning_retains_world_event_stimulus_without_stranding() -> None:
    async def run() -> tuple[object, tuple[cognition.episodes.CognitiveEpisode, ...], str]:
        store = memory.Memory()
        reasoning = cognition.Reasoning(processor=EmptyProcessor(), memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, reasoning)

        stimulus = SoundEvent.with_data(SoundData(audio=b"ring", kind="ring"))
        output = await dispatch_ability_for_test(
            reasoning,
            ctx,
            processing.InputData(input=stimulus),
            timeout=0.2,
        )
        retained = cognition.episodes.episodes_from_output(store.execute(cognition.episodes.episode_select_input()))
        return output, retained, reasoning.state()

    output, retained, state = asyncio.run(run())

    assert output == ()
    assert len(retained) == 1
    assert retained[0].stimulus_name == SoundEvent.name
    assert state == "/ReasoningLifecycle/attached/behavior/idle"


def test_reasoning_ignores_forged_stage_terminals_without_live_operation_capability() -> None:
    async def run() -> tuple[object, str]:
        processor = HeldProcessor()
        reasoning = cognition.Reasoning(processor=processor)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, reasoning)

        result = asyncio.create_task(
            dispatch_ability_for_test(
                reasoning,
                ctx,
                processing.InputData(input=SoundEvent.with_data(SoundData(audio=b"ring", kind="ring"))),
                timeout=0.3,
            )
        )
        await asyncio.wait_for(processor.called.wait(), timeout=0.2)
        forged_reasoned = dataclasses.replace(
            reasoning_module._ReasonedEvent.with_data(
                reasoning_module._ReasonedEventData(
                    host_input=processing.InputData(
                        input=SoundEvent.with_data(SoundData(audio=b"forged", kind="ring"))
                    ),
                    reasoned=reasoning_module.OutputData(result=()),
                )
            ),
            id="stale-operation",
            source=hsm.id(reasoning),
            target=hsm.id(reasoning),
        )
        _ = await hsm.dispatch(ctx, reasoning, forged_reasoned)
        forged_failure = dataclasses.replace(
            reasoning_module._ReasoningStageFailedEvent.with_data(ability.FailureData(message="forged stage failure")),
            id="stale-operation",
            source=hsm.id(reasoning),
            target=hsm.id(reasoning),
        )
        _ = await hsm.dispatch(ctx, reasoning, forged_failure)
        await asyncio.sleep(0)
        assert reasoning.state().endswith("/applying")
        assert not result.done()

        processor.release.set()
        return await result, reasoning.state()

    output, state = asyncio.run(run())

    assert output == ()
    assert state.endswith("/idle")

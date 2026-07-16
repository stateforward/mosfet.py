from bot.abilities import cognition
from bot.abilities import memory
from bot.abilities import processing
from bot.abilities.cognition import autonomy as autonomy_module
from bot.protocols import attachment
from bot.habit import storage as habit_storage

import asyncio
import collections.abc
import dataclasses
import typing

import hsm
import pytest

from bot import habit
from bot.world import SoundData, SoundEvent
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test
from tests.bot.abilities.cognition.metadata_contract import assert_metadata_is_not_coordination


def test_autonomy_never_uses_metadata_for_coordination() -> None:
    assert_metadata_is_not_coordination(autonomy_module)


_DELAYED_HABIT_SOURCE = """
input_event = hsm.event(
    name = "bot.habit.delayed_ring.input",
    schema = {"type": "object", "additionalProperties": True},
    description = "Input for a delayed habit.",
)
output_event = hsm.event(
    name = "bot.habit.delayed_ring.output",
    schema = {
        "type": "object",
        "properties": {"event": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["event"],
        "additionalProperties": True,
    },
    description = "Delayed habit selection.",
)
triggers = ["world.sound"]

def emit(event):
    hsm.dispatch(output_event, {"event": "bot.clear_focus", "reason": "delayed habit"})

habit = hsm.define(
    "DelayedRing",
    hsm.initial(hsm.target("/DelayedRing/idle")),
    hsm.state(
        "idle",
        hsm.transition(hsm.on(input_event), hsm.target("/DelayedRing/waiting")),
    ),
    hsm.state(
        "waiting",
        hsm.transition(hsm.after(seconds=0.01), hsm.effect("emit")),
    ),
)
""".strip()

_WAITING_HABIT_SOURCE = _DELAYED_HABIT_SOURCE.replace("seconds=0.01", "seconds=10")


class _Owner(hsm.Instance):
    model: typing.ClassVar[hsm.Model]
    lifecycle: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle = []


def _record(ctx: hsm.Context, instance: _Owner, event: hsm.Event[typing.Any]) -> None:
    del ctx
    instance.lifecycle.append(event)


_Owner.model = hsm.define(
    "AutonomyTestOwner",
    hsm.initial(hsm.target("/AutonomyTestOwner/ready")),
    hsm.state(
        "ready",
        hsm.transition(hsm.on(processing.CancelledEvent), hsm.effect(_record)),
    ),
)


async def _wait_until(predicate: typing.Callable[[], bool], *, timeout: float = 0.3) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def _turn(operation_id: str = "autonomy-turn") -> cognition.types.TurnData:
    return cognition.types.TurnData(
        input=cognition.InputData(
            stimulus=SoundEvent.with_data(SoundData(audio=b"ring", kind="ring")),
            abilities=(),
            actors={},
            focus=None,
            focus_candidates=(),
        ),
        operation_id=operation_id,
        generation="operation-token",
    )


def test_autonomy_waits_for_asynchronous_habit_terminal() -> None:
    async def run() -> tuple[object, str, int]:
        store = memory.Memory()
        installed = habit.start(
            _DELAYED_HABIT_SOURCE,
            name="DelayedRing",
            triggers=(SoundEvent.name,),
        )
        _ = store.execute(
            memory.InputData(statements=memory.compile_statements(*habit_storage.insert_habit_clauses(installed)))
        )
        autonomy = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)

        output = await dispatch_ability_for_test(
            autonomy,
            ctx,
            _turn(),
            timeout=0.2,
        )
        instances = autonomy.context().value(hsm.Keys.Instances)
        candidate_count = (
            sum(isinstance(actor, autonomy_module._CandidateRun) for actor in instances.values())
            if isinstance(instances, collections.abc.Mapping)
            else 0
        )
        return output, autonomy.state(), candidate_count

    output, state, candidate_count = asyncio.run(run())

    assert output == cognition.types.CompletionData(
        turn=_turn(),
        output=(
            cognition.types.EventData(
                event="bot.clear_focus",
                reason="delayed habit",
            ),
        ),
    )
    assert state == "/AutonomyLifecycle/attached/behavior/idle"
    assert candidate_count == 0


def test_autonomy_acknowledges_cancel_only_after_candidate_detaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        store = memory.Memory()
        installed = habit.start(
            _WAITING_HABIT_SOURCE,
            name="DelayedRing",
            triggers=(SoundEvent.name,),
        )
        _ = store.execute(
            memory.InputData(statements=memory.compile_statements(*habit_storage.insert_habit_clauses(installed)))
        )
        autonomy = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        owner = _Owner()
        _ = await hsm.started(ctx, owner, owner.model)
        await autonomy.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await _wait_until(lambda: autonomy.state().endswith("/idle"))

        operation_id = "cancelled-autonomy-turn"
        _ = await hsm.dispatch(
            ctx,
            autonomy,
            autonomy.input_event.with_data_and_id(
                _turn(operation_id),
                operation_id,
            ),
        )
        await _wait_until(lambda: autonomy.state().endswith("/running"))

        detached = asyncio.Event()
        original_detach = habit.Behavior.detach

        def delayed_detach(
            behavior: habit.Behavior,
            detach_ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> asyncio.Task[object]:
            async def detach_after_release() -> object:
                await detached.wait()
                return await original_detach(behavior, detach_ctx, event)

            return asyncio.create_task(detach_after_release())

        monkeypatch.setattr(habit.Behavior, "detach", delayed_detach)
        cancel = dataclasses.replace(
            processing.CancelEvent.with_data(
                processing.CancelData(operation_id=operation_id, token="exact-cancel-token")
            ),
            id=f"{operation_id}:autonomy",
            source=hsm.id(owner),
            target=hsm.id(autonomy),
        )
        _ = await hsm.dispatch(ctx, autonomy, cancel)
        await _wait_until(lambda: autonomy.state().endswith("/cancelling"))
        assert owner.lifecycle == []

        detached.set()
        await _wait_until(lambda: bool(owner.lifecycle))
        return owner.lifecycle, autonomy.state()

    lifecycle, state = asyncio.run(run())

    assert len(lifecycle) == 1
    assert lifecycle[0].id == "cancelled-autonomy-turn:autonomy"
    assert lifecycle[0].data == processing.CancelledData(
        operation_id="cancelled-autonomy-turn",
        token="exact-cancel-token",
    )
    assert state.endswith("/idle")

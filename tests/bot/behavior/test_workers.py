"""Runtime tests for the pooled isolated callback workers."""

import asyncio
import dataclasses
import os
import signal
import typing

import bot
import hsm
import pytest

from bot.behavior import runtime
from bot.behavior.workers import (
    CallbackWorkers,
    SlotBudgets,
    SlotEvaluationRequestData,
    SlotEvaluationResultData,
)

from tests.bot.behavior.support import greeting_behavior_source

_STOP_EVENT_NAME = "bot.behavior.workers.stop_request"
_SLOT_RESULT_EVENT_NAME = "bot.behavior.workers.slot.result"


def _budgets(**overrides: typing.Any) -> SlotBudgets:
    return SlotBudgets(**overrides)


def _guard_request(*, dispatch_allowed: bool = False) -> SlotEvaluationRequestData:
    return SlotEvaluationRequestData(
        program=greeting_behavior_source(),
        callback="has_text",
        event={"name": "conversation.greeting.recognized", "data": {"text": "hello"}},
        behavior_id="answer_greeting",
        dispatch_allowed=dispatch_allowed,
    )


def _effect_request() -> SlotEvaluationRequestData:
    return SlotEvaluationRequestData(
        program=greeting_behavior_source(),
        callback="emit_uppercase",
        event={"name": "conversation.greeting.recognized", "data": {"text": "hello"}},
        behavior_id="answer_greeting",
        dispatch_allowed=True,
    )


# Deliberately slow guard: roughly half a second of Starlark loop work on CI-class
# hardware (measured ~0.28s for 4M iterations), comfortably over a 0.2s deadline
# while staying far under the outer settle budget.
_BURN_SOURCE = """
input_event = hsm.event(
    name = "bot.behavior.burn.input",
    schema = {"type": "object"},
    description = "Burn input.",
    examples = [{}],
)
output_event = hsm.event(
    name = "bot.behavior.burn.output",
    schema = {"type": "object"},
    description = "Burn output.",
    examples = [{}],
)
triggers = ["bot.behavior.burn.input"]
description = "Deliberately slow guard used to exercise evaluation deadlines."

def burn(event):
    total = 0
    for i in range(2000):
        for j in range(3000):
            total = total + j
    return total == -1

behavior = hsm.define(
    "Burn",
    hsm.initial(hsm.target("/Burn/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.guard("burn"),
            hsm.effect("emit_uppercase"),
        ),
    ),
)
""".strip()


def _slow_request() -> SlotEvaluationRequestData:
    return SlotEvaluationRequestData(
        program=_BURN_SOURCE,
        callback="burn",
        event={"name": "bot.behavior.burn.input", "data": {}},
        behavior_id="burn",
        dispatch_allowed=False,
    )


async def _start_pool(pool: CallbackWorkers) -> CallbackWorkers:
    started = await bot.started(None, pool, pool.model)
    for _ in range(200):
        if pool.snapshot().ready:
            return started
        await asyncio.sleep(0.01)
    raise AssertionError(f"pool never became ready, state={started.state()}")


async def _stop_pool(pool: CallbackWorkers) -> str:
    stop = hsm.Event[None](name=_STOP_EVENT_NAME)
    _ = await pool.dispatch(
        pool.context(),
        dataclasses.replace(stop, source=hsm.id(pool), target=hsm.id(pool)),
    )
    for _ in range(400):
        state = pool.state()
        if state.endswith("/stopped") or state.endswith("/failed"):
            return state
        await asyncio.sleep(0.01)
    raise AssertionError(f"pool never stopped, state={pool.state()}")


def test_evaluate_returns_guard_boolean() -> None:
    async def main() -> None:
        pool = await _start_pool(CallbackWorkers(size=1))
        try:
            result = await pool.evaluate(_guard_request())
            assert result.result is True
            assert result.dispatches == ()
        finally:
            assert (await _stop_pool(pool)).endswith("/stopped")

    asyncio.run(main())


def test_evaluate_returns_effect_dispatches_and_declared_events() -> None:
    async def main() -> None:
        pool = await _start_pool(CallbackWorkers(size=1))
        try:
            result = await pool.evaluate(_effect_request())
            outcome, dispatches, declared = result.result, result.dispatches, result.declared_events
            assert outcome is None
            assert len(dispatches) == 1
            assert dispatches[0].data == {"text": "HELLO"}
            declared_names = {contract.name for contract in declared}
            assert "bot.behavior.answer_greeting.output" in declared_names
        finally:
            assert (await _stop_pool(pool)).endswith("/stopped")

    asyncio.run(main())


def test_requests_beyond_capacity_queue_until_a_slot_frees() -> None:
    async def main() -> None:
        pool = await _start_pool(CallbackWorkers(size=2))
        try:
            outcomes = await asyncio.gather(
                pool.evaluate(_guard_request()),
                pool.evaluate(_effect_request()),
                pool.evaluate(_guard_request()),
            )
            assert outcomes[0].result is True
            assert outcomes[1].result is None
            assert outcomes[2].result is True
            assert pool.snapshot().free_indices == (0, 1)
        finally:
            assert (await _stop_pool(pool)).endswith("/stopped")

    asyncio.run(main())


def test_timeout_kills_worker_and_replacement_serves_next_request() -> None:
    async def main() -> None:
        budgets = _budgets(evaluation_seconds=0.2)
        pool = CallbackWorkers(size=1, budgets=budgets)
        started = await _start_pool(pool)
        first_pid = started.worker_pid(0)
        assert first_pid is not None
        with pytest.raises(runtime.CallbackError):
            await pool.evaluate(_slow_request())
        for _ in range(200):
            replacement_pid = started.worker_pid(0)
            if replacement_pid is not None and replacement_pid != first_pid:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("slot never replaced its killed worker")
        try:
            result = await pool.evaluate(_guard_request())
            assert result.result is True
        finally:
            assert (await _stop_pool(pool)).endswith("/stopped")

    asyncio.run(main())


def test_slot_retires_worker_after_evaluation_limit_and_keeps_serving() -> None:
    async def main() -> None:
        budgets = _budgets(max_evaluations=2)
        pool = CallbackWorkers(size=1, budgets=budgets)
        started = await _start_pool(pool)
        pid_before = started.worker_pid(0)
        for _ in range(3):
            result = await pool.evaluate(_guard_request())
            assert result.result is True
        pid_after = started.worker_pid(0)
        assert pid_after is not None
        # Retirement happens between evaluations; by the third request a fresh
        # worker must be serving even though every answer stayed correct.
        assert pid_after != pid_before, "worker was not retired across the evaluation limit"
        assert (await _stop_pool(pool)).endswith("/stopped")

    asyncio.run(main())


def test_externally_killed_worker_fails_typed_then_pool_self_heals() -> None:
    async def main() -> None:
        pool = await _start_pool(CallbackWorkers(size=1))
        pid = pool.worker_pid(0)
        assert pid is not None
        os.kill(pid, signal.SIGKILL)
        with pytest.raises(runtime.CallbackError):
            await pool.evaluate(_guard_request())
        result = await pool.evaluate(_guard_request())
        assert result.result is True
        assert (await _stop_pool(pool)).endswith("/stopped")

    asyncio.run(main())


def test_forged_result_never_settles_an_operation() -> None:
    async def main() -> None:
        pool = await _start_pool(CallbackWorkers(size=1))
        try:
            forged_event = hsm.Event[SlotEvaluationResultData](
                name=_SLOT_RESULT_EVENT_NAME,
                schema=SlotEvaluationResultData,
            )
            forged = dataclasses.replace(
                forged_event.with_data(
                    SlotEvaluationResultData(
                        operation_id="callbackworkers:999", ok=True, result_kind="bool", result_value=False
                    )
                ),
                source=hsm.id(pool),
                target=hsm.id(pool),
            )
            _ = await pool.dispatch(pool.context(), forged)
            assert pool.snapshot().stale_results >= 1
            result = await pool.evaluate(_guard_request())
            assert result.result is True
        finally:
            assert (await _stop_pool(pool)).endswith("/stopped")

    asyncio.run(main())


def test_stop_with_in_flight_evaluation_settles_typed_and_terminates_workers() -> None:
    async def main() -> None:
        budgets = _budgets(evaluation_seconds=0.2)
        pool = CallbackWorkers(size=1, budgets=budgets)
        started = await _start_pool(pool)

        async def in_flight() -> object:
            try:
                return (await pool.evaluate(_slow_request())).result
            except runtime.CallbackError:
                return "failed"

        task = asyncio.ensure_future(in_flight())
        await asyncio.sleep(0.05)
        final_state = await _stop_pool(pool)
        outcome = await asyncio.wait_for(task, 5.0)
        assert outcome == "failed"
        assert final_state.endswith("/stopped")
        for _ in range(200):
            pid = started.worker_pid(0)
            if pid is None or not _pid_alive(pid):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("worker process survived pool stop")

    asyncio.run(main())


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

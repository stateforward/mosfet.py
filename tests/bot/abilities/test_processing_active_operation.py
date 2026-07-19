"""active_operation liveness is map membership, not state() (HSM-CONTEXT-001)."""

from __future__ import annotations

import asyncio

import hsm

from bot.abilities import processing


def test_active_operation_uses_map_membership_not_state() -> None:
    async def run() -> None:
        owner = hsm.Instance()
        await hsm.started(
            hsm.Context(),
            owner,
            hsm.define("Owner", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        op = await processing.start_operation(owner, "turn-1")
        assert processing.active_operation(owner, "turn-1") is op
        assert processing.active_operation_id(owner) == "turn-1"
        # Retire via finish_operation — must not depend on probing op.state().
        processing.finish_operation(owner.context(), owner, "turn-1")
        assert processing.active_operation(owner, "turn-1") is None
        assert processing.active_operation_id(owner) is None

    asyncio.run(run())

"""PhoneService production stop then attach restarts HSM (hsm 1.3.2 stop semantics)."""

from __future__ import annotations

import asyncio

import hsm

from bot.providers.livekit.phone import PhoneService
from bot.world import World


def _is_started(instance: hsm.Instance) -> bool:
    try:
        _ = hsm.id(instance)
    except hsm.ErrorValidatingModel:
        return False
    return True


def test_phone_service_production_stop_unstarts_machine() -> None:
    """``hsm.stop(service)`` makes ``hsm.id`` fail without any process-side hold."""

    async def run() -> tuple[bool, bool]:
        world = World()
        service = PhoneService()
        target = hsm.Instance()
        await hsm.started(
            world.context,
            target,
            hsm.define("T", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await service.attach(world, target)
        after_attach = _is_started(service)
        await hsm.stop(service)
        after_stop = _is_started(service)
        return after_attach, after_stop

    after_attach, after_stop = asyncio.run(run())
    assert after_attach is True
    assert after_stop is False


def test_phone_service_attach_after_stop_restarts_machine() -> None:
    """Stop via production API, then re-attach must restart without RuntimeError."""

    async def run() -> None:
        world = World()
        service = PhoneService()
        target = hsm.Instance()
        await hsm.started(
            world.context,
            target,
            hsm.define("T", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await service.attach(world, target)
        assert _is_started(service) is True

        await hsm.stop(service)
        assert _is_started(service) is False

        await service.attach(world, target)
        assert _is_started(service) is True
        await service.detach(world, target)

    asyncio.run(run())

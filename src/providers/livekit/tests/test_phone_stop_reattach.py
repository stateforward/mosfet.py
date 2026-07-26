"""PhoneService production stop then attach restarts HSM (hsm 1.3.2 stop semantics)."""

from __future__ import annotations

import asyncio

import hsm

import bot.lifecycle
from bot.providers.livekit.phone import PhoneService
from bot.environment import Environment


def test_phone_service_production_stop_unstarts_machine() -> None:
    """``hsm.stop(service)`` makes ``hsm.id`` fail without any process-side hold."""

    async def run() -> tuple[bool, bool]:
        environment = Environment()
        service = PhoneService()
        target = hsm.Instance()
        await hsm.started(
            environment,
            target,
            hsm.define("T", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await service.attach(environment, target)
        after_attach = bot.lifecycle.is_started(service)
        await hsm.stop(service)
        after_stop = bot.lifecycle.is_started(service)
        return after_attach, after_stop

    after_attach, after_stop = asyncio.run(run())
    assert after_attach is True
    assert after_stop is False


def test_phone_service_attach_after_stop_restarts_machine() -> None:
    """Stop via production API, then re-attach must restart without RuntimeError."""

    async def run() -> None:
        environment = Environment()
        service = PhoneService()
        target = hsm.Instance()
        await hsm.started(
            environment,
            target,
            hsm.define("T", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await service.attach(environment, target)
        assert bot.lifecycle.is_started(service) is True

        await hsm.stop(service)
        assert bot.lifecycle.is_started(service) is False

        await service.attach(environment, target)
        assert bot.lifecycle.is_started(service) is True
        await service.detach(environment, target)

    asyncio.run(run())

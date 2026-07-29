"""Device.stop then detach is typed/safe; reattach works after restart (hsm 1.3.2)."""

from __future__ import annotations

import asyncio

import hsm

import bot.lifecycle

from bot.device.device import Device
from bot.protocols import attachment
from bot.environment import Environment


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


class _EmptyFirmware(hsm.Instance):
    model = hsm.define("EmptyFirmware", hsm.initial(hsm.target("ready")), hsm.state("ready"))


class _TestDevice(Device):
    firmware_model = _EmptyFirmware.model

    def _create_firmware_instance(self, ctx: hsm.Context, event: hsm.Event) -> hsm.Instance:
        del ctx, event
        return _EmptyFirmware()


async def _wait_until(condition, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise TimeoutError("condition not met")


def test_device_stop_then_detach_does_not_dispatch() -> None:
    async def run() -> bool:
        environment = Environment()
        device = _TestDevice()
        _ = await hsm.started(environment, device, require_model(device.model))
        assert bot.lifecycle.is_started(device) is True
        owner = hsm.Instance()
        _ = await hsm.started(
            environment,
            owner,
            hsm.define("O", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await device.attach(
            environment,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        await _wait_until(lambda: device.state() == "/Device/attached" or bot.lifecycle.is_started(device))
        await device.stop(environment)
        assert bot.lifecycle.is_started(device) is False
        # Must not raise RuntimeError("dispatch requires a started HSM").
        await device.detach(
            environment,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)),
        )
        return bot.lifecycle.is_started(device)

    assert asyncio.run(run()) is False


def test_device_reattach_after_stop_requires_restart() -> None:
    async def run() -> None:
        environment = Environment()
        device = _TestDevice()
        _ = await hsm.started(environment, device, require_model(device.model))
        owner = hsm.Instance()
        _ = await hsm.started(
            environment,
            owner,
            hsm.define("O", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await device.stop(environment)
        assert bot.lifecycle.is_started(device) is False
        restarted = await device.restart(environment)
        assert restarted is device
        assert bot.lifecycle.is_started(device) is True
        await device.attach(
            environment,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )

    asyncio.run(run())

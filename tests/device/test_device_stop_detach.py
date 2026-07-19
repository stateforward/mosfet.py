"""Device.stop then detach is typed/safe; reattach works after restart (hsm 1.3.2)."""

from __future__ import annotations

import asyncio

import hsm

from bot.device.device import Device
from bot.protocols import attachment
from bot.world import World


class _EmptyFirmware(hsm.Instance):
    model = hsm.define("EmptyFirmware", hsm.initial(hsm.target("ready")), hsm.state("ready"))


class _TestDevice(Device):
    firmware_model = _EmptyFirmware.model

    def _create_firmware_instance(self, ctx: hsm.Context, event: hsm.Event) -> hsm.Instance:
        del ctx, event
        return _EmptyFirmware()

    def _can_activate(self, ctx: hsm.Context, event: hsm.Event) -> bool:
        del ctx, event
        return True


def _is_started(instance: hsm.Instance) -> bool:
    try:
        _ = hsm.id(instance)
    except hsm.ErrorValidatingModel:
        return False
    return True


async def _wait_until(condition, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise TimeoutError("condition not met")


def test_device_stop_then_detach_does_not_dispatch() -> None:
    async def run() -> bool:
        world = World()
        device = _TestDevice()
        _ = await hsm.started(world.context, device, device.model)
        assert _is_started(device) is True
        owner = hsm.Instance()
        _ = await hsm.started(
            world.context,
            owner,
            hsm.define("O", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await device.attach(
            world.context,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        await _wait_until(lambda: device.state().endswith("/active") or _is_started(device))
        await device.stop(world.context)
        assert _is_started(device) is False
        # Must not raise RuntimeError("dispatch requires a started HSM").
        await device.detach(
            world.context,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)),
        )
        return _is_started(device)

    assert asyncio.run(run()) is False


def test_device_reattach_after_stop_requires_restart() -> None:
    async def run() -> None:
        world = World()
        device = _TestDevice()
        _ = await hsm.started(world.context, device, device.model)
        owner = hsm.Instance()
        _ = await hsm.started(
            world.context,
            owner,
            hsm.define("O", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await device.stop(world.context)
        assert _is_started(device) is False
        restarted = await device.restart(world.context)
        assert restarted is device
        assert _is_started(device) is True
        await device.attach(
            world.context,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )

    asyncio.run(run())

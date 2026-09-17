"""Device.stop then detach is typed/safe; reattach works after restart (hsm 1.3.2)."""

from __future__ import annotations

import asyncio

import hsm

import mosfet.lifecycle

from mosfet.device import Device
from mosfet.protocols import attachment
from mosfet.environment import Environment


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


class _EmptyFirmware(hsm.Instance):
    model = mosfet.define("EmptyFirmware", hsm.initial(hsm.target("ready")), hsm.state("ready"))


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
        _ = await mosfet.started(environment, device, require_model(device.model))
        assert mosfet.lifecycle.is_started(device) is True
        owner = hsm.Instance()
        _ = await mosfet.started(
            environment,
            owner,
            mosfet.define("O", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await device.attach(
            environment,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        await _wait_until(lambda: device.state() == "/Device/attached" or mosfet.lifecycle.is_started(device))
        await device.stop(environment)
        assert mosfet.lifecycle.is_started(device) is False
        # Must not raise RuntimeError("dispatch requires a started HSM").
        await device.detach(
            environment,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)),
        )
        return mosfet.lifecycle.is_started(device)

    assert asyncio.run(run()) is False


def test_device_reattach_after_stop_requires_restart() -> None:
    async def run() -> None:
        environment = Environment()
        device = _TestDevice()
        _ = await mosfet.started(environment, device, require_model(device.model))
        owner = hsm.Instance()
        _ = await mosfet.started(
            environment,
            owner,
            mosfet.define("O", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await device.stop(environment)
        assert mosfet.lifecycle.is_started(device) is False
        restarted = await device.restart(environment)
        assert restarted is device
        assert mosfet.lifecycle.is_started(device) is True
        await device.attach(
            environment,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )

    asyncio.run(run())

"""Tests for the phone Display peripheral (mosfet.devices.phone.display)."""

from __future__ import annotations

import asyncio
import collections.abc
import typing

import hsm
import mosfet

from mosfet.device import Device
from mosfet.devices.phone import display as display_module
from mosfet.environment import Environment
from mosfet.protocols import attachment


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() >= deadline:
            raise TimeoutError("Timed out waiting for condition.")
        await asyncio.sleep(0)


def _caller_id(display: display_module.Display) -> str | None:
    attributes = display.take_snapshot().Attributes or {}
    return typing.cast(str | None, attributes.get("/Device/caller_id", "unset"))


def test_display_is_a_passive_output_peripheral() -> None:
    display = display_module.Display()

    assert isinstance(display, Device)
    assert display_module.Display.observation_name == "display"
    assert display_module.Display.required_bot_abilities == ()


def test_caller_id_event_sets_caller_id_while_detached() -> None:
    """A detached display still shows what its controller puts on the wire.

    Unlike the mouthpiece, showing a caller ID is not gated on attachment: the display shows
    exactly what firmware dispatches, whether or not a controller is currently attached to it.
    """

    async def run() -> str | None:
        display = display_module.Display()
        _ = await mosfet.started(None, display, typing.cast(hsm.Model, display.model))
        await wait_until(lambda: display.state() == "/Device/detached")

        await display.dispatch(
            display.context(),
            display_module.CallerIdEvent.with_data(display_module.CallerIdData(caller_id="Front desk")),
        )
        await asyncio.sleep(0)

        return _caller_id(display)

    assert asyncio.run(run()) == "Front desk"


def test_caller_id_event_sets_caller_id_while_attached() -> None:
    """Attachment is an ownership fact for the display, never a precondition for showing a caller id."""

    async def run() -> str | None:
        environment = Environment()
        display = display_module.Display()
        controller = hsm.Instance()
        _ = await mosfet.started(environment, display, typing.cast(hsm.Model, display.model))
        await display.attach(environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller)))
        await wait_until(lambda: display.state() == "/Device/attached")

        await display.dispatch(
            environment,
            display_module.CallerIdEvent.with_data(display_module.CallerIdData(caller_id="Front desk")),
        )
        await asyncio.sleep(0)

        return _caller_id(display)

    assert asyncio.run(run()) == "Front desk"


def test_caller_id_event_with_none_clears_a_previously_shown_caller() -> None:
    """``caller_id=None`` writes None outright; it never leaves the previous value lingering.

    ``CallerIdData`` documents null as a real, distinct instruction (the screen going blank), not
    the absence of one — this pins that the write actually lands rather than being treated as a
    no-op that leaves the prior caller showing.
    """

    async def run() -> tuple[str | None, str | None]:
        display = display_module.Display()
        _ = await mosfet.started(None, display, typing.cast(hsm.Model, display.model))
        await wait_until(lambda: display.state() == "/Device/detached")

        await display.dispatch(
            display.context(),
            display_module.CallerIdEvent.with_data(display_module.CallerIdData(caller_id="Front desk")),
        )
        await asyncio.sleep(0)
        shown = _caller_id(display)

        await display.dispatch(
            display.context(),
            display_module.CallerIdEvent.with_data(display_module.CallerIdData(caller_id=None)),
        )
        await asyncio.sleep(0)
        cleared = _caller_id(display)

        return shown, cleared

    shown, cleared = asyncio.run(run())
    assert shown == "Front desk"
    assert cleared is None


def test_caller_id_event_name_carries_the_phone_domain() -> None:
    """The event name is ``phone.display.caller_id`` — domain-scoped, not a package path."""

    assert display_module.CallerIdEvent.name == "phone.display.caller_id"

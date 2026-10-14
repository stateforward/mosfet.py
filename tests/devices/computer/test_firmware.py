from mosfet.devices import computer

import asyncio
import dataclasses
import typing

import hsm
import pydantic
import pytest

from mosfet.device import ObservedData, ObservedEvent
from mosfet.environment import Environment
from tests.devices.computer.remote import (
    DISPLAY_CODEC,
    DEVICE_ID,
    FakeStream,
    Owner,
    ports,
    settle,
    start,
    stream,
    wait_until,
)

_TRACE = {"traceparent": "00-00000000000000000000000000000000-0000000000000000-00"}
UP = computer.LinkUpData(device_id=DEVICE_ID, transport="livekit")
DOWN = computer.LinkDownData(device_id=DEVICE_ID, reason="transport_lost")


async def started_firmware(environment: Environment) -> tuple[computer.Firmware, Owner, computer.Ports, FakeStream]:
    owner = Owner()
    await start(environment, owner, "computer")
    wired = ports()
    await start(environment, wired.display, "display")
    display_stream = await stream(environment, wired.display, DISPLAY_CODEC, "display")
    firmware = computer.Firmware(device=owner, ports=wired, person_id="p-1")
    await start(environment, firmware, "firmware")
    await settle()
    return firmware, owner, wired, display_stream


def test_firmware_starts_unlinked() -> None:
    async def run() -> str:
        firmware, _, _, _ = await started_firmware(Environment())
        return firmware.state()

    assert asyncio.run(run()) == "/Computer/unlinked"


def test_link_up_redrives_the_display_and_is_handed_to_the_device() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]]]:
        environment = Environment()
        firmware, owner, _, display_stream = await started_firmware(environment)

        up = dataclasses.replace(computer.LinkUpEvent.with_data(UP), id="up-1", source="link", metadata=dict(_TRACE))
        _ = await firmware.dispatch(hsm.Context(), up)
        await wait_until(lambda: bool(owner.observed) and bool(display_stream.sent))
        return firmware.state(), owner.observed, display_stream.sent

    state, observed, sent = asyncio.run(run())

    assert state == "/Computer/linked"
    (handed,) = observed
    assert handed.name == ObservedEvent.name
    assert (handed.id, handed.source, handed.target) == ("up-1", "firmware", "computer")
    assert handed.metadata == _TRACE
    assert isinstance(handed.data, ObservedData)
    assert handed.data.observation.data is UP
    # The display was driven with what this computer shows (nothing yet), and sent it to its stream.
    (content,) = sent
    assert content.data == computer.ContentData()


def test_link_down_is_handed_to_the_device() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        environment = Environment()
        firmware, owner, _, _ = await started_firmware(environment)
        _ = await firmware.dispatch(hsm.Context(), computer.LinkUpEvent.with_data(UP))
        await wait_until(lambda: firmware.state() == "/Computer/linked")

        _ = await firmware.dispatch(hsm.Context(), computer.LinkDownEvent.with_data(DOWN))
        await wait_until(lambda: firmware.state() == "/Computer/unlinked")
        return firmware.state(), owner.observations()

    state, observations = asyncio.run(run())

    assert state == "/Computer/unlinked"
    assert [observation.data for observation in observations] == [UP, DOWN]


@pytest.mark.parametrize("person_id", ["", "p" * 201])
def test_firmware_rejects_an_invalid_person(person_id: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = computer.Firmware(device=Owner(), ports=ports(), person_id=person_id)

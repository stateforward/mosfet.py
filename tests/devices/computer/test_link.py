from mosfet.devices import audio, computer, hid

import asyncio
import dataclasses
import datetime
import pathlib
import typing
import uuid

import hsm
import mosfet.lifecycle
import pydantic
import pytest

from mosfet import abilities
from mosfet.environment import Environment
from mosfet.protocols import attachment, transport
from tests import metric_reader
from tests.devices.computer.remote import (
    DEVICE_ID,
    Controller,
    FakeTransport,
    connected,
    disconnected,
    ports,
    settle,
    start,
    wait_until,
)

_TIMEOUT = datetime.timedelta(seconds=1)


_REGISTER_TIMEOUT = datetime.timedelta(seconds=10)
type Request = typing.Callable[[hsm.Instance], hsm.Event[typing.Any]]


async def request(link: computer.Link, make: Request) -> hsm.Event[typing.Any]:
    """Send one attach or detach request, answered to ``reply``, to ``link`` and wait for its one terminal."""

    return await abilities.run_terminal_operation(
        hsm.Context(),
        child=link,
        request=lambda reply: dataclasses.replace(make(reply), id=uuid.uuid4().hex),
        terminals=(
            attachment.AttachCompleteEvent,
            attachment.AttachFailedEvent,
            attachment.DetachedEvent,
            attachment.DetachFailedEvent,
        ),
        timeout=_TIMEOUT,
    )


def attach(firmware: hsm.Instance) -> Request:
    return lambda reply: attachment.AttachEvent.with_data(attachment.AttachData(actor=firmware, reply_to=reply))


def detach(firmware: hsm.Instance) -> Request:
    return lambda reply: attachment.DetachEvent.with_data(attachment.DetachData(actor=firmware, reply_to=reply))


def new_link(
    transport_actor: hsm.Instance, *, register_timeout: datetime.timedelta = _REGISTER_TIMEOUT
) -> computer.Link:
    return computer.Link(
        transport=transport_actor,
        peer_id="studio-laptop-peer",
        device_id=DEVICE_ID,
        channels=computer.channels(ports()),
        register_timeout=register_timeout,
    )


class Linked(typing.NamedTuple):
    environment: Environment
    link: computer.Link
    transport: FakeTransport
    firmware: Controller


async def attached_link(*, register: str = "answer") -> Linked:
    environment = Environment()
    fake, firmware = FakeTransport(register=register), Controller()
    await start(environment, fake, "transport")
    await start(environment, firmware, "firmware")
    link = new_link(fake)
    await start(environment, link, "link")
    terminal = await request(link, attach(firmware))
    assert terminal.name == attachment.AttachCompleteEvent.name
    return Linked(environment, link, fake, firmware)


def told(firmware: Controller) -> list[object]:
    return [
        event.data
        for event in firmware.events
        if event.name in (computer.LinkUpEvent.name, computer.LinkDownEvent.name)
    ]


UP = computer.LinkUpData(device_id=DEVICE_ID, transport="websocket")


def down(reason: typing.Literal["device_left", "transport_lost", "superseded", "detached"]) -> computer.LinkDownData:
    return computer.LinkDownData(device_id=DEVICE_ID, reason=reason)


def test_ports_validate_the_device_id() -> None:
    assert ports().device_id == DEVICE_ID
    with pytest.raises(pydantic.ValidationError):
        _ = ports(device_id="Not A Device")


def test_channels_give_each_remote_port_its_own_stream_and_codec() -> None:
    bound = ports()

    table = {channel.name: channel for channel in computer.channels(bound)}

    assert {name: (channel.owner, channel.kind, channel.codec.name) for name, channel in table.items()} == {
        "keyboard": (bound.keyboard, "data", "json"),
        "pointer": (bound.pointer, "data", "json"),
        "display": (bound.display, "data", "json"),
        "camera": (bound.camera, "data", "still"),
        "microphone": (bound.microphone, "audio", "pcm"),
        "speaker": (bound.speaker, "audio", "pcm"),
    }
    key = hid.KeyEvent.with_data(hid.KeyData(action="down", key="a", code="KeyA"))
    assert table["keyboard"].codec.encodes(key) and not table["pointer"].codec.encodes(key)
    assert table["display"].codec.encodes(computer.ContentEvent) and table["display"].codec.encodes(
        computer.LayoutEvent
    )
    assert table["display"].codec.max_message_bytes == computer.MAX_MESSAGE_BYTES == 64 * 1024
    assert table["speaker"].codec.encodes(audio.OutputEvent) and not table["camera"].codec.encodes_any


def test_link_rejects_invalid_construction() -> None:
    fake = FakeTransport()
    table = computer.channels(ports())
    with pytest.raises(ValueError, match="register_timeout"):
        _ = new_link(fake, register_timeout=datetime.timedelta(0))
    with pytest.raises(pydantic.ValidationError):
        _ = computer.Link(transport=fake, peer_id="no spaces", device_id=DEVICE_ID, channels=table)
    with pytest.raises(pydantic.ValidationError):
        _ = computer.Link(transport=fake, peer_id="peer", device_id="Not A Device", channels=table)
    with pytest.raises(ValueError, match="once"):
        _ = computer.Link(transport=fake, peer_id="peer", device_id=DEVICE_ID, channels=(*table, table[0]))


def test_link_registers_its_peer_and_channel_table_then_answers_the_firmware() -> None:
    async def run() -> tuple[FakeTransport, hsm.Event[typing.Any]]:
        linked = await attached_link()
        return linked.transport, linked.transport.requests[0]

    fake, registration = asyncio.run(run())

    data = registration.data
    assert isinstance(data, transport.PeerAttachData)
    assert (registration.source, registration.target) == ("link", "transport")
    assert data.peer_id == "studio-laptop-peer" == fake.peer_id
    assert [channel.name for channel in data.channels] == [
        "keyboard",
        "pointer",
        "display",
        "camera",
        "microphone",
        "speaker",
    ]


@pytest.mark.parametrize("register", ["refuse", "silent"])
def test_a_link_the_transport_will_not_carry_refuses_its_firmware(register: str) -> None:
    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any], str]:
        environment = Environment()
        fake, firmware = FakeTransport(register=register), Controller()
        await start(environment, fake, "transport")
        await start(environment, firmware, "firmware")
        link = new_link(fake, register_timeout=datetime.timedelta(milliseconds=20))
        await start(environment, link, "link")
        attached = await request(link, attach(firmware))
        detached = await request(link, detach(firmware))
        return attached, detached, link.state()

    attached, detached, state = asyncio.run(run())

    assert attached.name == attachment.AttachFailedEvent.name
    assert isinstance(attached.data, attachment.FailedData)
    assert attached.data.kind == attachment.FailureKind.CONFLICT
    assert isinstance(detached.data, attachment.DetachedData) and detached.data.removed is False
    assert state == "/ComputerLink/refused"


def test_a_link_with_no_transport_is_unplugged_registers_nothing_and_never_comes_up() -> None:
    """DSL-28b: an unplugged link holds its one firmware and tells it nothing, since no peer can connect."""

    async def run() -> tuple[str, list[hsm.Event[typing.Any]], list[object], bool]:
        environment = Environment()
        firmware, second = Controller(), Controller()
        await start(environment, firmware, "firmware")
        await start(environment, second, "second")
        link = computer.Link(transport=None, peer_id="desk", device_id="desk", channels=computer.channels(ports()))
        await start(environment, link, "link")
        state = link.state()
        terminals = [
            await request(link, attach(firmware)),
            await request(link, attach(second)),
            await request(link, attach(firmware)),
            await request(link, detach(firmware)),
            await request(link, detach(firmware)),
        ]
        await settle()
        await hsm.stop(link)
        return state, terminals, told(firmware), mosfet.lifecycle.is_started(link)

    state, terminals, observations, started = asyncio.run(run())

    assert state == "/ComputerLink/unplugged"
    attached, refused, again, detached, absent = (terminal.data for terminal in terminals)
    assert isinstance(attached, attachment.AttachCompleteData) and attached.created is True
    assert isinstance(refused, attachment.FailedData) and refused.kind == attachment.FailureKind.CONFLICT
    assert isinstance(again, attachment.AttachCompleteData) and again.created is False
    assert isinstance(detached, attachment.DetachedData) and detached.removed is True
    assert isinstance(absent, attachment.DetachedData) and absent.removed is False
    assert observations == []
    assert started is False


def test_link_refuses_a_second_firmware_and_answers_the_first_again() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any]]:
        linked = await attached_link()
        second = Controller()
        await start(linked.environment, second, "second")
        return await request(linked.link, attach(second)), await request(linked.link, attach(linked.firmware))

    refused, again = asyncio.run(run())

    assert isinstance(refused.data, attachment.FailedData)
    assert refused.data.kind == attachment.FailureKind.CONFLICT
    assert isinstance(again.data, attachment.AttachCompleteData) and again.data.created is False


def test_a_connection_is_accepted_and_the_link_comes_up() -> None:
    async def run() -> tuple[list[str], list[object], str]:
        linked = await attached_link()
        _ = await linked.transport.dispatch(hsm.Context(), connected(linked.transport))
        await wait_until(lambda: bool(told(linked.firmware)))
        return linked.transport.accepted(), told(linked.firmware), linked.link.state()

    accepted, observations, state = asyncio.run(run())

    assert accepted == ["c1"]
    assert observations == [UP]
    assert state == "/ComputerLink/registered/linked"


def test_a_connection_with_no_firmware_behind_it_is_let_go() -> None:
    async def run() -> tuple[list[str], list[tuple[str, str]]]:
        environment = Environment()
        fake = FakeTransport()
        await start(environment, fake, "transport")
        link = new_link(fake)
        await start(environment, link, "link")
        await wait_until(lambda: link.state() == "/ComputerLink/registered/detached")
        _ = await fake.dispatch(hsm.Context(), connected(fake))
        await wait_until(lambda: bool(fake.closed()))
        return fake.accepted(), fake.closed()

    assert asyncio.run(run()) == ([], [("c1", "released")])


def test_a_newer_connection_supersedes_the_current_one() -> None:
    async def run() -> tuple[list[str], list[tuple[str, str]], list[object]]:
        linked = await attached_link()
        _ = await linked.transport.dispatch(hsm.Context(), connected(linked.transport, "c1"))
        await wait_until(lambda: len(told(linked.firmware)) == 1)
        _ = await linked.transport.dispatch(hsm.Context(), connected(linked.transport, "c2"))
        await wait_until(lambda: len(told(linked.firmware)) == 3)
        # The superseded connection's late end is stale and changes nothing.
        _ = await linked.transport.dispatch(hsm.Context(), disconnected(linked.transport, "c1"))
        await settle()
        return linked.transport.accepted(), linked.transport.closed(), told(linked.firmware)

    accepted, closed, observations = asyncio.run(run())

    assert accepted == ["c1", "c2"]
    assert closed == [("c1", "superseded")]
    assert observations == [UP, down("superseded"), UP]


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("left", "device_left"), ("transport_lost", "transport_lost"), ("closed", "transport_lost")],
)
def test_a_lost_connection_takes_the_link_down_and_the_next_one_brings_it_up(
    reason: typing.Literal["left", "transport_lost", "closed"],
    expected: typing.Literal["device_left", "transport_lost"],
) -> None:
    async def run() -> tuple[list[object], list[str]]:
        linked = await attached_link()
        _ = await linked.transport.dispatch(hsm.Context(), connected(linked.transport, "c1"))
        _ = await linked.transport.dispatch(hsm.Context(), disconnected(linked.transport, "c1", reason))
        _ = await linked.transport.dispatch(hsm.Context(), connected(linked.transport, "c2"))
        await wait_until(lambda: len(told(linked.firmware)) == 3)
        return told(linked.firmware), linked.transport.accepted()

    observations, accepted = asyncio.run(run())

    assert observations == [UP, down(expected), UP]
    assert accepted == ["c1", "c2"]


@pytest.mark.parametrize("connect", [False, True])
def test_detach_is_answered_once_and_lets_the_current_connection_go(connect: bool) -> None:
    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any], list[tuple[str, str]], list[object], str]:
        linked = await attached_link()
        if connect:
            _ = await linked.transport.dispatch(hsm.Context(), connected(linked.transport))
            await wait_until(lambda: bool(told(linked.firmware)))
        first = await request(linked.link, detach(linked.firmware))
        second = await request(linked.link, detach(linked.firmware))
        return first, second, linked.transport.closed(), told(linked.firmware), linked.link.state()

    first, second, closed, observations, state = asyncio.run(run())

    assert isinstance(first.data, attachment.DetachedData) and first.data.removed is True
    assert isinstance(second.data, attachment.DetachedData) and second.data.removed is False
    assert closed == ([("c1", "released")] if connect else [])
    assert observations == ([UP, down("detached")] if connect else [])
    assert state == "/ComputerLink/registered/detached"


def test_stopping_the_link_unregisters_it_from_its_transport() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        linked = await attached_link()
        await hsm.stop(linked.link)
        await wait_until(lambda: len(linked.transport.requests) == 2)
        return linked.transport.requests

    _, unregistration = asyncio.run(run())

    assert unregistration.name == attachment.DetachEvent.name
    assert (unregistration.source, unregistration.target) == ("link", "transport")


def test_linked_sessions_are_counted_by_transport(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        linked = await attached_link()
        _ = await linked.transport.dispatch(hsm.Context(), connected(linked.transport, "c1"))
        _ = await linked.transport.dispatch(hsm.Context(), connected(linked.transport, "c2"))
        await wait_until(lambda: len(told(linked.firmware)) == 3)
        await hsm.stop(linked.link)

    def active(metrics: metric_reader.Reader) -> dict[object, float]:
        return {
            point.attributes.get("transport"): point.value
            for point in metrics.points("bot.devices.computer.link.session.active")
        }

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = active(metrics)
        asyncio.run(run())
        after = active(metrics)

    assert after.get("websocket", 0) - before.get("websocket", 0) == 0
    assert set(after) <= {"websocket"}

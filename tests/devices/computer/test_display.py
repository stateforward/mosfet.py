from mosfet.devices import computer

import asyncio
import dataclasses
import json
import typing

import pathlib

import hsm
import mosfet
import pytest

from mosfet.device import Device
from mosfet.environment import Environment
from tests import metric_reader
from tests.devices.computer.remote import (
    DISPLAY_CODEC,
    Controller,
    FakeStream,
    EndStreamEvent,
    EndData,
    settle,
    start,
    stream,
    wait_until,
    wire,
)

_TRACE = {"traceparent": "00-00000000000000000000000000000000-0000000000000000-00"}
CONTENT = computer.ContentData(
    elements=(
        computer.TextElement(id="greeting", text="Hello!", role="heading"),
        computer.FieldElement(id="name", placeholder="Your name", focused=True),
        computer.ButtonElement(id="ok", label="OK"),
    )
)
LAYOUT = computer.LayoutData(boxes=(computer.BoxData(id="ok", x=0.4, y=0.8, width=0.2, height=0.1),))


class Screen(Device):
    """A device that owns one computer display, so the display's observation folds into its snapshot."""

    def __init__(self, display: computer.Display) -> None:
        super().__init__(peripherals=(display,))


async def started_display(environment: Environment) -> computer.Display:
    display = computer.Display()
    await start(environment, display, "display")
    return display


def content(content_id: str) -> hsm.Event[typing.Any]:
    return dataclasses.replace(
        computer.ContentEvent.with_data(CONTENT), id=content_id, source="firmware", metadata=dict(_TRACE)
    )


def test_display_keeps_its_content_and_sends_it_one_way_to_the_stream_it_owns() -> None:
    async def run() -> tuple[object, FakeStream]:
        environment = Environment()
        display = await started_display(environment)
        owned = await stream(environment, display, DISPLAY_CODEC, "display")
        await settle()

        _ = await display.dispatch(hsm.Context(), content("content-1"))
        await wait_until(lambda: bool(owned.sent))
        return (display.take_snapshot().Attributes or {}).get("/Device/content"), owned

    shown, owned = asyncio.run(run())

    assert shown == CONTENT
    (forwarded,) = owned.sent
    assert forwarded.data is CONTENT
    assert (forwarded.id, forwarded.source, forwarded.target) == ("content-1", "display", "stream-display")
    assert forwarded.metadata == _TRACE
    assert owned.messages == [DISPLAY_CODEC.encode(forwarded)]


def _unowned(metrics: metric_reader.Reader) -> float:
    return sum(
        point.value
        for point in metrics.points("bot.transform.dropped.count")
        if point.attributes.get("reason") == "unowned" and point.attributes.get("codec") == "json"
    )


@pytest.mark.parametrize("ended", ["never_owned", "own_stream_ended"])
def test_a_display_with_no_stream_keeps_its_content_and_counts_the_send_dropped(
    ended: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[object, list[hsm.Event[typing.Any]]]:
        environment = Environment()
        display = await started_display(environment)
        sent: list[hsm.Event[typing.Any]] = []
        if ended == "own_stream_ended":
            owned = await stream(environment, display, DISPLAY_CODEC, "display")
            await settle()
            _ = await owned.dispatch(hsm.Context(), EndStreamEvent.with_data(EndData(reason="reset")))
            await settle()
            sent = owned.sent
        _ = await display.dispatch(hsm.Context(), content("content-1"))
        await settle()
        return (display.take_snapshot().Attributes or {}).get("/Device/content"), sent

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = _unowned(metrics)
        shown, sent = asyncio.run(run())
        after = _unowned(metrics)

    assert shown == CONTENT
    assert sent == []
    assert after - before == 1


def test_a_stale_stream_ending_leaves_the_newer_stream_owned() -> None:
    async def run() -> tuple[FakeStream, FakeStream]:
        environment = Environment()
        display = await started_display(environment)
        older = await stream(environment, display, DISPLAY_CODEC, "display")
        await settle()
        newer = FakeStream(owner=display, codec=DISPLAY_CODEC, channel="display")
        await start(environment, newer, "stream-display-2")
        await settle()
        _ = await older.dispatch(hsm.Context(), EndStreamEvent.with_data(EndData()))
        await settle()

        _ = await display.dispatch(hsm.Context(), content("content-2"))
        await wait_until(lambda: bool(newer.sent))
        return older, newer

    older, newer = asyncio.run(run())

    assert older.sent == []
    assert [event.id for event in newer.sent] == ["content-2"]


def test_display_content_folds_into_its_owner_snapshot_without_bytes() -> None:
    async def run() -> hsm.Snapshot:
        environment = Environment()
        display = computer.Display()
        screen = Screen(display)
        _ = await mosfet.started(environment, screen, typing.cast(hsm.Model, screen.model))
        await wait_until(lambda: screen.state() == "/Device/detached")

        _ = await display.dispatch(hsm.Context(), computer.ContentEvent.with_data(CONTENT))
        await settle()
        return screen.take_snapshot()

    snapshot = asyncio.run(run())

    assert snapshot.Attributes is not None
    folded = typing.cast(dict[str, object], snapshot.Attributes["display"])
    assert folded == {"content": CONTENT}
    schema = json.dumps(computer.ContentData.model_json_schema())
    assert "binary" not in schema and "base64" not in schema


def test_attached_display_hands_the_remote_layout_to_its_controller() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        display = await started_display(environment)
        controller = Controller()
        await start(environment, controller, "controller")
        await wire(environment, display, controller)
        await wait_until(controller.attached)

        layout = dataclasses.replace(computer.LayoutEvent.with_data(LAYOUT), id="layout-1", source="stream-display")
        _ = await display.dispatch(hsm.Context(), layout)
        await wait_until(lambda: bool(controller.named(computer.LayoutEvent)))
        return controller.named(computer.LayoutEvent)

    (forwarded,) = asyncio.run(run())

    assert forwarded.data is LAYOUT
    assert (forwarded.id, forwarded.source, forwarded.target) == ("layout-1", "display", "controller")


def test_unattached_display_keeps_the_remote_layout_to_itself() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        display = await started_display(environment)
        controller = Controller()
        await start(environment, controller, "controller")

        _ = await display.dispatch(hsm.Context(), computer.LayoutEvent.with_data(LAYOUT))
        await settle()
        return controller.named(computer.LayoutEvent)

    assert asyncio.run(run()) == []


def test_a_newly_owned_stream_is_sent_what_the_display_shows() -> None:
    """Content driven before any stream was owned reaches the remote screen once one is handed over."""

    async def run() -> FakeStream:
        environment = Environment()
        display = await started_display(environment)
        _ = await display.dispatch(hsm.Context(), content("content-1"))
        await settle()

        owned = await stream(environment, display, DISPLAY_CODEC, "display")
        await wait_until(lambda: bool(owned.sent))
        await settle()
        return owned

    owned = asyncio.run(run())

    (shown,) = owned.sent
    assert shown.data == CONTENT
    assert (shown.source, shown.target) == ("display", "stream-display")


def test_a_display_that_shows_nothing_sends_nothing_to_a_new_stream() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        display = await started_display(environment)
        owned = await stream(environment, display, DISPLAY_CODEC, "display")
        await settle()
        return owned.sent

    assert asyncio.run(run()) == []

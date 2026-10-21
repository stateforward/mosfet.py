"""``companion_example.ui``: the browser bridge, a room's far side whose devices are a page in a browser.

Every session runs over loopback sockets, in the kitchen ``support`` sets up: a real ``websocket.Server`` and a
real ``computer.Computer`` on a counter, held by a recording holder, with an ear in the house. The bridge is the
real ``ui.serve`` page server and a real ``room.Room`` on the bridge's ``ui.Browser`` devices; the page is a
WebSocket client the test drives, as the page's script would, with the page's own origin. The process's exit
codes are checked by running ``python -m companion_example.ui`` as a child process.
"""

import asyncio
import collections.abc
import contextlib
import dataclasses
import json
import os
import struct
import sys
import typing

import mosfet
import pytest
import websockets
import websockets.asyncio.client
import websockets.exceptions
from mosfet import environment, media
from mosfet.devices import computer
from mosfet.environment import space
from companion_example import room, ui

from tests import support

# The smallest JPEG signature the camera's stills are checked by, then an end-of-image marker.
JPEG = b"\xff\xd8\xff\xe0" + bytes(16) + b"\xff\xd9"
_CHILD_BOUND = 60.0


@dataclasses.dataclass
class Page:
    """The page's end of the bridge's socket: what it sends, and everything the bridge told it, in order."""

    socket: websockets.asyncio.client.ClientConnection
    texts: list[dict[str, typing.Any]] = dataclasses.field(default_factory=list)
    sounds: list[bytes] = dataclasses.field(default_factory=list)

    async def read(self) -> None:
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            async for message in self.socket:
                if isinstance(message, bytes):
                    self.sounds.append(message)
                else:
                    self.texts.append(typing.cast(dict[str, typing.Any], json.loads(message)))

    def states(self) -> list[str]:
        return [str(text["state"]) for text in self.texts if text["type"] == "state"]

    def dropped(self) -> list[tuple[str, str]]:
        return [(str(text["what"]), str(text["why"])) for text in self.texts if text["type"] == "dropped"]


@dataclasses.dataclass
class Bridge:
    """The bridge to the kitchen, its page server and the page connected to it."""

    house: support.Kitchen
    session: support.Session
    port: int
    page: Page


@contextlib.asynccontextmanager
async def bridge(*, camera: bool = False) -> collections.abc.AsyncGenerator[Bridge]:
    """The kitchen, the bridge's room linked to its computer, and a page connected and told the room is linked."""

    house = await support.kitchen()
    browser = ui.Browser()
    try:
        async with ui.serve(browser, port=0, page=ui.page(camera=camera)) as server:
            port = typing.cast(tuple[str, int], next(iter(server.sockets)).getsockname())[1]
            closed: asyncio.Future[room.ClosedData] = asyncio.get_running_loop().create_future()

            def on_closed(data: room.ClosedData) -> None:
                browser.closed(data)
                closed.set_result(data)

            machine = room.Room(
                url=f"ws://127.0.0.1:{house.port}",
                peer=support.PEER,
                token=support.TOKENS.issue(support.PEER),
                microphone=browser.microphone,
                speaker=browser.speaker,
                keyboard=browser.keyboard,
                camera=browser.camera if camera else None,
                on_closed=on_closed,
                connect_timeout=support.BOUND,
                open_timeout=support.BOUND,
            )
            _ = await mosfet.started(None, machine, room.Room.model)
            session = support.Session(machine=machine, closed=closed)
            async with websockets.asyncio.client.connect(
                f"ws://127.0.0.1:{port}/ws", origin=websockets.Origin(f"http://127.0.0.1:{port}")
            ) as socket:
                page = Page(socket=socket)
                reading = asyncio.create_task(page.read())
                try:
                    await support.wait_until(lambda: house.holder.linked() and "linked" in page.states())
                    yield Bridge(house=house, session=session, port=port, page=page)
                finally:
                    if not closed.done():
                        _ = await session.leave()
                    await socket.close()
                    await reading
    finally:
        await house.close()


def test_what_the_page_hears_is_put_into_the_house_by_the_rooms_computer() -> None:
    async def run() -> list[environment.SoundData]:
        async with bridge() as linked:
            await linked.page.socket.send(bytes((ui.MICROPHONE,)) + support.HEARD)
            await support.wait_until(lambda: bool(linked.house.ear.sounds()))
            return linked.house.ear.sounds()

    (sound,) = asyncio.run(run())

    assert isinstance(sound, computer.MicrophoneSoundData)
    assert (sound.audio, sound.sample_rate_hz, sound.channels) == (support.HEARD, 16_000, 1)


def test_what_the_computer_plays_reaches_the_page_with_its_rate() -> None:
    async def run() -> list[bytes]:
        async with bridge() as linked:
            voice = environment.SoundData(
                audio=support.SPOKEN, media_type="audio/pcm", sample_rate_hz=24_000, channels=1
            )
            await linked.house.house.broadcast(environment.SoundEvent.with_data(voice), origin=space.Position(0.0, 0.0))
            await support.wait_until(lambda: bool(linked.page.sounds))
            return linked.page.sounds

    (sound,) = asyncio.run(run())

    rate, channels = struct.unpack_from("<IB", sound)
    assert (rate, channels, sound[5:]) == (24_000, 1, support.SPOKEN)


def test_a_line_typed_on_the_page_reaches_whoever_holds_the_computer_with_no_field() -> None:
    async def run() -> list[computer.TextEnteredData]:
        async with bridge() as linked:
            await linked.page.socket.send(json.dumps({"type": "line", "text": "Hello, the companion"}))
            await support.wait_until(lambda: bool(linked.house.holder.entered()))
            return linked.house.holder.entered()

    (entered,) = asyncio.run(run())

    assert (entered.device_id, entered.field_id, entered.text) == (support.PEER, None, "Hello, the companion")


def test_what_the_page_sends_that_the_bridge_cannot_take_is_dropped_and_the_page_told() -> None:
    async def run() -> tuple[list[tuple[str, str]], list[computer.TextEnteredData], list[environment.SoundData]]:
        async with bridge() as linked:
            socket = linked.page.socket
            await socket.send(json.dumps({"type": "line", "text": ""}))
            await socket.send(json.dumps({"type": "line", "text": "x" * (room.MAX_LINE + 1)}))
            await socket.send("not json")
            # Odd PCM, a still with no camera, and a tag nobody knows.
            await socket.send(bytes((ui.MICROPHONE,)) + b"\x01")
            await socket.send(bytes((ui.CAMERA,)) + JPEG)
            await socket.send(b"\x09")
            await support.wait_until(lambda: len(linked.page.dropped()) == 6)
            return linked.page.dropped(), linked.house.holder.entered(), linked.house.ear.sounds()

    dropped, entered, sounds = asyncio.run(run())

    assert dropped == [
        ("line", "invalid"),
        ("line", "invalid"),
        ("message", "invalid"),
        ("microphone", "invalid"),
        ("camera", "not_linked"),
        ("message", "invalid"),
    ]
    assert (entered, sounds) == ([], [])


def test_with_a_camera_the_pages_stills_reach_the_house() -> None:
    async def run() -> list[environment.VisualData]:
        async with bridge(camera=True) as linked:
            await linked.page.socket.send(bytes((ui.CAMERA,)) + JPEG)
            await support.wait_until(lambda: bool(linked.house.ear.sights()))
            return linked.house.ear.sights()

    (sight,) = asyncio.run(run())

    assert sight.image == media.Image(content=JPEG, media_type="image/jpeg")


def test_when_the_room_closes_the_page_is_told_how() -> None:
    async def run() -> list[dict[str, typing.Any]]:
        async with bridge() as linked:
            _ = await linked.session.leave()
            await support.wait_until(lambda: "closed" in linked.page.states())
            return linked.page.texts

    texts = asyncio.run(run())

    assert {"type": "state", "state": "closed", "reason": "left", "code": None} in texts


def _get(port: int, path: str) -> tuple[int, dict[str, str], str]:
    async def run() -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        return response

    head, _, body = asyncio.run(run()).partition(b"\r\n\r\n")
    status, *fields = head.decode("latin-1").split("\r\n")
    headers = {name.lower(): value.strip() for name, _, value in (field.partition(":") for field in fields)}
    return int(status.split()[1]), headers, body.decode()


def test_the_page_is_served_at_slash_and_the_ui_binds_loopback_only() -> None:
    async def run() -> tuple[list[str], int]:
        async with ui.serve(ui.Browser(), port=0, page=ui.page(camera=True)) as server:
            sockets = list(server.sockets)
            port = typing.cast(tuple[str, int], sockets[0].getsockname())[1]
            served = await asyncio.to_thread(_get, port, "/")
            missing = await asyncio.to_thread(_get, port, "/elsewhere")
            pages.extend((served, missing))
            return [typing.cast(tuple[str, int], socket.getsockname())[0] for socket in sockets], port

    pages: list[tuple[int, dict[str, str], str]] = []
    hosts, _ = asyncio.run(run())

    assert hosts == ["127.0.0.1"]
    (status, headers, body), (missing, _, _) = pages
    assert status == 200
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert "<title>the companion</title>" in body
    assert '"camera": true' in body
    assert missing == 404


def test_a_page_from_another_origin_is_refused() -> None:
    async def run() -> list[int]:
        refused: list[int] = []
        async with ui.serve(ui.Browser(), port=0, page=ui.page(camera=False)) as server:
            port = typing.cast(tuple[str, int], next(iter(server.sockets)).getsockname())[1]
            for origin in (websockets.Origin("http://evil.example"), None):
                with pytest.raises(websockets.exceptions.InvalidStatus) as rejected:
                    async with websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}/ws", origin=origin):
                        pass
                refused.append(rejected.value.response.status_code)
        return refused

    assert asyncio.run(run()) == [403, 403]


def test_main_refuses_bad_flags_with_64_and_a_missing_secret_with_78(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("COMPUTER_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("COMPUTER_TOKEN_SECRET_FILE", raising=False)

    assert ui.main(["--url", "http://127.0.0.1:8765"]) == 64
    assert ui.main(["--peer", "Not A Room"]) == 64
    assert ui.main(["--port", "70000"]) == 64
    # The page is served on loopback only: there is no host to choose.
    assert ui.main(["--host", "0.0.0.0"]) == 64
    assert ui.main(["--nonsense"]) == 64
    assert ui.main(["--peer", support.PEER]) == 78
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "computer_token_secret" in captured.err


async def _child(url: str, secret: str) -> tuple[int, str]:
    environ = {
        **os.environ,
        "COMPUTER_TOKEN_SECRET": secret,
        "BOT_OTEL_DISABLED": "1",
    }
    environ.pop("COMPUTER_TOKEN_SECRET_FILE", None)
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "companion_example.ui",
        "--url",
        url,
        "--peer",
        support.PEER,
        "--port",
        "0",
        env=environ,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    async with asyncio.timeout(_CHILD_BOUND):
        out, err = await child.communicate()
    assert out == b""
    assert child.returncode is not None
    return child.returncode, err.decode()


def test_a_refused_token_ends_the_ui_1() -> None:
    async def run() -> tuple[int, str]:
        house = await support.kitchen()
        try:
            return await _child(f"ws://127.0.0.1:{house.port}", "another-secret")
        finally:
            await house.close()

    code, err = asyncio.run(run())

    assert code == 1
    assert "refused (4401)" in err


def test_a_ui_with_nothing_to_connect_to_ends_1() -> None:
    code, err = asyncio.run(_child(f"ws://127.0.0.1:{support.free_port()}", "computer-secret"))

    assert code == 1
    assert "abnormal" in err

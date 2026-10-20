"""The kitchen the room's far side and the browser bridge are tested in, and the room's fake devices.

Willy's side is a real ``websocket.Server`` and a real ``computer.Computer`` set on a counter in the kitchen,
held by a recording holder, in a house (an ``environment.Environment``) where a listener hears everything put
into it. The room's sound card, camera and keyboard are fakes, the one seam the far side has: a microphone the
test speaks into from a thread of its own, a speaker that keeps what it played, a camera the test shows stills
to, and a keyboard the test types lines on.
"""

import asyncio
import collections.abc
import dataclasses
import datetime
import socket as sockets
import threading
import typing

import hsm
import mosfet
import pydantic
from mosfet import environment, media
from mosfet.devices import computer
from mosfet.environment import space
from mosfet.protocols import attachment
from mosfet.providers import websocket
from willy_example import room

PEER = "kitchen"
TOKENS = websocket.HmacTokens(pydantic.SecretStr("computer-secret"))
BOUND = datetime.timedelta(seconds=2)
ATTACHED = "/Device/attached"
LISTENING = "/Server/listening"
COUNTER = environment.Object(id="counter", placement=space.Placement(position=space.Position(4.0, 2.0)))
# A tenth of a second of the room, and of Willy's voice at its speech engine's rate.
HEARD = b"\x07\x00" * 1_600
SPOKEN = b"\x09\x00" * 2_400
# The smallest PNG: one transparent pixel.
PIXEL = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    + "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


def free_port() -> int:
    with sockets.socket(sockets.AF_INET, sockets.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return typing.cast(tuple[str, int], probe.getsockname())[1]


class Recorder(hsm.Instance):
    """Stands in for the bot holding the computer, or for an ear in the house: keeps every event it is given."""

    seen: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.seen = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Recorder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.seen.append(event)

    def sounds(self) -> list[environment.SoundData]:
        return [event.data for event in self.seen if isinstance(event.data, environment.SoundData)]

    def sights(self) -> list[environment.VisualData]:
        return [event.data for event in self.seen if isinstance(event.data, environment.VisualData)]

    def linked(self) -> bool:
        return any(
            isinstance(event.data, mosfet.InputEventData)
            and event.data.observation is not None
            and event.data.observation.event == computer.LinkUpEvent.name
            for event in self.seen
        )

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Recorder",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(
                hsm.on(mosfet.InputEvent, environment.SoundEvent, environment.VisualEvent), hsm.effect(_record)
            ),
        ),
    )


class Microphone:
    """The room's microphone: the test speaks into it, from a thread of its own as a sound card does."""

    def __init__(self, *, fails: bool = False) -> None:
        self.heard: collections.abc.Callable[[bytes], None] | None = None
        self.stopped: int = 0
        self._fails: bool = fails

    def start(self, heard: collections.abc.Callable[[bytes], None]) -> None:
        if self._fails:
            raise OSError("no input device")
        self.heard = heard

    def stop(self) -> None:
        self.stopped += 1

    def say(self, pcm: bytes) -> None:
        heard = self.heard
        assert heard is not None
        speaking = threading.Thread(target=heard, args=(pcm,))
        speaking.start()
        speaking.join()


@dataclasses.dataclass
class Played:
    pcm: bytes
    sample_rate_hz: int
    channels: int


class Speaker:
    """The room's speaker: keeps everything it played."""

    def __init__(self) -> None:
        self.played: list[Played] = []
        self.stopped: int = 0

    def play(self, pcm: bytes, *, sample_rate_hz: int, channels: int) -> None:
        self.played.append(Played(pcm=pcm, sample_rate_hz=sample_rate_hz, channels=channels))

    def stop(self) -> None:
        self.stopped += 1


class Camera:
    """The room's camera: the test shows it stills, from a thread of its own."""

    def __init__(self) -> None:
        self.seen: collections.abc.Callable[[media.Image], None] | None = None
        self.stopped: int = 0

    def start(self, seen: collections.abc.Callable[[media.Image], None]) -> None:
        self.seen = seen

    def stop(self) -> None:
        self.stopped += 1

    def show(self, image: media.Image) -> None:
        seen = self.seen
        assert seen is not None
        showing = threading.Thread(target=seen, args=(image,))
        showing.start()
        showing.join()


@dataclasses.dataclass
class Kitchen:
    """Willy's side: a websocket server, the kitchen's computer on the counter, its holder, and an ear in the house."""

    house: environment.Environment
    port: int
    server: websocket.Server
    device: computer.Computer
    holder: Recorder
    ear: Recorder

    async def close(self) -> None:
        await hsm.stop(self.device)
        await hsm.stop(self.server)


async def kitchen() -> Kitchen:
    house = environment.Environment()
    port = free_port()
    server = websocket.Server(host="127.0.0.1", port=port, verifier=TOKENS)
    _ = await mosfet.started(house, server, websocket.Server.model)
    await wait_until(lambda: server.state() == LISTENING)
    device = computer.Computer(id=PEER, transport=server, peer_id=PEER, placement=COUNTER.surface())
    _ = await mosfet.started(house, device, typing.cast(hsm.Model, device.model))
    holder = Recorder()
    _ = await mosfet.started(house, holder, Recorder.model)
    _ = await hsm.dispatch(house, device, attachment.AttachEvent.with_data(attachment.AttachData(actor=holder)))
    await wait_until(lambda: device.state() == ATTACHED)
    ear = Recorder()
    _ = await mosfet.started(house, ear, Recorder.model)
    house.join(ear)
    return Kitchen(house=house, port=port, server=server, device=device, holder=holder, ear=ear)


@dataclasses.dataclass
class Session:
    machine: room.Room
    closed: asyncio.Future[room.ClosedData]

    async def leave(self) -> room.ClosedData:
        _ = await self.machine.dispatch(self.machine.context(), room.LeaveEvent.with_data(room.LeaveData()))
        return await self.ended()

    async def ended(self) -> room.ClosedData:
        try:
            return await asyncio.wait_for(asyncio.shield(self.closed), 5)
        finally:
            await hsm.stop(self.machine)


async def enter(
    port: int,
    microphone: Microphone,
    speaker: Speaker,
    *,
    camera: Camera | None = None,
    token: pydantic.SecretStr | None = None,
) -> Session:
    closed: asyncio.Future[room.ClosedData] = asyncio.get_running_loop().create_future()
    machine = room.Room(
        url=f"ws://127.0.0.1:{port}",
        peer=PEER,
        token=TOKENS.issue(PEER) if token is None else token,
        microphone=microphone,
        speaker=speaker,
        camera=camera,
        on_closed=closed.set_result,
        connect_timeout=BOUND,
        open_timeout=BOUND,
    )
    _ = await mosfet.started(None, machine, room.Room.model)
    return Session(machine=machine, closed=closed)

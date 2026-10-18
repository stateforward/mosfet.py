"""``willy_example.room``: one room's far side of its computer, against the real websocket server.

Every session runs over loopback sockets: a real ``websocket.Server`` and a real ``computer.Computer`` set on
a counter in the kitchen, held by a recording holder, in a house (an ``environment.Environment``) where a
listener hears everything put into it. The room's sound card and camera are fakes, the one seam the far side
has: a microphone the test speaks into from a thread of its own, a speaker that keeps what it played, and a
camera the test shows stills to. The ``sounddevice`` backends are tested over a fake ``sounddevice`` module, so
no sound card is opened.
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
import pytest
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


def test_what_the_rooms_microphone_hears_is_put_into_the_house_by_its_computer() -> None:
    async def run() -> tuple[list[environment.SoundData], room.ClosedData, Microphone]:
        house = await kitchen()
        microphone, speaker = Microphone(), Speaker()
        try:
            session = await enter(house.port, microphone, speaker)
            await wait_until(lambda: house.holder.linked() and microphone.heard is not None)
            microphone.say(HEARD)
            await wait_until(lambda: bool(house.ear.sounds()))
            closed = await session.leave()
            return house.ear.sounds(), closed, microphone
        finally:
            await house.close()

    sounds, closed, microphone = asyncio.run(run())

    (sound,) = sounds
    # The computer's own microphone sound: never relayed back to a speaker, and nobody stamped on it.
    assert isinstance(sound, computer.MicrophoneSoundData)
    assert (sound.audio, sound.media_type, sound.sample_rate_hz, sound.channels) == (HEARD, "audio/pcm", 16_000, 1)
    assert sound.participants == ()
    assert (closed.reason, closed.exit_code) == ("left", 0)
    assert microphone.stopped == 1


def test_what_the_computer_hears_in_the_house_comes_out_of_the_rooms_speaker() -> None:
    async def run() -> tuple[list[Played], room.ClosedData, Speaker]:
        house = await kitchen()
        microphone, speaker = Microphone(), Speaker()
        try:
            session = await enter(house.port, microphone, speaker)
            await wait_until(lambda: house.holder.linked() and microphone.heard is not None)
            # Willy's voice, said somewhere in the house: the computer's mouthpiece hears it.
            voice = environment.SoundData(audio=SPOKEN, media_type="audio/pcm", sample_rate_hz=24_000, channels=1)
            await house.house.broadcast(environment.SoundEvent.with_data(voice), origin=space.Position(0.0, 0.0))
            await wait_until(lambda: bool(speaker.played))
            closed = await session.leave()
            return speaker.played, closed, speaker
        finally:
            await house.close()

    played, closed, speaker = asyncio.run(run())

    assert played == [Played(pcm=SPOKEN, sample_rate_hz=24_000, channels=1)]
    assert (closed.reason, closed.exit_code) == ("left", 0)
    assert speaker.stopped == 1


def test_with_a_camera_the_room_shows_the_house_what_it_sees() -> None:
    async def run() -> tuple[list[environment.VisualData], Camera]:
        house = await kitchen()
        microphone, speaker, camera = Microphone(), Speaker(), Camera()
        try:
            session = await enter(house.port, microphone, speaker, camera=camera)
            await wait_until(lambda: house.holder.linked() and camera.seen is not None)
            camera.show(media.Image(content=PIXEL, media_type="image/png"))
            await wait_until(lambda: bool(house.ear.sights()))
            _ = await session.leave()
            return house.ear.sights(), camera
        finally:
            await house.close()

    sights, camera = asyncio.run(run())

    (sight,) = sights
    assert sight.image == media.Image(content=PIXEL, media_type="image/png")
    assert camera.stopped == 1


def test_a_room_whose_token_is_refused_ends_1_and_hears_nothing() -> None:
    async def run() -> tuple[room.ClosedData, Microphone]:
        house = await kitchen()
        microphone = Microphone()
        try:
            wrong = websocket.HmacTokens(pydantic.SecretStr("another-secret")).issue(PEER)
            session = await enter(house.port, microphone, Speaker(), token=wrong)
            return await session.ended(), microphone
        finally:
            await house.close()

    closed, microphone = asyncio.run(run())

    assert (closed.reason, closed.code, closed.exit_code) == ("refused", 4401, 1)
    assert microphone.heard is None or microphone.stopped == 1


def test_a_room_with_nothing_to_connect_to_ends_1() -> None:
    async def run() -> room.ClosedData:
        session = await enter(free_port(), Microphone(), Speaker())
        return await session.ended()

    closed = asyncio.run(run())

    assert (closed.reason, closed.exit_code) == ("abnormal", 1)


def test_a_sound_card_that_cannot_open_ends_the_session_1() -> None:
    async def run() -> room.ClosedData:
        house = await kitchen()
        try:
            session = await enter(house.port, Microphone(fails=True), Speaker())
            return await session.ended()
        finally:
            await house.close()

    closed = asyncio.run(run())

    assert (closed.reason, closed.exit_code) == ("abnormal", 1)


def test_a_room_url_and_name_are_checked_when_it_is_built() -> None:
    def build(url: str, peer: str) -> None:
        _ = room.Room(
            url=url,
            peer=peer,
            token=TOKENS.issue(PEER),
            microphone=Microphone(),
            speaker=Speaker(),
            on_closed=lambda closed: None,
        )

    for url, peer in (("http://127.0.0.1:8765", PEER), ("ws://127.0.0.1:8765/v1", PEER), ("ws://h:1", "Kitchen!")):
        with pytest.raises(ValueError):
            build(url, peer)


def test_main_refuses_bad_flags_with_64_and_a_missing_secret_with_78(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("COMPUTER_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("COMPUTER_TOKEN_SECRET_FILE", raising=False)

    assert room.main(["--url", "http://127.0.0.1:8765"]) == 64
    assert room.main(["--peer", "Not A Room"]) == 64
    assert room.main(["--nonsense"]) == 64
    assert room.main(["--peer", PEER]) == 78
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "computer_token_secret" in captured.err


class _Stream:
    def __init__(self, callback: collections.abc.Callable[..., None], **opened: object) -> None:
        self.callback: collections.abc.Callable[..., None] = callback
        self.opened: dict[str, object] = opened
        self.started: bool = False
        self.closed: bool = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class SoundDevice:
    """A fake ``sounddevice`` module: it opens nothing, and keeps every stream it was asked for."""

    def __init__(self) -> None:
        self.streams: list[_Stream] = []

    def RawInputStream(self, *, callback: collections.abc.Callable[..., None], **opened: object) -> _Stream:
        self.streams.append(_Stream(callback, **opened))
        return self.streams[-1]

    def RawOutputStream(self, *, callback: collections.abc.Callable[..., None], **opened: object) -> _Stream:
        self.streams.append(_Stream(callback, **opened))
        return self.streams[-1]


def test_the_sounddevice_microphone_captures_16_khz_mono_blocks_and_hands_each_on() -> None:
    card = SoundDevice()
    heard: list[bytes] = []
    microphone = room.SoundDeviceMicrophone(load=lambda: card)

    microphone.start(heard.append)
    (stream,) = card.streams
    stream.callback(memoryview(HEARD), len(HEARD) // 2, None, None)
    microphone.stop()

    assert stream.opened == {"samplerate": 16_000, "blocksize": 3_200, "channels": 1, "dtype": "int16"}
    assert heard == [HEARD]
    assert stream.closed and not stream.started


def test_the_sounddevice_speaker_plays_in_order_pads_with_silence_and_reopens_for_another_rate() -> None:
    card = SoundDevice()
    speaker = room.SoundDeviceSpeaker(load=lambda: card, max_queued_seconds=1.0)

    speaker.play(b"\x01\x00\x02\x00", sample_rate_hz=24_000, channels=1)
    (first,) = card.streams
    filled = bytearray(8)
    first.callback(filled, 4, None, None)
    speaker.play(b"\x03\x00", sample_rate_hz=16_000, channels=1)
    speaker.stop()

    assert first.opened == {"samplerate": 24_000, "channels": 1, "dtype": "int16"}
    assert bytes(filled) == b"\x01\x00\x02\x00" + bytes(4)
    assert first.closed
    (_, second) = card.streams
    assert second.opened["samplerate"] == 16_000 and second.closed


def test_the_sounddevice_speaker_lets_the_oldest_go_past_what_it_may_queue() -> None:
    card = SoundDevice()
    speaker = room.SoundDeviceSpeaker(load=lambda: card, max_queued_seconds=0.0001)

    # 0.0001 s at 24 kHz is two samples: four bytes.
    speaker.play(b"\x01\x00\x02\x00\x03\x00", sample_rate_hz=24_000, channels=1)
    (stream,) = card.streams
    filled = bytearray(6)
    stream.callback(filled, 3, None, None)

    assert bytes(filled) == b"\x02\x00\x03\x00" + bytes(2)

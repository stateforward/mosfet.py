"""``willy_example.room``: one room's far side of its computer, against the real websocket server.

Every session runs over loopback sockets, in the kitchen ``support`` sets up: a real ``websocket.Server`` and a
real ``computer.Computer`` on a counter, held by a recording holder, with an ear in the house. The room's sound
card, camera and keyboard are ``support``'s fakes. The ``sounddevice`` backends are tested over a fake
``sounddevice`` module, so no sound card is opened.
"""

import asyncio
import collections.abc

import pydantic
import pytest
from mosfet import environment, media
from mosfet.devices import computer
from mosfet.environment import space
from mosfet.providers import websocket
from willy_example import room

from tests import support


def test_what_the_rooms_microphone_hears_is_put_into_the_house_by_its_computer() -> None:
    async def run() -> tuple[list[environment.SoundData], room.ClosedData, support.Microphone]:
        house = await support.kitchen()
        microphone, speaker = support.Microphone(), support.Speaker()
        try:
            session = await support.enter(house.port, microphone, speaker)
            await support.wait_until(lambda: house.holder.linked() and microphone.heard is not None)
            microphone.say(support.HEARD)
            await support.wait_until(lambda: bool(house.ear.sounds()))
            closed = await session.leave()
            return house.ear.sounds(), closed, microphone
        finally:
            await house.close()

    sounds, closed, microphone = asyncio.run(run())

    (sound,) = sounds
    # The computer's own microphone sound: never relayed back to a speaker, and nobody stamped on it.
    assert isinstance(sound, computer.MicrophoneSoundData)
    assert (sound.audio, sound.media_type, sound.sample_rate_hz, sound.channels) == (
        support.HEARD,
        "audio/pcm",
        16_000,
        1,
    )
    assert sound.participants == ()
    assert (closed.reason, closed.exit_code) == ("left", 0)
    assert microphone.stopped == 1


def test_what_the_computer_hears_in_the_house_comes_out_of_the_rooms_speaker() -> None:
    async def run() -> tuple[list[support.Played], room.ClosedData, support.Speaker]:
        house = await support.kitchen()
        microphone, speaker = support.Microphone(), support.Speaker()
        try:
            session = await support.enter(house.port, microphone, speaker)
            await support.wait_until(lambda: house.holder.linked() and microphone.heard is not None)
            # Willy's voice, said somewhere in the house: the computer's mouthpiece hears it.
            voice = environment.SoundData(
                audio=support.SPOKEN, media_type="audio/pcm", sample_rate_hz=24_000, channels=1
            )
            await house.house.broadcast(environment.SoundEvent.with_data(voice), origin=space.Position(0.0, 0.0))
            await support.wait_until(lambda: bool(speaker.played))
            closed = await session.leave()
            return speaker.played, closed, speaker
        finally:
            await house.close()

    played, closed, speaker = asyncio.run(run())

    assert played == [support.Played(pcm=support.SPOKEN, sample_rate_hz=24_000, channels=1)]
    assert (closed.reason, closed.exit_code) == ("left", 0)
    assert speaker.stopped == 1


def test_with_a_camera_the_room_shows_the_house_what_it_sees() -> None:
    async def run() -> tuple[list[environment.VisualData], support.Camera]:
        house = await support.kitchen()
        microphone, speaker, camera = support.Microphone(), support.Speaker(), support.Camera()
        try:
            session = await support.enter(house.port, microphone, speaker, camera=camera)
            await support.wait_until(lambda: house.holder.linked() and camera.seen is not None)
            camera.show(media.Image(content=support.PIXEL, media_type="image/png"))
            await support.wait_until(lambda: bool(house.ear.sights()))
            _ = await session.leave()
            return house.ear.sights(), camera
        finally:
            await house.close()

    sights, camera = asyncio.run(run())

    (sight,) = sights
    assert sight.image == media.Image(content=support.PIXEL, media_type="image/png")
    assert camera.stopped == 1


def test_a_line_typed_on_the_rooms_keyboard_reaches_whoever_holds_the_computer_with_no_field() -> None:
    async def run() -> tuple[list[computer.TextEnteredData], support.Keyboard]:
        house = await support.kitchen()
        microphone, keyboard = support.Microphone(), support.Keyboard()
        try:
            session = await support.enter(house.port, microphone, support.Speaker(), keyboard=keyboard)
            await support.wait_until(lambda: house.holder.linked() and keyboard.typed is not None)
            keyboard.type("Hello, Willy")
            keyboard.type("x" * (room.MAX_LINE + 1))
            keyboard.type("Are you there?")
            await support.wait_until(lambda: len(house.holder.entered()) == 2)
            _ = await session.leave()
            return house.holder.entered(), keyboard
        finally:
            await house.close()

    entered, keyboard = asyncio.run(run())

    # Typed with nothing focused: the line reaches the holder with no field. A line too long to type is not typed.
    assert [(line.field_id, line.text, line.device_id) for line in entered] == [
        (None, "Hello, Willy", support.PEER),
        (None, "Are you there?", support.PEER),
    ]
    assert keyboard.stopped == 1


def test_a_room_whose_token_is_refused_ends_1_and_hears_nothing() -> None:
    async def run() -> tuple[room.ClosedData, support.Microphone]:
        house = await support.kitchen()
        microphone = support.Microphone()
        try:
            wrong = websocket.HmacTokens(pydantic.SecretStr("another-secret")).issue(support.PEER)
            session = await support.enter(house.port, microphone, support.Speaker(), token=wrong)
            return await session.ended(), microphone
        finally:
            await house.close()

    closed, microphone = asyncio.run(run())

    assert (closed.reason, closed.code, closed.exit_code) == ("refused", 4401, 1)
    assert microphone.heard is None or microphone.stopped == 1


def test_a_room_with_nothing_to_connect_to_ends_1() -> None:
    async def run() -> room.ClosedData:
        session = await support.enter(support.free_port(), support.Microphone(), support.Speaker())
        return await session.ended()

    closed = asyncio.run(run())

    assert (closed.reason, closed.exit_code) == ("abnormal", 1)


def test_a_sound_card_that_cannot_open_ends_the_session_1() -> None:
    async def run() -> room.ClosedData:
        house = await support.kitchen()
        try:
            session = await support.enter(house.port, support.Microphone(fails=True), support.Speaker())
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
            token=support.TOKENS.issue(support.PEER),
            microphone=support.Microphone(),
            speaker=support.Speaker(),
            on_closed=lambda closed: None,
        )

    for url, peer in (
        ("http://127.0.0.1:8765", support.PEER),
        ("ws://127.0.0.1:8765/v1", support.PEER),
        ("ws://h:1", "support.Kitchen!"),
    ):
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
    assert room.main(["--peer", support.PEER]) == 78
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
    stream.callback(memoryview(support.HEARD), len(support.HEARD) // 2, None, None)
    microphone.stop()

    assert stream.opened == {"samplerate": 16_000, "blocksize": 3_200, "channels": 1, "dtype": "int16"}
    assert heard == [support.HEARD]
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

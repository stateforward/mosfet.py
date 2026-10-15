"""Unit tests for ``speech_server.Transcription``: the streaming WebSocket wire as a ``hearing.speech.stream`` session.

The socket is a scripted fake: the test plays the server's frames and closes, and reads what the session sent.
"""

from __future__ import annotations

import asyncio
import collections.abc
import datetime
import json
import typing

import hsm
import pytest

from mosfet.abilities.hearing.speech import stream
from mosfet.define import define
from mosfet.protocols import attachment
from mosfet.providers import speech_server
from mosfet.providers.speech_server import transcription
from mosfet.start import started

_FRAME = 1_600  # 50 ms of s16le mono at 16 kHz


class FakeSocket:
    """A server, as the session sees it: frames the test plays, and everything the session sent."""

    def __init__(self) -> None:
        self.sent: list[str | bytes] = []
        self.closed: bool = False
        self._incoming: asyncio.Queue[str | bytes | transcription.Closed] = asyncio.Queue()

    def play(self, frame: dict[str, object] | str | bytes) -> None:
        self._incoming.put_nowait(json.dumps(frame) if isinstance(frame, dict) else frame)

    def drop(self, code: int | None = None) -> None:
        self._incoming.put_nowait(transcription.Closed(code))

    async def send(self, message: str | bytes) -> None:
        if self.closed:
            raise transcription.Closed(None)
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        item = await self._incoming.get()
        if isinstance(item, transcription.Closed):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True
        self._incoming.put_nowait(transcription.Closed(1000))

    def texts(self) -> list[dict[str, object]]:
        return [typing.cast(dict[str, object], json.loads(item)) for item in self.sent if isinstance(item, str)]

    def frames(self) -> list[bytes]:
        return [item for item in self.sent if isinstance(item, bytes)]


class Server:
    """The connect function a session takes: one fake socket per connection, all kept."""

    def __init__(self, *, refuse: int = 0) -> None:
        self.urls: list[str] = []
        self.sockets: list[FakeSocket] = []
        self._refuse: int = refuse

    async def __call__(self, url: str) -> transcription.Socket:
        self.urls.append(url)
        if self._refuse:
            self._refuse -= 1
            raise OSError("connection refused")
        socket = FakeSocket()
        self.sockets.append(socket)
        return socket


class Controller(hsm.Instance):
    """What a session's controller hears: every event it is sent."""

    events: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Controller", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)

    model: typing.ClassVar[hsm.Model] = define(
        "Controller",
        hsm.initial(hsm.target("/Controller/hearing")),
        hsm.state("hearing", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def outputs(self, event: hsm.Event[typing.Any]) -> list[object]:
        return [typing.cast(object, item.data) for item in self.events if item.name == event.name]


async def _wait(condition: collections.abc.Callable[[], bool], *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.002)
    raise AssertionError("condition not met")


_BACKOFF = (datetime.timedelta(seconds=1), datetime.timedelta(seconds=30))
_CONNECT_TIMEOUT = datetime.timedelta(seconds=5)


async def _attached(
    server: Server,
    *,
    backoff: tuple[datetime.timedelta, datetime.timedelta] = _BACKOFF,
    connect_timeout: datetime.timedelta = _CONNECT_TIMEOUT,
) -> tuple[speech_server.Transcription, Controller, hsm.Context]:
    ctx = hsm.Context()
    session = speech_server.Transcription(
        url="ws://stt.test:8001", connect=server, backoff=backoff, connect_timeout=connect_timeout
    )
    _ = await started(ctx, session, session.model)
    controller = Controller()
    _ = await started(ctx, controller, controller.model)
    _ = await session.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=controller), "a"))
    await _wait(lambda: bool(controller.outputs(attachment.AttachCompleteEvent)))
    return session, controller, ctx


async def _say(ctx: hsm.Context, session: speech_server.Transcription, event: hsm.Event[typing.Any]) -> None:
    _ = await hsm.dispatch(ctx, session, event)


def _audio(seconds: float, *, rate: int = 16_000, channels: int = 1) -> hsm.Event[stream.AudioData]:
    frames = round(seconds * rate)
    return stream.AudioEvent.with_data(
        stream.AudioData(audio=b"\x01\x00" * frames * channels, sample_rate_hz=rate, channels=channels)
    )


def _producing(producing: bool) -> hsm.Event[stream.ProducingData]:
    return stream.ProducingEvent.with_data(stream.ProducingData(producing=producing))


def test_the_session_streams_to_the_documented_endpoint() -> None:
    async def run() -> str:
        server = Server()
        session, _, _ = await _attached(server)
        await _wait(lambda: bool(server.urls))
        await hsm.stop(session)
        return server.urls[0]

    url = asyncio.run(run())
    assert url == "ws://stt.test:8001/v1/audio/stream?sample_rate=16000&language=en&chunk=0.56&turns=1"


def test_audio_waits_for_ready_and_is_sent_in_whole_50_ms_frames_of_16_khz_mono() -> None:
    async def run() -> tuple[FakeSocket, Controller]:
        server = Server()
        session, controller, ctx = await _attached(server)
        await _wait(lambda: bool(server.sockets))
        socket = server.sockets[0]
        await _say(ctx, session, _audio(0.06))  # held: the server is not ready
        await asyncio.sleep(0.02)
        assert socket.frames() == []
        socket.play({"type": "ready", "chunk_seconds": 0.56, "sample_rate": 16000, "format": "s16le", "turns": True})
        await _wait(lambda: bool(controller.outputs(stream.ReadyEvent)))
        # 48 kHz stereo is converted; 0.06 s held + 0.06 s now = 0.12 s = two 50 ms frames and a remainder.
        await _say(ctx, session, _audio(0.06, rate=48_000, channels=2))
        await _wait(lambda: len(socket.frames()) == 2)
        await hsm.stop(session)
        return socket, controller

    socket, controller = asyncio.run(run())
    assert [len(frame) for frame in socket.frames()] == [_FRAME, _FRAME]
    assert controller.outputs(stream.ReadyEvent) == [stream.ReadyData()]


def test_the_server_is_told_the_bot_is_speaking_only_on_change() -> None:
    async def run() -> FakeSocket:
        server = Server()
        session, controller, ctx = await _attached(server)
        await _say(ctx, session, _producing(True))  # before ready: told once ready
        await _wait(lambda: bool(server.sockets))
        socket = server.sockets[0]
        socket.play({"type": "ready"})
        await _wait(lambda: bool(controller.outputs(stream.ReadyEvent)))
        for producing in (True, True, False, False, True):
            await _say(ctx, session, _producing(producing))
        await _wait(lambda: len(socket.texts()) == 3)
        await asyncio.sleep(0.02)
        assert len(socket.texts()) == 3
        await hsm.stop(session)
        return socket

    socket = asyncio.run(run())
    assert [text for text in socket.texts() if text["type"] == "agent"] == [
        {"type": "agent", "speaking": True},
        {"type": "agent", "speaking": False},
        {"type": "agent", "speaking": True},
    ]


def test_every_documented_server_frame_maps_to_its_session_output_or_is_dropped() -> None:
    frames: list[dict[str, object]] = [
        {"type": "ready"},
        {"type": "speech_start", "t": 1.0},
        {"type": "partial", "text": "what is"},
        {"type": "eot", "t": 1.4, "p": 0.3, "last_word": "is"},
        {"type": "resumed", "t": 1.5},
        {
            "type": "final",
            "text": "what is the time",
            "audio_id": "x",
            "confidence": 0.91,
            "words": [{"text": "what", "t": 1.1}, {"text": "time", "t": 1.9}],
            "signals": {},
        },
        {"type": "speech_end", "t": 2.0},
        {
            "type": "turn_end",
            "t": 2.6,
            "reason": "likely",
            "p": 0.88,
            "resumed": False,
            "rescored": False,
            "speakers": [],
            "text": "what is the time",
        },
        {"type": "rejoin", "t": 3.0, "gap_s": 0.4, "agent_speaking": False},
        {"type": "speech_start", "t": 4.0},
        {"type": "interrupt", "t": 4.9},
    ]

    async def run() -> Controller:
        server = Server()
        session, controller, _ = await _attached(server)
        await _wait(lambda: bool(server.sockets))
        for frame in frames:
            server.sockets[0].play(frame)
        await _wait(lambda: bool(controller.outputs(stream.OverlapEvent)))
        await hsm.stop(session)
        return controller

    controller = asyncio.run(run())
    names = [event.name for event in controller.events if event.name.startswith("bot.ability.hearing.speech.stream.")]
    assert names == [
        stream.ReadyEvent.name,
        stream.VoiceStartedEvent.name,
        stream.PartialEvent.name,
        stream.TranscriptEvent.name,
        stream.VoiceEndedEvent.name,
        stream.TurnEndedEvent.name,
        stream.VoiceStartedEvent.name,
        stream.OverlapEvent.name,
    ]
    assert controller.outputs(stream.TranscriptEvent) == [
        stream.TranscriptData(text="what is the time", confidence=0.91, start_seconds=1.1, end_seconds=1.9)
    ]
    assert controller.outputs(stream.TurnEndedEvent) == [
        stream.TurnEndedData(at_seconds=2.6, probability=0.88, text="what is the time")
    ]
    (overlap,) = controller.outputs(stream.OverlapEvent)
    assert isinstance(overlap, stream.OverlapData)
    assert overlap.at_seconds == 4.9 and overlap.voiced_seconds == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("frame", "close", "kind"),
    [
        (None, 1013, "overloaded"),
        (None, 1003, "invalid_input"),
        (None, None, "unavailable"),
        ("not json", None, "invalid_response"),
        ({"type": "who knows"}, None, "invalid_response"),
        ({"type": "error", "message": "boom", "error": "x"}, None, "unavailable"),
    ],
)
def test_a_lost_server_is_a_typed_failure_then_a_backoff_and_a_reconnect(
    frame: str | dict[str, object] | None, close: int | None, kind: stream.FailureKind
) -> None:
    async def run() -> tuple[Controller, Server]:
        server = Server()
        session, controller, _ = await _attached(
            server, backoff=(datetime.timedelta(milliseconds=20), datetime.timedelta(milliseconds=40))
        )
        await _wait(lambda: bool(server.sockets))
        socket = server.sockets[0]
        socket.play({"type": "ready"})
        if frame is None:
            socket.drop(close)
        else:
            socket.play(frame)
        await _wait(lambda: len(server.sockets) == 2)
        await hsm.stop(session)
        return controller, server

    controller, server = asyncio.run(run())
    assert controller.outputs(stream.FailedEvent) == [stream.FailedData(kind=kind)]
    assert server.sockets[0].closed


def _turn(socket: FakeSocket) -> None:
    """One turn, in the server's own clock: every connection's clock starts at zero."""

    frames: tuple[dict[str, object], ...] = (
        {"type": "speech_start", "t": 0.5},
        {"type": "final", "text": "hello", "confidence": 0.9, "words": [{"text": "hello", "t": 0.6}]},
        {"type": "speech_end", "t": 0.8},
        {"type": "turn_end", "t": 1.0, "p": 0.9, "text": "hello"},
        {"type": "speech_start", "t": 1.2},
        {"type": "interrupt", "t": 2.1},
    )
    for frame in frames:
        socket.play(frame)


def _times(controller: Controller) -> list[tuple[str, float]]:
    times: list[tuple[str, float]] = []
    for event in controller.events:
        data = typing.cast(object, event.data)
        match data:
            case stream.VoiceStartedData() | stream.VoiceEndedData() | stream.TurnEndedData() | stream.OverlapData():
                times.append((event.name.rsplit(".", 1)[-1], data.at_seconds))
            case stream.TranscriptData():
                times.append(("transcript", data.start_seconds))
            case _:
                pass
    return times


def test_times_after_a_reconnect_are_seconds_of_the_audio_this_session_has_received() -> None:
    async def run() -> tuple[Controller, list[stream.OverlapData]]:
        server = Server()
        session, controller, ctx = await _attached(
            server, backoff=(datetime.timedelta(milliseconds=20), datetime.timedelta(milliseconds=40))
        )
        await _wait(lambda: bool(server.sockets))
        server.sockets[0].play({"type": "ready"})
        await _wait(lambda: bool(controller.outputs(stream.ReadyEvent)))
        await _say(ctx, session, _audio(3.0))
        server.sockets[0].drop()
        await _wait(lambda: bool(controller.outputs(stream.FailedEvent)))
        # Heard while reconnecting: held, then sent first on the new connection, whose clock starts at it.
        await _say(ctx, session, _audio(1.0))
        await _wait(lambda: len(server.sockets) == 2)
        server.sockets[1].play({"type": "ready"})
        await _wait(lambda: len(controller.outputs(stream.ReadyEvent)) == 2)
        _turn(server.sockets[1])
        await _wait(lambda: bool(controller.outputs(stream.OverlapEvent)))
        await hsm.stop(session)
        return controller, typing.cast(list[stream.OverlapData], controller.outputs(stream.OverlapEvent))

    controller, overlaps = asyncio.run(run())
    assert _times(controller) == [
        ("voice_started", pytest.approx(3.5)),
        ("transcript", pytest.approx(3.6)),
        ("voice_ended", pytest.approx(3.8)),
        ("turn_ended", pytest.approx(4.0)),
        ("voice_started", pytest.approx(4.2)),
        ("overlap", pytest.approx(5.1)),
    ]
    assert overlaps[0].voiced_seconds == pytest.approx(0.9)


def test_audio_let_go_while_the_server_is_not_ready_still_counts_toward_session_time() -> None:
    async def run() -> Controller:
        server = Server()
        session, controller, ctx = await _attached(server)
        await _wait(lambda: bool(server.sockets))
        # 3 s before ready: the oldest 1 s is let go, the newest 2 s are sent once the server is ready.
        for _ in range(6):
            await _say(ctx, session, _audio(0.5))
        server.sockets[0].play({"type": "ready"})
        await _wait(lambda: bool(controller.outputs(stream.ReadyEvent)))
        _turn(server.sockets[0])
        await _wait(lambda: bool(controller.outputs(stream.OverlapEvent)))
        await hsm.stop(session)
        return controller

    controller = asyncio.run(run())
    assert _times(controller) == [
        ("voice_started", pytest.approx(1.5)),
        ("transcript", pytest.approx(1.6)),
        ("voice_ended", pytest.approx(1.8)),
        ("turn_ended", pytest.approx(2.0)),
        ("voice_started", pytest.approx(2.2)),
        ("overlap", pytest.approx(3.1)),
    ]


class StuckSocket(FakeSocket):
    """A server that stops taking audio: every binary send waits forever, so the send queue fills."""

    @typing.override
    async def send(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            _ = await asyncio.Event().wait()
        await super().send(message)


def test_audio_let_go_past_the_send_queue_moves_later_server_times_by_its_length() -> None:
    async def run() -> Controller:
        sockets: list[FakeSocket] = []

        async def connect(url: str) -> transcription.Socket:
            del url
            sockets.append(StuckSocket())
            return sockets[-1]

        ctx = hsm.Context()
        session = speech_server.Transcription(url="ws://stt.test:8001", connect=connect)
        controller = Controller()
        for actor in (session, controller):
            _ = await started(ctx, actor, actor.model)
        _ = await session.attach(
            ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=controller), "a")
        )
        await _wait(lambda: bool(sockets))
        sockets[0].play({"type": "ready"})
        await _wait(lambda: bool(controller.outputs(stream.ReadyEvent)))
        # 0.1 s chunks: one is taken by the stuck send, 512 fill the queue, and the last 2 are let go.
        await _say(ctx, session, _audio(0.1))
        await asyncio.sleep(0.05)
        for _ in range(514):
            await _say(ctx, session, _audio(0.1))
        sockets[0].play({"type": "speech_start", "t": 50.0})
        sockets[0].play({"type": "speech_end", "t": 52.0})
        await _wait(lambda: bool(controller.outputs(stream.VoiceEndedEvent)))
        await hsm.stop(session)
        return controller

    controller = asyncio.run(run())
    # Server second 51.3 is where the queue was full: the 0.2 s let go there is session time the server missed.
    assert _times(controller) == [("voice_started", pytest.approx(50.0)), ("voice_ended", pytest.approx(52.2))]


class GatedSocket(FakeSocket):
    """A server that takes one audio frame each time the test lets it."""

    def __init__(self) -> None:
        super().__init__()
        self.gate: asyncio.Semaphore = asyncio.Semaphore(0)

    @typing.override
    async def send(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self.gate.acquire()
        await super().send(message)


async def _ready_on(
    socket: FakeSocket,
) -> tuple[speech_server.Transcription, Controller, hsm.Context]:
    async def connect(url: str) -> transcription.Socket:
        del url
        return socket

    ctx = hsm.Context()
    session = speech_server.Transcription(url="ws://stt.test:8001", connect=connect)
    controller = Controller()
    for actor in (session, controller):
        _ = await started(ctx, actor, actor.model)
    _ = await session.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=controller), "a"))
    await _wait(lambda: bool(controller.outputs(attachment.AttachCompleteEvent)))
    socket.play({"type": "ready"})
    await _wait(lambda: bool(controller.outputs(stream.ReadyEvent)))
    return session, controller, ctx


def test_a_long_stall_lets_go_of_audio_without_moving_the_times_the_server_heard_before_it() -> None:
    async def run() -> Controller:
        socket = StuckSocket()
        session, controller, ctx = await _ready_on(socket)
        # 0.1 s chunks: one is taken by the stuck send and 512 fill the queue (51.3 s); then 200 are let go (20 s).
        await _say(ctx, session, _audio(0.1))
        await asyncio.sleep(0.05)
        for _ in range(712):
            await _say(ctx, session, _audio(0.1))
        socket.play({"type": "speech_start", "t": 0.5})
        socket.play({"type": "speech_end", "t": 52.0})
        await _wait(lambda: bool(controller.outputs(stream.VoiceEndedEvent)))
        await hsm.stop(session)
        return controller

    controller = asyncio.run(run())
    assert _times(controller) == [("voice_started", pytest.approx(0.5)), ("voice_ended", pytest.approx(72.0))]


def test_audio_let_go_on_and_off_for_long_keeps_the_clock_bounded_and_its_origin_exact() -> None:
    async def run() -> Controller:
        socket = GatedSocket()
        session, controller, ctx = await _ready_on(socket)
        # 50 ms chunks (one frame each): one is taken by the held send and 512 fill the queue (25.65 s).
        for _ in range(513):
            await _say(ctx, session, _audio(0.05))
        # Then, 100 times over, one chunk is let go and the server takes a frame, making room for the next chunk.
        for taken in range(1, 101):
            await _say(ctx, session, _audio(0.05))
            socket.gate.release()
            await _wait(lambda: len(socket.frames()) == taken)
            await _say(ctx, session, _audio(0.05))
        # Before any audio was let go, server and session time agree; after the last, they are 5 s apart.
        socket.play({"type": "speech_start", "t": 0.5})
        socket.play({"type": "speech_end", "t": 25.0})
        socket.play({"type": "speech_start", "t": 30.62})
        await _wait(lambda: len(controller.outputs(stream.VoiceStartedEvent)) == 2)
        await hsm.stop(session)
        return controller

    controller = asyncio.run(run())
    assert _times(controller) == [
        ("voice_started", pytest.approx(0.5)),
        ("voice_ended", pytest.approx(25.0)),
        ("voice_started", pytest.approx(35.62)),
    ]


def test_an_unreachable_server_is_retried_with_a_growing_backoff() -> None:
    async def run() -> tuple[Controller, Server]:
        server = Server(refuse=2)
        session, controller, _ = await _attached(
            server, backoff=(datetime.timedelta(milliseconds=20), datetime.timedelta(milliseconds=40))
        )
        await _wait(lambda: len(server.sockets) == 1)
        await hsm.stop(session)
        return controller, server

    controller, server = asyncio.run(run())
    assert len(server.urls) == 3
    assert controller.outputs(stream.FailedEvent) == [stream.FailedData(kind="unavailable")] * 2


def test_a_server_that_never_says_ready_times_out() -> None:
    async def run() -> Controller:
        server = Server()
        session, controller, _ = await _attached(
            server,
            connect_timeout=datetime.timedelta(milliseconds=30),
            backoff=(datetime.timedelta(seconds=10), datetime.timedelta(seconds=10)),
        )
        await _wait(lambda: bool(controller.outputs(stream.FailedEvent)))
        await hsm.stop(session)
        return controller

    controller = asyncio.run(run())
    assert controller.outputs(stream.FailedEvent) == [stream.FailedData(kind="timeout")]


def test_detaching_says_end_and_closes_the_socket() -> None:
    async def run() -> FakeSocket:
        server = Server()
        session, controller, ctx = await _attached(server)
        await _wait(lambda: bool(server.sockets))
        socket = server.sockets[0]
        socket.play({"type": "ready"})
        await _wait(lambda: bool(controller.outputs(stream.ReadyEvent)))
        _ = await session.detach(
            ctx, attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=controller), "d")
        )
        await _wait(lambda: socket.closed)
        await _wait(lambda: bool(controller.outputs(attachment.DetachedEvent)))
        await hsm.stop(session)
        return socket

    socket = asyncio.run(run())
    assert socket.texts()[-1] == {"type": "end"}
    assert socket.closed


def test_a_second_controller_is_refused() -> None:
    async def run() -> Controller:
        server = Server()
        session, _, ctx = await _attached(server)
        other = Controller()
        _ = await started(ctx, other, other.model)
        _ = await session.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=other), "b"))
        await _wait(lambda: bool(other.outputs(attachment.AttachFailedEvent)))
        await hsm.stop(session)
        return other

    other = asyncio.run(run())
    assert other.outputs(attachment.AttachCompleteEvent) == []


def test_construction_refuses_what_cannot_stream() -> None:
    builds: list[collections.abc.Callable[[], speech_server.Transcription]] = [
        lambda: speech_server.Transcription(url="http://stt.test"),
        lambda: speech_server.Transcription(url="ws://stt.test", language=""),
        lambda: speech_server.Transcription(url="ws://stt.test", frame=datetime.timedelta(0)),
        lambda: speech_server.Transcription(
            url="ws://stt.test", backoff=(datetime.timedelta(seconds=2), datetime.timedelta(seconds=1))
        ),
    ]
    for build in builds:
        with pytest.raises(ValueError):
            _ = build()

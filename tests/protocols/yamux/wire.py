"""Test doubles around yamux: an actor that records what it is told, a gated owner, a pipe between two
sessions, a note codec, and readers for the frames a session or stream writes.

``Recorder`` stands in for a connection, a session (under a bare stream) or a stream owner: it keeps every
event delivered to it. ``Gate`` is an owner whose deliveries finish only when the test opens them, so a
test can see that window credit waits for the owner. ``Pipe`` is the connection on both ends of an
in-process socket: what one session transmits, the other receives, and each pipe hands a stream the peer
opened to the owner the test names.
"""

import asyncio
import collections.abc
import dataclasses
import struct
import typing

import hsm
import mosfet
import pydantic

from mosfet.protocols import transform, yamux


class NoteData(pydantic.BaseModel):
    """A short note one owner sends another over a stream."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    text: str


class OtherData(pydantic.BaseModel):
    """An event no test codec registers."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    text: str


NoteEvent = hsm.Event[NoteData](name="test.yamux.note", schema=NoteData)
OtherEvent = hsm.Event[OtherData](name="test.yamux.other", schema=OtherData)


def codec(max_message_bytes: int = 1024) -> transform.JsonCodec:
    return transform.JsonCodec(NoteEvent, max_message_bytes=max_message_bytes)


def note(text: str) -> hsm.Event[NoteData]:
    return NoteEvent.with_data(NoteData(text=text))


def framed(message: bytes) -> bytes:
    """``message`` as it travels on a stream: a 4-byte big-endian length, then the message."""

    return struct.pack("!I", len(message)) + message


def message(text: str) -> bytes:
    """The framed stream bytes carrying ``note(text)``."""

    return framed(codec().encode(note(text)))


class Recorder(hsm.Instance):
    """Keeps every event delivered to it, in order."""

    events: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Recorder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Recorder",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def named(self, event: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        return [received for received in self.events if received.name == event.name]

    def frames(self) -> list[yamux.FrameData]:
        """Every frame transmitted or written to this recorder, decoded, in order."""

        return [
            yamux.decode_frame(received.data.frame)[0]
            for received in self.events
            if isinstance(received.data, yamux.TransmitData | yamux.FrameOutData)
        ]

    def notes(self) -> list[str]:
        return [received.data.text for received in self.events if isinstance(received.data, NoteData)]

    def ended(self) -> list[str]:
        """Why each stream this owner owned ended, in order."""

        return [received.data.reason for received in self.events if isinstance(received.data, transform.EndedData)]

    def session_ended(self) -> list[str]:
        """Why each session this connection carried ended, in order."""

        return [received.data.reason for received in self.events if isinstance(received.data, yamux.EndedData)]

    def streams_ended(self) -> list[int]:
        """The ids of the streams that told this session they are over, in order."""

        return [
            received.data.stream_id
            for received in self.events
            if received.name == yamux.StreamEndedEvent.name and isinstance(received.data, yamux.StreamIdData)
        ]


class Gate(hsm.Instance):
    """An owner whose deliveries complete only when the test opens them."""

    events: list[hsm.Event[typing.Any]]
    _pending: list[asyncio.Future[bool]]

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self._pending = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        del ctx
        self.events.append(event)
        done: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending.append(done)
        return done

    def open(self) -> None:
        """Finish every delivery so far."""

        pending, self._pending = self._pending, []
        for done in pending:
            done.set_result(True)

    def notes(self) -> list[str]:
        return [received.data.text for received in self.events if isinstance(received.data, NoteData)]


class Pipe(hsm.Instance):
    """One end of an in-process socket between two sessions.

    What its session transmits is received by the peer session. A stream the peer opens is handed to
    ``owner`` with the note codec on ``channel``.
    """

    peer: yamux.Session | None
    owner: hsm.Instance | None
    ended: list[str]

    def __init__(self, *, owner: hsm.Instance | None = None) -> None:
        super().__init__()
        self.peer = None
        self.owner = owner
        self.ended = []

    @staticmethod
    def _carry(ctx: hsm.Context, instance: "Pipe", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, yamux.TransmitData)
        assert instance.peer is not None
        frame, rest = yamux.decode_frame(data.frame)
        assert rest == b""
        _ = hsm.dispatch(ctx, instance.peer, yamux.event_from_frame(frame))

    @staticmethod
    def _hand_over(ctx: hsm.Context, instance: "Pipe", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, yamux.StreamAcceptedData)
        assert instance.owner is not None
        _ = hsm.dispatch(
            ctx,
            data.stream,
            yamux.HandOverEvent.with_data(yamux.HandOverData(owner=instance.owner, codec=codec(), channel="notes")),
        )

    @staticmethod
    def _end(ctx: hsm.Context, instance: "Pipe", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, yamux.EndedData)
        instance.ended.append(data.reason)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Pipe",
        hsm.initial(hsm.target("carrying")),
        hsm.state(
            "carrying",
            hsm.transition(hsm.on(yamux.TransmitEvent), hsm.effect(_carry)),
            hsm.transition(hsm.on(yamux.StreamAcceptedEvent), hsm.effect(_hand_over)),
            hsm.transition(hsm.on(yamux.EndedEvent), hsm.effect(_end)),
        ),
    )


async def start[T: hsm.Instance](machine: T, machine_id: str) -> T:
    model = typing.cast(hsm.Model, getattr(type(machine), "model"))
    return await mosfet.started(None, machine, model, hsm.Config(id=machine_id))


async def settle(turns: int = 20) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


def from_peer[T](event: hsm.Event[T], sender: hsm.Instance) -> hsm.Event[T]:
    """``event`` as ``sender`` dispatches it: its source stamped with the sender's id."""

    return dataclasses.replace(event, source=hsm.id(sender))


def frame_event(frame: yamux.FrameData) -> hsm.Event[typing.Any]:
    return yamux.event_from_frame(frame)

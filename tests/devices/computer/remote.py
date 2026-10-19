"""Test doubles for a computer's far side: a transport, its streams, a controller, a holder, and an owner.

``FakeTransport`` is a transport as ``mosfet.protocols.transport`` describes one, in process: it answers a
link's ``PeerAttachData`` registration (or refuses it, or stays silent), keeps the channel table, records
the link's ``peer.accept`` / ``peer.close`` commands and its unregistration, and lets the test play the
remote peer by re-addressing ``peer.connected`` / ``peer.disconnected`` to the registered endpoint.

``FakeStream`` is one stream's host without yamux: it applies its codec between bytes the test plays in
and the owner, stamping ``source``/``target``/``id`` as a real host does, tells a duplex owner
``protocol.transform.owned`` / ``ended``, and keeps every event its owner sends it, encoded.
"""

import asyncio
import collections.abc
import dataclasses
import datetime
import struct
import typing
import uuid

import hsm
import mosfet
import pydantic
from pydantic.json_schema import SkipJsonSchema

from mosfet import media
from mosfet.device import Device, ObservedData, ObservedEvent
from mosfet.devices import computer, hid
from mosfet.environment import Environment
from mosfet.protocols import attachment, transform, transport
from mosfet.telemetry.hsm import Traced

DEVICE_ID = "studio-laptop"
PERSON = (mosfet.Participant(ref="p-1", role=mosfet.ParticipantRole.SOURCE),)
"""Who uses the computer the tests build (``person_id="p-1"``), as it stamps them: the source."""
DISPLAY_CODEC = transform.JsonCodec(
    computer.ContentEvent, computer.LayoutEvent, max_message_bytes=computer.MAX_MESSAGE_BYTES
)


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() >= deadline:
            raise TimeoutError("Timed out waiting for condition.")
        await asyncio.sleep(0)


async def settle(turns: int = 10) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


class PlayData(pydantic.BaseModel):
    """What the remote peer does, played by the test through its transport."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    event: SkipJsonSchema[pydantic.SkipValidation[hsm.Event[typing.Any]]]


PlayEvent = hsm.Event[PlayData](name="test.transport.play", schema=PlayData)


class FakeTransport(Traced, attachment.Attachment):
    """One transport, in process. ``register`` says how it answers a registration: answer, refuse, or silent."""

    _attachment_limit: typing.ClassVar[int | None] = 1
    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str
    requests: list[hsm.Event[typing.Any]]
    commands: list[hsm.Event[typing.Any]]
    peer_id: str
    channels: tuple[transport.ChannelData, ...]
    _register: str

    def __init__(self, *, register: str = "answer") -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=1)
        self._attachment_request_id = ""
        self.requests = []
        self.commands = []
        self.peer_id = ""
        self.channels = ()
        self._register = register

    @typing.override
    def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> collections.abc.Awaitable[bool]:
        return self.dispatch(ctx, event)

    @typing.override
    def detach(self, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> collections.abc.Awaitable[bool]:
        return self.dispatch(ctx, event)

    @staticmethod
    def _keep(ctx: hsm.Context, instance: "FakeTransport", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.requests.append(event)

    @staticmethod
    def _command(ctx: hsm.Context, instance: "FakeTransport", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.commands.append(event)

    @staticmethod
    def _answers(ctx: hsm.Context, instance: "FakeTransport", event: hsm.Event[typing.Any]) -> bool:
        return (
            instance._register == "answer"
            and isinstance(event.data, transport.PeerAttachData)
            and attachment.Attachment._can_attach(ctx, instance, event)
        )

    @staticmethod
    def _refuses(ctx: hsm.Context, instance: "FakeTransport", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._register == "refuse"

    @staticmethod
    def _bind(ctx: hsm.Context, instance: "FakeTransport", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, transport.PeerAttachData)
        instance.peer_id = data.peer_id
        instance.channels = data.channels

    @staticmethod
    def _play(ctx: hsm.Context, instance: "FakeTransport", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, PlayData)
        (endpoint,) = instance._attachments
        _ = hsm.dispatch(
            ctx,
            endpoint,
            dataclasses.replace(data.event, id=uuid.uuid4().hex, source=hsm.id(instance), target=hsm.id(endpoint)),
        )

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "FakeTransport",
        hsm.initial(hsm.target("serving")),
        hsm.state(
            "serving",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(_answers),
                hsm.effect(
                    _keep, attachment.Attachment._attach, _bind, attachment.Attachment._dispatch_attach_complete_created
                ),
            ),
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(_refuses),
                hsm.effect(_keep, attachment.Attachment._dispatch_attach_failed),
            ),
            hsm.transition(hsm.on(attachment.AttachEvent), hsm.effect(_keep)),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._detach_last_attachment),
                hsm.effect(
                    _keep, attachment.Attachment._detach, attachment.Attachment._dispatch_detach_complete_removed
                ),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.effect(_keep, attachment.Attachment._dispatch_detach_complete_absent),
            ),
            hsm.transition(hsm.on(transport.AcceptPeerEvent, transport.ClosePeerEvent), hsm.effect(_command)),
            hsm.transition(hsm.on(PlayEvent), hsm.effect(_play)),
        ),
    )

    def accepted(self) -> list[str]:
        """Connection ids the link accepted, in order."""

        return [event.data.connection_id for event in self.commands if isinstance(event.data, transport.AcceptPeerData)]

    def closed(self) -> list[tuple[str, str]]:
        """Connection ids the link let go, with why, in order."""

        return [
            (event.data.connection_id, event.data.reason)
            for event in self.commands
            if isinstance(event.data, transport.ClosePeerData)
        ]


def connected(transport_actor: FakeTransport, connection_id: str = "c1") -> hsm.Event[typing.Any]:
    """The remote peer connects, authenticated, on ``connection_id``."""

    return PlayEvent.with_data(
        PlayData(
            event=transport.PeerConnectedEvent.with_data(
                transport.PeerConnectedData(
                    peer_id=transport_actor.peer_id,
                    connection_id=connection_id,
                    transport="websocket",
                    connection=transport_actor,
                )
            )
        )
    )


def disconnected(
    transport_actor: FakeTransport,
    connection_id: str = "c1",
    reason: typing.Literal["left", "transport_lost", "closed"] = "left",
) -> hsm.Event[typing.Any]:
    """The connection ``connection_id`` is gone."""

    return PlayEvent.with_data(
        PlayData(
            event=transport.PeerDisconnectedEvent.with_data(
                transport.PeerDisconnectedData(
                    peer_id=transport_actor.peer_id, connection_id=connection_id, transport="websocket", reason=reason
                )
            )
        )
    )


class BytesData(pydantic.BaseModel):
    """One whole message the remote peer sends on a stream."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    message: bytes


class EndData(pydantic.BaseModel):
    """The stream ends."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    reason: typing.Literal["finished", "reset", "closed"] = "finished"


PeerSendEvent = hsm.Event[BytesData](name="test.stream.peer_send", schema=BytesData)
EndStreamEvent = hsm.Event[EndData](name="test.stream.end", schema=EndData)


class FakeStream(Traced):
    """One stream's host: decodes what the test plays in to its owner, encodes what its owner sends."""

    _owner: hsm.Instance
    _codec: transform.Codec
    _channel: str
    sent: list[hsm.Event[typing.Any]]
    messages: list[bytes]
    failures: list[str]

    def __init__(self, *, owner: hsm.Instance, codec: transform.Codec, channel: str) -> None:
        super().__init__()
        self._owner = owner
        self._codec = codec
        self._channel = channel
        self.sent = []
        self.messages = []
        self.failures = []

    @staticmethod
    def _tell_owner(ctx: hsm.Context, instance: "FakeStream", told: hsm.Event[typing.Any]) -> None:
        owner = instance._owner
        _ = hsm.dispatch(
            ctx, owner, dataclasses.replace(told, id=uuid.uuid4().hex, source=hsm.id(instance), target=hsm.id(owner))
        )

    @staticmethod
    def _owned(ctx: hsm.Context, instance: "FakeStream", event: hsm.Event[typing.Any]) -> None:
        del event
        if instance._codec.encodes_any:
            FakeStream._tell_owner(
                ctx,
                instance,
                transform.OwnedEvent.with_data(transform.OwnedData(stream=instance, channel=instance._channel)),
            )

    @staticmethod
    def _decode(ctx: hsm.Context, instance: "FakeStream", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, BytesData)
        try:
            decoded = instance._codec.decode(data.message)
        except transform.DecodeError as error:
            instance.failures.append(error.reason)
            return
        FakeStream._tell_owner(ctx, instance, decoded)

    @staticmethod
    def _ended(ctx: hsm.Context, instance: "FakeStream", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, EndData)
        if instance._codec.encodes_any:
            FakeStream._tell_owner(
                ctx,
                instance,
                transform.EndedEvent.with_data(
                    transform.EndedData(stream=instance, channel=instance._channel, reason=data.reason)
                ),
            )

    @staticmethod
    def _encodes(ctx: hsm.Context, instance: "FakeStream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return instance._codec.encodes(event)

    @staticmethod
    def _encode(ctx: hsm.Context, instance: "FakeStream", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.sent.append(event)
        instance.messages.append(instance._codec.encode(event))

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "FakeStream",
        hsm.initial(hsm.target("open")),
        hsm.state(
            "open",
            hsm.entry(_owned),
            hsm.transition(hsm.on(PeerSendEvent), hsm.effect(_decode)),
            hsm.transition(hsm.on(EndStreamEvent), hsm.effect(_ended), hsm.target("../ended")),
            hsm.transition(hsm.on(hsm.AnyEvent), hsm.guard(_encodes), hsm.effect(_encode)),
        ),
        hsm.state("ended"),
    )

    def named(self, event: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        return [sent for sent in self.sent if sent.name == event.name]


def peer_send(message: bytes) -> hsm.Event[typing.Any]:
    """The remote peer sends ``message`` on a stream."""

    return PeerSendEvent.with_data(BytesData(message=message))


def json_message(event: hsm.Event[typing.Any]) -> bytes:
    """``event`` as one JSON stream message."""

    return transform.JsonCodec(event, max_message_bytes=computer.MAX_MESSAGE_BYTES).encode(event)


def pcm_message(pcm: bytes, *, sample_rate_hz: int = 16_000, channels: int = 1) -> bytes:
    """``pcm`` as one PCM stream message."""

    return struct.pack("!IH", sample_rate_hz, channels) + pcm


def still_message(image: media.Image) -> bytes:
    """``image`` as one camera stream message."""

    media_type = image.media_type.encode("ascii")
    return bytes([len(media_type)]) + media_type + image.content


async def start(environment: Environment, machine: hsm.Instance, machine_id: str) -> None:
    model = typing.cast(hsm.Model, getattr(type(machine), "model", None))
    _ = await mosfet.started(environment, machine, model, hsm.Config(id=machine_id))


async def stream(environment: Environment, owner: hsm.Instance, codec: transform.Codec, channel: str) -> FakeStream:
    """Start a stream owned by ``owner``, named ``stream-<channel>``."""

    hosted = FakeStream(owner=owner, codec=codec, channel=channel)
    await start(environment, hosted, f"stream-{channel}")
    return hosted


async def open_streams(environment: Environment, transport_actor: FakeTransport) -> dict[str, FakeStream]:
    """Once the link accepted a connection, hand every registered channel's stream to its owner."""

    await wait_until(lambda: bool(transport_actor.accepted()))
    return {
        channel.name: await stream(environment, channel.owner, channel.codec, channel.name)
        for channel in transport_actor.channels
    }


class Controller(Device):
    """Stands in for whatever a port is wired to: keeps every event delivered to it."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[hsm.Event[typing.Any]] = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        del ctx
        self.events.append(event)
        done: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        done.set_result(True)
        return done

    def named(self, event: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        return [received for received in self.events if received.name == event.name]

    def attached(self) -> bool:
        return bool(self.named(attachment.AttachCompleteEvent))


class Holder(hsm.Instance):
    """Stands in for a bot holding a computer: keeps each ``bot.input`` it is told."""

    told: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.told = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Holder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.told.append(event)

    def observations(self) -> list[tuple[str, object]]:
        return [
            (event.data.observation.event, event.data.observation.data)
            for event in self.told
            if isinstance(event.data, mosfet.InputEventData) and event.data.observation is not None
        ]

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Holder",
        hsm.initial(hsm.target("holding")),
        hsm.state("holding", hsm.transition(hsm.on(mosfet.InputEvent), hsm.effect(_record))),
    )


async def wire(environment: Environment, port: Device, controller: hsm.Instance) -> None:
    """Attach ``controller`` to ``port`` the way firmware wires itself at bring-up."""

    _ = await hsm.dispatch(environment, port, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller)))


class Owner(hsm.Instance):
    """Stands in for the computer firmware is installed in: keeps each observation firmware hands it."""

    observed: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.observed = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Owner", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.observed.append(event)

    def observations(self) -> list[hsm.Event[typing.Any]]:
        return [event.data.observation for event in self.observed if isinstance(event.data, ObservedData)]

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Owner",
        hsm.initial(hsm.target("owning")),
        hsm.state("owning", hsm.transition(hsm.on(ObservedEvent), hsm.effect(_record))),
    )


def ports(device_id: str = DEVICE_ID) -> computer.Ports:
    """A computer's ports, owning no stream yet."""

    return computer.Ports(
        device_id=device_id,
        keyboard=hid.Keyboard(),
        pointer=hid.Pointer(),
        camera=computer.Camera(),
        microphone=computer.Microphone(),
        display=computer.Display(),
        speaker=computer.Speaker(),
        mouthpiece=computer.Mouthpiece(),
    )

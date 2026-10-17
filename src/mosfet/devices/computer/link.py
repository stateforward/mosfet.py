"""A computer's link: it registers the computer with a transport and decides which connection is current.

The link carries no message itself. Each of the computer's ports owns one stream, and whatever
carries that stream decodes into the port and encodes what the port sends (``protocol.transform``).
The link registers the computer's peer and its channel table — :func:`channels`, which port owns
which stream through which codec — with one transport, answers ``attachment.attach`` /
``attachment.detach`` for the computer's firmware, accepts one connection at a time (a newer one
supersedes it), and tells the firmware ``computer.link.up`` / ``computer.link.down``.
"""

from mosfet.devices import audio, hid

from .camera import Camera
from .codec import StillCodec
from .display import Display
from .events import (
    MAX_MESSAGE_BYTES,
    ContentEvent,
    LayoutEvent,
    LinkDownData,
    LinkDownEvent,
    LinkUpData,
    LinkUpEvent,
    PeerId,
)
from .microphone import Microphone
from .mouthpiece import Mouthpiece
from .speaker import Speaker

import collections.abc
import dataclasses
import datetime
import logging
import typing
import uuid

import hsm
import mosfet.device
import pydantic
from opentelemetry import metrics

from mosfet.define import define
from mosfet.protocols import attachment, transform, transport
from mosfet.telemetry import observer
from mosfet.telemetry.hsm import Traced

_DEFAULT_REGISTER_TIMEOUT = datetime.timedelta(seconds=10)

_LOG = logging.getLogger(__name__)
_DEVICE_ID: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(mosfet.device.DeviceId)
_PEER_ID: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(PeerId)
# Attribute: the transport's own low-cardinality name, stamped by the transport.
_SESSIONS = metrics.get_meter("bot.devices.computer").create_up_down_counter(
    "bot.devices.computer.link.session.active",
    unit="{session}",
    description="Remote machines a computer link holds linked right now, by transport.",
)
_DISCONNECT_REASONS: dict[str, typing.Literal["device_left", "transport_lost"]] = {
    "left": "device_left",
    "transport_lost": "transport_lost",
    "closed": "transport_lost",
}


@dataclasses.dataclass(frozen=True)
class Ports:
    """One computer's ports, by role.

    Six are remote-side, each owning one stream of the link — keyboard, pointer, camera and
    microphone inbound, display both ways, speaker outbound — and the mouthpiece is bot-side, wired
    only to the computer's firmware.

    Raises:
        pydantic.ValidationError: If ``device_id`` is not a valid ``mosfet.device.DeviceId``.
    """

    device_id: mosfet.device.DeviceId
    keyboard: hid.Keyboard
    pointer: hid.Pointer
    camera: Camera
    microphone: Microphone
    display: Display
    speaker: Speaker
    mouthpiece: Mouthpiece

    def __post_init__(self) -> None:
        _ = _DEVICE_ID.validate_python(self.device_id)


def channels(ports: Ports) -> tuple[transport.ChannelData, ...]:
    """The computer's wire: which port owns each channel's stream, through which codec.

    ``keyboard``, ``pointer`` and ``display`` carry JSON messages of at most :data:`MAX_MESSAGE_BYTES`
    (display content out, layout in); ``camera`` carries binary stills; ``microphone`` and ``speaker``
    carry binary PCM. The mouthpiece has no channel: it is in the bot's room, not on the wire.
    """

    pcm = audio.PcmCodec()
    return (
        transport.ChannelData(
            name="keyboard",
            owner=ports.keyboard,
            codec=transform.JsonCodec(hid.KeyEvent, max_message_bytes=MAX_MESSAGE_BYTES),
            kind="data",
        ),
        transport.ChannelData(
            name="pointer",
            owner=ports.pointer,
            codec=transform.JsonCodec(hid.PointerEvent, max_message_bytes=MAX_MESSAGE_BYTES),
            kind="data",
        ),
        transport.ChannelData(
            name="display",
            owner=ports.display,
            codec=transform.JsonCodec(ContentEvent, LayoutEvent, max_message_bytes=MAX_MESSAGE_BYTES),
            kind="data",
        ),
        transport.ChannelData(name="camera", owner=ports.camera, codec=StillCodec(), kind="data"),
        transport.ChannelData(name="microphone", owner=ports.microphone, codec=pcm, kind="audio"),
        transport.ChannelData(name="speaker", owner=ports.speaker, codec=pcm, kind="audio"),
    )


class Link(Traced, attachment.Attachment):
    """One computer's link to its remote machine, over whichever transport it is given.

    On start it registers with ``transport`` for ``peer_id``, handing over the channel table, and
    holds the firmware's ``attachment.attach`` / ``attachment.detach`` until the transport answers. A
    refusal, or no answer within ``register_timeout``, leaves the link refused: every attach is then
    answered ``attach.failed``. Registered, it attaches one firmware. A connection the transport
    reports while a firmware is attached is accepted (``protocol.transport.peer.accept``), and only
    then does the transport hand its streams to the ports; the link is up. A newer connection
    supersedes the current one (``peer.close{superseded}``, link down, then up). A connection
    reported while no firmware is attached is let go (``peer.close{released}``). Stopping the link
    unregisters it.

    The link never reads a message, a port, or a stream: identity is the registration and the
    transport's stamp.
    """

    _attachment_limit: typing.ClassVar[int | None] = 1
    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str

    # Addresses: dispatched to, never read.
    _transport: hsm.Instance
    _peer_id: str
    _device_id: str
    _channels: tuple[transport.ChannelData, ...]
    _register_timeout: datetime.timedelta
    # The current connection; written only by _bind_connection / _unbind_connection.
    _connection: hsm.Instance | None
    _connection_id: str | None
    _transport_name: str | None

    def __init__(
        self,
        *,
        transport: hsm.Instance,
        peer_id: str,
        device_id: str,
        channels: collections.abc.Iterable[transport.ChannelData],
        register_timeout: datetime.timedelta = _DEFAULT_REGISTER_TIMEOUT,
    ) -> None:
        """Build an unstarted link for ``device_id``'s remote machine, known to ``transport`` as ``peer_id``.

        Raises:
            ValueError: If ``register_timeout`` is not positive, ``peer_id`` is not a valid
                :data:`PeerId`, ``device_id`` is not a valid ``mosfet.device.DeviceId``, or ``channels``
                names a channel twice (``pydantic.ValidationError`` is a ``ValueError``).
        """

        super().__init__()
        if register_timeout <= datetime.timedelta():
            raise ValueError("register_timeout must be a positive duration.")
        table = tuple(channels)
        names = [channel.name for channel in table]
        if len(names) != len(set(names)):
            raise ValueError("Link channels must name each channel once.")
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self._transport = transport
        self._peer_id = _PEER_ID.validate_python(peer_id)
        self._device_id = _DEVICE_ID.validate_python(device_id)
        self._channels = table
        self._register_timeout = register_timeout
        self._connection = None
        self._connection_id = None
        self._transport_name = None

    @typing.override
    def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> collections.abc.Awaitable[bool]:
        """Deliver one ``attachment.attach`` request; resolves to whether it was delivered.

        The answer goes to the request's ``reply_to`` (else its ``actor``), never a return value.
        """

        return self.dispatch(ctx, event)

    @typing.override
    def detach(self, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> collections.abc.Awaitable[bool]:
        """Deliver one ``attachment.detach`` request; resolves to whether it was delivered."""

        return self.dispatch(ctx, event)

    # -- registration --------------------------------------------------------------------------

    @staticmethod
    def _register(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> None:
        """Ask the transport to carry this peer, with the channel table its streams are handed to."""

        target = instance._transport
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                attachment.AttachEvent.with_data(
                    transport.PeerAttachData(
                        actor=instance,
                        peer_id=instance._peer_id,
                        channels=instance._channels,
                        reply_to=instance,
                        timeout=instance._register_timeout,
                    )
                ),
                id=uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(target),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _unregister(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> None:
        """Let the transport forget this peer; whatever it answers, the link is going."""

        target = instance._transport
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                attachment.DetachEvent.with_data(attachment.DetachData(actor=instance)),
                id=uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(target),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _from_transport(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> bool:
        """Correlate a registration answer with the one transport this link registered with."""

        del ctx
        return event.source == hsm.id(instance._transport)

    @staticmethod
    def _register_timeout_value(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._register_timeout

    @staticmethod
    def _log_refused(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        data = event.data
        kind = data.kind if isinstance(data, attachment.FailedData) else "timeout"
        _LOG.warning("computer link registration refused by its transport kind=%s", kind)

    # -- firmware ------------------------------------------------------------------------------

    @staticmethod
    def _accepts_firmware(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> bool:
        return isinstance(event.data, attachment.AttachData) and attachment.Attachment._can_attach(ctx, instance, event)

    @staticmethod
    def _tell_firmware(
        ctx: hsm.Context, instance: "Link", trigger: hsm.Event[typing.Any], told: hsm.Event[typing.Any]
    ) -> None:
        firmware = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            firmware,
            dataclasses.replace(
                told,
                id=trigger.id or uuid.uuid4().hex,
                source=hsm.id(instance),
                target=attachment.Attachment._actor_id(firmware),
                metadata=dict(trigger.metadata),
            ),
        )

    # -- connections ---------------------------------------------------------------------------

    @staticmethod
    def _bind_connection(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> None:
        """Take the reported connection as current and accept it, so its streams reach the ports."""

        data = event.data
        assert isinstance(data, transport.PeerConnectedData)
        instance._connection = data.connection
        instance._connection_id = data.connection_id
        instance._transport_name = data.transport
        _ = hsm.dispatch(
            ctx,
            data.connection,
            dataclasses.replace(
                transport.AcceptPeerEvent.with_data(
                    transport.AcceptPeerData(peer_id=data.peer_id, connection_id=data.connection_id)
                ),
                id=uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(data.connection),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _unbind_connection(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._connection = None
        instance._connection_id = None
        instance._transport_name = None

    @staticmethod
    def _close(
        ctx: hsm.Context,
        instance: "Link",
        trigger: hsm.Event[typing.Any],
        connection: hsm.Instance,
        peer_id: str,
        connection_id: str,
        reason: typing.Literal["superseded", "released"],
    ) -> None:
        _ = hsm.dispatch(
            ctx,
            connection,
            dataclasses.replace(
                transport.ClosePeerEvent.with_data(
                    transport.ClosePeerData(peer_id=peer_id, connection_id=connection_id, reason=reason)
                ),
                id=uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(connection),
                metadata=dict(trigger.metadata),
            ),
        )

    @staticmethod
    def _close_reported(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> None:
        """Let go of a connection nothing is attached to receive."""

        data = event.data
        assert isinstance(data, transport.PeerConnectedData)
        Link._close(ctx, instance, event, data.connection, data.peer_id, data.connection_id, "released")

    @staticmethod
    def _close_current(
        ctx: hsm.Context,
        instance: "Link",
        event: hsm.Event[typing.Any],
        reason: typing.Literal["superseded", "released"],
    ) -> None:
        connection, connection_id = instance._connection, instance._connection_id
        assert connection is not None and connection_id is not None
        Link._close(ctx, instance, event, connection, instance._peer_id, connection_id, reason)

    @staticmethod
    def _is_other_connection(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, transport.PeerConnectedData) and data.connection_id != instance._connection_id

    @staticmethod
    def _is_current_connection(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, transport.PeerDisconnectedData) and data.connection_id == instance._connection_id

    # -- link up / down ------------------------------------------------------------------------

    @staticmethod
    def _link_up(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> None:
        """Tell the firmware the remote machine is connected, over the transport that reported it."""

        transport_name = instance._transport_name
        assert transport_name is not None
        _SESSIONS.add(1, {"transport": transport_name})
        up = LinkUpData.model_validate({"device_id": instance._device_id, "transport": transport_name})
        Link._tell_firmware(ctx, instance, event, LinkUpEvent.with_data(up))

    @staticmethod
    def _link_closed(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        transport_name = instance._transport_name
        assert transport_name is not None
        _SESSIONS.add(-1, {"transport": transport_name})

    @staticmethod
    def _link_down(
        ctx: hsm.Context,
        instance: "Link",
        event: hsm.Event[typing.Any],
        reason: typing.Literal["device_left", "transport_lost", "superseded", "detached"],
    ) -> None:
        down = LinkDownData(device_id=instance._device_id, reason=reason)
        Link._tell_firmware(ctx, instance, event, LinkDownEvent.with_data(down))

    @staticmethod
    def _supersede(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> None:
        """The newer connection wins: let the current one go and tell the firmware it went down."""

        Link._close_current(ctx, instance, event, "superseded")
        Link._link_down(ctx, instance, event, "superseded")

    @staticmethod
    def _disconnected(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, transport.PeerDisconnectedData)
        Link._link_down(ctx, instance, event, _DISCONNECT_REASONS[data.reason])

    @staticmethod
    def _release(ctx: hsm.Context, instance: "Link", event: hsm.Event[typing.Any]) -> None:
        """The firmware is going: the link goes down and the current connection is let go."""

        Link._link_down(ctx, instance, event, "detached")
        Link._close_current(ctx, instance, event, "released")

    model: typing.ClassVar[hsm.Model] = define(
        "ComputerLink",
        hsm.initial(hsm.target("registering")),
        hsm.state(
            "registering",
            hsm.entry(_register),
            # The firmware's requests wait for the transport's answer.
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.transition(
                hsm.on(attachment.AttachCompleteEvent),
                hsm.guard(_from_transport),
                hsm.target("../registered"),
            ),
            hsm.transition(
                hsm.on(attachment.AttachFailedEvent),
                hsm.guard(_from_transport),
                hsm.effect(_log_refused),
                hsm.target("../refused"),
            ),
            hsm.transition(
                hsm.after(_register_timeout_value),
                hsm.effect(_log_refused),
                hsm.target("../refused"),
            ),
        ),
        # Nothing carries this computer: the root answers every attach failed and every detach absent.
        hsm.state("refused"),
        hsm.state(
            "registered",
            # Runs on hsm.stop as well, which exits the active configuration.
            hsm.exit(_unregister),
            hsm.initial(hsm.target("detached")),
            hsm.state(
                "detached",
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(_accepts_firmware),
                    hsm.effect(attachment.Attachment._attach, attachment.Attachment._dispatch_attach_complete_created),
                    hsm.target("../awaiting_peer"),
                ),
                hsm.transition(hsm.on(transport.PeerConnectedEvent), hsm.effect(_close_reported)),
            ),
            hsm.state(
                "awaiting_peer",
                hsm.transition(
                    hsm.on(transport.PeerConnectedEvent),
                    hsm.effect(_bind_connection),
                    hsm.target("../linked"),
                ),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(attachment.Attachment._detach_last_attachment),
                    hsm.effect(attachment.Attachment._detach, attachment.Attachment._dispatch_detach_complete_removed),
                    hsm.target("../detached"),
                ),
            ),
            hsm.state(
                "linked",
                hsm.entry(_link_up),
                hsm.exit(_link_closed),
                hsm.transition(
                    # External self-transition: exit and entry run again, so down, then up.
                    hsm.on(transport.PeerConnectedEvent),
                    hsm.guard(_is_other_connection),
                    hsm.effect(_supersede, _bind_connection),
                    hsm.target("../linked"),
                ),
                hsm.transition(
                    hsm.on(transport.PeerDisconnectedEvent),
                    hsm.guard(_is_current_connection),
                    hsm.effect(_disconnected, _unbind_connection),
                    hsm.target("../awaiting_peer"),
                ),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(attachment.Attachment._detach_last_attachment),
                    hsm.effect(
                        _release,
                        _unbind_connection,
                        attachment.Attachment._detach,
                        attachment.Attachment._dispatch_detach_complete_removed,
                    ),
                    hsm.target("../detached"),
                ),
            ),
        ),
        hsm.transition(
            hsm.on(attachment.AttachEvent),
            hsm.guard(attachment.Attachment._is_attached),
            hsm.effect(attachment.Attachment._dispatch_attach_complete_existing),
        ),
        hsm.transition(
            hsm.on(attachment.AttachEvent),
            hsm.guard(attachment.Attachment._is_attach_request),
            hsm.effect(attachment.Attachment._dispatch_attach_failed),
        ),
        hsm.transition(
            hsm.on(attachment.DetachEvent),
            hsm.effect(attachment.Attachment._dispatch_detach_complete_absent),
        ),
        hsm.observe(observer),
    )
    """Registering, then refused or registered: detached, awaiting a peer, or linked to one connection."""


__all__ = ["Link", "Ports", "channels"]

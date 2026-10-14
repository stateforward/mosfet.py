"""Carry one remote peer's channels over a LiveKit room: a device-agnostic transport.

One room per transport, one registered peer per room. The bot joins the room as one participant
and the peer (a browser, a kiosk) joins it as another. ``RoomTransport`` owns the room's bytes
for that peer and knows nothing of what they mean: an endpoint registers for the peer with
``attachment.attach`` carrying a ``transport.PeerAttachData`` (the peer's identity and its channel
table), and every channel's owner and codec come from that table.

Each connection attempt gets a fresh room handle and a fresh ``RoomAudioTrackPath`` on it. The path
owns the room connection (connect, the published local audio track, disconnect); the transport
reaches it only by ``RoomAudioConnectEvent`` / ``RoomAudioDisconnectEvent``, its
``connection_sink``, and ``hsm.stop``. A path that ends in its dead ``failed`` leaf is stopped and
discarded, never reused.

Wire:

* a data channel is a topic named after the channel: one reliable data packet, addressed to the
  peer only, or one byte stream (used outbound for a message over 14 KiB) is one codec message;
* an audio channel inbound is the peer's audio track whose publication is named after the channel,
  outbound the path's published local track. Its messages are a 6-byte big-endian ``!IH`` header
  (sample rate in hertz, channel count) followed by signed 16-bit little-endian PCM.

SDK callbacks are the admission boundary: the sender must be the registered peer, the topic a
registered data channel, and the size within ``max_message_bytes`` before anything is decoded. A
refused message is counted (``bot.transport.rejected.count{reason}``) and dispatches nothing.
"""

from __future__ import annotations

from mosfet.devices import audio

import asyncio
import collections.abc
import dataclasses
import datetime
import logging
import struct
import typing
import uuid
import weakref

import hsm
import pydantic
from livekit import rtc
from opentelemetry import metrics

from mosfet.define import define
from mosfet.media import MAX_IMAGE_BYTES
from mosfet.protocols import attachment, transform, transport
from mosfet.start import started
from mosfet.telemetry import observer, span
from mosfet.telemetry.hsm import Traced

from .audio import AudioBridge, create_audio_bridge
from .pcm_batch import RemotePcmBatcher
from .room_audio import (
    AudioStreamFactory,
    LocalAudioTrackFactory,
    LocalParticipant,
    RoomAudioConnectData,
    RoomAudioConnectEvent,
    RoomAudioConnectedData,
    RoomAudioDisconnectEvent,
    RoomAudioTrackPath,
)

_LOG = logging.getLogger(__name__)
_SCOPE = "mosfet.providers.livekit"
_COMPONENT = "livekit.transport"
_TRANSPORT = "livekit"

# LiveKit splits data packets above ~15 KB; a message over this goes as a byte stream instead.
_MAX_PACKET_BYTES = 14 * 1024
# Bounds on what one attempt holds (PY-MEM-001): unread byte streams, unsent messages, and the
# decoded bytes per channel an owner has not yet processed.
_STREAM_QUEUE_LIMIT = 4
_OUTBOUND_QUEUE_LIMIT = 256
_IN_FLIGHT_LIMIT = 256 * 1024
_PCM_HEADER = struct.Struct("!IH")
_PCM_MEDIA_TYPE = "audio/pcm"

_DEFAULT_CONNECT_TIMEOUT = datetime.timedelta(seconds=10)
_DEFAULT_RELEASE_TIMEOUT = datetime.timedelta(seconds=5)
_DEFAULT_RECONNECT_INITIAL = datetime.timedelta(seconds=1)
_DEFAULT_RECONNECT_MAX = datetime.timedelta(seconds=30)
_DEFAULT_MAX_MESSAGE_BYTES = MAX_IMAGE_BYTES + 64 * 1024
# 2**30 initial delays is already far past any reconnect_max; capping the exponent keeps the
# timedelta product finite however long the transport keeps retrying.
_MAX_BACKOFF_EXPONENT = 30

# Transport-neutral instruments, told apart by ``transport``. Every attribute is a closed set from
# code; identities, peer ids, channel names, room names, tokens and content are never attributes.
_METER = metrics.get_meter(_SCOPE)
_PEERS = _METER.create_up_down_counter(
    "bot.transport.peer.active",
    unit="{peer}",
    description="Remote peers a transport holds reachable right now, by transport.",
)
_CONNECTS = _METER.create_counter(
    "bot.transport.connect.count",
    unit="{attempt}",
    description="Transport connection attempts, by transport and outcome: connected, failed, or timed_out.",
)
_REJECTED = _METER.create_counter(
    "bot.transport.rejected.count",
    unit="{message}",
    description="Inbound messages a transport refused at its boundary, by transport and reason.",
)
_TRANSFERRED = _METER.create_counter(
    "bot.transport.transfer.size",
    unit="By",
    description="Message bytes a transport carried, by transport and direction.",
)
_SEND_DROPPED = _METER.create_counter(
    "bot.transport.send.dropped.count",
    unit="{message}",
    description="Outbound messages a transport did not send, by transport and reason.",
)

type _Outcome = typing.Literal["connected", "failed", "timed_out"]
type _Rejection = typing.Literal["wrong_peer", "refused_channel", "oversize", "backlog"]
type _SendDrop = typing.Literal["oversize", "backlog"]
type _Outbound = tuple[str, transport.ChannelKind, bytes]


class ByteStreamWriter(typing.Protocol):
    """The writer ``stream_bytes`` hands back: write the bytes, then close the stream."""

    def write(self, data: bytes) -> collections.abc.Awaitable[None]:
        """Write one chunk of the stream."""
        ...

    def aclose(self) -> collections.abc.Awaitable[None]:
        """Close the stream; the receiver sees it complete."""
        ...


class DataParticipant(LocalParticipant, typing.Protocol):
    """The bot's LiveKit participant, as a room transport uses it: the room path's, plus data out."""

    def publish_data(
        self,
        payload: bytes | str,
        *,
        reliable: bool = True,
        destination_identities: list[str] = ...,
        topic: str = "",
    ) -> collections.abc.Awaitable[None]:
        """Send one data packet to the named participants on ``topic``."""
        ...

    def stream_bytes(
        self,
        name: str,
        *,
        total_size: int | None = None,
        mime_type: str = "application/octet-stream",
        attributes: dict[str, str] | None = None,
        stream_id: str | None = None,
        destination_identities: list[str] | None = None,
        topic: str = "",
    ) -> collections.abc.Awaitable[ByteStreamWriter]:
        """Open a byte stream to the named participants on ``topic``."""
        ...


type ByteStreamHandler = collections.abc.Callable[[typing.Any, str], None]


class DataRoom(typing.Protocol):
    """The LiveKit room shape a room transport binds: its room path's ``RoomHandle``, plus data and presence.

    Everything here exists on ``livekit.rtc.Room`` 1.1.10, and a ``DataRoom`` is a
    ``room_audio.RoomHandle``. The transport never calls ``connect`` or ``disconnect`` itself: those
    belong to the attempt's room path.
    """

    local_participant: DataParticipant
    remote_participants: collections.abc.Mapping[str, object]

    def connect(self, url: str, token: str) -> collections.abc.Awaitable[None]:
        """Connect to a LiveKit room (called by the room path only)."""
        ...

    def disconnect(self) -> collections.abc.Awaitable[None]:
        """Disconnect from a LiveKit room (called by the room path only)."""
        ...

    def on(
        self,
        event: str,
        callback: collections.abc.Callable[..., object] | None = None,
    ) -> collections.abc.Callable[..., object]:
        """Register a LiveKit room event callback."""
        ...

    def off(self, event: str, callback: collections.abc.Callable[..., object]) -> None:
        """Unregister a LiveKit room event callback."""
        ...

    def register_byte_stream_handler(self, topic: str, handler: ByteStreamHandler) -> None:
        """Hand each incoming byte stream on ``topic`` to ``handler`` with its sender's identity."""
        ...

    def unregister_byte_stream_handler(self, topic: str) -> None:
        """Stop taking byte streams on ``topic``."""
        ...


type RoomFactory = collections.abc.Callable[[], DataRoom]
type BridgeFactory = collections.abc.Callable[[], AudioBridge[rtc.AudioFrame]]


class _ByteStreamInfo(typing.Protocol):
    size: int | None


class _ByteStreamReader(typing.Protocol):
    info: _ByteStreamInfo

    def __aiter__(self) -> collections.abc.AsyncIterator[bytes]: ...


class _DataPacket(typing.Protocol):
    data: bytes
    participant: object | None
    topic: str | None


class _AttemptData(pydantic.BaseModel):
    """Which connection attempt an observation belongs to; a stale attempt's is ignored."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    attempt: int = pydantic.Field(ge=1, description="The transport's attempt counter when the observation was made.")


class _PeerLeftData(_AttemptData):
    """The registered peer left the room."""


class _TrackData(_AttemptData):
    channel: str = pydantic.Field(description="Publication name of the peer's subscribed audio track.")


class _MessageData(_AttemptData):
    channel: str = pydantic.Field(description="The registered data channel (topic) the message arrived on.")
    message: bytes = pydantic.Field(description="One admitted inbound message, not yet decoded.")


class _AudioReceivedData(_AttemptData):
    chunk: audio.InputData = pydantic.Field(description="Batched PCM from the peer's audio track.")


class _DeliveredData(pydantic.BaseModel):
    """An owner finished processing one decoded message: its bytes are no longer in flight."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    connection_id: str = pydantic.Field(description="The connection the message was delivered on.")
    channel: str = pydantic.Field(description="The channel whose in-flight count drops.")
    size: int = pydantic.Field(ge=0, description="Bytes of the delivered message.")


# Private: each is dispatched by this class's own closures, activities or callbacks to itself only.
_PathConnectedEvent = hsm.Event[_AttemptData](name="bot.provider.livekit.transport.path.connected", schema=_AttemptData)
_PathDisconnectedEvent = hsm.Event[_AttemptData](
    name="bot.provider.livekit.transport.path.disconnected", schema=_AttemptData
)
_PathStartFailedEvent = hsm.Event[_AttemptData](
    name="bot.provider.livekit.transport.path.start_failed", kind=hsm.ErrorEventKind, schema=_AttemptData
)
_AttemptReleasedEvent = hsm.Event[_AttemptData](
    name="bot.provider.livekit.transport.attempt.released", kind=hsm.CompletionEventKind, schema=_AttemptData
)
_PeerPresentEvent = hsm.Event[_AttemptData](name="bot.provider.livekit.transport.peer.present", schema=_AttemptData)
_PeerLeftEvent = hsm.Event[_PeerLeftData](name="bot.provider.livekit.transport.peer.left", schema=_PeerLeftData)
_RoomDownEvent = hsm.Event[_AttemptData](name="bot.provider.livekit.transport.room.down", schema=_AttemptData)
_TrackEvent = hsm.Event[_TrackData](name="bot.provider.livekit.transport.track", schema=_TrackData)
_MessageEvent = hsm.Event[_MessageData](name="bot.provider.livekit.transport.message", schema=_MessageData)
_AudioReceivedEvent = hsm.Event[_AudioReceivedData](
    name="bot.provider.livekit.transport.audio", schema=_AudioReceivedData
)
_DeliveredEvent = hsm.Event[_DeliveredData](name="bot.provider.livekit.transport.delivered", schema=_DeliveredData)


def _reject(reason: _Rejection) -> None:
    _REJECTED.add(1, {"transport": _TRANSPORT, "reason": reason})


def _drop_send(reason: _SendDrop) -> None:
    _SEND_DROPPED.add(1, {"transport": _TRANSPORT, "reason": reason})


def _transferred(size: int, direction: typing.Literal["in", "out"]) -> None:
    _TRANSFERRED.add(size, {"transport": _TRANSPORT, "direction": direction})


def _record_connect(outcome: _Outcome) -> None:
    """Count one attempt's outcome and leave a span for it; the outcome is the only attribute."""

    _CONNECTS.add(1, {"transport": _TRANSPORT, "outcome": outcome})
    with span.operation(
        "transport.connect",
        scope=_SCOPE,
        component=_COMPONENT,
        stage="connect",
        attributes={"bot.transport.name": _TRANSPORT, "bot.transport.outcome": outcome},
    ) as active:
        if outcome != "connected":
            span.record_failure(active, outcome)


def _identity(participant: object) -> str | None:
    identity = typing.cast(object, getattr(participant, "identity", None))
    return identity if isinstance(identity, str) else None


def _dispatch_own(owner: weakref.ref[RoomTransport], event: hsm.Event[typing.Any]) -> None:
    """Hand an SDK observation to the transport that registered for it, if it is still there."""

    live = owner()
    if live is None:
        return
    _ = hsm.dispatch(live.context(), live, event)


class RoomTransport(Traced, attachment.Attachment):
    """One remote peer's channels over a LiveKit room of its own.

    It answers ``attachment.attach`` for one endpoint registering one peer with a
    ``transport.PeerAttachData`` (a plain ``attachment.AttachData`` is refused, as is a second
    endpoint: one registered peer per room). It answers as soon as the peer is bound and the first
    connection attempt begins, and ``attachment.detach`` as soon as the binding is removed; releasing
    the room continues inside the transport, bounded by ``release_timeout``.

    Whenever the peer is in the room it tells the endpoint ``protocol.transport.peer.connected``
    (``connection_id`` unique per attempt and presence, ``connection`` this transport) and, exactly
    once for each, ``protocol.transport.peer.disconnected``: ``left`` when the peer leaves,
    ``closed`` when the endpoint closes it or unregisters, ``transport_lost`` otherwise. Nothing is
    handed to the channel owners until the endpoint accepts that connection
    (``protocol.transport.peer.accept``). Accepted, each owner whose codec encodes anything is told
    ``protocol.transform.owned`` with this transport as its stream, and ``protocol.transform.ended``
    when the connection goes; inbound messages decode through their channel's codec to their
    channel's owner, and whatever an owner sends here goes out on the channel it owns whose codec
    encodes it. An owner holds at most 256 KiB of decoded messages it has not finished processing per
    channel; past that, messages are dropped and counted (``flow_control``). Closing a connection
    (``protocol.transport.peer.close``) leaves the room, since a participant cannot evict another, and
    joins again after the backoff, where the peer is offered as a new connection.

    A lost room, a path that tears itself down, or a connect that fails or times out releases the
    attempt and retries on a fresh room after an exponential backoff from ``reconnect_initial`` to
    ``reconnect_max``, reset once the peer is reachable.

    Transports are shared actors: the runtime root owns this one (starts and stops it), and a device
    link registers with it by message.
    """

    _attachment_limit: typing.ClassVar[int | None] = 1
    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str

    _url: str
    _token: pydantic.SecretStr
    _connect_timeout: datetime.timedelta
    _release_timeout: datetime.timedelta
    _reconnect_initial: datetime.timedelta
    _reconnect_max: datetime.timedelta
    _max_message_bytes: int
    _room_factory: RoomFactory
    _bridge_factory: BridgeFactory
    _stream_factory: AudioStreamFactory | None
    _local_track_factory: LocalAudioTrackFactory | None
    _loop: asyncio.AbstractEventLoop | None
    # The registered peer and its channel table; written only by _bind_peer / _unbind_peer.
    _peer_id: str | None
    _channels: dict[str, transport.ChannelData]
    # The current attempt: its number, room handle, audio bridge, audio batcher, path, the room
    # callbacks it registered, its byte streams waiting to be read, and its named audio channel.
    _attempt: int
    _room: DataRoom | None
    _bridge: AudioBridge[rtc.AudioFrame] | None
    _batcher: RemotePcmBatcher | None
    _path: RoomAudioTrackPath | None
    _track_callback: collections.abc.Callable[..., object] | None
    _room_callbacks: dict[str, collections.abc.Callable[..., object]]
    _stream_topics: tuple[str, ...]
    _streams: asyncio.Queue[tuple[str, _ByteStreamReader]]
    _audio_channel: str | None
    # The current presence of the peer within the attempt; written by _peer_up / _peer_down.
    _presence: int
    _connection_id: str | None
    # The accepted connection: owner addresses, decoded bytes in flight per channel, unsent messages.
    _owner_ids: dict[str, str]
    _senders: dict[str, tuple[transport.ChannelData, ...]]
    _in_flight: dict[str, int]
    _outbound: asyncio.Queue[_Outbound]
    _release_for: typing.Literal["retry", "detach"]
    _reconnects: int

    def __init__(
        self,
        *,
        url: str,
        token: pydantic.SecretStr,
        connect_timeout: datetime.timedelta = _DEFAULT_CONNECT_TIMEOUT,
        release_timeout: datetime.timedelta = _DEFAULT_RELEASE_TIMEOUT,
        reconnect_initial: datetime.timedelta = _DEFAULT_RECONNECT_INITIAL,
        reconnect_max: datetime.timedelta = _DEFAULT_RECONNECT_MAX,
        max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES,
        room_factory: RoomFactory | None = None,
        bridge_factory: BridgeFactory | None = None,
        stream_factory: AudioStreamFactory | None = None,
        local_track_factory: LocalAudioTrackFactory | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Build an unstarted transport to the room ``url`` admits ``token`` to.

        The peer is whoever registers: its ``PeerAttachData.peer_id`` is its participant identity in
        that room, and only that participant is heard. ``max_message_bytes`` bounds one inbound or
        outbound message. ``room_factory`` builds one room handle per attempt (default ``rtc.Room``);
        ``bridge_factory`` builds the local audio bridge per attempt (default a 48 kHz mono LiveKit
        audio source).

        Raises:
            ValueError: If ``url`` is empty, a timeout is not positive, ``reconnect_initial`` exceeds
                ``reconnect_max``, or ``max_message_bytes`` is not positive.
        """

        super().__init__()
        if not url:
            raise ValueError("url must be a non-empty LiveKit server URL.")
        for name, value in (
            ("connect_timeout", connect_timeout),
            ("release_timeout", release_timeout),
            ("reconnect_initial", reconnect_initial),
            ("reconnect_max", reconnect_max),
        ):
            if value <= datetime.timedelta():
                raise ValueError(f"{name} must be a positive duration.")
        if reconnect_initial > reconnect_max:
            raise ValueError("reconnect_initial must not exceed reconnect_max.")
        if max_message_bytes < 1:
            raise ValueError("max_message_bytes must be positive.")

        def default_room() -> DataRoom:
            return typing.cast(DataRoom, typing.cast(object, rtc.Room(loop=loop)))

        def default_bridge() -> AudioBridge[rtc.AudioFrame]:
            return create_audio_bridge(loop=loop)

        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self._url = url
        self._token = token
        self._connect_timeout = connect_timeout
        self._release_timeout = release_timeout
        self._reconnect_initial = reconnect_initial
        self._reconnect_max = reconnect_max
        self._max_message_bytes = max_message_bytes
        self._room_factory = room_factory if room_factory is not None else default_room
        self._bridge_factory = bridge_factory if bridge_factory is not None else default_bridge
        self._stream_factory = stream_factory
        self._local_track_factory = local_track_factory
        self._loop = loop
        self._peer_id = None
        self._channels = {}
        self._attempt = 0
        self._room = None
        self._bridge = None
        self._batcher = None
        self._path = None
        self._track_callback = None
        self._room_callbacks = {}
        self._stream_topics = ()
        self._streams = asyncio.Queue(maxsize=_STREAM_QUEUE_LIMIT)
        self._audio_channel = None
        self._presence = 0
        self._connection_id = None
        self._owner_ids = {}
        self._senders = {}
        self._in_flight = {}
        self._outbound = asyncio.Queue(maxsize=_OUTBOUND_QUEUE_LIMIT)
        self._release_for = "retry"
        self._reconnects = 0

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

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop this transport, then the attempt it still owns, so no room path outlives it."""

        path = self._path
        batcher = self._batcher
        await super().stop(ctx)
        if batcher is not None:
            await batcher.aclose()
        if path is not None:
            await hsm.stop(path, ctx)

    # -- guards -----------------------------------------------------------------------------------

    @staticmethod
    def _accepts_peer(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> bool:
        return isinstance(event.data, transport.PeerAttachData) and attachment.Attachment._can_attach(
            ctx, instance, event
        )

    @staticmethod
    def _is_current_attempt(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, _AttemptData) and data.attempt == instance._attempt

    @staticmethod
    def _is_current_connection(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> bool:
        """Correlate the endpoint's accept or close with the connection this transport offered last."""

        del ctx
        data = event.data
        return (
            isinstance(data, transport.AcceptPeerData | transport.ClosePeerData)
            and data.connection_id == instance._connection_id
            and data.peer_id == instance._peer_id
        )

    @staticmethod
    def _is_current_delivery(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, _DeliveredData) and data.connection_id == instance._connection_id

    @staticmethod
    def _is_owner_send(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> bool:
        """Whether a channel owner sent an event its own channel's codec encodes (a pure name lookup)."""

        del ctx
        return any(channel.codec.encodes(event) for channel in instance._senders.get(event.source, ()))

    @staticmethod
    def _releasing_for_detach(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._release_for == "detach"

    # -- timers -----------------------------------------------------------------------------------

    @staticmethod
    def _connect_timeout_value(
        ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, event
        return instance._connect_timeout

    @staticmethod
    def _release_timeout_value(
        ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, event
        return instance._release_timeout

    @staticmethod
    def _backoff(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        exponent = min(instance._reconnects, _MAX_BACKOFF_EXPONENT)
        return min(instance._reconnect_initial * (1 << exponent), instance._reconnect_max)

    # -- registration bookkeeping -----------------------------------------------------------------

    @staticmethod
    def _bind_peer(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, transport.PeerAttachData)
        instance._peer_id = data.peer_id
        instance._channels = {channel.name: channel for channel in data.channels}

    @staticmethod
    def _unbind_peer(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._peer_id = None
        instance._channels = {}

    @staticmethod
    def _release_for_detach(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._release_for = "detach"

    @staticmethod
    def _release_for_retry(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._release_for = "retry"

    @staticmethod
    def _reset_reconnects(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._reconnects = 0

    @staticmethod
    def _count_reconnect(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """The backoff timer fired: the next wait, if this attempt fails too, is twice as long."""

        del ctx, event
        instance._reconnects += 1

    # -- one attempt ------------------------------------------------------------------------------

    @staticmethod
    def _begin_attempt(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """Build this attempt's room handle, audio bridge, audio batcher and room path.

        Every SDK-facing closure carries the attempt number, so whatever a discarded room or path
        reports late is ignored by the attempt guards. The peer's audio track subscriptions are heard
        from before the room connects, so the track that the path reads is always named.
        """

        del ctx, event
        peer_id = instance._peer_id
        assert peer_id is not None
        instance._attempt += 1
        attempt = instance._attempt
        owner = weakref.ref(instance)
        room = instance._room_factory()
        bridge = instance._bridge_factory()

        async def deliver_audio(chunk: audio.InputData) -> None:
            _dispatch_own(owner, _AudioReceivedEvent.with_data(_AudioReceivedData(attempt=attempt, chunk=chunk)))

        batcher = RemotePcmBatcher(emit=deliver_audio, loop=instance._loop)

        async def remote_audio(chunk: audio.InputData) -> None:
            await batcher.push(chunk)

        def connection(data: RoomAudioConnectedData) -> None:
            # The path reports a published track when it connects and none once it is disconnected.
            observed = _PathConnectedEvent if data.local_track_sid is not None else _PathDisconnectedEvent
            _dispatch_own(owner, observed.with_data(_AttemptData(attempt=attempt)))

        def on_track_subscribed(track: object, publication: object, participant: object) -> object:
            kind = typing.cast(object, getattr(track, "kind", None))
            if _identity(participant) != peer_id or kind != rtc.TrackKind.KIND_AUDIO:
                return None
            name = typing.cast(object, getattr(publication, "name", None))
            _dispatch_own(
                owner, _TrackEvent.with_data(_TrackData(attempt=attempt, channel=name if isinstance(name, str) else ""))
            )
            return None

        _ = room.on("track_subscribed", on_track_subscribed)
        instance._room = room
        instance._bridge = bridge
        instance._batcher = batcher
        instance._track_callback = on_track_subscribed
        instance._audio_channel = None
        instance._presence = 0
        instance._path = RoomAudioTrackPath(
            bridge=bridge,
            remote_audio_sink=remote_audio,
            connection_sink=connection,
            room=room,
            stream_factory=instance._stream_factory,
            local_track_factory=instance._local_track_factory,
            loop=instance._loop,
            participant_identity=peer_id,
        )

    @staticmethod
    async def _run_start_path(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """Start the attempt's path under the transport's lifetime, then ask it to connect.

        The outcome comes back through the path's connection sink, bounded by ``connect_timeout``.
        """

        del event
        path = instance._path
        assert path is not None
        attempt = instance._attempt
        try:
            # The path outlives this activity: it is started under the transport's own context
            # (HSM-CONTEXT-001) and stopped when the attempt is released.
            _ = await started(instance.context(), path, RoomAudioTrackPath.model)
        except Exception as error:
            _LOG.warning("room transport could not start its room path kind=%s", type(error).__name__)
            _ = hsm.dispatch(ctx, instance, _PathStartFailedEvent.with_data(_AttemptData(attempt=attempt)))
            return
        connect = RoomAudioConnectData(url=instance._url, token=instance._token.get_secret_value())
        _ = hsm.dispatch(
            ctx,
            path,
            dataclasses.replace(RoomAudioConnectEvent.with_data(connect), source=hsm.id(instance), target=hsm.id(path)),
        )

    @staticmethod
    def _disconnect_path(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """Ask the attempt's started path to leave the room; a path already down or dead ignores it."""

        del event
        path = instance._path
        assert path is not None
        _ = hsm.dispatch(ctx, path, dataclasses.replace(RoomAudioDisconnectEvent, source=hsm.id(instance)))

    @staticmethod
    async def _run_stop_path(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """Stop the attempt's batcher and path (valid from any path state, ``failed`` and unstarted included)."""

        del event
        attempt = instance._attempt
        batcher = instance._batcher
        path = instance._path
        if batcher is not None:
            await batcher.aclose()
        if path is not None:
            await hsm.stop(path, ctx)
        _ = hsm.dispatch(ctx, instance, _AttemptReleasedEvent.with_data(_AttemptData(attempt=attempt)))

    @staticmethod
    def _discard_attempt(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """Let go of the released attempt: the next one never reuses its room or path."""

        del ctx, event
        room = instance._room
        callback = instance._track_callback
        if room is not None and callback is not None:
            room.off("track_subscribed", callback)
        instance._room = None
        instance._bridge = None
        instance._batcher = None
        instance._path = None
        instance._track_callback = None
        instance._audio_channel = None

    @staticmethod
    def _count_connected(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _record_connect("connected")

    @staticmethod
    def _count_connect_failed(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _record_connect("failed")

    @staticmethod
    def _count_connect_timed_out(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _record_connect("timed_out")

    # -- the connected room -----------------------------------------------------------------------

    @staticmethod
    def _bind_room(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """Listen to the attempt's room: the room going down, the peer coming and going, its messages.

        Each callback admits at the SDK edge (sender, topic, size) and dispatches only what it
        admitted, stamped with this attempt. Every registered data channel's topic takes byte streams.
        """

        del ctx, event
        room = instance._room
        peer_id = instance._peer_id
        assert room is not None and peer_id is not None
        attempt = instance._attempt
        data_channels = frozenset(name for name, channel in instance._channels.items() if channel.kind == "data")
        max_message_bytes = instance._max_message_bytes
        owner = weakref.ref(instance)
        streams: asyncio.Queue[tuple[str, _ByteStreamReader]] = asyncio.Queue(maxsize=_STREAM_QUEUE_LIMIT)
        instance._streams = streams

        def on_disconnected(*args: object) -> object:
            del args
            _dispatch_own(owner, _RoomDownEvent.with_data(_AttemptData(attempt=attempt)))
            return None

        def on_participant_connected(participant: object) -> object:
            if _identity(participant) == peer_id:
                _dispatch_own(owner, _PeerPresentEvent.with_data(_AttemptData(attempt=attempt)))
            return None

        def on_participant_disconnected(participant: object) -> object:
            if _identity(participant) == peer_id:
                _dispatch_own(owner, _PeerLeftEvent.with_data(_PeerLeftData(attempt=attempt)))
            return None

        def on_data_received(received: object) -> object:
            packet = typing.cast(_DataPacket, received)
            if _identity(packet.participant) != peer_id:
                _reject("wrong_peer")
                return None
            topic = packet.topic
            if topic is None or topic not in data_channels:
                _reject("refused_channel")
                return None
            if len(packet.data) > max_message_bytes:
                _reject("oversize")
                return None
            _transferred(len(packet.data), "in")
            _dispatch_own(
                owner, _MessageEvent.with_data(_MessageData(attempt=attempt, channel=topic, message=packet.data))
            )
            return None

        def stream_handler(channel: str) -> ByteStreamHandler:
            def on_byte_stream(received: object, participant_identity: str) -> None:
                reader = typing.cast(_ByteStreamReader, received)
                if participant_identity != peer_id:
                    _reject("wrong_peer")
                    return
                size = reader.info.size
                if size is None or size > max_message_bytes:
                    _reject("oversize")
                    return
                try:
                    streams.put_nowait((channel, reader))
                except asyncio.QueueFull:
                    # More byte streams in flight than the transport reads at once.
                    _reject("backlog")

            return on_byte_stream

        callbacks: dict[str, collections.abc.Callable[..., object]] = {
            "disconnected": on_disconnected,
            "participant_connected": on_participant_connected,
            "participant_disconnected": on_participant_disconnected,
            "data_received": on_data_received,
        }
        for name, callback in callbacks.items():
            _ = room.on(name, callback)
        topics = tuple(sorted(data_channels))
        for topic in topics:
            room.register_byte_stream_handler(topic, stream_handler(topic))
        instance._room_callbacks = callbacks
        instance._stream_topics = topics
        if peer_id in room.remote_participants:
            _dispatch_own(owner, _PeerPresentEvent.with_data(_AttemptData(attempt=attempt)))

    @staticmethod
    def _unbind_room(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        room = instance._room
        callbacks = instance._room_callbacks
        topics = instance._stream_topics
        instance._room_callbacks = {}
        instance._stream_topics = ()
        if room is None:
            return
        for name, callback in callbacks.items():
            room.off(name, callback)
        for topic in topics:
            room.unregister_byte_stream_handler(topic)

    @staticmethod
    async def _run_read_streams(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """Read admitted byte streams one at a time; each whole stream is one message."""

        del event
        streams = instance._streams
        attempt = instance._attempt
        max_message_bytes = instance._max_message_bytes
        while True:
            channel, reader = await streams.get()
            content = bytearray()
            try:
                async for chunk in reader:
                    content += chunk
                    if len(content) > max_message_bytes:
                        break
            except Exception as error:
                # A stream the SDK could not finish is a lost message, not a fault of the transport.
                _LOG.warning("room transport byte stream failed kind=%s", type(error).__name__)
                continue
            if len(content) > max_message_bytes:
                _reject("oversize")
                continue
            _transferred(len(content), "in")
            _ = hsm.dispatch(
                ctx,
                instance,
                _MessageEvent.with_data(_MessageData(attempt=attempt, channel=channel, message=bytes(content))),
            )

    @staticmethod
    def _name_audio_track(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """The peer's audio track carries the audio channel its publication is named after, if registered."""

        del ctx
        data = event.data
        assert isinstance(data, _TrackData)
        channel = instance._channels.get(data.channel)
        if channel is None or channel.kind != "audio":
            _reject("refused_channel")
            instance._audio_channel = None
            return
        instance._audio_channel = channel.name

    @staticmethod
    def _drop_unowned(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """A message with no accepted connection to go to: counted against its channel's codec."""

        del ctx
        data = event.data
        name = data.channel if isinstance(data, _MessageData) else instance._audio_channel
        channel = instance._channels.get(name) if name is not None else None
        if channel is not None:
            transform.telemetry.dropped(channel.codec, "unowned")

    # -- the peer ---------------------------------------------------------------------------------

    @staticmethod
    def _tell_endpoint(
        ctx: hsm.Context, instance: RoomTransport, trigger: hsm.Event[typing.Any], told: hsm.Event[typing.Any]
    ) -> None:
        endpoint = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            endpoint,
            dataclasses.replace(
                told,
                id=uuid.uuid4().hex,
                source=hsm.id(instance),
                target=attachment.Attachment._actor_id(endpoint),
                metadata=dict(trigger.metadata),
            ),
        )

    @staticmethod
    def _peer_up(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """The peer is in the room: offer it to the endpoint as a new connection and reset the backoff."""

        peer_id = instance._peer_id
        assert peer_id is not None
        instance._presence += 1
        connection_id = f"{instance._attempt}.{instance._presence}"
        instance._connection_id = connection_id
        instance._reconnects = 0
        _PEERS.add(1, {"transport": _TRANSPORT})
        RoomTransport._tell_endpoint(
            ctx,
            instance,
            event,
            transport.PeerConnectedEvent.with_data(
                transport.PeerConnectedData(
                    peer_id=peer_id, connection_id=connection_id, transport=_TRANSPORT, connection=instance
                )
            ),
        )

    @staticmethod
    def _peer_down(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """The connection is gone, for whichever reason the trigger's payload type says."""

        peer_id, connection_id = instance._peer_id, instance._connection_id
        assert peer_id is not None and connection_id is not None
        data = event.data
        reason: typing.Literal["left", "transport_lost", "closed"] = "transport_lost"
        if isinstance(data, _PeerLeftData):
            reason = "left"
        elif isinstance(data, transport.ClosePeerData | attachment.DetachData):
            reason = "closed"
        instance._connection_id = None
        _PEERS.add(-1, {"transport": _TRANSPORT})
        RoomTransport._tell_endpoint(
            ctx,
            instance,
            event,
            transport.PeerDisconnectedEvent.with_data(
                transport.PeerDisconnectedData(
                    peer_id=peer_id, connection_id=connection_id, transport=_TRANSPORT, reason=reason
                )
            ),
        )

    # -- the accepted connection ------------------------------------------------------------------

    @staticmethod
    def _tell_owners(
        ctx: hsm.Context,
        instance: RoomTransport,
        trigger: hsm.Event[typing.Any],
        build: collections.abc.Callable[[str], hsm.Event[typing.Any]],
    ) -> None:
        """Tell every owner whose codec encodes anything about the channel it owns."""

        for name, channel in instance._channels.items():
            if not channel.codec.encodes_any:
                continue
            _ = hsm.dispatch(
                ctx,
                channel.owner,
                dataclasses.replace(
                    build(name),
                    id=uuid.uuid4().hex,
                    source=hsm.id(instance),
                    target=instance._owner_ids[name],
                    metadata=dict(trigger.metadata),
                ),
            )

    @staticmethod
    def _own_streams(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """The endpoint accepted: hand every channel to its owner, with nothing in flight or unsent."""

        owner_ids = {
            name: attachment.Attachment._actor_id(channel.owner) for name, channel in instance._channels.items()
        }
        senders: dict[str, tuple[transport.ChannelData, ...]] = {}
        for name, channel in instance._channels.items():
            senders[owner_ids[name]] = (*senders.get(owner_ids[name], ()), channel)
        instance._owner_ids = owner_ids
        instance._senders = senders
        instance._in_flight = {}
        instance._outbound = asyncio.Queue(maxsize=_OUTBOUND_QUEUE_LIMIT)
        RoomTransport._tell_owners(
            ctx,
            instance,
            event,
            lambda name: transform.OwnedEvent.with_data(transform.OwnedData(stream=instance, channel=name)),
        )

    @staticmethod
    def _end_streams(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """The connection is gone: tell the owners their channels ended, and why."""

        data = event.data
        # The peer finished; the room or path under the connection went (reset); or the endpoint
        # closed it, unregistered, or this transport stopped (closed).
        reason: typing.Literal["finished", "reset", "closed"] = "closed"
        if isinstance(data, _PeerLeftData):
            reason = "finished"
        elif isinstance(data, _AttemptData):
            reason = "reset"
        RoomTransport._tell_owners(
            ctx,
            instance,
            event,
            lambda name: transform.EndedEvent.with_data(
                transform.EndedData(stream=instance, channel=name, reason=reason)
            ),
        )
        instance._senders = {}

    @staticmethod
    def _deliver(
        ctx: hsm.Context,
        instance: RoomTransport,
        trigger: hsm.Event[typing.Any],
        channel: transport.ChannelData,
        message: bytes,
    ) -> None:
        """Decode one message to its channel's owner, and count it in flight until the owner is done."""

        connection_id = instance._connection_id
        assert connection_id is not None
        in_flight = instance._in_flight.get(channel.name, 0)
        if in_flight and in_flight + len(message) > _IN_FLIGHT_LIMIT:
            transform.telemetry.dropped(channel.codec, "flow_control")
            return
        try:
            decoded = channel.codec.decode(message)
        except transform.DecodeError as error:
            transform.telemetry.decode_failed(channel.codec, error.reason)
            return
        transform.telemetry.transferred(channel.codec, "in", len(message))
        done = hsm.dispatch(
            ctx,
            channel.owner,
            dataclasses.replace(
                decoded,
                id=uuid.uuid4().hex,
                source=hsm.id(instance),
                target=instance._owner_ids[channel.name],
                metadata=dict(trigger.metadata),
            ),
        )
        instance._in_flight[channel.name] = in_flight + len(message)
        owner = weakref.ref(instance)
        delivered = _DeliveredEvent.with_data(
            _DeliveredData(connection_id=connection_id, channel=channel.name, size=len(message))
        )

        def credit(finished: asyncio.Future[bool]) -> None:
            if not finished.cancelled():
                _ = finished.exception()
            _dispatch_own(owner, delivered)

        asyncio.ensure_future(done).add_done_callback(credit)

    @staticmethod
    def _deliver_message(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _MessageData)
        RoomTransport._deliver(ctx, instance, event, instance._channels[data.channel], data.message)

    @staticmethod
    def _deliver_audio(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """Pack batched PCM as ``!IH`` + PCM messages that fit the channel's codec, and deliver each."""

        data = event.data
        assert isinstance(data, _AudioReceivedData)
        name = instance._audio_channel
        if name is None:
            _reject("refused_channel")
            return
        channel = instance._channels[name]
        chunk = data.chunk
        if chunk.sample_rate_hz is None or chunk.channels is None:
            transform.telemetry.decode_failed(channel.codec, "malformed")
            return
        _transferred(len(chunk.audio), "in")
        header = _PCM_HEADER.pack(chunk.sample_rate_hz, chunk.channels)
        frame_bytes = 2 * chunk.channels
        step = max((channel.codec.max_message_bytes - _PCM_HEADER.size) // frame_bytes, 1) * frame_bytes
        for offset in range(0, len(chunk.audio), step):
            RoomTransport._deliver(ctx, instance, event, channel, header + chunk.audio[offset : offset + step])

    @staticmethod
    def _credit(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _DeliveredData)
        instance._in_flight[data.channel] = max(instance._in_flight.get(data.channel, 0) - data.size, 0)

    @staticmethod
    def _send(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """Encode what an owner sent through the codec of the channel it owns, and queue it for the room."""

        del ctx
        channel = next(channel for channel in instance._senders.get(event.source, ()) if channel.codec.encodes(event))
        try:
            message = channel.codec.encode(event)
        except transform.EncodeError:
            transform.telemetry.dropped(channel.codec, "encode_failed")
            return
        if len(message) > instance._max_message_bytes:
            _drop_send("oversize")
            return
        try:
            instance._outbound.put_nowait((channel.name, channel.kind, message))
        except asyncio.QueueFull:
            _drop_send("backlog")
            return
        transform.telemetry.transferred(channel.codec, "out", len(message))

    @staticmethod
    async def _run_outbound(ctx: hsm.Context, instance: RoomTransport, event: hsm.Event[typing.Any]) -> None:
        """Send queued messages to the peer, in order, while the connection is accepted."""

        del ctx, event
        outbound = instance._outbound
        room = instance._room
        bridge = instance._bridge
        peer_id = instance._peer_id
        assert room is not None and bridge is not None and peer_id is not None
        participant = room.local_participant
        destination = [peer_id]
        while True:
            channel, kind, message = await outbound.get()
            try:
                if kind == "audio":
                    sample_rate_hz, channels = typing.cast(tuple[int, int], _PCM_HEADER.unpack_from(message))
                    await bridge.publish_audio(
                        audio.OutputData(
                            audio=message[_PCM_HEADER.size :],
                            media_type=_PCM_MEDIA_TYPE,
                            sample_rate_hz=sample_rate_hz,
                            channels=channels,
                        )
                    )
                elif len(message) > _MAX_PACKET_BYTES:
                    writer = await participant.stream_bytes(
                        channel, total_size=len(message), destination_identities=destination, topic=channel
                    )
                    await writer.write(message)
                    await writer.aclose()
                else:
                    await participant.publish_data(
                        message, reliable=True, destination_identities=destination, topic=channel
                    )
                _transferred(len(message), "out")
            except Exception as error:
                # A write the room refused loses that one message; the room itself reports a lost
                # connection.
                _LOG.warning("room transport write failed kind=%s", type(error).__name__)

    _detach_effects: typing.ClassVar[tuple[collections.abc.Callable[..., None], ...]] = (
        attachment.Attachment._detach,
        _unbind_peer,
        _release_for_detach,
        attachment.Attachment._dispatch_detach_complete_removed,
    )

    model: typing.ClassVar[hsm.Model] = define(
        "RoomTransport",
        hsm.initial(hsm.target("/RoomTransport/detached")),
        hsm.state(
            "detached",
            hsm.entry(_reset_reconnects),
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(_accepts_peer),
                hsm.effect(
                    attachment.Attachment._attach,
                    _bind_peer,
                    attachment.Attachment._dispatch_attach_complete_created,
                ),
                hsm.target("/RoomTransport/starting_path"),
            ),
        ),
        hsm.state(
            "starting_path",
            hsm.entry(_begin_attempt),
            hsm.activity(_run_start_path),
            # The peer's audio track may be subscribed while the room connects.
            hsm.defer(_TrackEvent),
            hsm.transition(
                hsm.on(_PathConnectedEvent),
                hsm.guard(_is_current_attempt),
                hsm.effect(_count_connected),
                hsm.target("/RoomTransport/connected"),
            ),
            hsm.transition(
                # The connect failed and the path cleaned itself up to disconnected.
                hsm.on(_PathDisconnectedEvent),
                hsm.guard(_is_current_attempt),
                hsm.effect(_count_connect_failed),
                hsm.target("/RoomTransport/releasing/stopping_path"),
            ),
            hsm.transition(
                # The path never started, so there is nothing to disconnect.
                hsm.on(_PathStartFailedEvent),
                hsm.guard(_is_current_attempt),
                hsm.effect(_count_connect_failed),
                hsm.target("/RoomTransport/releasing/stopping_path"),
            ),
            hsm.transition(
                # Also covers a connect that ended in the path's dead `failed`, where no sink fires.
                hsm.after(_connect_timeout_value),
                hsm.effect(_count_connect_timed_out),
                hsm.target("/RoomTransport/releasing"),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._detach_last_attachment),
                hsm.effect(*_detach_effects),
                hsm.target("/RoomTransport/releasing"),
            ),
        ),
        hsm.state(
            "connected",
            hsm.entry(_bind_room),
            hsm.exit(_unbind_room),
            hsm.activity(_run_read_streams),
            hsm.initial(hsm.target("/RoomTransport/connected/awaiting_peer")),
            hsm.state(
                "awaiting_peer",
                hsm.transition(
                    hsm.on(_PeerPresentEvent),
                    hsm.guard(_is_current_attempt),
                    hsm.target("/RoomTransport/connected/present"),
                ),
            ),
            hsm.state(
                "present",
                hsm.entry(_peer_up),
                hsm.exit(_peer_down),
                hsm.initial(hsm.target("/RoomTransport/connected/present/offered")),
                hsm.state(
                    "offered",
                    hsm.transition(
                        hsm.on(transport.AcceptPeerEvent),
                        hsm.guard(_is_current_connection),
                        hsm.target("/RoomTransport/connected/present/accepted"),
                    ),
                ),
                hsm.state(
                    "accepted",
                    hsm.entry(_own_streams),
                    hsm.exit(_end_streams),
                    hsm.activity(_run_outbound),
                    hsm.transition(hsm.on(_MessageEvent), hsm.guard(_is_current_attempt), hsm.effect(_deliver_message)),
                    hsm.transition(
                        hsm.on(_AudioReceivedEvent), hsm.guard(_is_current_attempt), hsm.effect(_deliver_audio)
                    ),
                    hsm.transition(hsm.on(_DeliveredEvent), hsm.guard(_is_current_delivery), hsm.effect(_credit)),
                    # Lowest priority (HSM-CATCHALL-001): only an event a channel owner sent, named
                    # as its own channel's codec encodes, matches; lifecycle events never do.
                    hsm.transition(hsm.on(hsm.AnyEvent), hsm.guard(_is_owner_send), hsm.effect(_send)),
                ),
                hsm.transition(
                    # A participant cannot evict another: leave the room and offer the peer anew.
                    hsm.on(transport.ClosePeerEvent),
                    hsm.guard(_is_current_connection),
                    hsm.target("/RoomTransport/releasing"),
                ),
            ),
            hsm.transition(
                hsm.on(_PeerLeftEvent),
                hsm.guard(_is_current_attempt),
                hsm.target("/RoomTransport/connected/awaiting_peer"),
            ),
            hsm.transition(hsm.on(_TrackEvent), hsm.guard(_is_current_attempt), hsm.effect(_name_audio_track)),
            hsm.transition(
                hsm.on(_RoomDownEvent),
                hsm.guard(_is_current_attempt),
                hsm.target("/RoomTransport/releasing"),
            ),
            hsm.transition(
                hsm.on(_PathDisconnectedEvent),
                hsm.guard(_is_current_attempt),
                hsm.target("/RoomTransport/releasing/stopping_path"),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._detach_last_attachment),
                hsm.effect(*_detach_effects),
                hsm.target("/RoomTransport/releasing"),
            ),
            hsm.transition(
                hsm.on(_MessageEvent, _AudioReceivedEvent),
                hsm.guard(_is_current_attempt),
                hsm.effect(_drop_unowned),
            ),
        ),
        hsm.state(
            "releasing",
            # A new endpoint waits until the old attempt is released.
            hsm.defer(attachment.AttachEvent),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._detach_last_attachment),
                hsm.effect(*_detach_effects),
            ),
            hsm.initial(hsm.target("/RoomTransport/releasing/disconnecting_path")),
            hsm.state(
                "disconnecting_path",
                hsm.entry(_disconnect_path),
                hsm.transition(
                    hsm.on(_PathDisconnectedEvent),
                    hsm.guard(_is_current_attempt),
                    hsm.target("/RoomTransport/releasing/stopping_path"),
                ),
                hsm.transition(
                    # The path's disconnect failed and it ended in `failed`, where no sink fires.
                    hsm.after(_release_timeout_value),
                    hsm.target("/RoomTransport/releasing/stopping_path"),
                ),
            ),
            hsm.state(
                "stopping_path",
                hsm.activity(_run_stop_path),
                hsm.transition(
                    hsm.on(_AttemptReleasedEvent),
                    hsm.guard(_is_current_attempt),
                    hsm.effect(_discard_attempt),
                    hsm.target("/RoomTransport/releasing/released"),
                ),
                hsm.transition(
                    hsm.after(_release_timeout_value),
                    hsm.effect(_count_connect_timed_out, _discard_attempt),
                    hsm.target("/RoomTransport/releasing/released"),
                ),
            ),
            hsm.choice(
                "released",
                hsm.transition(
                    hsm.guard(_releasing_for_detach),
                    hsm.effect(_release_for_retry),
                    hsm.target("/RoomTransport/detached"),
                ),
                hsm.transition(hsm.target("/RoomTransport/reconnect_wait")),
            ),
        ),
        hsm.state(
            "reconnect_wait",
            hsm.transition(
                hsm.after(_backoff),
                hsm.effect(_count_reconnect),
                hsm.target("/RoomTransport/starting_path"),
            ),
            hsm.transition(
                # No attempt is alive here: the binding goes and the transport is detached at once.
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._detach_last_attachment),
                hsm.effect(
                    attachment.Attachment._detach,
                    _unbind_peer,
                    attachment.Attachment._dispatch_detach_complete_removed,
                ),
                hsm.target("/RoomTransport/detached"),
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
    """Detached, then one attempt at a time: starting its path, connected (awaiting the peer, the peer
    present and offered or accepted), releasing it, and waiting to reconnect."""


__all__ = [
    "BridgeFactory",
    "ByteStreamHandler",
    "ByteStreamWriter",
    "DataParticipant",
    "DataRoom",
    "RoomFactory",
    "RoomTransport",
]

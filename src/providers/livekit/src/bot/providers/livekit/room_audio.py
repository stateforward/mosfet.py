from __future__ import annotations

from bot.devices import audio as audio_device

import asyncio
import collections.abc
import datetime
import typing
import weakref

import hsm
import bot
import pydantic
from livekit import rtc

from bot.telemetry import observer, span

from .audio import AudioBridge, AudioFrame

_DEFAULT_OPERATION_TIMEOUT = datetime.timedelta(seconds=30)
_SCOPE = "bot.providers.livekit"
_COMPONENT = "livekit.room_audio"


class RoomAudioError(RuntimeError):
    """Raised when a LiveKit room audio track path fails."""

    failure_kind: str

    def __init__(self, message: str, *, failure_kind: str = "unknown") -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


class RoomAudioConnectData(pydantic.BaseModel):
    """Connection request for publishing and subscribing stateforward.bot audio through a LiveKit room."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "LiveKit room credentials and local track name for a stateforward.bot audio bridge.",
            "examples": [
                {
                    "url": "wss://livekit.example.com",
                    "token": "livekit-jwt",
                    "track_name": "alice-audio",
                }
            ],
        },
    )

    url: str = pydantic.Field(
        min_length=1,
        description="LiveKit server URL used by the SDK room connection.",
        examples=["wss://livekit.example.com"],
    )
    token: str = pydantic.Field(
        min_length=1,
        description="LiveKit access token for the room participant. Do not log or expose this value.",
        examples=["livekit-jwt"],
    )
    track_name: str = pydantic.Field(
        default="bot-audio",
        min_length=1,
        description="Name for the local LiveKit audio track published from the stateforward.bot bridge source.",
        examples=["alice-audio"],
    )


class RoomAudioConnectedData(pydantic.BaseModel):
    """Completion payload for a connected room audio track path."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "LiveKit room audio path connection result.",
            "examples": [{"local_track_sid": "TR_123"}],
        },
    )

    local_track_sid: str | None = pydantic.Field(
        default=None,
        description="LiveKit track SID assigned to the published local audio track when available.",
        examples=["TR_123"],
    )


class RoomAudioFailureData(pydantic.BaseModel):
    """FailureData payload for LiveKit room audio track path operations."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Normalized LiveKit room audio path failure.",
            "examples": [{"failure_kind": "connect_failed"}],
        },
    )

    failure_kind: str = pydantic.Field(
        min_length=1,
        description="Stable low-cardinality failure kind for the room audio path operation.",
        examples=["connect_failed", "disconnect_failed", "remote_audio_failed"],
    )


class RoomAudioTrackSubscribedData(pydantic.BaseModel):
    """Private provider observation for a LiveKit room track subscription."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        json_schema_extra={
            "description": "Private LiveKit track subscription observation consumed by the room audio path.",
            "examples": [{"track": "<livekit-audio-track>"}],
        },
    )

    track: object = pydantic.Field(
        description="LiveKit SDK track object from the room callback. The value is not logged or serialized.",
    )


RoomAudioConnectEvent = hsm.Event[RoomAudioConnectData](
    name="bot.provider.livekit.room_audio.connect",
    schema=RoomAudioConnectData,
)
RoomAudioDisconnectEvent = hsm.Event[None](
    name="bot.provider.livekit.room_audio.disconnect",
)
_RoomAudioConnectedEvent = hsm.Event[RoomAudioConnectedData](
    name="bot.provider.livekit.room_audio.connected",
    kind=hsm.CompletionEventKind,
    schema=RoomAudioConnectedData,
)
_RoomAudioDisconnectedEvent = hsm.Event[None](
    name="bot.provider.livekit.room_audio.disconnected",
    kind=hsm.CompletionEventKind,
)
_RoomAudioFailedEvent = hsm.Event[RoomAudioFailureData](
    name="bot.provider.livekit.room_audio.failed",
    kind=hsm.ErrorEventKind,
    schema=RoomAudioFailureData,
)
_RoomAudioTrackSubscribedEvent = hsm.Event[RoomAudioTrackSubscribedData](
    name="bot.provider.livekit.room_audio.track_subscribed",
    schema=RoomAudioTrackSubscribedData,
)
_RoomAudioRemoteAudioEndedEvent = hsm.Event[None](
    name="bot.provider.livekit.room_audio.remote_audio.ended",
    kind=hsm.CompletionEventKind,
)
_RoomAudioLocalTrackUnpublishedEvent = hsm.Event[None](
    name="bot.provider.livekit.room_audio.local_track.unpublished",
    kind=hsm.CompletionEventKind,
)
_RoomAudioRoomDisconnectedEvent = hsm.Event[None](
    name="bot.provider.livekit.room_audio.room.disconnected",
    kind=hsm.CompletionEventKind,
)
_RoomAudioListenerUnregisteredEvent = hsm.Event[None](
    name="bot.provider.livekit.room_audio.listener.unregistered",
    kind=hsm.CompletionEventKind,
)


class TrackPublication(typing.Protocol):
    """LiveKit track publication shape needed by the room audio path."""

    @property
    def sid(self) -> str:
        """LiveKit track SID assigned by the room."""
        ...


class LocalParticipant(typing.Protocol):
    """LiveKit local participant shape used to publish audio and to signal call setup.

    The participant is both the mouth and the line: audio goes out on a published track, and
    addressed call-setup messages go out (and come back) over the same participant's RPC channel.
    """

    def publish_track(
        self,
        track: object,
        options: object | None = None,
    ) -> collections.abc.Awaitable[TrackPublication]:
        """Publish a local LiveKit track."""
        ...

    def unpublish_track(self, track_sid: str) -> collections.abc.Awaitable[None]:
        """Unpublish a local LiveKit track by SID."""
        ...

    def perform_rpc(
        self,
        *,
        destination_identity: str,
        method: str,
        payload: str,
        response_timeout: float | None = None,
    ) -> collections.abc.Awaitable[str]:
        """Send one addressed RPC to a participant and await its response payload."""
        ...

    def register_rpc_method(
        self,
        method_name: str,
        handler: collections.abc.Callable[[object], str],
    ) -> object:
        """Answer ``method_name`` with ``handler``; the handler receives the RPC invocation data."""
        ...

    def unregister_rpc_method(self, method: str) -> None:
        """Stop answering ``method``."""
        ...


class RoomHandle(typing.Protocol):
    """LiveKit room shape consumed by the stateforward.bot room audio path."""

    @property
    def local_participant(self) -> LocalParticipant:
        """Local participant used to publish audio."""
        ...

    def connect(self, url: str, token: str) -> collections.abc.Awaitable[None]:
        """Connect to a LiveKit room."""
        ...

    def disconnect(self) -> collections.abc.Awaitable[None]:
        """Disconnect from a LiveKit room."""
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


type AudioStreamFactory = collections.abc.Callable[[object], collections.abc.AsyncIterable[object]]
type RoomAudioConnectionSink = collections.abc.Callable[[RoomAudioConnectedData], None]
type LocalAudioTrackFactory = collections.abc.Callable[[str, object], object]
type RemoteAudioSink = collections.abc.Callable[[audio_device.AudioInputData], collections.abc.Awaitable[None]]


def _livekit_room(loop: asyncio.AbstractEventLoop | None = None) -> RoomHandle:
    return typing.cast(RoomHandle, typing.cast(object, rtc.Room(loop=loop)))


def _livekit_audio_stream(track: object) -> collections.abc.AsyncIterable[object]:
    return typing.cast(collections.abc.AsyncIterable[object], rtc.AudioStream(typing.cast(rtc.Track, track)))


def _livekit_local_audio_track(name: str, source: object) -> object:
    return rtc.LocalAudioTrack.create_audio_track(name, typing.cast(rtc.AudioSource, source))


def _require_positive_timeout(value: datetime.timedelta) -> None:
    if value <= datetime.timedelta():
        raise ValueError("operation_timeout must be a positive duration.")


def _stable_failure_kind(error: BaseException, fallback: str) -> str:
    if isinstance(error, RoomAudioError):
        return error.failure_kind
    return fallback


def _stream_item_frame(item: object) -> AudioFrame:
    frame = getattr(item, "frame", item)
    return typing.cast(AudioFrame, frame)


def _is_audio_track(track: object) -> bool:
    kind = typing.cast(object, getattr(track, "kind", None))
    return kind == rtc.TrackKind.KIND_AUDIO


def _has_connect_data(ctx: hsm.Context, instance: "RoomAudioTrackPath", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, RoomAudioConnectData)


def _has_connected_data(
    ctx: hsm.Context,
    instance: "RoomAudioTrackPath",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, RoomAudioConnectedData)


def _has_failure_data(ctx: hsm.Context, instance: "RoomAudioTrackPath", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, RoomAudioFailureData)


def _has_audio_track_subscribed_data(
    ctx: hsm.Context,
    instance: "RoomAudioTrackPath",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, RoomAudioTrackSubscribedData) and _is_audio_track(data.track)


def _has_non_audio_track_subscribed_data(
    ctx: hsm.Context,
    instance: "RoomAudioTrackPath",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, RoomAudioTrackSubscribedData) and not _is_audio_track(data.track)


class RoomAudioTrackPath(hsm.Instance):
    """LiveKit room/SIP audio track path for a stateforward.bot-owned audio bridge."""

    _bridge: AudioBridge[rtc.AudioFrame]
    _remote_audio_sink: RemoteAudioSink
    _connection_sink: RoomAudioConnectionSink | None
    _room: RoomHandle
    _stream_factory: AudioStreamFactory
    _local_track_factory: LocalAudioTrackFactory
    _operation_timeout: datetime.timedelta
    _track_subscribed_callback: collections.abc.Callable[..., object]
    _local_track_sid: str | None
    _track_listener_registered: bool

    def __init__(
        self,
        *,
        bridge: AudioBridge[rtc.AudioFrame],
        remote_audio_sink: RemoteAudioSink,
        connection_sink: RoomAudioConnectionSink | None = None,
        room: RoomHandle | None = None,
        stream_factory: AudioStreamFactory | None = None,
        local_track_factory: LocalAudioTrackFactory | None = None,
        operation_timeout: datetime.timedelta = _DEFAULT_OPERATION_TIMEOUT,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        _require_positive_timeout(operation_timeout)
        instance_ref = weakref.ref(self)

        def handle_track_subscribed(track: object, publication: object, participant: object) -> object:
            del publication, participant
            live_instance = instance_ref()
            if live_instance is None:
                return None
            _ = live_instance.dispatch(
                live_instance.context(),
                _RoomAudioTrackSubscribedEvent.with_data(RoomAudioTrackSubscribedData(track=track)),
            )
            return None

        self._bridge = bridge
        self._remote_audio_sink = remote_audio_sink
        self._connection_sink = connection_sink
        self._room = room if room is not None else _livekit_room(loop)
        self._stream_factory = stream_factory if stream_factory is not None else _livekit_audio_stream
        self._local_track_factory = (
            local_track_factory if local_track_factory is not None else _livekit_local_audio_track
        )
        self._operation_timeout = operation_timeout
        self._track_subscribed_callback = handle_track_subscribed
        self._local_track_sid = None
        self._track_listener_registered = False

    def connect_room(self, ctx: hsm.Context, data: RoomAudioConnectData) -> collections.abc.Awaitable[None]:
        """Start the LiveKit room audio track path."""

        return self.dispatch(ctx, RoomAudioConnectEvent.with_data(data))

    def disconnect_room(self, ctx: hsm.Context) -> collections.abc.Awaitable[None]:
        """Stop the LiveKit room audio track path."""

        return self.dispatch(ctx, RoomAudioDisconnectEvent)

    @staticmethod
    def _operation_timeout_value(
        ctx: hsm.Context,
        instance: "RoomAudioTrackPath",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, event
        return instance._operation_timeout

    async def _connect_room_audio(self, data: RoomAudioConnectData) -> RoomAudioConnectedData:
        # Two SDK round trips under one span: joining the room and publishing the local track.
        # Neither the URL, the token, nor the assigned SID may become an attribute.
        with span.operation(
            "bot.provider.livekit.room_audio.connect",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="connect",
        ) as active:
            await self._room.connect(data.url, data.token)
            active.set_attribute("bot.room.joined", True)
            track = self._local_track_factory(data.track_name, self._bridge.source_writer.source)
            publication = await self._room.local_participant.publish_track(track)
            active.set_attribute("bot.local_track.published", True)
            return RoomAudioConnectedData(local_track_sid=publication.sid)

    async def _unpublish_local_track(self) -> None:
        track_sid = self._local_track_sid
        assert track_sid is not None
        await self._room.local_participant.unpublish_track(track_sid)

    async def _disconnect_room(self) -> None:
        await self._room.disconnect()

    def _unregister_track_subscription_listener(self) -> None:
        self._room.off("track_subscribed", self._track_subscribed_callback)
        self._track_listener_registered = False

    @staticmethod
    def _has_local_track_sid(
        ctx: hsm.Context,
        instance: "RoomAudioTrackPath",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._local_track_sid is not None

    @staticmethod
    def _has_registered_track_listener(
        ctx: hsm.Context,
        instance: "RoomAudioTrackPath",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._track_listener_registered

    @staticmethod
    def _register_track_subscription_listener(
        ctx: hsm.Context,
        instance: "RoomAudioTrackPath",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx, event
        _ = instance._room.on("track_subscribed", instance._track_subscribed_callback)
        instance._track_listener_registered = True

    @staticmethod
    def _set_connected(ctx: hsm.Context, instance: "RoomAudioTrackPath", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, RoomAudioConnectedData)
        instance._local_track_sid = data.local_track_sid
        if instance._connection_sink is not None:
            instance._connection_sink(data)

    @staticmethod
    def _clear_connected(ctx: hsm.Context, instance: "RoomAudioTrackPath", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._local_track_sid = None
        if instance._connection_sink is not None:
            instance._connection_sink(RoomAudioConnectedData(local_track_sid=None))

    async def _forward_remote_audio(self, track: object) -> None:
        # One span for the whole subscription (it ends when the remote track does, carrying the
        # frame total), and one per frame so a single sound off the wire has a trace of its own
        # from decode through the sink that batches it. Never the track SID or the frame bytes.
        with span.operation(
            "bot.provider.livekit.room_audio.remote_audio",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="receive",
        ) as subscription:
            frames = 0
            async for item in self._stream_factory(track):
                with span.operation(
                    "bot.provider.livekit.room_audio.frame",
                    scope=_SCOPE,
                    component=_COMPONENT,
                    stage="decode",
                ):
                    frame = _stream_item_frame(item)
                    audio = await self._bridge.receive_frame(frame)
                    await self._remote_audio_sink(audio)
                frames += 1
                subscription.set_attribute("bot.frames.count", frames)

    @staticmethod
    async def _run_connect_room_audio(
        ctx: hsm.Context,
        instance: "RoomAudioTrackPath",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, RoomAudioConnectData)
        try:
            connected = await instance._connect_room_audio(data)
        except Exception as error:
            failure = RoomAudioFailureData(failure_kind=_stable_failure_kind(error, "connect_failed"))
            _ = hsm.dispatch(ctx, instance, _RoomAudioFailedEvent.with_data(failure))
            return
        _ = hsm.dispatch(ctx, instance, _RoomAudioConnectedEvent.with_data(connected))

    @staticmethod
    async def _run_unpublish_local_track(
        ctx: hsm.Context,
        instance: "RoomAudioTrackPath",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        try:
            await instance._unpublish_local_track()
        except Exception as error:
            failure = RoomAudioFailureData(failure_kind=_stable_failure_kind(error, "disconnect_failed"))
            _ = hsm.dispatch(ctx, instance, _RoomAudioFailedEvent.with_data(failure))
            return
        _ = hsm.dispatch(ctx, instance, _RoomAudioLocalTrackUnpublishedEvent)

    @staticmethod
    async def _run_disconnect_room(
        ctx: hsm.Context,
        instance: "RoomAudioTrackPath",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        try:
            await instance._disconnect_room()
        except Exception as error:
            failure = RoomAudioFailureData(failure_kind=_stable_failure_kind(error, "disconnect_failed"))
            _ = hsm.dispatch(ctx, instance, _RoomAudioFailedEvent.with_data(failure))
            return
        _ = hsm.dispatch(ctx, instance, _RoomAudioRoomDisconnectedEvent)

    @staticmethod
    async def _run_unregister_track_subscription_listener(
        ctx: hsm.Context,
        instance: "RoomAudioTrackPath",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        try:
            instance._unregister_track_subscription_listener()
        except Exception as error:
            failure = RoomAudioFailureData(failure_kind=_stable_failure_kind(error, "disconnect_failed"))
            _ = hsm.dispatch(ctx, instance, _RoomAudioFailedEvent.with_data(failure))
            return
        _ = hsm.dispatch(ctx, instance, _RoomAudioListenerUnregisteredEvent)

    @staticmethod
    async def _run_receive_remote_audio(
        ctx: hsm.Context,
        instance: "RoomAudioTrackPath",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, RoomAudioTrackSubscribedData)
        try:
            await instance._forward_remote_audio(data.track)
        except Exception as error:
            failure = RoomAudioFailureData(failure_kind=_stable_failure_kind(error, "remote_audio_failed"))
            _ = hsm.dispatch(ctx, instance, _RoomAudioFailedEvent.with_data(failure))
            return
        _ = hsm.dispatch(ctx, instance, _RoomAudioRemoteAudioEndedEvent)

    model: typing.ClassVar[hsm.Model] = bot.define(
        "RoomAudioTrackPath",
        hsm.initial(hsm.target("/RoomAudioTrackPath/disconnected")),
        hsm.state(
            "disconnected",
            hsm.transition(
                hsm.on(RoomAudioConnectEvent),
                hsm.guard(_has_connect_data),
                hsm.target("/RoomAudioTrackPath/connecting"),
            ),
        ),
        hsm.state(
            "connecting",
            hsm.entry(_register_track_subscription_listener),
            hsm.defer(_RoomAudioTrackSubscribedEvent),
            hsm.activity(_run_connect_room_audio),
            hsm.transition(
                hsm.on(RoomAudioDisconnectEvent),
                hsm.target("/RoomAudioTrackPath/disconnecting"),
            ),
            hsm.transition(
                hsm.on(_RoomAudioConnectedEvent),
                hsm.guard(_has_connected_data),
                hsm.effect(_set_connected),
                hsm.target("/RoomAudioTrackPath/connected"),
            ),
            hsm.transition(
                hsm.on(_RoomAudioFailedEvent),
                hsm.guard(_has_failure_data),
                hsm.target("/RoomAudioTrackPath/disconnecting"),
            ),
            hsm.transition(
                hsm.after(_operation_timeout_value),
                hsm.target("/RoomAudioTrackPath/disconnecting"),
            ),
        ),
        hsm.state(
            "connected",
            hsm.transition(
                hsm.on(_RoomAudioTrackSubscribedEvent),
                hsm.guard(_has_audio_track_subscribed_data),
                hsm.target("/RoomAudioTrackPath/receiving_remote_audio"),
            ),
            hsm.transition(
                hsm.on(_RoomAudioTrackSubscribedEvent),
                hsm.guard(_has_non_audio_track_subscribed_data),
            ),
            hsm.transition(
                hsm.on(_RoomAudioFailedEvent),
                hsm.guard(_has_failure_data),
                hsm.target("/RoomAudioTrackPath/disconnecting"),
            ),
            hsm.transition(
                hsm.on(RoomAudioDisconnectEvent),
                hsm.target("/RoomAudioTrackPath/disconnecting"),
            ),
        ),
        hsm.state(
            "receiving_remote_audio",
            hsm.activity(_run_receive_remote_audio),
            hsm.transition(
                hsm.on(_RoomAudioRemoteAudioEndedEvent),
                hsm.target("/RoomAudioTrackPath/connected"),
            ),
            hsm.transition(
                hsm.on(_RoomAudioTrackSubscribedEvent),
                hsm.guard(_has_audio_track_subscribed_data),
            ),
            hsm.transition(
                hsm.on(_RoomAudioTrackSubscribedEvent),
                hsm.guard(_has_non_audio_track_subscribed_data),
            ),
            hsm.transition(
                hsm.on(_RoomAudioFailedEvent),
                hsm.guard(_has_failure_data),
                hsm.target("/RoomAudioTrackPath/disconnecting"),
            ),
            hsm.transition(
                hsm.on(RoomAudioDisconnectEvent),
                hsm.target("/RoomAudioTrackPath/disconnecting"),
            ),
        ),
        hsm.state(
            "disconnecting",
            hsm.initial(hsm.target("/RoomAudioTrackPath/disconnecting/routing_disconnect_cleanup")),
            hsm.choice(
                "routing_disconnect_cleanup",
                hsm.transition(
                    hsm.guard(_has_local_track_sid),
                    hsm.target("/RoomAudioTrackPath/disconnecting/unpublishing_local_track"),
                ),
                hsm.transition(hsm.target("/RoomAudioTrackPath/disconnecting/disconnecting_room")),
            ),
            hsm.state(
                "unpublishing_local_track",
                hsm.activity(_run_unpublish_local_track),
                hsm.transition(
                    hsm.on(_RoomAudioLocalTrackUnpublishedEvent),
                    hsm.target("/RoomAudioTrackPath/disconnecting/disconnecting_room"),
                ),
                hsm.transition(
                    hsm.on(_RoomAudioFailedEvent),
                    hsm.guard(_has_failure_data),
                    hsm.target("/RoomAudioTrackPath/disconnecting/disconnecting_room_after_failure"),
                ),
                hsm.transition(
                    hsm.after(_operation_timeout_value),
                    hsm.target("/RoomAudioTrackPath/disconnecting/disconnecting_room_after_failure"),
                ),
            ),
            hsm.state(
                "disconnecting_room",
                hsm.activity(_run_disconnect_room),
                hsm.transition(
                    hsm.on(_RoomAudioRoomDisconnectedEvent),
                    hsm.target("/RoomAudioTrackPath/disconnecting/routing_listener_cleanup"),
                ),
                hsm.transition(
                    hsm.on(_RoomAudioFailedEvent),
                    hsm.guard(_has_failure_data),
                    hsm.target("/RoomAudioTrackPath/disconnecting/routing_listener_cleanup_after_failure"),
                ),
                hsm.transition(
                    hsm.after(_operation_timeout_value),
                    hsm.target("/RoomAudioTrackPath/disconnecting/routing_listener_cleanup_after_failure"),
                ),
            ),
            hsm.choice(
                "routing_listener_cleanup",
                hsm.transition(
                    hsm.guard(_has_registered_track_listener),
                    hsm.target("/RoomAudioTrackPath/disconnecting/unregistering_listener"),
                ),
                hsm.transition(hsm.effect(_clear_connected), hsm.target("/RoomAudioTrackPath/disconnected")),
            ),
            hsm.state(
                "unregistering_listener",
                hsm.activity(_run_unregister_track_subscription_listener),
                hsm.transition(
                    hsm.on(_RoomAudioListenerUnregisteredEvent),
                    hsm.effect(_clear_connected),
                    hsm.target("/RoomAudioTrackPath/disconnected"),
                ),
                hsm.transition(
                    hsm.on(_RoomAudioFailedEvent),
                    hsm.guard(_has_failure_data),
                    hsm.target("/RoomAudioTrackPath/failed"),
                ),
                hsm.transition(
                    hsm.after(_operation_timeout_value),
                    hsm.target("/RoomAudioTrackPath/failed"),
                ),
            ),
            hsm.state(
                "disconnecting_room_after_failure",
                hsm.activity(_run_disconnect_room),
                hsm.transition(
                    hsm.on(_RoomAudioRoomDisconnectedEvent),
                    hsm.target("/RoomAudioTrackPath/disconnecting/routing_listener_cleanup_after_failure"),
                ),
                hsm.transition(
                    hsm.on(_RoomAudioFailedEvent),
                    hsm.guard(_has_failure_data),
                    hsm.target("/RoomAudioTrackPath/disconnecting/routing_listener_cleanup_after_failure"),
                ),
                hsm.transition(
                    hsm.after(_operation_timeout_value),
                    hsm.target("/RoomAudioTrackPath/disconnecting/routing_listener_cleanup_after_failure"),
                ),
            ),
            hsm.choice(
                "routing_listener_cleanup_after_failure",
                hsm.transition(
                    hsm.guard(_has_registered_track_listener),
                    hsm.target("/RoomAudioTrackPath/disconnecting/unregistering_listener_after_failure"),
                ),
                hsm.transition(hsm.target("/RoomAudioTrackPath/failed")),
            ),
            hsm.state(
                "unregistering_listener_after_failure",
                hsm.activity(_run_unregister_track_subscription_listener),
                hsm.transition(
                    hsm.on(_RoomAudioListenerUnregisteredEvent),
                    hsm.target("/RoomAudioTrackPath/failed"),
                ),
                hsm.transition(
                    hsm.on(_RoomAudioFailedEvent),
                    hsm.guard(_has_failure_data),
                    hsm.target("/RoomAudioTrackPath/failed"),
                ),
                hsm.transition(
                    hsm.after(_operation_timeout_value),
                    hsm.target("/RoomAudioTrackPath/failed"),
                ),
            ),
        ),
        hsm.state("failed"),
        hsm.observe(observer),
    )


__all__ = [
    "AudioStreamFactory",
    "LocalAudioTrackFactory",
    "RemoteAudioSink",
    "RoomAudioConnectEvent",
    "RoomAudioDisconnectEvent",
    "RoomAudioConnectData",
    "RoomAudioConnectedData",
    "RoomAudioError",
    "RoomAudioFailureData",
    "RoomAudioTrackPath",
    "RoomHandle",
]

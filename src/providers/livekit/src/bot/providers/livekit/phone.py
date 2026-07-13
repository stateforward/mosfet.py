from __future__ import annotations


from bot.devices import audio
from bot.devices import phone

import asyncio
import collections.abc
import dataclasses
import datetime
import typing
import weakref

import hsm
import pydantic
import bot.device
from livekit import rtc

from bot.telemetry import observer
from bot.world import World, require_world_scope

from .audio import AudioBridge, create_audio_bridge
from .room_audio import (
    AudioStreamFactory,
    LocalAudioTrackFactory,
    RoomAudioConnectData,
    RoomAudioConnectedData,
    RoomAudioTrackPath,
    RoomHandle,
)

_DEFAULT_OPERATION_TIMEOUT = datetime.timedelta(seconds=30)
_DEFAULT_ANSWER_TIMEOUT = datetime.timedelta(seconds=30)
_DEFAULT_TRANSFER_TIMEOUT = datetime.timedelta(seconds=30)
_PHONE_SERVICE_TARGET_DESCRIPTION = (
    "Process-local phone event target attached by the owning phone service lifecycle. This target has no JSON "
    "representation and is not a provider configuration field."
)
_PHONE_SERVICE_TARGET_JSON_SCHEMA: pydantic.json_schema.JsonSchemaValue = {
    "not": {},
    "description": _PHONE_SERVICE_TARGET_DESCRIPTION,
}


def _phone_service_target_from_runtime_value(data: object) -> hsm.Instance:
    if isinstance(data, hsm.Instance):
        return data
    raise ValueError("Phone service attachment target must be the phone-owned HSM event target.")


_PhoneServiceTarget = typing.Annotated[
    hsm.Instance,
    pydantic.BeforeValidator(_phone_service_target_from_runtime_value),
    pydantic.WithJsonSchema(_PHONE_SERVICE_TARGET_JSON_SCHEMA),
]


class _PhoneServiceAttachmentData(pydantic.BaseModel):
    """Private attachment payload connecting a LiveKit phone service to a phone event target."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    target: _PhoneServiceTarget = pydantic.Field(
        description=_PHONE_SERVICE_TARGET_DESCRIPTION,
    )


_ACTIVE_OPERATION_EVENT_ATTRIBUTE = "active_operation_event"
_PRIVATE_OPERATION_ID_METADATA = "bot.provider.livekit.phone.operation_id"
_ServiceAttachedEvent = hsm.Event[_PhoneServiceAttachmentData](
    name="bot.provider.livekit.phone.attached",
    schema=_PhoneServiceAttachmentData,
)
_ServiceDetachedEvent = hsm.Event[_PhoneServiceAttachmentData](
    name="bot.provider.livekit.phone.detached",
    schema=_PhoneServiceAttachmentData,
)
_ServiceAttachmentRejectedEvent = hsm.Event[_PhoneServiceAttachmentData](
    name="bot.provider.livekit.phone.attachment.rejected",
    kind=hsm.ErrorEventKind,
    schema=_PhoneServiceAttachmentData,
)


class PhoneServiceError(RuntimeError):
    """Raised when LiveKit phone service work fails before a neutral phone event can be emitted."""

    failure_kind: phone.FailureKind

    def __init__(self, message: str, *, failure_kind: phone.FailureKind = "unknown") -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


def _raise_unavailable_call_control() -> typing.NoReturn:
    raise PhoneServiceError(
        "LiveKit server-side SIP call control is not available in the installed provider dependency.",
        failure_kind="provider_unavailable",
    )


ServiceIncomingCallEvent = hsm.Event[phone.IncomingCallData](
    name="bot.provider.livekit.phone.incoming_call",
    schema=phone.IncomingCallData,
)
ServiceMediaReadyEvent = hsm.Event[phone.MediaReadyData](
    name="bot.provider.livekit.phone.media_ready",
    schema=phone.MediaReadyData,
)
ServiceRemoteHangUpEvent = hsm.Event[phone.RemoteHangUpData](
    name="bot.provider.livekit.phone.remote_hang_up",
    schema=phone.RemoteHangUpData,
)
ServiceCallFailedEvent = hsm.Event[phone.CallFailedData](
    name="bot.provider.livekit.phone.call_failed",
    schema=phone.CallFailedData,
)
ServiceTransferCompletedEvent = hsm.Event[phone.TransferCompletedData](
    name="bot.provider.livekit.phone.transfer_completed",
    schema=phone.TransferCompletedData,
)
ServiceTransferFailedEvent = hsm.Event[phone.TransferFailedData](
    name="bot.provider.livekit.phone.transfer_failed",
    schema=phone.TransferFailedData,
)
# Private: room remote PCM enters the service machine; delivery is gated by HSM guards.
_RemoteAudioReceivedEvent = hsm.Event[audio.AudioInputData](
    name="bot.provider.livekit.phone.remote_audio.received",
    schema=audio.AudioInputData,
)
_RoomAudioStatusEvent = hsm.Event[RoomAudioConnectedData](
    name="bot.provider.livekit.phone.room_audio.status",
    schema=RoomAudioConnectedData,
)

_AnswerCompletedEvent = hsm.Event[phone.CallConnectedData](
    name="bot.provider.livekit.phone.answer.completed",
    kind=hsm.CompletionEventKind,
    schema=phone.CallConnectedData,
)
_DialCompletedEvent = hsm.Event[phone.CallConnectedData](
    name="bot.provider.livekit.phone.dial.completed",
    kind=hsm.CompletionEventKind,
    schema=phone.CallConnectedData,
)
_AnswerFailedEvent = hsm.Event[phone.CallFailedData](
    name="bot.provider.livekit.phone.answer.failed",
    kind=hsm.ErrorEventKind,
    schema=phone.CallFailedData,
)
_DialFailedEvent = hsm.Event[phone.CallFailedData](
    name="bot.provider.livekit.phone.dial.failed",
    kind=hsm.ErrorEventKind,
    schema=phone.CallFailedData,
)
_DeclineCompletedEvent = hsm.Event[phone.CallIdData](
    name="bot.provider.livekit.phone.decline.completed",
    kind=hsm.CompletionEventKind,
    schema=phone.CallIdData,
)
_DeclineFailedEvent = hsm.Event[phone.CallFailedData](
    name="bot.provider.livekit.phone.decline.failed",
    kind=hsm.ErrorEventKind,
    schema=phone.CallFailedData,
)
_HangUpCompletedEvent = hsm.Event[phone.CallIdData](
    name="bot.provider.livekit.phone.hang_up.completed",
    kind=hsm.CompletionEventKind,
    schema=phone.CallIdData,
)
_HangUpFailedEvent = hsm.Event[phone.CallFailedData](
    name="bot.provider.livekit.phone.hang_up.failed",
    kind=hsm.ErrorEventKind,
    schema=phone.CallFailedData,
)
_TransferAcceptedEvent = hsm.Event[phone.TransferAcceptedData](
    name="bot.provider.livekit.phone.transfer.accepted",
    kind=hsm.CompletionEventKind,
    schema=phone.TransferAcceptedData,
)
_TransferRequestFailedEvent = hsm.Event[phone.TransferFailedData](
    name="bot.provider.livekit.phone.transfer.request_failed",
    kind=hsm.ErrorEventKind,
    schema=phone.TransferFailedData,
)

_PHONE_REQUEST_EVENT_NAMES = frozenset(
    {
        phone.ServiceDialRequestedEvent.name,
        phone.ServiceAnswerRequestedEvent.name,
        phone.ServiceDeclineRequestedEvent.name,
        phone.ServiceHangUpRequestedEvent.name,
        phone.ServiceTransferRequestedEvent.name,
    }
)

_PhoneServiceEffect = collections.abc.Callable[
    [hsm.Context, "PhoneService", hsm.Event[typing.Any]],
    None,
]
_PhoneServiceAfter = collections.abc.Callable[
    [hsm.Context, "PhoneService", hsm.Event[typing.Any]],
    datetime.timedelta,
]


def _require_positive_timeout(value: datetime.timedelta) -> None:
    if value <= datetime.timedelta():
        raise ValueError("operation_timeout must be a positive duration.")


def _failure_kind(error: Exception) -> phone.FailureKind:
    if isinstance(error, PhoneServiceError):
        return error.failure_kind
    return "unknown"


def _public_metadata(metadata: collections.abc.Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in metadata.items() if key != _PRIVATE_OPERATION_ID_METADATA}


def _with_trigger_metadata[T](event: hsm.Event[T], trigger: hsm.Event[typing.Any]) -> hsm.Event[T]:
    return dataclasses.replace(event, metadata=dict(trigger.metadata))


def _with_operation_metadata[T](event: hsm.Event[T], trigger: hsm.Event[typing.Any]) -> hsm.Event[T]:
    metadata = dict(trigger.metadata)
    if trigger.id:
        metadata[_PRIVATE_OPERATION_ID_METADATA] = trigger.id
    return dataclasses.replace(event, metadata=metadata)


def _active_operation_event(instance: "PhoneService") -> hsm.Event[typing.Any] | None:
    value, ok = typing.cast(tuple[object, bool], instance.get(_ACTIVE_OPERATION_EVENT_ATTRIBUTE))
    if not ok or not isinstance(value, hsm.Event):
        return None
    return value


def _set_active_operation_event(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
    del ctx
    _ = instance.set(_ACTIVE_OPERATION_EVENT_ATTRIBUTE, _with_operation_metadata(event, event))


def _clear_active_operation_event(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
    del ctx, event
    _ = instance.set(_ACTIVE_OPERATION_EVENT_ATTRIBUTE, None)


def _active_operation_call_id(instance: "PhoneService") -> str | None:
    active = _active_operation_event(instance)
    if active is None:
        return None
    data = active.data
    if isinstance(
        data,
        phone.DialData | phone.AnswerCallData | phone.DeclineCallData | phone.HangUpCallData | phone.TransferCallData,
    ):
        return data.call_id
    return None


def _active_operation_transfer(instance: "PhoneService") -> phone.TransferCallData | None:
    active = _active_operation_event(instance)
    if active is None or not isinstance(active.data, phone.TransferCallData):
        return None
    return active.data


def _matches_active_call(instance: "PhoneService", call_id: str) -> bool:
    return _active_operation_call_id(instance) == call_id


def _matches_active_operation_metadata(instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    active = _active_operation_event(instance)
    if active is None:
        return False
    active_id = active.metadata.get(_PRIVATE_OPERATION_ID_METADATA)
    return active_id is not None and event.metadata.get(_PRIVATE_OPERATION_ID_METADATA) == active_id


def _matches_active_transfer(
    instance: "PhoneService",
    data: phone.PhoneTransferData
    | phone.PhoneTransferFailedData
    | phone.TransferAcceptedData
    | phone.TransferCompletedData
    | phone.TransferFailedData,
) -> bool:
    active = _active_operation_transfer(instance)
    return (
        active is not None
        and active.call_id == data.call_id
        and active.transfer_id == data.transfer_id
        and active.target == data.target
    )


def _has_incoming_call(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.IncomingCallData)


def _has_media_ready(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.MediaReadyData)


def _has_passive_call_observation(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return (event.name == ServiceIncomingCallEvent.name and isinstance(event.data, phone.IncomingCallData)) or (
        event.name == ServiceMediaReadyEvent.name and isinstance(event.data, phone.MediaReadyData)
    )


def _has_remote_hang_up(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.RemoteHangUpData)


def _has_call_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.CallFailedData)


def _has_transfer_completed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.TransferCompletedData)


def _has_transfer_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.TransferFailedData)


def _has_transfer_terminal_observation(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return (
        event.name == ServiceTransferCompletedEvent.name and isinstance(event.data, phone.TransferCompletedData)
    ) or (event.name == ServiceTransferFailedEvent.name and isinstance(event.data, phone.TransferFailedData))


def _has_answer_request(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.AnswerCallData)


def _has_dial_request(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.DialData)


def _has_decline_request(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.DeclineCallData)


def _has_hang_up_request(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.HangUpCallData)


def _has_transfer_request(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.TransferCallData)


def _has_provider_terminal_hang_up(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    data = event.data
    return (
        event.name == phone.HungUpEvent.name
        and isinstance(data, phone.PhoneHungUpData)
        and data.outcome in {"declined", "failed", "local_hang_up", "remote_hang_up"}
        and _matches_active_call(instance, data.call_id)
    )


def _has_provider_transfer_completed(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    return (
        event.name == phone.CallTransferCompletedEvent.name
        and isinstance(event.data, phone.PhoneTransferData)
        and _matches_active_transfer(instance, event.data)
    )


def _has_provider_transfer_failed(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    return (
        event.name == phone.CallTransferFailedEvent.name
        and isinstance(event.data, phone.PhoneTransferFailedData)
        and _matches_active_transfer(instance, event.data)
    )


def _has_current_remote_hang_up(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    return isinstance(event.data, phone.RemoteHangUpData) and _matches_active_call(instance, event.data.call_id)


def _has_current_call_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    return isinstance(event.data, phone.CallFailedData) and _matches_active_call(instance, event.data.call_id)


def _has_current_transfer_completed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    return isinstance(event.data, phone.TransferCompletedData) and _matches_active_transfer(instance, event.data)


def _has_current_transfer_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    return isinstance(event.data, phone.TransferFailedData) and _matches_active_transfer(instance, event.data)


def _has_uncorrelated_remote_hang_up(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    return _active_operation_event(instance) is None and _has_remote_hang_up(ctx, instance, event)


def _has_uncorrelated_call_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    return _active_operation_event(instance) is None and _has_call_failed(ctx, instance, event)


def _has_uncorrelated_transfer_completed(
    ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]
) -> bool:
    return _active_operation_event(instance) is None and _has_transfer_completed(ctx, instance, event)


def _has_uncorrelated_transfer_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    return _active_operation_event(instance) is None and _has_transfer_failed(ctx, instance, event)


def _has_active_call_operation(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, event
    active = _active_operation_event(instance)
    return active is not None and isinstance(
        active.data, phone.DialData | phone.AnswerCallData | phone.DeclineCallData | phone.HangUpCallData
    )


def _has_active_transfer_operation(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, event
    active = _active_operation_event(instance)
    return active is not None and isinstance(active.data, phone.TransferCallData)


async def _await_call_control_operation(
    operation: collections.abc.Awaitable[None],
) -> phone.FailureKind | None:
    try:
        await operation
        return None
    except Exception as error:
        return _failure_kind(error)


def _has_call_connected_completion(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    return (
        isinstance(event.data, phone.CallConnectedData)
        and _matches_active_call(instance, event.data.call_id)
        and _matches_active_operation_metadata(instance, event)
    )


def _has_call_id_completion(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    return (
        isinstance(event.data, phone.CallIdData)
        and _matches_active_call(instance, event.data.call_id)
        and _matches_active_operation_metadata(instance, event)
    )


def _has_transfer_accepted_completion(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    return (
        isinstance(event.data, phone.TransferAcceptedData)
        and _matches_active_transfer(instance, event.data)
        and _matches_active_operation_metadata(instance, event)
    )


def _has_call_failed_completion(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    return (
        isinstance(event.data, phone.CallFailedData)
        and _matches_active_call(instance, event.data.call_id)
        and _matches_active_operation_metadata(instance, event)
    )


def _has_transfer_failed_completion(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    return (
        isinstance(event.data, phone.TransferFailedData)
        and _matches_active_transfer(instance, event.data)
        and _matches_active_operation_metadata(instance, event)
    )


def _ignore_passive_call_observation_transition() -> hsm.Element:
    return hsm.transition(
        hsm.on(ServiceIncomingCallEvent, ServiceMediaReadyEvent),
        hsm.guard(_has_passive_call_observation),
    )


def _ignore_transfer_terminal_observation_transition() -> hsm.Element:
    return hsm.transition(
        hsm.on(ServiceTransferCompletedEvent, ServiceTransferFailedEvent),
        hsm.guard(_has_transfer_terminal_observation),
    )


def _active_call_operation_resolution_transitions(
    *,
    emit_remote_hang_up: _PhoneServiceEffect,
    emit_call_failed: _PhoneServiceEffect,
) -> tuple[hsm.Element, ...]:
    return (
        hsm.transition(
            hsm.on(ServiceRemoteHangUpEvent),
            hsm.guard(_has_current_remote_hang_up),
            hsm.effect(emit_remote_hang_up),
            hsm.effect(_clear_active_operation_event),
            hsm.target("/PhoneService/ready"),
        ),
        hsm.transition(
            hsm.on(ServiceCallFailedEvent),
            hsm.guard(_has_current_call_failed),
            hsm.effect(emit_call_failed),
            hsm.effect(_clear_active_operation_event),
            hsm.target("/PhoneService/ready"),
        ),
        hsm.transition(
            hsm.on(phone.HungUpEvent),
            hsm.guard(_has_provider_terminal_hang_up),
            hsm.effect(_clear_active_operation_event),
            hsm.target("/PhoneService/ready"),
        ),
    )


def _active_call_operation_timeout_transition(
    *,
    operation_timeout: _PhoneServiceAfter,
    emit_active_call_timeout: _PhoneServiceEffect,
) -> hsm.Element:
    return hsm.transition(
        hsm.after(operation_timeout),
        hsm.guard(_has_active_call_operation),
        hsm.effect(emit_active_call_timeout),
        hsm.effect(_clear_active_operation_event),
        hsm.target("/PhoneService/ready"),
    )


class MediaSnapshot(pydantic.BaseModel):
    """Low-cardinality room-media proof counters owned by a LiveKit phone service."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Observable LiveKit phone-service media counters without callback internals.",
            "examples": [
                {
                    "local_track_sid": "TR_123",
                    "remote_audio_chunks": 2,
                    "remote_audio_bytes": 3840,
                    "remote_audio_dropped_chunks": 1,
                    "remote_audio_dropped_bytes": 960,
                }
            ],
        },
    )

    local_track_sid: str | None = pydantic.Field(
        default=None,
        description="LiveKit track SID for the local published audio track when connected.",
        examples=["TR_123"],
    )
    remote_audio_chunks: int = pydantic.Field(
        ge=0,
        description="Number of remote audio chunks delivered to phone firmware as ServiceAudioReceived.",
        examples=[2],
    )
    remote_audio_bytes: int = pydantic.Field(
        ge=0,
        description="Number of remote audio bytes delivered to phone firmware as ServiceAudioReceived.",
        examples=[3840],
    )
    remote_audio_dropped_chunks: int = pydantic.Field(
        ge=0,
        description=(
            "Number of remote audio chunks rejected by PhoneService HSM guards "
            "(no media call id and/or no attached phone target)."
        ),
        examples=[1],
    )
    remote_audio_dropped_bytes: int = pydantic.Field(
        ge=0,
        description="Number of remote audio bytes rejected by PhoneService HSM guards.",
        examples=[960],
    )


class PhoneService(hsm.Instance):
    """stateforward.bot-native LiveKit phone service that adapts LiveKit call control and room media to phone firmware events."""

    _operation_timeout: datetime.timedelta
    _loop: asyncio.AbstractEventLoop | None
    _room: RoomHandle | None
    _room_media_configured: bool
    _stream_factory: AudioStreamFactory | None
    _local_track_factory: LocalAudioTrackFactory | None
    _room_connect: RoomAudioConnectData | None
    _bridge: AudioBridge[typing.Any] | None
    _track_path: RoomAudioTrackPath | None
    _local_track_sid: str | None
    _remote_audio_chunks: int
    _remote_audio_bytes: int
    _remote_audio_dropped_chunks: int
    _remote_audio_dropped_bytes: int
    _media_call_id: str | None
    _delivering_remote_audio: bool
    _attached_phone_target_ref: weakref.ReferenceType[hsm.Instance] | None
    _presence_bound: bool
    _presence_joined_callback: collections.abc.Callable[..., object] | None
    _presence_left_callback: collections.abc.Callable[..., object] | None
    _presence_call_ids: dict[str, str]

    def __init__(
        self,
        *,
        url: str | None = None,
        token: str | None = None,
        track_name: str = "bot-audio",
        operation_timeout: datetime.timedelta = _DEFAULT_OPERATION_TIMEOUT,
        room: RoomHandle | None = None,
        stream_factory: AudioStreamFactory | None = None,
        local_track_factory: LocalAudioTrackFactory | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        _require_positive_timeout(operation_timeout)
        if (url is None) != (token is None):
            raise ValueError("url and token must both be provided or both omitted.")
        if not track_name:
            raise ValueError("track_name must be a non-empty string.")
        self._operation_timeout = operation_timeout
        self._loop = loop
        self._room = room
        self._room_media_configured = room is not None or url is not None
        self._stream_factory = stream_factory
        self._local_track_factory = local_track_factory
        self._room_connect = (
            None if url is None or token is None else RoomAudioConnectData(url=url, token=token, track_name=track_name)
        )
        self._bridge = None
        self._track_path = None
        self._local_track_sid = None
        self._remote_audio_chunks = 0
        self._remote_audio_bytes = 0
        self._remote_audio_dropped_chunks = 0
        self._remote_audio_dropped_bytes = 0
        self._media_call_id = None
        self._delivering_remote_audio = False
        self._attached_phone_target_ref = None
        self._presence_bound = False
        self._presence_joined_callback = None
        self._presence_left_callback = None
        self._presence_call_ids = {}

    def _room_media_enabled(self) -> bool:
        """True when this service is configured for LiveKit room media (not bare call-control stub)."""

        # Do not treat a lazy media bridge as room-media: bare PhoneService() still reports
        # call control unavailable until url/token, an injected room, or a live room connect.
        return self._room_media_configured or self._local_track_sid is not None

    @staticmethod
    def _participant_identity(participant: object) -> str:
        identity = getattr(participant, "identity", None)
        if isinstance(identity, str) and identity:
            return identity
        sid = getattr(participant, "sid", None)
        if isinstance(sid, str) and sid:
            return sid
        return "remote"

    @staticmethod
    def _call_id_for_participant(identity: str) -> str:
        return f"livekit:{identity}"

    def _bind_room_presence(self, room: RoomHandle) -> None:
        """Map remote room presence to phone call observations (ring / remote hang-up).

        Does not auto-answer: the bot (or operator) decides whether to answer.
        """

        if self._presence_bound:
            return
        service_ref = weakref.ref(self)

        def on_participant_connected(participant: object) -> object:
            live_service = service_ref()
            if live_service is None or not live_service.state():
                return None
            live_service._offer_incoming_call_for_participant(participant)
            return None

        def on_participant_disconnected(participant: object) -> object:
            live_service = service_ref()
            if live_service is None or not live_service.state():
                return None
            live_service._hang_up_for_participant(participant)
            return None

        _ = room.on("participant_connected", on_participant_connected)
        _ = room.on("participant_disconnected", on_participant_disconnected)
        self._presence_joined_callback = on_participant_connected
        self._presence_left_callback = on_participant_disconnected
        self._presence_bound = True

    def _scan_existing_remote_participants(self, room: RoomHandle) -> None:
        """Ring for remotes already in the room when the bot joins late."""

        remote_participants = getattr(room, "remote_participants", None)
        if remote_participants is None:
            return
        values: collections.abc.Iterable[object]
        if isinstance(remote_participants, collections.abc.Mapping):
            values = typing.cast(collections.abc.Mapping[object, object], remote_participants).values()
        elif isinstance(remote_participants, collections.abc.Iterable):
            values = typing.cast(collections.abc.Iterable[object], remote_participants)
        else:
            return
        for participant in values:
            self._offer_incoming_call_for_participant(participant)

    def _offer_incoming_call_for_participant(self, participant: object) -> None:
        if not self.state():
            return
        # One active media call at a time for this room-phone mapping.
        if self._media_call_id is not None:
            return
        identity = PhoneService._participant_identity(participant)
        call_id = PhoneService._call_id_for_participant(identity)
        self._presence_call_ids[identity] = call_id
        _ = self.dispatch(
            self.context(),
            ServiceIncomingCallEvent.with_data(phone.IncomingCallData(call_id=call_id, display_hint=identity)),
        )

    def _hang_up_for_participant(self, participant: object) -> None:
        if not self.state():
            return
        identity = PhoneService._participant_identity(participant)
        call_id = self._presence_call_ids.pop(identity, PhoneService._call_id_for_participant(identity))
        if self._media_call_id is not None and self._media_call_id != call_id:
            return
        _ = self.dispatch(
            self.context(),
            ServiceRemoteHangUpEvent.with_data(phone.RemoteHangUpData(call_id=call_id)),
        )

    def _emit_media_ready_if_track_live(self, call_id: str) -> None:
        """If the room local track is already published, advance firmware to media_ready."""

        if self._local_track_sid is None:
            return
        phone_event_target = PhoneService._phone_event_target(self)
        if phone_event_target is None:
            return
        _ = phone_event_target.dispatch(
            phone_event_target.context(),
            dataclasses.replace(
                phone.ServiceMediaReadyEvent.with_data(phone.MediaReadyData(call_id=call_id)),
                source=hsm.id(self),
                target=hsm.id(phone_event_target),
            ),
        )

    def _ensure_media(self) -> tuple[AudioBridge[typing.Any], RoomAudioTrackPath]:
        if self._bridge is not None and self._track_path is not None:
            return self._bridge, self._track_path
        service_ref = weakref.ref(self)

        async def consume_remote_audio(audio_input: audio.AudioInputData) -> None:
            """Ingress: publish remote PCM into PhoneService RTC (guards decide deliver vs drop).

            Do not gate on ``context().is_done()``: after device firmware attach, the service
            context can report done while the machine is still live and accepting events.
            Use started state instead; HSM guards decide deliver vs drop.
            """

            live_service = service_ref()
            if live_service is None or not live_service.state():
                return
            await live_service.receive_remote_audio(live_service.context(), audio_input)

        def record_connected(data: RoomAudioConnectedData) -> None:
            live_service = service_ref()
            if live_service is None or not live_service.state():
                return
            _ = live_service.dispatch(live_service.context(), _RoomAudioStatusEvent.with_data(data))

        bridge = create_audio_bridge(loop=self._loop)
        room = self._room
        if room is None:
            room = typing.cast(RoomHandle, typing.cast(object, rtc.Room(loop=self._loop)))
            self._room = room
        track_path = RoomAudioTrackPath(
            bridge=bridge,
            remote_audio_sink=consume_remote_audio,
            connection_sink=record_connected,
            room=room,
            stream_factory=self._stream_factory,
            local_track_factory=self._local_track_factory,
            operation_timeout=self._operation_timeout,
            loop=self._loop,
        )
        self._bridge = bridge
        self._track_path = track_path
        if self._room is not None:
            self._bind_room_presence(self._room)
        return bridge, track_path

    async def dial(self, request: phone.DialData) -> phone.CallConnectedData:
        """Dial an outbound LiveKit/SIP call. Default installation reports call control unavailable."""

        del request
        _raise_unavailable_call_control()

    async def answer_call(self, request: phone.AnswerCallData) -> None:
        """Answer a call.

        Room-media installations complete answer locally (bot decides to answer; no SIP
        gateway required). Pure call-control stubs without room media still report unavailable.
        """

        del request
        if not self._room_media_enabled():
            _raise_unavailable_call_control()

    async def decline_call(self, request: phone.DeclineCallData) -> None:
        """Decline a ringing call. Room-media installations complete decline locally."""

        del request
        if not self._room_media_enabled():
            _raise_unavailable_call_control()

    async def hang_up_call(self, request: phone.HangUpCallData) -> None:
        """Hang up an active call. Room-media installations complete hang-up locally."""

        del request
        if not self._room_media_enabled():
            _raise_unavailable_call_control()

    async def transfer_call(self, request: phone.TransferCallData) -> None:
        """Transfer an active LiveKit/SIP call. Default installation reports call control unavailable."""

        del request
        _raise_unavailable_call_control()

    async def attach(self, world: World, target: hsm.Instance) -> None:
        """Attach this service to the phone-owned firmware target."""

        require_world_scope(world, target, participant="Phone service target")
        _, track_path = self._ensure_media()
        if not track_path.state():
            _ = await hsm.started(world.context, track_path, track_path.model)
        # Start this service before room connect so participant_connected can ring.
        if not self.state():
            _ = await hsm.started(world.context, self, self.model)
        require_world_scope(world, self, participant="PhoneService")
        await self.dispatch(world.context, _ServiceAttachedEvent.with_data(_PhoneServiceAttachmentData(target=target)))
        target_ref = self._attached_phone_target_ref
        current_target = None if target_ref is None else target_ref()
        if current_target is not target:
            raise PhoneServiceError(
                "LiveKit phone service is already attached to another phone event target.",
                failure_kind="provider_unavailable",
            )
        if self._room_connect is not None and self._local_track_sid is None:
            await track_path.connect_room(track_path.context(), self._room_connect)
            room = self._room
            assert room is not None
            self._scan_existing_remote_participants(room)

    async def detach(self, world: World, target: hsm.Instance) -> None:
        """Detach this service from the phone-owned firmware target."""

        if not self.state():
            return
        require_world_scope(world, target, participant="Phone service target")
        require_world_scope(world, self, participant="PhoneService")
        await self.dispatch(world.context, _ServiceDetachedEvent.with_data(_PhoneServiceAttachmentData(target=target)))

    async def connect_room(
        self,
        *,
        url: str,
        token: str,
        track_name: str = "bot-audio",
    ) -> None:
        """Connect the privately owned LiveKit room audio path for this service."""

        _, track_path = self._ensure_media()
        if not track_path.state():
            _ = await hsm.started(self.context() if self.state() else None, track_path, track_path.model)
        await track_path.connect_room(
            track_path.context(),
            RoomAudioConnectData(url=url, token=token, track_name=track_name),
        )
        room = self._room
        assert room is not None
        self._scan_existing_remote_participants(room)

    async def disconnect_room(self) -> None:
        """Disconnect room audio from the underlying LiveKit room."""

        _, track_path = self._ensure_media()
        await track_path.disconnect_room(track_path.context())

    async def publish_audio(self, output: audio.AudioOutputData) -> None:
        """Publish generated or encoded audio through the local LiveKit audio track."""

        bridge, _ = self._ensure_media()
        await bridge.publish_audio(output)

    def media_snapshot(self) -> MediaSnapshot:
        """Return low-cardinality room-media proof counters for this service."""

        _ = self._ensure_media()
        return MediaSnapshot(
            local_track_sid=self._local_track_sid,
            remote_audio_chunks=self._remote_audio_chunks,
            remote_audio_bytes=self._remote_audio_bytes,
            remote_audio_dropped_chunks=self._remote_audio_dropped_chunks,
            remote_audio_dropped_bytes=self._remote_audio_dropped_bytes,
        )

    @staticmethod
    def _operation_timeout_value(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, event
        return instance._operation_timeout

    @staticmethod
    def _matches_service_attachment(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _PhoneServiceAttachmentData):
            return False
        target_ref = instance._attached_phone_target_ref
        return target_ref is not None and target_ref() is data.target

    @staticmethod
    def _has_conflicting_service_attachment(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _PhoneServiceAttachmentData):
            return False
        target_ref = instance._attached_phone_target_ref
        current_target = None if target_ref is None else target_ref()
        return current_target is not None and current_target is not data.target

    @staticmethod
    def _has_attachable_service_target(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _PhoneServiceAttachmentData):
            return False
        target_ref = instance._attached_phone_target_ref
        return target_ref is None or target_ref() is None

    @staticmethod
    def _set_phone_event_target(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _PhoneServiceAttachmentData)
        instance._attached_phone_target_ref = weakref.ref(data.target)

    @staticmethod
    def _clear_phone_event_target(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._attached_phone_target_ref = None

    @staticmethod
    def _apply_room_audio_status(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, RoomAudioConnectedData)
        instance._local_track_sid = data.local_track_sid
        room = instance._room
        if room is not None:
            instance._bind_room_presence(room)
            instance._scan_existing_remote_participants(room)
        call_id = instance._media_call_id
        phone_event_target = PhoneService._phone_event_target(instance)
        if call_id is None or phone_event_target is None or data.local_track_sid is None:
            return
        _ = phone_event_target.dispatch(
            ctx,
            dataclasses.replace(
                phone.ServiceMediaReadyEvent.with_data(phone.MediaReadyData(call_id=call_id)),
                source=hsm.id(instance),
                target=hsm.id(phone_event_target),
            ),
        )

    @staticmethod
    def _has_remote_audio(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, audio.AudioInputData)

    @staticmethod
    def _phone_event_target(instance: "PhoneService") -> hsm.Instance | None:
        target_ref = instance._attached_phone_target_ref
        if target_ref is None:
            return None
        return target_ref()

    @staticmethod
    def _can_deliver_remote_audio(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        if not isinstance(event.data, audio.AudioInputData):
            return False
        if instance._media_call_id is None:
            return False
        return PhoneService._phone_event_target(instance) is not None

    @staticmethod
    def _cannot_deliver_remote_audio(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
        return PhoneService._has_remote_audio(ctx, instance, event) and not PhoneService._can_deliver_remote_audio(
            ctx, instance, event
        )

    @staticmethod
    def _deliver_remote_audio_to_phone(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, audio.AudioInputData)
        call_id = instance._media_call_id
        phone_event_target = PhoneService._phone_event_target(instance)
        assert call_id is not None
        assert phone_event_target is not None
        instance._remote_audio_chunks += 1
        instance._remote_audio_bytes += len(data.audio)
        instance._delivering_remote_audio = True
        try:
            _ = phone_event_target.dispatch(
                ctx,
                dataclasses.replace(
                    phone.ServiceAudioReceivedEvent.with_data(
                        phone.ServiceAudioData(
                            call_id=call_id,
                            audio=data.audio,
                            media_type=data.media_type,
                            sample_rate_hz=data.sample_rate_hz,
                            channels=data.channels,
                        )
                    ),
                    source=hsm.id(instance),
                    target=hsm.id(phone_event_target),
                    metadata=_public_metadata(event.metadata),
                ),
            )
        finally:
            instance._delivering_remote_audio = False

    @staticmethod
    def _drop_remote_audio(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, audio.AudioInputData)
        instance._remote_audio_dropped_chunks += 1
        instance._remote_audio_dropped_bytes += len(data.audio)

    @staticmethod
    def _dispatch_attachment_rejected(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _PhoneServiceAttachmentData)
        _ = hsm.dispatch(ctx, instance, _ServiceAttachmentRejectedEvent.with_data(data))

    @staticmethod
    def _emit_phone_event(
        ctx: hsm.Context,
        instance: "PhoneService",
        trigger: hsm.Event[typing.Any],
        event: hsm.Event[typing.Any],
    ) -> None:
        target_ref = instance._attached_phone_target_ref
        assert target_ref is not None
        phone_event_target = target_ref()
        assert phone_event_target is not None
        _ = phone_event_target.dispatch(
            ctx,
            dataclasses.replace(
                event,
                source=hsm.id(instance),
                target=hsm.id(phone_event_target),
                metadata=_public_metadata(trigger.metadata),
            ),
        )

    @staticmethod
    def _emit_incoming_call(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.IncomingCallData)
        instance._media_call_id = data.call_id
        PhoneService._emit_phone_event(ctx, instance, event, phone.IncomingCallEvent.with_data(data))

    @staticmethod
    def _emit_media_ready(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.MediaReadyData)
        instance._media_call_id = data.call_id
        PhoneService._emit_phone_event(ctx, instance, event, phone.ServiceMediaReadyEvent.with_data(data))

    @staticmethod
    def _emit_remote_hang_up(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.RemoteHangUpData)
        instance._media_call_id = None
        PhoneService._emit_phone_event(ctx, instance, event, phone.RemoteHangUpEvent.with_data(data))

    @staticmethod
    def _emit_call_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.CallFailedData)
        instance._media_call_id = None
        PhoneService._emit_phone_event(ctx, instance, event, phone.CallFailedEvent.with_data(data))

    @staticmethod
    def _emit_transfer_accepted(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.TransferAcceptedData)
        PhoneService._emit_phone_event(ctx, instance, event, phone.TransferAcceptedEvent.with_data(data))

    @staticmethod
    def _emit_transfer_completed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.TransferCompletedData)
        PhoneService._emit_phone_event(ctx, instance, event, phone.ServiceTransferCompletedEvent.with_data(data))

    @staticmethod
    def _emit_transfer_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.TransferFailedData)
        PhoneService._emit_phone_event(ctx, instance, event, phone.ServiceTransferFailedEvent.with_data(data))

    @staticmethod
    def _emit_call_connected(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.CallConnectedData)
        instance._media_call_id = data.call_id
        PhoneService._emit_phone_event(ctx, instance, event, phone.CallConnectedEvent.with_data(data))

    @staticmethod
    def _emit_active_call_timeout(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        active = _active_operation_event(instance)
        assert active is not None
        data = active.data
        assert isinstance(data, phone.DialData | phone.AnswerCallData | phone.DeclineCallData | phone.HangUpCallData)
        failure = phone.CallFailedData(call_id=data.call_id, failure_kind="timeout")
        PhoneService._emit_call_failed(
            ctx,
            instance,
            _with_trigger_metadata(ServiceCallFailedEvent.with_data(failure), active),
        )

    @staticmethod
    def _emit_active_transfer_timeout(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        active = _active_operation_event(instance)
        assert active is not None
        assert isinstance(active.data, phone.TransferCallData)
        data = active.data
        failure = phone.TransferFailedData(
            call_id=data.call_id,
            transfer_id=data.transfer_id,
            target=data.target,
            failure_kind="timeout",
        )
        PhoneService._emit_transfer_failed(
            ctx,
            instance,
            _with_trigger_metadata(ServiceTransferFailedEvent.with_data(failure), active),
        )

    @staticmethod
    async def _run_answer_call(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, phone.AnswerCallData)
        failure_kind = await _await_call_control_operation(instance.answer_call(data))
        if failure_kind is not None:
            failure = phone.CallFailedData(call_id=data.call_id, failure_kind=failure_kind)
            _ = hsm.dispatch(ctx, instance, _with_operation_metadata(_AnswerFailedEvent.with_data(failure), event))
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            _with_operation_metadata(
                _AnswerCompletedEvent.with_data(phone.CallConnectedData(call_id=data.call_id)),
                event,
            ),
        )
        # After answer completes, if the room local track is already live, advance media_ready
        # (participant may have joined after the bot already connected the room).
        if instance._local_track_sid is not None:
            _ = hsm.dispatch(
                ctx,
                instance,
                ServiceMediaReadyEvent.with_data(phone.MediaReadyData(call_id=data.call_id)),
            )

    @staticmethod
    async def _run_dial(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.DialData)
        try:
            connected = await instance.dial(data)
        except Exception as error:
            failure = phone.CallFailedData(call_id=data.call_id, failure_kind=_failure_kind(error))
            _ = hsm.dispatch(ctx, instance, _with_operation_metadata(_DialFailedEvent.with_data(failure), event))
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            _with_operation_metadata(_DialCompletedEvent.with_data(connected), event),
        )

    @staticmethod
    async def _run_decline_call(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, phone.DeclineCallData)
        failure_kind = await _await_call_control_operation(instance.decline_call(data))
        if failure_kind is not None:
            failure = phone.CallFailedData(call_id=data.call_id, failure_kind=failure_kind)
            _ = hsm.dispatch(ctx, instance, _with_operation_metadata(_DeclineFailedEvent.with_data(failure), event))
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            _with_operation_metadata(_DeclineCompletedEvent.with_data(phone.CallIdData(call_id=data.call_id)), event),
        )

    @staticmethod
    async def _run_hang_up_call(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, phone.HangUpCallData)
        failure_kind = await _await_call_control_operation(instance.hang_up_call(data))
        if failure_kind is not None:
            failure = phone.CallFailedData(call_id=data.call_id, failure_kind=failure_kind)
            _ = hsm.dispatch(ctx, instance, _with_operation_metadata(_HangUpFailedEvent.with_data(failure), event))
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            _with_operation_metadata(_HangUpCompletedEvent.with_data(phone.CallIdData(call_id=data.call_id)), event),
        )

    @staticmethod
    async def _run_transfer_call(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, phone.TransferCallData)
        failure_kind = await _await_call_control_operation(instance.transfer_call(data))
        if failure_kind is not None:
            failure = phone.TransferFailedData(
                call_id=data.call_id,
                transfer_id=data.transfer_id,
                target=data.target,
                failure_kind=failure_kind,
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                _with_operation_metadata(_TransferRequestFailedEvent.with_data(failure), event),
            )
            return
        accepted = phone.TransferAcceptedData(call_id=data.call_id, transfer_id=data.transfer_id, target=data.target)
        _ = hsm.dispatch(ctx, instance, _with_operation_metadata(_TransferAcceptedEvent.with_data(accepted), event))

    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        """Accept phone firmware provider requests through the `PhoneService` protocol."""

        if event.name == audio.OutputEvent.name:
            # Local speaker audio offered as call uplink. Skip while delivering remote audio
            # so ServiceAudioReceived → speaker does not loop back onto the LiveKit track.
            if self._delivering_remote_audio:
                return
            data = event.data
            if not isinstance(data, audio.AudioOutputData):
                return
            _ = asyncio.ensure_future(self.publish_audio(data))
            return
        if event.name == phone.HungUpEvent.name:
            self._media_call_id = None
        if event.name not in _PHONE_REQUEST_EVENT_NAMES and event.name not in {
            phone.HungUpEvent.name,
            ServiceTransferCompletedEvent.name,
            ServiceTransferFailedEvent.name,
        }:
            return
        target_ref = self._attached_phone_target_ref
        current_target = None if target_ref is None else target_ref()
        if current_target is None or event.target != hsm.id(self):
            return
        if event.source == hsm.id(current_target):
            _ = self.dispatch(ctx, event)
            return
        if event.name in _PHONE_REQUEST_EVENT_NAMES:
            instances = ctx.value(hsm.Keys.Instances)
            source_instance: hsm.Instance | None = None
            if event.source and isinstance(instances, collections.abc.Mapping):
                source = typing.cast(collections.abc.Mapping[object, object], instances).get(event.source)
                if isinstance(source, hsm.Instance):
                    source_instance = source
            if source_instance is None:
                return
            if isinstance(event.data, phone.TransferCallData):
                failure_event = phone.ServiceTransferFailedEvent.with_data(
                    phone.TransferFailedData(
                        call_id=event.data.call_id,
                        transfer_id=event.data.transfer_id,
                        target=event.data.target,
                        failure_kind="provider_unavailable",
                    )
                )
            elif isinstance(
                event.data, phone.DialData | phone.AnswerCallData | phone.DeclineCallData | phone.HangUpCallData
            ):
                failure_event = phone.CallFailedEvent.with_data(
                    phone.CallFailedData(call_id=event.data.call_id, failure_kind="provider_unavailable")
                )
            else:
                return
            _ = hsm.dispatch(
                ctx,
                source_instance,
                dataclasses.replace(
                    failure_event,
                    source=hsm.id(self),
                    target=event.source,
                    metadata=_public_metadata(event.metadata),
                ),
            )
            return

    def incoming_call(self, ctx: hsm.Context, data: phone.IncomingCallData) -> collections.abc.Awaitable[None]:
        """Report an incoming LiveKit/SIP call to the provider-neutral phone firmware."""

        return self.dispatch(ctx, ServiceIncomingCallEvent.with_data(data))

    def receive_remote_audio(self, ctx: hsm.Context, data: audio.AudioInputData) -> collections.abc.Awaitable[None]:
        """Publish remote room PCM into PhoneService. Delivery is gated by HSM guards.

        Remote audio is delivered to phone firmware as `ServiceAudioReceived` only when a media
        call id is active and a phone event target is attached; otherwise it is dropped and
        counted on `media_snapshot().remote_audio_dropped_*`.
        """

        return self.dispatch(ctx, _RemoteAudioReceivedEvent.with_data(data))

    def media_ready(self, ctx: hsm.Context, data: phone.MediaReadyData) -> collections.abc.Awaitable[None]:
        """Report that LiveKit media is ready for the active call."""

        return self.dispatch(ctx, ServiceMediaReadyEvent.with_data(data))

    def remote_hang_up(self, ctx: hsm.Context, data: phone.RemoteHangUpData) -> collections.abc.Awaitable[None]:
        """Report that the remote LiveKit/SIP party ended the call."""

        return self.dispatch(ctx, ServiceRemoteHangUpEvent.with_data(data))

    def call_failed(self, ctx: hsm.Context, data: phone.CallFailedData) -> collections.abc.Awaitable[None]:
        """Report a normalized LiveKit call failure to phone firmware."""

        return self.dispatch(ctx, ServiceCallFailedEvent.with_data(data))

    def transfer_completed(
        self, ctx: hsm.Context, data: phone.TransferCompletedData
    ) -> collections.abc.Awaitable[None]:
        """Report that LiveKit completed the active transfer."""

        return self.dispatch(ctx, ServiceTransferCompletedEvent.with_data(data))

    def transfer_failed(self, ctx: hsm.Context, data: phone.TransferFailedData) -> collections.abc.Awaitable[None]:
        """Report that LiveKit rejected or failed the active transfer."""

        return self.dispatch(ctx, ServiceTransferFailedEvent.with_data(data))

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "PhoneService",
        hsm.attribute(_ACTIVE_OPERATION_EVENT_ATTRIBUTE),
        hsm.initial(hsm.target("/PhoneService/unconnected")),
        hsm.transition(
            hsm.on(_ServiceAttachedEvent),
            hsm.guard(_matches_service_attachment),
        ),
        hsm.transition(
            hsm.on(_ServiceAttachedEvent),
            hsm.guard(_has_conflicting_service_attachment),
            hsm.effect(_dispatch_attachment_rejected),
        ),
        hsm.transition(
            hsm.on(_ServiceAttachmentRejectedEvent),
        ),
        hsm.transition(
            hsm.on(_RoomAudioStatusEvent),
            hsm.effect(_apply_room_audio_status),
        ),
        hsm.transition(
            hsm.on(_ServiceDetachedEvent),
            hsm.guard(_matches_service_attachment),
            hsm.effect(_clear_active_operation_event),
            hsm.effect(_clear_phone_event_target),
            hsm.target("/PhoneService/unconnected"),
        ),
        # Remote room audio: deliver only when media call + phone target are ready; else drop observably.
        hsm.transition(
            hsm.on(_RemoteAudioReceivedEvent),
            hsm.guard(_can_deliver_remote_audio),
            hsm.effect(_deliver_remote_audio_to_phone),
        ),
        hsm.transition(
            hsm.on(_RemoteAudioReceivedEvent),
            hsm.guard(_cannot_deliver_remote_audio),
            hsm.effect(_drop_remote_audio),
        ),
        hsm.state(
            "unconnected",
            hsm.transition(
                hsm.on(_ServiceAttachedEvent),
                hsm.guard(_has_attachable_service_target),
                hsm.effect(_set_phone_event_target),
                hsm.target("/PhoneService/ready"),
            ),
        ),
        hsm.state(
            "ready",
            hsm.transition(
                hsm.on(ServiceIncomingCallEvent),
                hsm.guard(_has_incoming_call),
                hsm.effect(_emit_incoming_call),
            ),
            hsm.transition(
                hsm.on(ServiceMediaReadyEvent),
                hsm.guard(_has_media_ready),
                hsm.effect(_emit_media_ready),
            ),
            hsm.transition(
                hsm.on(ServiceRemoteHangUpEvent),
                hsm.guard(_has_current_remote_hang_up),
                hsm.effect(_clear_active_operation_event),
                hsm.effect(_emit_remote_hang_up),
            ),
            hsm.transition(
                hsm.on(ServiceRemoteHangUpEvent),
                hsm.guard(_has_uncorrelated_remote_hang_up),
                hsm.effect(_emit_remote_hang_up),
            ),
            hsm.transition(
                hsm.on(ServiceCallFailedEvent),
                hsm.guard(_has_current_call_failed),
                hsm.effect(_clear_active_operation_event),
                hsm.effect(_emit_call_failed),
            ),
            hsm.transition(
                hsm.on(ServiceCallFailedEvent),
                hsm.guard(_has_uncorrelated_call_failed),
                hsm.effect(_emit_call_failed),
            ),
            hsm.transition(
                hsm.on(ServiceTransferCompletedEvent),
                hsm.guard(_has_current_transfer_completed),
                hsm.effect(_clear_active_operation_event),
                hsm.effect(_emit_transfer_completed),
            ),
            hsm.transition(
                hsm.on(ServiceTransferCompletedEvent),
                hsm.guard(_has_uncorrelated_transfer_completed),
                hsm.effect(_emit_transfer_completed),
            ),
            hsm.transition(
                hsm.on(ServiceTransferFailedEvent),
                hsm.guard(_has_current_transfer_failed),
                hsm.effect(_clear_active_operation_event),
                hsm.effect(_emit_transfer_failed),
            ),
            hsm.transition(
                hsm.on(ServiceTransferFailedEvent),
                hsm.guard(_has_uncorrelated_transfer_failed),
                hsm.effect(_emit_transfer_failed),
            ),
            hsm.transition(
                hsm.on(phone.HungUpEvent),
                hsm.guard(_has_provider_terminal_hang_up),
                hsm.effect(_clear_active_operation_event),
            ),
            hsm.transition(
                hsm.on(phone.CallTransferCompletedEvent),
                hsm.guard(_has_provider_transfer_completed),
                hsm.effect(_clear_active_operation_event),
            ),
            hsm.transition(
                hsm.on(phone.CallTransferFailedEvent),
                hsm.guard(_has_provider_transfer_failed),
                hsm.effect(_clear_active_operation_event),
            ),
            hsm.transition(
                hsm.on(phone.ServiceDialRequestedEvent),
                hsm.guard(_has_dial_request),
                hsm.effect(_set_active_operation_event),
                hsm.target("/PhoneService/dialing"),
            ),
            hsm.transition(
                hsm.on(phone.ServiceAnswerRequestedEvent),
                hsm.guard(_has_answer_request),
                hsm.effect(_set_active_operation_event),
                hsm.target("/PhoneService/answering"),
            ),
            hsm.transition(
                hsm.on(phone.ServiceDeclineRequestedEvent),
                hsm.guard(_has_decline_request),
                hsm.effect(_set_active_operation_event),
                hsm.target("/PhoneService/declining"),
            ),
            hsm.transition(
                hsm.on(phone.ServiceHangUpRequestedEvent),
                hsm.guard(_has_hang_up_request),
                hsm.effect(_set_active_operation_event),
                hsm.target("/PhoneService/hanging_up"),
            ),
            hsm.transition(
                hsm.on(phone.ServiceTransferRequestedEvent),
                hsm.guard(_has_transfer_request),
                hsm.effect(_set_active_operation_event),
                hsm.target("/PhoneService/transferring"),
            ),
        ),
        hsm.state(
            "dialing",
            _ignore_passive_call_observation_transition(),
            hsm.transition(hsm.on(phone.ServiceDialRequestedEvent), hsm.guard(_has_dial_request)),
            hsm.transition(hsm.on(phone.ServiceAnswerRequestedEvent), hsm.guard(_has_answer_request)),
            hsm.transition(hsm.on(phone.ServiceDeclineRequestedEvent), hsm.guard(_has_decline_request)),
            hsm.transition(
                hsm.on(phone.ServiceHangUpRequestedEvent),
                hsm.guard(_has_hang_up_request),
                hsm.effect(_set_active_operation_event),
                hsm.target("/PhoneService/hanging_up"),
            ),
            hsm.transition(hsm.on(phone.ServiceTransferRequestedEvent), hsm.guard(_has_transfer_request)),
            hsm.activity(_run_dial),
            *_active_call_operation_resolution_transitions(
                emit_remote_hang_up=_emit_remote_hang_up,
                emit_call_failed=_emit_call_failed,
            ),
            _ignore_transfer_terminal_observation_transition(),
            hsm.transition(
                hsm.on(_DialCompletedEvent),
                hsm.guard(_has_call_connected_completion),
                hsm.effect(_emit_call_connected),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(_DialFailedEvent),
                hsm.guard(_has_call_failed_completion),
                hsm.effect(_emit_call_failed),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            _active_call_operation_timeout_transition(
                operation_timeout=_operation_timeout_value,
                emit_active_call_timeout=_emit_active_call_timeout,
            ),
        ),
        hsm.state(
            "answering",
            _ignore_passive_call_observation_transition(),
            hsm.transition(hsm.on(phone.ServiceDialRequestedEvent), hsm.guard(_has_dial_request)),
            hsm.transition(hsm.on(phone.ServiceAnswerRequestedEvent), hsm.guard(_has_answer_request)),
            hsm.transition(
                hsm.on(phone.ServiceDeclineRequestedEvent),
                hsm.guard(_has_decline_request),
                hsm.effect(_set_active_operation_event),
                hsm.target("/PhoneService/declining"),
            ),
            hsm.transition(
                hsm.on(phone.ServiceHangUpRequestedEvent),
                hsm.guard(_has_hang_up_request),
                hsm.effect(_set_active_operation_event),
                hsm.target("/PhoneService/hanging_up"),
            ),
            hsm.activity(_run_answer_call),
            *_active_call_operation_resolution_transitions(
                emit_remote_hang_up=_emit_remote_hang_up,
                emit_call_failed=_emit_call_failed,
            ),
            _ignore_transfer_terminal_observation_transition(),
            hsm.transition(
                hsm.on(_AnswerCompletedEvent),
                hsm.guard(_has_call_connected_completion),
                hsm.effect(_emit_call_connected),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(_AnswerFailedEvent),
                hsm.guard(_has_call_failed_completion),
                hsm.effect(_emit_call_failed),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            _active_call_operation_timeout_transition(
                operation_timeout=_operation_timeout_value,
                emit_active_call_timeout=_emit_active_call_timeout,
            ),
        ),
        hsm.state(
            "declining",
            _ignore_passive_call_observation_transition(),
            hsm.transition(hsm.on(phone.ServiceDialRequestedEvent), hsm.guard(_has_dial_request)),
            hsm.transition(hsm.on(phone.ServiceAnswerRequestedEvent), hsm.guard(_has_answer_request)),
            hsm.transition(hsm.on(phone.ServiceDeclineRequestedEvent), hsm.guard(_has_decline_request)),
            hsm.transition(
                hsm.on(phone.ServiceHangUpRequestedEvent),
                hsm.guard(_has_hang_up_request),
                hsm.effect(_set_active_operation_event),
                hsm.target("/PhoneService/hanging_up"),
            ),
            hsm.transition(hsm.on(phone.ServiceTransferRequestedEvent), hsm.guard(_has_transfer_request)),
            hsm.activity(_run_decline_call),
            *_active_call_operation_resolution_transitions(
                emit_remote_hang_up=_emit_remote_hang_up,
                emit_call_failed=_emit_call_failed,
            ),
            _ignore_transfer_terminal_observation_transition(),
            hsm.transition(
                hsm.on(_DeclineCompletedEvent),
                hsm.guard(_has_call_id_completion),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(_DeclineFailedEvent),
                hsm.guard(_has_call_failed_completion),
                hsm.effect(_emit_call_failed),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            _active_call_operation_timeout_transition(
                operation_timeout=_operation_timeout_value,
                emit_active_call_timeout=_emit_active_call_timeout,
            ),
        ),
        hsm.state(
            "hanging_up",
            _ignore_passive_call_observation_transition(),
            hsm.transition(hsm.on(phone.ServiceDialRequestedEvent), hsm.guard(_has_dial_request)),
            hsm.transition(hsm.on(phone.ServiceAnswerRequestedEvent), hsm.guard(_has_answer_request)),
            hsm.transition(hsm.on(phone.ServiceDeclineRequestedEvent), hsm.guard(_has_decline_request)),
            hsm.transition(hsm.on(phone.ServiceHangUpRequestedEvent), hsm.guard(_has_hang_up_request)),
            hsm.transition(hsm.on(phone.ServiceTransferRequestedEvent), hsm.guard(_has_transfer_request)),
            hsm.activity(_run_hang_up_call),
            *_active_call_operation_resolution_transitions(
                emit_remote_hang_up=_emit_remote_hang_up,
                emit_call_failed=_emit_call_failed,
            ),
            _ignore_transfer_terminal_observation_transition(),
            hsm.transition(
                hsm.on(_HangUpCompletedEvent),
                hsm.guard(_has_call_id_completion),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(_HangUpFailedEvent),
                hsm.guard(_has_call_failed_completion),
                hsm.effect(_emit_call_failed),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            _active_call_operation_timeout_transition(
                operation_timeout=_operation_timeout_value,
                emit_active_call_timeout=_emit_active_call_timeout,
            ),
        ),
        hsm.state(
            "transferring",
            _ignore_passive_call_observation_transition(),
            hsm.transition(hsm.on(phone.ServiceDialRequestedEvent), hsm.guard(_has_dial_request)),
            hsm.transition(hsm.on(phone.ServiceAnswerRequestedEvent), hsm.guard(_has_answer_request)),
            hsm.transition(hsm.on(phone.ServiceDeclineRequestedEvent), hsm.guard(_has_decline_request)),
            hsm.transition(
                hsm.on(phone.ServiceHangUpRequestedEvent),
                hsm.guard(_has_hang_up_request),
                hsm.effect(_set_active_operation_event),
                hsm.target("/PhoneService/hanging_up"),
            ),
            hsm.transition(hsm.on(phone.ServiceTransferRequestedEvent), hsm.guard(_has_transfer_request)),
            hsm.activity(_run_transfer_call),
            *_active_call_operation_resolution_transitions(
                emit_remote_hang_up=_emit_remote_hang_up,
                emit_call_failed=_emit_call_failed,
            ),
            hsm.transition(
                hsm.on(ServiceTransferCompletedEvent),
                hsm.guard(_has_current_transfer_completed),
                hsm.effect(_emit_transfer_completed),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(ServiceTransferFailedEvent),
                hsm.guard(_has_current_transfer_failed),
                hsm.effect(_emit_transfer_failed),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(phone.CallTransferCompletedEvent),
                hsm.guard(_has_provider_transfer_completed),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(phone.CallTransferFailedEvent),
                hsm.guard(_has_provider_transfer_failed),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(_TransferAcceptedEvent),
                hsm.guard(_has_transfer_accepted_completion),
                hsm.effect(_emit_transfer_accepted),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(_TransferRequestFailedEvent),
                hsm.guard(_has_transfer_failed_completion),
                hsm.effect(_emit_transfer_failed),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.after(_operation_timeout_value),
                hsm.guard(_has_active_transfer_operation),
                hsm.effect(_emit_active_transfer_timeout),
                hsm.effect(_clear_active_operation_event),
                hsm.target("/PhoneService/ready"),
            ),
        ),
        hsm.observe(observer),
    )


class Phone(phone.Phone):
    """Convenience LiveKit phone: core ``Phone`` with a LiveKit ``PhoneService`` from room credentials."""

    def __init__(
        self,
        bots: collections.abc.Iterable[hsm.Instance] = (),
        *,
        url: str | None = None,
        token: str | None = None,
        track_name: str = "bot-audio",
        microphone: audio.Microphone | None = None,
        speaker: audio.Speaker | None = None,
        peripherals: collections.abc.Iterable[bot.device.Device] = (),
        operation_timeout: datetime.timedelta = _DEFAULT_OPERATION_TIMEOUT,
        answer_timeout: datetime.timedelta = _DEFAULT_ANSWER_TIMEOUT,
        transfer_timeout: datetime.timedelta = _DEFAULT_TRANSFER_TIMEOUT,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__(
            bots=bots,
            microphone=microphone,
            speaker=speaker,
            peripherals=peripherals,
            service=PhoneService(
                url=url,
                token=token,
                track_name=track_name,
                operation_timeout=operation_timeout,
                loop=loop,
            ),
            answer_timeout=answer_timeout,
            transfer_timeout=transfer_timeout,
        )


__all__ = [
    "ServiceCallFailedEvent",
    "ServiceIncomingCallEvent",
    "ServiceMediaReadyEvent",
    "ServiceRemoteHangUpEvent",
    "ServiceTransferCompletedEvent",
    "ServiceTransferFailedEvent",
    "MediaSnapshot",
    "Phone",
    "PhoneService",
    "PhoneServiceError",
]

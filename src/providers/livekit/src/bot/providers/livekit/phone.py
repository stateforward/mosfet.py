from __future__ import annotations


from bot.devices import audio
from bot.devices import phone

import asyncio
import collections.abc
import dataclasses
import datetime
import logging
import typing
import weakref

import hsm
import pydantic
import bot.device
from livekit import rtc

from bot import lifecycle
from bot.telemetry import observer
from bot.world import World, require_world_scope

from .audio import AudioBridge, create_audio_bridge
from .pcm_batch import RemotePcmBatcher
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


_LOG = logging.getLogger(__name__)


class _PhoneServiceAttachmentData(pydantic.BaseModel):
    """Private attachment payload connecting a LiveKit phone service to a phone event target."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    target: _PhoneServiceTarget = pydantic.Field(
        description=_PHONE_SERVICE_TARGET_DESCRIPTION,
    )


_ACTIVE_OPERATION_ATTRIBUTE = "active_operation"
_ActiveRequestData = (
    phone.DialData
    | phone.AnswerCallData
    | phone.DeclineCallData
    | phone.HangUpCallData
    | phone.TransferCallData
)


@dataclasses.dataclass(frozen=True, slots=True)
class _ActiveOperation:
    """In-flight call-control identity: envelope id + typed request data only (HSM-COMPLETION-001)."""

    id: str
    data: _ActiveRequestData


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


class _LocalAudioPublishFailedData(pydantic.BaseModel):
    """Why one local playout chunk never made it onto the LiveKit track."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Uplink publish failure for a single local audio chunk.",
            "examples": [{"message": "LiveKit audio frames only support raw PCM input.", "chunk_bytes": 960}],
        },
    )

    message: str = pydantic.Field(
        min_length=1,
        description="Human-readable reason the chunk could not be published.",
        examples=["LiveKit audio frames only support raw PCM input from stateforward.bot audio output data."],
    )
    media_type: str | None = pydantic.Field(
        default=None,
        description="Media type the rejected chunk declared, which is usually why it was rejected.",
        examples=["audio/wav"],
    )
    chunk_bytes: int = pydantic.Field(
        ge=0,
        description="Size of the chunk that was dropped, in bytes.",
        examples=[960],
    )


_LocalAudioPublishFailedEvent = hsm.Event[_LocalAudioPublishFailedData](
    name="bot.provider.livekit.phone.local_audio.publish_failed",
    kind=hsm.ErrorEventKind,
    schema=_LocalAudioPublishFailedData,
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


def _with_operation_correlation[T](
    event: hsm.Event[T],
    *,
    operation_id: str,
    metadata: collections.abc.Mapping[str, object] | None = None,
) -> hsm.Event[T]:
    """Stamp private completion/failure with the active request envelope id; copy telemetry metadata only."""

    stamped: hsm.Event[T]
    if operation_id and event.data is not None:
        stamped = event.with_data_and_id(event.data, operation_id)
    elif operation_id:
        stamped = dataclasses.replace(event, id=operation_id)
    else:
        stamped = event
    return dataclasses.replace(stamped, metadata=dict(metadata or {}))


def _with_trigger_correlation[T](event: hsm.Event[T], trigger: hsm.Event[typing.Any]) -> hsm.Event[T]:
    return _with_operation_correlation(event, operation_id=trigger.id, metadata=trigger.metadata)


def _active_operation(instance: "PhoneService") -> _ActiveOperation | None:
    value, ok = typing.cast(tuple[object, bool], instance.get(_ACTIVE_OPERATION_ATTRIBUTE))
    if not ok or not isinstance(value, _ActiveOperation):
        return None
    return value


def _set_active_operation(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
    del ctx
    data = event.data
    if not event.id or not isinstance(
        data,
        phone.DialData | phone.AnswerCallData | phone.DeclineCallData | phone.HangUpCallData | phone.TransferCallData,
    ):
        _ = instance.set(_ACTIVE_OPERATION_ATTRIBUTE, None)
        return
    _ = instance.set(_ACTIVE_OPERATION_ATTRIBUTE, _ActiveOperation(id=event.id, data=data))


def _clear_active_operation(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
    del ctx, event
    _ = instance.set(_ACTIVE_OPERATION_ATTRIBUTE, None)


def _active_operation_call_id(instance: "PhoneService") -> str | None:
    active = _active_operation(instance)
    if active is None:
        return None
    return active.data.call_id


def _active_operation_transfer(instance: "PhoneService") -> phone.TransferCallData | None:
    active = _active_operation(instance)
    if active is None or not isinstance(active.data, phone.TransferCallData):
        return None
    return active.data


def _matches_active_call(instance: "PhoneService", call_id: str) -> bool:
    return _active_operation_call_id(instance) == call_id


def _matches_active_operation_id(instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    active = _active_operation(instance)
    if active is None or not active.id:
        return False
    return event.id == active.id


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
    # Multi-trigger transition: discriminate by typed payload, not event.name.
    return isinstance(event.data, phone.IncomingCallData | phone.MediaReadyData)


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
    return isinstance(event.data, phone.TransferCompletedData | phone.TransferFailedData)


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


def _has_hung_up(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    """True for any committed hang-up observation (all HangUpOutcome values)."""

    del ctx, instance
    return isinstance(event.data, phone.PhoneHungUpData)


def _has_provider_terminal_hang_up(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    """True when hang-up ends the active call op (not mid-transfer ``transferred`` hang-ups)."""

    del ctx
    data = event.data
    return (
        isinstance(data, phone.PhoneHungUpData)
        and data.outcome in {"declined", "failed", "local_hang_up", "remote_hang_up"}
        and _matches_active_call(instance, data.call_id)
    )


def _publish_local_audio_uplink(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> None:
    del ctx
    data = event.data
    assert isinstance(data, audio.AudioOutputData)
    published = asyncio.ensure_future(instance.publish_audio(data))

    def _surface_failure(done: asyncio.Future[None]) -> None:
        # Without this the encoder's exception dies in an orphaned task: the bot goes mute and the
        # only trace is an unretrieved-task warning. A publish that fails is a typed outcome.
        if done.cancelled():
            return
        error = done.exception()
        if error is None:
            return
        _ = hsm.dispatch(
            instance.context(),
            instance,
            _LocalAudioPublishFailedEvent.with_data(
                _LocalAudioPublishFailedData(
                    message=str(error) or type(error).__name__,
                    media_type=data.media_type,
                    chunk_bytes=len(data.audio),
                )
            ),
        )

    published.add_done_callback(_surface_failure)


def _has_provider_transfer_completed(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    return isinstance(event.data, phone.PhoneTransferData) and _matches_active_transfer(instance, event.data)


def _has_provider_transfer_failed(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    return isinstance(event.data, phone.PhoneTransferFailedData) and _matches_active_transfer(instance, event.data)


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
    return _active_operation(instance) is None and _has_remote_hang_up(ctx, instance, event)


def _has_uncorrelated_call_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    return _active_operation(instance) is None and _has_call_failed(ctx, instance, event)


def _has_uncorrelated_transfer_completed(
    ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]
) -> bool:
    return _active_operation(instance) is None and _has_transfer_completed(ctx, instance, event)


def _has_uncorrelated_transfer_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    return _active_operation(instance) is None and _has_transfer_failed(ctx, instance, event)


def _has_active_call_operation(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, event
    active = _active_operation(instance)
    return active is not None and isinstance(
        active.data, phone.DialData | phone.AnswerCallData | phone.DeclineCallData | phone.HangUpCallData
    )


def _has_active_transfer_operation(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, event
    active = _active_operation(instance)
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
        and _matches_active_operation_id(instance, event)
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
        and _matches_active_operation_id(instance, event)
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
        and _matches_active_operation_id(instance, event)
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
        and _matches_active_operation_id(instance, event)
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
        and _matches_active_operation_id(instance, event)
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
    on_hung_up: _PhoneServiceEffect,
) -> tuple[hsm.Element, ...]:
    """Resolve remote hang-up / call-failed / committed HungUp while an owner call op is active.

    ``on_hung_up`` MUST clear media (``_media_call_id``) as well as active-op bookkeeping —
    not only ``_clear_active_operation``. Active-op states (dialing/answering/…) share this
    path; ready has its own HungUp transition. Passing media-clear here closes the stale
    media-session gap when hang-up arrives before the op completes.
    """

    return (
        hsm.transition(
            hsm.on(ServiceRemoteHangUpEvent),
            hsm.guard(_has_current_remote_hang_up),
            hsm.effect(emit_remote_hang_up),
            hsm.effect(_clear_active_operation),
            hsm.target("/PhoneService/ready"),
        ),
        hsm.transition(
            hsm.on(ServiceCallFailedEvent),
            hsm.guard(_has_current_call_failed),
            hsm.effect(emit_call_failed),
            hsm.effect(_clear_active_operation),
            hsm.target("/PhoneService/ready"),
        ),
        hsm.transition(
            hsm.on(phone.HungUpEvent),
            hsm.guard(_has_provider_terminal_hang_up),
            hsm.effect(on_hung_up),
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
        hsm.effect(_clear_active_operation),
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
    _remote_pcm_batcher: RemotePcmBatcher | None
    _local_track_sid: str | None
    _remote_audio_chunks: int
    _remote_audio_bytes: int
    _remote_audio_dropped_chunks: int
    _remote_audio_dropped_bytes: int
    _local_audio_failed_chunks: int
    _local_audio_failed_bytes: int
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
        self._remote_pcm_batcher = None
        self._local_track_sid = None
        self._remote_audio_chunks = 0
        self._remote_audio_bytes = 0
        self._remote_audio_dropped_chunks = 0
        self._remote_audio_dropped_bytes = 0
        self._local_audio_failed_chunks = 0
        self._local_audio_failed_bytes = 0
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

    def _sdk_ingress_open(self) -> bool:
        """True when attach effects opened provider ingress (not a ``state()`` probe).

        HSM-CONTEXT-001: delivery policy is machine-owned attach lifetime, never
        ``instance.state()`` readiness sampling from SDK callbacks.
        """

        return self._attached_phone_target_ref is not None

    def _bind_room_presence(self, room: RoomHandle) -> None:
        """Map remote room presence to phone call observations (ring / remote hang-up).

        Does not auto-answer: the bot (or operator) decides whether to answer.
        """

        if self._presence_bound:
            return
        service_ref = weakref.ref(self)

        def on_participant_connected(participant: object) -> object:
            live_service = service_ref()
            if live_service is None or not live_service._sdk_ingress_open():
                return None
            live_service._offer_incoming_call_for_participant(participant)
            return None

        def on_participant_disconnected(participant: object) -> object:
            live_service = service_ref()
            if live_service is None or not live_service._sdk_ingress_open():
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
            values = remote_participants
        else:
            return
        for participant in values:
            self._offer_incoming_call_for_participant(participant)

    def _offer_incoming_call_for_participant(self, participant: object) -> None:
        if not self._sdk_ingress_open():
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
        if not self._sdk_ingress_open():
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

        async def deliver_batched_remote_audio(audio_input: audio.AudioInputData) -> None:
            """Publish one batched utterance into PhoneService (guards decide deliver vs drop).

            HSM-CONTEXT-001: do not gate on ``context().is_done()`` or ``state()``. Attach
            lifetime opens ingress; HSM guards decide deliver vs drop once dispatched.
            """

            live_service = service_ref()
            if live_service is None or not live_service._sdk_ingress_open():
                return
            await live_service.receive_remote_audio(live_service.context(), audio_input)

        async def consume_remote_audio(audio_input: audio.AudioInputData) -> None:
            """Ingress: batch ~10 ms LiveKit frames into utterance-sized PCM for Listening."""

            live_service = service_ref()
            if live_service is None or not live_service._sdk_ingress_open():
                return
            batcher = live_service._remote_pcm_batcher
            if batcher is None:
                await deliver_batched_remote_audio(audio_input)
                return
            await batcher.push(audio_input)

        def record_connected(data: RoomAudioConnectedData) -> None:
            live_service = service_ref()
            if live_service is None or not live_service._sdk_ingress_open():
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
        self._remote_pcm_batcher = RemotePcmBatcher(
            emit=deliver_batched_remote_audio,
            loop=self._loop,
        )
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
        if not lifecycle.is_started(track_path):
            _ = await hsm.started(world, track_path, track_path.model)
        # Start this service before room connect so participant_connected can ring.
        if not lifecycle.is_started(self):
            _ = await hsm.started(world, self, self.model)
        require_world_scope(world, self, participant="PhoneService")
        await self.dispatch(world, _ServiceAttachedEvent.with_data(_PhoneServiceAttachmentData(target=target)))
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

        # Idempotent when already stopped; no firmware reply channel on this API.
        if not lifecycle.is_started(self):
            return
        require_world_scope(world, target, participant="Phone service target")
        require_world_scope(world, self, participant="PhoneService")
        await self.dispatch(world, _ServiceDetachedEvent.with_data(_PhoneServiceAttachmentData(target=target)))

    async def connect_room(
        self,
        *,
        url: str,
        token: str,
        track_name: str = "bot-audio",
    ) -> None:
        """Connect the privately owned LiveKit room audio path for this service."""

        _, track_path = self._ensure_media()
        if not lifecycle.is_started(track_path):
            # Parent the track path under this service only while the service is live.
            parent = self.context() if lifecycle.is_started(self) else None
            _ = await hsm.started(parent, track_path, track_path.model)
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

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop this service and its privately owned room track path if started."""

        track_path = self._track_path
        # Ingress opens when attach holds a target ref; clear on stop so SDK callbacks
        # cannot deliver after the service is stopped (only detach effect cleared it before).
        self._attached_phone_target_ref = None
        await hsm.Instance.stop(self, ctx)
        if track_path is None:
            return
        # Track path is created by _ensure_media before start; only stop if started.
        if not lifecycle.is_started(track_path):
            return
        await hsm.stop(track_path, ctx)

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
    def _has_local_audio_uplink(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """True when local speaker playout should uplink (exact AudioOutputData, not remote service audio)."""

        del ctx
        # Exact type: ServiceAudioData subclasses AudioOutputData and must not uplink as local playout.
        return type(event.data) is audio.AudioOutputData and not instance._delivering_remote_audio

    @staticmethod
    def _clear_media_on_hung_up(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Any HungUp observation ends the media session (matches prior publish-side clear)."""

        del ctx
        if isinstance(event.data, phone.PhoneHungUpData):
            instance._media_call_id = None

    @staticmethod
    def _clear_media_and_maybe_active_operation(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Always clear media on HungUp; clear active call op only for terminal (non-transfer) outcomes."""

        PhoneService._clear_media_on_hung_up(ctx, instance, event)
        if _has_provider_terminal_hang_up(ctx, instance, event):
            _clear_active_operation(ctx, instance, event)

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
                    metadata=dict(event.metadata),
                ),
            )
        finally:
            instance._delivering_remote_audio = False

    @staticmethod
    def _note_local_audio_publish_failure(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _LocalAudioPublishFailedData)
        instance._local_audio_failed_chunks += 1
        instance._local_audio_failed_bytes += data.chunk_bytes
        # Operationally load-bearing: this is the difference between "the bot is mute" and knowing
        # why. Media type is low-cardinality; the message is the provider's own rejection reason.
        _LOG.error(
            "livekit local audio uplink publish failed media_type=%s chunk_bytes=%d failed_chunks=%d reason=%s",
            data.media_type,
            data.chunk_bytes,
            instance._local_audio_failed_chunks,
            data.message,
        )

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
        emitted = dataclasses.replace(
            event,
            source=hsm.id(instance),
            target=hsm.id(phone_event_target),
            metadata=dict(trigger.metadata),
        )
        if trigger.id:
            emitted = emitted.with_data_and_id(emitted.data, trigger.id)
        _ = phone_event_target.dispatch(ctx, emitted)

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
        active = _active_operation(instance)
        assert active is not None
        data = active.data
        assert isinstance(data, phone.DialData | phone.AnswerCallData | phone.DeclineCallData | phone.HangUpCallData)
        failure = phone.CallFailedData(call_id=data.call_id, failure_kind="timeout")
        # Stamp envelope id from the active request (same correlation path as activity completions).
        PhoneService._emit_call_failed(
            ctx,
            instance,
            _with_operation_correlation(
                ServiceCallFailedEvent.with_data(failure),
                operation_id=active.id,
            ),
        )

    @staticmethod
    def _emit_active_transfer_timeout(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        active = _active_operation(instance)
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
            _with_operation_correlation(
                ServiceTransferFailedEvent.with_data(failure),
                operation_id=active.id,
            ),
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
            _ = hsm.dispatch(ctx, instance, _with_trigger_correlation(_AnswerFailedEvent.with_data(failure), event))
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            _with_trigger_correlation(
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
            _ = hsm.dispatch(ctx, instance, _with_trigger_correlation(_DialFailedEvent.with_data(failure), event))
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            _with_trigger_correlation(_DialCompletedEvent.with_data(connected), event),
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
            _ = hsm.dispatch(ctx, instance, _with_trigger_correlation(_DeclineFailedEvent.with_data(failure), event))
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            _with_trigger_correlation(_DeclineCompletedEvent.with_data(phone.CallIdData(call_id=data.call_id)), event),
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
            _ = hsm.dispatch(ctx, instance, _with_trigger_correlation(_HangUpFailedEvent.with_data(failure), event))
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            _with_trigger_correlation(_HangUpCompletedEvent.with_data(phone.CallIdData(call_id=data.call_id)), event),
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
                _with_trigger_correlation(_TransferRequestFailedEvent.with_data(failure), event),
            )
            return
        accepted = phone.TransferAcceptedData(call_id=data.call_id, transfer_id=data.transfer_id, target=data.target)
        _ = hsm.dispatch(ctx, instance, _with_trigger_correlation(_TransferAcceptedEvent.with_data(accepted), event))

    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        """Firmware ingress: deliver into this service HSM.

        Delivery is the gate (``dispatch`` / ``dispatch_to``). Do not re-check name, target, or
        source here or in guards for routing — transitions match or they do not; payload guards
        only express domain conditions (typed data, active call/transfer correlation).
        """

        _ = self.dispatch(ctx, event)

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
        hsm.attribute(_ACTIVE_OPERATION_ATTRIBUTE),
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
            hsm.effect(_clear_active_operation),
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
        # Local speaker playout offered as call uplink (suppressed while delivering remote audio).
        hsm.transition(
            hsm.on(audio.OutputEvent),
            hsm.guard(_has_local_audio_uplink),
            hsm.effect(_publish_local_audio_uplink),
        ),
        # ... and what happened to it when the track would not take it.
        hsm.transition(
            hsm.on(_LocalAudioPublishFailedEvent),
            hsm.effect(_note_local_audio_publish_failure),
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
                hsm.effect(_clear_active_operation),
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
                hsm.effect(_clear_active_operation),
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
                hsm.effect(_clear_active_operation),
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
                hsm.effect(_clear_active_operation),
                hsm.effect(_emit_transfer_failed),
            ),
            hsm.transition(
                hsm.on(ServiceTransferFailedEvent),
                hsm.guard(_has_uncorrelated_transfer_failed),
                hsm.effect(_emit_transfer_failed),
            ),
            hsm.transition(
                hsm.on(phone.HungUpEvent),
                hsm.guard(_has_hung_up),
                hsm.effect(_clear_media_and_maybe_active_operation),
            ),
            hsm.transition(
                hsm.on(phone.CallTransferCompletedEvent),
                hsm.guard(_has_provider_transfer_completed),
                hsm.effect(_clear_active_operation),
            ),
            hsm.transition(
                hsm.on(phone.CallTransferFailedEvent),
                hsm.guard(_has_provider_transfer_failed),
                hsm.effect(_clear_active_operation),
            ),
            hsm.transition(
                hsm.on(phone.ServiceDialRequestedEvent),
                hsm.guard(_has_dial_request),
                hsm.effect(_set_active_operation),
                hsm.target("/PhoneService/dialing"),
            ),
            hsm.transition(
                hsm.on(phone.ServiceAnswerRequestedEvent),
                hsm.guard(_has_answer_request),
                hsm.effect(_set_active_operation),
                hsm.target("/PhoneService/answering"),
            ),
            hsm.transition(
                hsm.on(phone.ServiceDeclineRequestedEvent),
                hsm.guard(_has_decline_request),
                hsm.effect(_set_active_operation),
                hsm.target("/PhoneService/declining"),
            ),
            hsm.transition(
                hsm.on(phone.ServiceHangUpRequestedEvent),
                hsm.guard(_has_hang_up_request),
                hsm.effect(_set_active_operation),
                hsm.target("/PhoneService/hanging_up"),
            ),
            hsm.transition(
                hsm.on(phone.ServiceTransferRequestedEvent),
                hsm.guard(_has_transfer_request),
                hsm.effect(_set_active_operation),
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
                hsm.effect(_set_active_operation),
                hsm.target("/PhoneService/hanging_up"),
            ),
            hsm.transition(hsm.on(phone.ServiceTransferRequestedEvent), hsm.guard(_has_transfer_request)),
            hsm.activity(_run_dial),
            *_active_call_operation_resolution_transitions(
                emit_remote_hang_up=_emit_remote_hang_up,
                emit_call_failed=_emit_call_failed,
                on_hung_up=_clear_media_and_maybe_active_operation,
            ),
            _ignore_transfer_terminal_observation_transition(),
            hsm.transition(
                hsm.on(_DialCompletedEvent),
                hsm.guard(_has_call_connected_completion),
                hsm.effect(_emit_call_connected),
                hsm.effect(_clear_active_operation),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(_DialFailedEvent),
                hsm.guard(_has_call_failed_completion),
                hsm.effect(_emit_call_failed),
                hsm.effect(_clear_active_operation),
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
                hsm.effect(_set_active_operation),
                hsm.target("/PhoneService/declining"),
            ),
            hsm.transition(
                hsm.on(phone.ServiceHangUpRequestedEvent),
                hsm.guard(_has_hang_up_request),
                hsm.effect(_set_active_operation),
                hsm.target("/PhoneService/hanging_up"),
            ),
            hsm.activity(_run_answer_call),
            *_active_call_operation_resolution_transitions(
                emit_remote_hang_up=_emit_remote_hang_up,
                emit_call_failed=_emit_call_failed,
                on_hung_up=_clear_media_and_maybe_active_operation,
            ),
            _ignore_transfer_terminal_observation_transition(),
            hsm.transition(
                hsm.on(_AnswerCompletedEvent),
                hsm.guard(_has_call_connected_completion),
                hsm.effect(_emit_call_connected),
                hsm.effect(_clear_active_operation),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(_AnswerFailedEvent),
                hsm.guard(_has_call_failed_completion),
                hsm.effect(_emit_call_failed),
                hsm.effect(_clear_active_operation),
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
                hsm.effect(_set_active_operation),
                hsm.target("/PhoneService/hanging_up"),
            ),
            hsm.transition(hsm.on(phone.ServiceTransferRequestedEvent), hsm.guard(_has_transfer_request)),
            hsm.activity(_run_decline_call),
            *_active_call_operation_resolution_transitions(
                emit_remote_hang_up=_emit_remote_hang_up,
                emit_call_failed=_emit_call_failed,
                on_hung_up=_clear_media_and_maybe_active_operation,
            ),
            _ignore_transfer_terminal_observation_transition(),
            hsm.transition(
                hsm.on(_DeclineCompletedEvent),
                hsm.guard(_has_call_id_completion),
                hsm.effect(_clear_active_operation),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(_DeclineFailedEvent),
                hsm.guard(_has_call_failed_completion),
                hsm.effect(_emit_call_failed),
                hsm.effect(_clear_active_operation),
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
                on_hung_up=_clear_media_and_maybe_active_operation,
            ),
            _ignore_transfer_terminal_observation_transition(),
            hsm.transition(
                hsm.on(_HangUpCompletedEvent),
                hsm.guard(_has_call_id_completion),
                hsm.effect(_clear_active_operation),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(_HangUpFailedEvent),
                hsm.guard(_has_call_failed_completion),
                hsm.effect(_emit_call_failed),
                hsm.effect(_clear_active_operation),
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
                hsm.effect(_set_active_operation),
                hsm.target("/PhoneService/hanging_up"),
            ),
            hsm.transition(hsm.on(phone.ServiceTransferRequestedEvent), hsm.guard(_has_transfer_request)),
            hsm.activity(_run_transfer_call),
            *_active_call_operation_resolution_transitions(
                emit_remote_hang_up=_emit_remote_hang_up,
                emit_call_failed=_emit_call_failed,
                on_hung_up=_clear_media_and_maybe_active_operation,
            ),
            hsm.transition(
                hsm.on(ServiceTransferCompletedEvent),
                hsm.guard(_has_current_transfer_completed),
                hsm.effect(_emit_transfer_completed),
                hsm.effect(_clear_active_operation),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(ServiceTransferFailedEvent),
                hsm.guard(_has_current_transfer_failed),
                hsm.effect(_emit_transfer_failed),
                hsm.effect(_clear_active_operation),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(phone.CallTransferCompletedEvent),
                hsm.guard(_has_provider_transfer_completed),
                hsm.effect(_clear_active_operation),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.on(phone.CallTransferFailedEvent),
                hsm.guard(_has_provider_transfer_failed),
                hsm.effect(_clear_active_operation),
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
                hsm.effect(_clear_active_operation),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.transition(
                hsm.after(_operation_timeout_value),
                hsm.guard(_has_active_transfer_operation),
                hsm.effect(_emit_active_transfer_timeout),
                hsm.effect(_clear_active_operation),
                hsm.target("/PhoneService/ready"),
            ),
        ),
        hsm.observe(observer),
    )


class Phone(phone.Phone):
    """Convenience LiveKit phone: core ``Phone`` with a LiveKit ``PhoneService`` from room credentials."""

    def __init__(
        self,
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

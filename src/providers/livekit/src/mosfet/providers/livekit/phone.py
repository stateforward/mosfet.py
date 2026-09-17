from __future__ import annotations


from mosfet.devices import audio
from mosfet.devices import phone

import asyncio
import collections.abc
import dataclasses
import datetime
import logging
import typing
import weakref

import hsm
import pydantic
import mosfet.device
from livekit import rtc

import mosfet.lifecycle
from mosfet.telemetry import inject_context, observer, span
from mosfet.environment import Environment, require_environment_scope

from . import signaling
from .audio import AudioBridge, create_audio_bridge
from .pcm_batch import RemotePcmBatcher
from .room_audio import (
    AudioStreamFactory,
    LocalAudioTrackFactory,
    LocalParticipant,
    RoomAudioConnectData,
    RoomAudioConnectedData,
    RoomAudioTrackPath,
    RoomHandle,
)

_DEFAULT_OPERATION_TIMEOUT = datetime.timedelta(seconds=30)
_DEFAULT_ANSWER_TIMEOUT = datetime.timedelta(seconds=30)
_DEFAULT_TRANSFER_TIMEOUT = datetime.timedelta(seconds=30)
# How long call setup has to reach the other phone and be acknowledged. Seconds, not the
# half-minute an operation gets, because this budget covers a message crossing the room — not
# the ring that follows it. Ring time belongs to phone firmware: how long to let it ring before
# giving up is the caller's patience, not the exchange's.
_DEFAULT_SETUP_TIMEOUT = datetime.timedelta(seconds=5)
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
_SCOPE = "mosfet.providers.livekit"
_COMPONENT = "livekit.phone"


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
# Firmware-stamped service requests. Every one but dial carries the call it is about; a dial has
# no call yet, because the exchange assigns it on connect.
_ActiveRequestData = (
    phone.DialData
    | phone.AnswerRequestData
    | phone.DeclineRequestData
    | phone.HangUpRequestData
    | phone.TransferRequestData
)
_ActiveCallRequestData = phone.AnswerRequestData | phone.DeclineRequestData | phone.HangUpRequestData


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
        examples=["LiveKit audio frames only support raw PCM input from stateforward.mosfet audio output data."],
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
_RemoteAudioReceivedEvent = hsm.Event[audio.InputData](
    name="bot.provider.livekit.phone.remote_audio.received",
    schema=audio.InputData,
)
_RoomAudioStatusEvent = hsm.Event[RoomAudioConnectedData](
    name="bot.provider.livekit.phone.room_audio.status",
    schema=RoomAudioConnectedData,
)


class _PeerLeftData(pydantic.BaseModel):
    """One remote participant is no longer in the room.

    A transduced observation and nothing more. Whether a departure is a dial that can never be
    answered, a far end that crashed mid-call, or a stranger leaving a room this phone is not
    talking into is decided by topology, which is the only thing that knows what this phone is
    doing. The callback that raises this reports; it does not interpret.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "A LiveKit participant that is no longer present in the room.",
            "examples": [{"identity": "agent-b"}],
        },
    )

    identity: str = pydantic.Field(
        min_length=1,
        description="LiveKit participant identity that left the room.",
        examples=["agent-b"],
    )


# Private: room presence and the call-setup wire enter the service machine; guards decide meaning.
_PeerLeftEvent = hsm.Event[_PeerLeftData](
    name="bot.provider.livekit.phone.peer.left",
    schema=_PeerLeftData,
)
_SetupDeliveredEvent = hsm.Event[None](
    name="bot.provider.livekit.phone.setup.delivered",
    kind=hsm.CompletionEventKind,
)
_SetupAcceptedEvent = hsm.Event[phone.CallConnectedData](
    name="bot.provider.livekit.phone.setup.accepted",
    schema=phone.CallConnectedData,
)
_SetupDeclinedEvent = hsm.Event[phone.CallIdData](
    name="bot.provider.livekit.phone.setup.declined",
    schema=phone.CallIdData,
)

_AnswerCompletedEvent = hsm.Event[phone.CallConnectedData](
    name="bot.provider.livekit.phone.answer.completed",
    kind=hsm.CompletionEventKind,
    schema=phone.CallConnectedData,
)
_AnswerFailedEvent = hsm.Event[phone.CallFailedData](
    name="bot.provider.livekit.phone.answer.failed",
    kind=hsm.ErrorEventKind,
    schema=phone.CallFailedData,
)
_DialFailedEvent = hsm.Event[phone.DialFailedData](
    name="bot.provider.livekit.phone.dial.failed",
    kind=hsm.ErrorEventKind,
    schema=phone.DialFailedData,
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
        _ActiveRequestData,
    ):
        _ = instance.set(_ACTIVE_OPERATION_ATTRIBUTE, None)
        return
    _ = instance.set(_ACTIVE_OPERATION_ATTRIBUTE, _ActiveOperation(id=event.id, data=data))


def _clear_active_operation(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
    del ctx, event
    _ = instance.set(_ACTIVE_OPERATION_ATTRIBUTE, None)


def _active_operation_call_id(instance: "PhoneService") -> str | None:
    """Call the active request is about, or None while dialing (no call assigned yet)."""

    active = _active_operation(instance)
    if active is None or not isinstance(active.data, phone.CallIdData):
        return None
    return active.data.call_id


def _dial_call_id(instance: "PhoneService") -> str | None:
    """Call id the in-flight dial minted, or None when no dial is in flight.

    A SIP user agent mints the Call-ID when it sends the INVITE, and every later message about
    that call carries it back. Here the dial operation's envelope id *is* that handle: it already
    identifies the attempt end to end, so deriving the call from it means the two phones agree on
    one name without either side storing a second one.
    """

    active = _active_operation(instance)
    if active is None or not active.id or not isinstance(active.data, phone.DialData):
        return None
    return f"livekit:{active.id}"


def _active_operation_transfer(instance: "PhoneService") -> phone.TransferRequestData | None:
    active = _active_operation(instance)
    if active is None or not isinstance(active.data, phone.TransferRequestData):
        return None
    return active.data


def _matches_active_call(instance: "PhoneService", call_id: str) -> bool:
    active = _active_operation_call_id(instance)
    return active is not None and active == call_id


def _matches_active_operation_id(instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    active = _active_operation(instance)
    if active is None or not active.id:
        return False
    return event.id == active.id


def _matches_active_transfer(
    instance: "PhoneService",
    data: phone.TransferData
    | phone.CallTransferFailedData
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
    return isinstance(event.data, phone.AnswerRequestData)


def _has_dial_request(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.DialData)


def _has_decline_request(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.DeclineRequestData)


def _has_hang_up_request(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.HangUpRequestData)


def _has_transfer_request(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    return isinstance(event.data, phone.TransferRequestData)


def _has_hung_up(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    """True for any committed hang-up observation (all HangUpOutcome values)."""

    del ctx, instance
    return isinstance(event.data, phone.HungUpData)


def _has_provider_terminal_hang_up(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    """True when hang-up ends the active call op (not mid-transfer ``transferred`` hang-ups)."""

    del ctx
    data = event.data
    return (
        isinstance(data, phone.HungUpData)
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
    assert isinstance(data, audio.OutputData)
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
                    message="Local audio chunk publish failed.",
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
    return isinstance(event.data, phone.TransferData) and _matches_active_transfer(instance, event.data)


def _has_provider_transfer_failed(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    return isinstance(event.data, phone.CallTransferFailedData) and _matches_active_transfer(instance, event.data)


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
    return active is not None and isinstance(active.data, phone.DialData | _ActiveCallRequestData)


def _has_active_transfer_operation(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
    del ctx, event
    active = _active_operation(instance)
    return active is not None and isinstance(active.data, phone.TransferRequestData)


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


def _has_setup_delivered(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    """The acked setup belongs to *this* dial — correlated by envelope, since no call exists yet."""

    del ctx
    return _matches_active_operation_id(instance, event)


def _has_setup_accepted(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    """The callee answered *this* attempt: the call it names is the one this dial minted."""

    del ctx
    data = event.data
    return isinstance(data, phone.CallConnectedData) and data.call_id == _dial_call_id(instance)


def _has_setup_declined(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    """The callee refused *this* attempt."""

    del ctx
    data = event.data
    return isinstance(data, phone.CallIdData) and data.call_id == _dial_call_id(instance)


def _has_abandoned_dial(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    """Firmware gave up on the attempt — nobody answered, or the operator hung up mid-ring.

    Ring time is the caller's patience and firmware owns it, so the exchange learns the attempt
    is over the same way a real one does: the calling handset stops asking.
    """

    del ctx
    return isinstance(event.data, phone.NoCallData) and _dial_call_id(instance) is not None


def _has_dial_failure(
    ctx: hsm.Context,
    instance: "PhoneService",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx
    return isinstance(event.data, phone.DialFailedData) and _matches_active_operation_id(instance, event)


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
                    "local_audio_failed_chunks": 1,
                    "local_audio_failed_bytes": 960,
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
    local_audio_failed_chunks: int = pydantic.Field(
        default=0,
        ge=0,
        description=(
            "Number of local playout chunks that failed to publish onto the LiveKit track, "
            "usually because the audio was not raw PCM. The outbound mirror of "
            "remote_audio_dropped_chunks: a mute bot is observable here rather than only in logs."
        ),
        examples=[1],
    )
    local_audio_failed_bytes: int = pydantic.Field(
        default=0,
        ge=0,
        description="Number of local playout bytes that failed to publish onto the LiveKit track.",
        examples=[960],
    )


class PhoneService(hsm.Instance):
    """stateforward.mosfet-native LiveKit phone service that adapts LiveKit call control and room media to phone firmware events."""

    _operation_timeout: datetime.timedelta
    _uplink_sample_rate_hz: int
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
    # Who this phone is on a call with, or dialing. One peer on one call, the way a handset knows
    # exactly one far end — not a dictionary of everyone who happens to be in the room.
    _call_peer_identity: str | None
    _delivering_remote_audio: bool
    _attached_phone_target_ref: weakref.ReferenceType[hsm.Instance] | None
    _setup_timeout: datetime.timedelta
    _presence_bound: bool
    _presence_left_callback: collections.abc.Callable[..., object] | None
    _signaling_participant: LocalParticipant | None
    _dial_plan: signaling.DialPlan | None

    def __init__(
        self,
        *,
        url: str | None = None,
        token: str | None = None,
        track_name: str = "bot-audio",
        uplink_sample_rate_hz: int = 48_000,
        operation_timeout: datetime.timedelta = _DEFAULT_OPERATION_TIMEOUT,
        setup_timeout: datetime.timedelta = _DEFAULT_SETUP_TIMEOUT,
        room: RoomHandle | None = None,
        stream_factory: AudioStreamFactory | None = None,
        local_track_factory: LocalAudioTrackFactory | None = None,
        dial_plan: signaling.DialPlan | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        if operation_timeout <= datetime.timedelta():
            raise ValueError("operation_timeout must be a positive duration.")
        if setup_timeout <= datetime.timedelta():
            raise ValueError("setup_timeout must be a positive duration.")
        if (url is None) != (token is None):
            raise ValueError("url and token must both be provided or both omitted.")
        if not track_name:
            raise ValueError("track_name must be a non-empty string.")
        self._operation_timeout = operation_timeout
        self._setup_timeout = setup_timeout
        self._uplink_sample_rate_hz = uplink_sample_rate_hz
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
        self._call_peer_identity = None
        self._delivering_remote_audio = False
        self._attached_phone_target_ref = None
        self._presence_bound = False
        self._presence_left_callback = None
        self._signaling_participant = None
        self._dial_plan = dial_plan

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

    def _sdk_ingress_open(self) -> bool:
        """True when attach effects opened provider ingress (not a ``state()`` probe).

        HSM-CONTEXT-001: delivery policy is machine-owned attach lifetime, never
        ``instance.state()`` readiness sampling from SDK callbacks.
        """

        return self._attached_phone_target_ref is not None

    def _bind_room_presence(self, room: RoomHandle) -> None:
        """Transduce room presence: a participant vanishing is the line going dead.

        Arrival is not a call — being in the room is being reachable, and a phone that rang for
        everyone who walked into the exchange would have no callers, only neighbours. Departure
        still matters, because it is the only way to notice a far end that crashed instead of
        hanging up.

        The callback decides nothing. It reports who left; topology decides what that means for
        the call this phone is actually on.
        """

        if self._presence_bound:
            return
        service_ref = weakref.ref(self)

        def on_participant_disconnected(participant: object) -> object:
            live_service = service_ref()
            if live_service is None or not live_service._sdk_ingress_open():
                return None
            _ = live_service.dispatch(
                live_service.context(),
                _PeerLeftEvent.with_data(
                    _PeerLeftData(identity=PhoneService._participant_identity(participant)),
                ),
            )
            return None

        _ = room.on("participant_disconnected", on_participant_disconnected)
        self._presence_left_callback = on_participant_disconnected
        self._presence_bound = True

    def _unbind_room_presence(self) -> None:
        """Stop observing participant departure so a stopped service cannot receive callbacks."""

        if not self._presence_bound:
            return
        self._presence_bound = False
        callback = self._presence_left_callback
        self._presence_left_callback = None
        room = self._room
        if room is not None and callback is not None:
            room.off("participant_disconnected", callback)

    def _bind_room_signaling(self, room: RoomHandle) -> None:
        """Answer the four call-setup methods for as long as this phone is on the room.

        Registration waits for the room because ``Room.local_participant`` does not exist before
        connect: a phone has no line until it is plugged in.
        """

        if self._signaling_participant is not None:
            return
        participant = room.local_participant
        service_ref = weakref.ref(self)

        def handler_for(
            build: collections.abc.Callable[[str, signaling.MessageData], hsm.Event[typing.Any]],
        ) -> collections.abc.Callable[[object], str]:
            def handle(data: object) -> str:
                caller_identity, message = signaling.invocation(data)
                live_service = service_ref()
                if live_service is None or not live_service._sdk_ingress_open():
                    raise rtc.RpcError(
                        rtc.RpcError.ErrorCode.RECIPIENT_NOT_FOUND,
                        "This LiveKit phone is not attached to a handset and cannot take calls.",
                    )
                # Answer the wire now, and let the phone ring on its own time. Awaiting the
                # dispatch would hold the caller's setup transaction open across a ring, a sound
                # stimulus, and possibly a whole cognition turn — long past any RPC deadline.
                # The message arrived off the wire, so the event minted from it is stamped with
                # this arrival's context; without it every observation of an inbound call starts
                # its own trace. Caller identity and call id stay out of the attributes.
                with span.operation(
                    "bot.provider.livekit.phone.signaling.receive",
                    scope=_SCOPE,
                    component=_COMPONENT,
                    stage="signal_in",
                ):
                    _ = live_service.dispatch(
                        live_service.context(),
                        inject_context(build(caller_identity, message)),
                    )
                return message.model_dump_json()

            return handle

        _ = participant.register_rpc_method(
            signaling.SetupMethod,
            handler_for(
                lambda caller_identity, message: ServiceIncomingCallEvent.with_data(
                    phone.IncomingCallData(call_id=message.call_id, caller=caller_identity),
                ),
            ),
        )
        _ = participant.register_rpc_method(
            signaling.AcceptMethod,
            handler_for(
                lambda caller_identity, message: _SetupAcceptedEvent.with_data(
                    phone.CallConnectedData(call_id=message.call_id),
                ),
            ),
        )
        _ = participant.register_rpc_method(
            signaling.DeclineMethod,
            handler_for(
                lambda caller_identity, message: _SetupDeclinedEvent.with_data(
                    phone.CallIdData(call_id=message.call_id),
                ),
            ),
        )
        _ = participant.register_rpc_method(
            signaling.ByeMethod,
            handler_for(
                lambda caller_identity, message: ServiceRemoteHangUpEvent.with_data(
                    phone.RemoteHangUpData(call_id=message.call_id),
                ),
            ),
        )
        self._signaling_participant = participant

    def _unbind_room_signaling(self) -> None:
        """Stop answering call setup. The phone is off the line; nothing may ring it."""

        participant = self._signaling_participant
        if participant is None:
            return
        self._signaling_participant = None
        for method in signaling.Methods:
            participant.unregister_rpc_method(method)

    async def _signal_peer(self, method: str, call_id: str) -> None:
        """Tell the other end what this phone just did.

        No peer and no line means there is nothing to tell: a room-media phone with no signalling
        wire still completes its own side locally, exactly as it did before there was one.
        """

        participant = self._signaling_participant
        peer_identity = self._call_peer_identity
        if participant is None or peer_identity is None:
            return
        # `method` is one of the four call-setup constants, so it is a dimension, not a label.
        # The peer identity (a phone number here) and the call id never become attributes.
        with span.operation(
            "bot.provider.livekit.phone.signaling.send",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="signal_out",
            attributes={"bot.signaling.method": method},
        ):
            try:
                _ = await participant.perform_rpc(
                    destination_identity=peer_identity,
                    method=method,
                    payload=signaling.MessageData(call_id=call_id).model_dump_json(),
                    response_timeout=self._setup_timeout.total_seconds(),
                )
            except rtc.RpcError as error:
                raise PhoneServiceError(
                    "LiveKit phone signaling RPC failed.",
                    failure_kind=signaling.failure_kind(error),
                ) from error

    def _ensure_media(self) -> tuple[AudioBridge[typing.Any], RoomAudioTrackPath]:
        if self._bridge is not None and self._track_path is not None:
            return self._bridge, self._track_path
        service_ref = weakref.ref(self)

        async def deliver_batched_remote_audio(audio_input: audio.InputData) -> None:
            """Publish one batched utterance into PhoneService (guards decide deliver vs drop).

            HSM-CONTEXT-001: do not gate on ``context().is_done()`` or ``state()``. Attach
            lifetime opens ingress; HSM guards decide deliver vs drop once dispatched.
            """

            live_service = service_ref()
            if live_service is None or not live_service._sdk_ingress_open():
                return
            await live_service.receive_remote_audio(live_service.context(), audio_input)

        async def consume_remote_audio(audio_input: audio.InputData) -> None:
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

        # A LiveKit audio source is fixed at construction, so the rate the robot speaks at has to
        # be declared here rather than discovered from the first frame: the local track is
        # published at connect time, before anything is ever spoken. A frame that disagrees is
        # refused outright ("sample_rate and num_channels don't match"), which is a mute call.
        bridge = create_audio_bridge(sample_rate_hz=self._uplink_sample_rate_hz, loop=self._loop)
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

    async def dial(self, request: phone.DialData) -> None:
        """Address setup at the dialled number's participant identity and return once it is ringing.

        On this SFU fiction the participant identity **is** the phone number (same normalized
        digit form ``DialData`` already validated). Setup is sent to ``request.number`` unless an
        optional :class:`~signaling.DialPlan` remaps aliases for tests or rare provisioning.

        When a plan is present and returns None, that is a wrong number answered by the exchange
        with nothing put on the wire. When no plan (or the plan maps to an identity), the room
        decides presence: no participant with that identity yields ``remote_unavailable``.

        Returning does **not** mean connected. The ack on setup means the far end is ringing; the
        connect arrives later, as an accept, because whether to answer is the callee's decision
        and nothing here may make it for them.
        """

        call_id = _dial_call_id(self)
        participant = self._signaling_participant
        if call_id is None or participant is None:
            raise PhoneServiceError(
                "This LiveKit phone is not on a connected room, so there is no line to dial out on.",
                failure_kind="provider_unavailable",
            )
        dial_plan = self._dial_plan
        if dial_plan is None:
            endpoint = request.number
        else:
            endpoint = dial_plan.endpoint(request.number)
            if endpoint is None:
                # Deliberately says nothing about which number: a wrong number is all a caller is told,
                # and a log line is not the place to start keeping a record of who was dialled.
                raise PhoneServiceError(
                    "No line is registered against the number dialled.",
                    failure_kind="remote_unavailable",
                )
        # A phone that has dialled knows who it dialled, before it knows whether they will answer.
        self._call_peer_identity = endpoint
        try:
            _ = await participant.perform_rpc(
                destination_identity=endpoint,
                method=signaling.SetupMethod,
                payload=signaling.MessageData(call_id=call_id).model_dump_json(),
                response_timeout=self._setup_timeout.total_seconds(),
            )
        except rtc.RpcError as error:
            self._call_peer_identity = None
            raise PhoneServiceError(
                "LiveKit phone call setup failed.",
                failure_kind=signaling.failure_kind(error),
            ) from error

    async def answer_call(self, request: phone.AnswerRequestData) -> None:
        """Answer a call, and tell the caller so their phone stops ringing and connects.

        Room-media installations complete answer locally (the bot decides to answer; no SIP
        gateway required). Pure call-control stubs without room media still report unavailable.
        """

        if not self._room_media_enabled():
            _raise_unavailable_call_control()
        await self._signal_peer(signaling.AcceptMethod, request.call_id)

    async def decline_call(self, request: phone.DeclineRequestData) -> None:
        """Decline a ringing call, and tell the caller they were refused rather than unreachable."""

        if not self._room_media_enabled():
            _raise_unavailable_call_control()
        await self._signal_peer(signaling.DeclineMethod, request.call_id)

    async def hang_up_call(self, request: phone.HangUpRequestData) -> None:
        """Hang up an active call, and tell the other end.

        Signalling is the only way the far end can learn this: a bot that hangs up stays in the
        room, so nothing about its presence changes when the call ends.
        """

        if not self._room_media_enabled():
            _raise_unavailable_call_control()
        await self._signal_peer(signaling.ByeMethod, request.call_id)

    async def transfer_call(self, request: phone.TransferRequestData) -> None:
        """Transfer an active LiveKit/SIP call. Default installation reports call control unavailable."""

        del request
        _raise_unavailable_call_control()

    async def attach(self, environment: Environment, target: hsm.Instance) -> None:
        """Attach this service to the phone-owned firmware target."""

        require_environment_scope(environment, target, participant="Phone service target")
        _, track_path = self._ensure_media()
        # Start this service before room connect so its signaling handlers are registered by the
        # time setup can arrive; an unattached phone answers RPC with RECIPIENT_NOT_FOUND.
        if not mosfet.lifecycle.is_started(self):
            _ = await mosfet.started(environment, self, self.model, owner=target)
        if not mosfet.lifecycle.is_started(track_path):
            _ = await mosfet.started(environment, track_path, track_path.model, owner=self)
        require_environment_scope(environment, self, participant="PhoneService")
        await self.dispatch(environment, _ServiceAttachedEvent.with_data(_PhoneServiceAttachmentData(target=target)))
        target_ref = self._attached_phone_target_ref
        current_target = None if target_ref is None else target_ref()
        if current_target is not target:
            raise PhoneServiceError(
                "LiveKit phone service is already attached to another phone event target.",
                failure_kind="provider_unavailable",
            )
        _ = mosfet.register(self, self.model, owner=target)
        if self._room_connect is not None and self._local_track_sid is None:
            await track_path.connect_room(track_path.context(), self._room_connect)

    async def detach(self, environment: Environment, target: hsm.Instance) -> None:
        """Detach this service from the phone-owned firmware target."""

        # Idempotent when already stopped; no firmware reply channel on this API.
        if not mosfet.lifecycle.is_started(self):
            return
        require_environment_scope(environment, target, participant="Phone service target")
        require_environment_scope(environment, self, participant="PhoneService")
        await self.dispatch(environment, _ServiceDetachedEvent.with_data(_PhoneServiceAttachmentData(target=target)))
        target_ref = self._attached_phone_target_ref
        current_target = None if target_ref is None else target_ref()
        if current_target is None:
            _ = mosfet.register(self, self.model, clear_owner=True)

    async def connect_room(
        self,
        *,
        url: str,
        token: str,
        track_name: str = "bot-audio",
    ) -> None:
        """Connect the privately owned LiveKit room audio path for this service."""

        _, track_path = self._ensure_media()
        if not mosfet.lifecycle.is_started(track_path):
            # Parent the track path under this service only while the service is live.
            parent = self.context() if mosfet.lifecycle.is_started(self) else None
            _ = await mosfet.started(parent, track_path, track_path.model, owner=self)
        await track_path.connect_room(
            track_path.context(),
            RoomAudioConnectData(url=url, token=token, track_name=track_name),
        )

    async def disconnect_room(self) -> None:
        """Disconnect room audio from the underlying LiveKit room."""

        _, track_path = self._ensure_media()
        self._unbind_room_presence()
        self._unbind_room_signaling()
        await track_path.disconnect_room(track_path.context())

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop this service and its privately owned room track path if started."""

        track_path = self._track_path
        # Ingress opens when attach holds a target ref; clear on stop so SDK callbacks
        # cannot deliver after the service is stopped (only detach effect cleared it before).
        self._attached_phone_target_ref = None
        self._unbind_room_presence()
        self._unbind_room_signaling()
        if mosfet.lifecycle.is_started(self):
            await hsm.Instance.stop(self, ctx)
        if self.model is not None:
            _ = mosfet.register(self, self.model, clear_owner=True)
        if track_path is None:
            return
        # Track path is created by _ensure_media before start; only stop if started.
        if mosfet.lifecycle.is_started(track_path):
            await hsm.stop(track_path, ctx)
        if track_path.model is not None:
            _ = mosfet.register(track_path, track_path.model, clear_owner=True)

    async def publish_audio(self, output: audio.OutputData) -> None:
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
            local_audio_failed_chunks=self._local_audio_failed_chunks,
            local_audio_failed_bytes=self._local_audio_failed_bytes,
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
    def _setup_timeout_value(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, event
        return instance._setup_timeout

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
            # The line goes live with the room: before connect there is no local participant to
            # register call setup on, and after it this phone can both ring and be rung.
            instance._bind_room_signaling(room)
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
        return isinstance(event.data, audio.InputData)

    @staticmethod
    def _phone_event_target(instance: "PhoneService") -> hsm.Instance | None:
        target_ref = instance._attached_phone_target_ref
        if target_ref is None:
            return None
        return target_ref()

    @staticmethod
    def _can_deliver_remote_audio(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        if not isinstance(event.data, audio.InputData):
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
        """True when local speaker playout should uplink (exact OutputData, not remote service audio)."""

        del ctx
        # Exact type: ServiceAudioData subclasses OutputData and must not uplink as local playout.
        return type(event.data) is audio.OutputData and not instance._delivering_remote_audio

    @staticmethod
    def _clear_media_on_hung_up(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Any HungUp observation ends the media session (matches prior publish-side clear)."""

        del ctx
        if isinstance(event.data, phone.HungUpData):
            instance._media_call_id = None
            instance._call_peer_identity = None

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
        assert isinstance(data, audio.InputData)
        call_id = instance._media_call_id
        phone_event_target = PhoneService._phone_event_target(instance)
        assert call_id is not None
        assert phone_event_target is not None
        instance._remote_audio_chunks += 1
        instance._remote_audio_bytes += len(data.audio)
        instance._delivering_remote_audio = True
        try:
            with span.operation(
                "bot.provider.livekit.phone.remote_audio.deliver",
                scope=_SCOPE,
                component=_COMPONENT,
                stage="deliver",
            ) as active:
                active.set_attribute("bot.audio.bytes", len(data.audio))
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
        assert isinstance(data, audio.InputData)
        instance._remote_audio_dropped_chunks += 1
        instance._remote_audio_dropped_bytes += len(data.audio)
        # Refusing delivery is decisive and was previously invisible: the counters said how much
        # was dropped, nothing said why. Same order the guard checks in, so the kind names the
        # first condition that failed.
        # The kind is a closed vocabulary, not an error: dropping is the modelled outcome of a
        # guard, so the span is `ok` and the reason is the attribute that answers "why".
        kind = "media_call_inactive" if instance._media_call_id is None else "phone_not_attached"
        with span.operation(
            "bot.provider.livekit.phone.remote_audio.drop",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="deliver",
            attributes={"bot.remote_audio.drop_reason": kind},
        ) as active:
            active.set_attribute("bot.audio.bytes", len(data.audio))
        if instance._remote_audio_dropped_chunks == 1:
            # Once per call: a phone that drops the first chunk drops every chunk after it for the
            # same reason, and the difference between a deaf bot and a working one is this line.
            _LOG.warning("livekit remote audio dropped reason=%s", kind)

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
        # Caller ID is who the call is with. The callee adopts both the caller's call id and the
        # caller themself from setup, so from here on both phones name the same call and the same
        # peer.
        instance._call_peer_identity = data.caller
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
        instance._call_peer_identity = None
        PhoneService._emit_phone_event(ctx, instance, event, phone.RemoteHangUpEvent.with_data(data))

    @staticmethod
    def _emit_dial_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.DialFailedData)
        instance._media_call_id = None
        instance._call_peer_identity = None
        PhoneService._emit_phone_event(ctx, instance, event, phone.ServiceDialFailedEvent.with_data(data))

    @staticmethod
    def _emit_call_failed(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.CallFailedData)
        instance._media_call_id = None
        instance._call_peer_identity = None
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
        # Stamp who the call is with at the boundary: the peer this phone adopted from setup or
        # the dial plan. None when the far end was never identified — never invented.
        PhoneService._emit_phone_event(
            ctx,
            instance,
            event,
            phone.CallConnectedEvent.with_data(
                phone.CallConnectedData(call_id=data.call_id, party=instance._call_peer_identity)
            ),
        )

    @staticmethod
    def _emit_media_ready_for_connected_call(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Media for a call that just connected, when this phone's track is already published.

        Media is ready once two things hold: a connected call, and a published local track. They
        arrive in either order and whichever lands second is what makes media ready. This is the
        half where the call lands second — both when the callee answers and when the caller's
        setup is accepted, because either is the same moment for the phone it happens on.
        ``_apply_room_audio_status`` is the half where the track lands second.
        """

        data = event.data
        assert isinstance(data, phone.CallConnectedData)
        if instance._local_track_sid is None:
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                ServiceMediaReadyEvent.with_data(phone.MediaReadyData(call_id=data.call_id)),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _has_dial_peer_left(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
        """The endpoint this dial is ringing has left the room: nobody is there to answer."""

        del ctx
        data = event.data
        return (
            isinstance(data, _PeerLeftData)
            and _dial_call_id(instance) is not None
            and instance._call_peer_identity == data.identity
        )

    @staticmethod
    def _has_call_peer_left(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> bool:
        """The far end of the call this phone is on left the room — the line went dead."""

        del ctx
        data = event.data
        return (
            isinstance(data, _PeerLeftData)
            and instance._media_call_id is not None
            and instance._call_peer_identity == data.identity
        )

    @staticmethod
    def _emit_dial_failed_on_peer_loss(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        """A dial whose callee vanished never becomes a call, so there is none to fail."""

        active = _active_operation(instance)
        assert active is not None
        PhoneService._emit_dial_failed(
            ctx,
            instance,
            _with_operation_correlation(
                _DialFailedEvent.with_data(phone.DialFailedData(failure_kind="remote_unavailable")),
                operation_id=active.id,
                metadata=event.metadata,
            ),
        )

    @staticmethod
    def _emit_setup_declined(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        """A refusal is not an absence: somebody was there, and they said no."""

        active = _active_operation(instance)
        assert active is not None
        PhoneService._emit_dial_failed(
            ctx,
            instance,
            _with_operation_correlation(
                _DialFailedEvent.with_data(phone.DialFailedData(failure_kind="call_declined")),
                operation_id=active.id,
                metadata=event.metadata,
            ),
        )

    @staticmethod
    def _report_peer_hang_up(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        """A far end that vanished mid-call hung up without saying so; say it for them."""

        call_id = instance._media_call_id
        assert call_id is not None
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                ServiceRemoteHangUpEvent.with_data(phone.RemoteHangUpData(call_id=call_id)),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _cancel_dial(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        """Stop a ringing callee once the caller has given up on the attempt."""

        del ctx, event
        call_id = _dial_call_id(instance)
        if call_id is None:
            return
        cancelled = asyncio.ensure_future(instance._signal_peer(signaling.ByeMethod, call_id))
        instance._call_peer_identity = None

        def _note_undelivered(done: asyncio.Future[None]) -> None:
            if done.cancelled():
                return
            error = done.exception()
            if error is None:
                return
            # Operationally load-bearing: the callee keeps ringing for a call nobody is on.
            _LOG.warning("livekit phone abandoned dial not cancelled at the callee reason=%s", error)

        cancelled.add_done_callback(_note_undelivered)

    @staticmethod
    def _unbind_signaling(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._unbind_room_presence()
        instance._unbind_room_signaling()

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
        assert isinstance(data, phone.DialData | _ActiveCallRequestData)
        if not isinstance(data, phone.CallIdData):
            # A dial that timed out has no call to fail: the exchange never assigned one.
            PhoneService._emit_dial_failed(
                ctx,
                instance,
                _with_operation_correlation(
                    _DialFailedEvent.with_data(phone.DialFailedData(failure_kind="timeout")),
                    operation_id=active.id,
                ),
            )
            return
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
        assert isinstance(active.data, phone.TransferRequestData)
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
        assert isinstance(data, phone.AnswerRequestData)
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

    @staticmethod
    async def _run_dial(ctx: hsm.Context, instance: "PhoneService", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, phone.DialData)
        try:
            await instance.dial(data)
        except Exception as error:
            failure = phone.DialFailedData(failure_kind=_failure_kind(error))
            _ = hsm.dispatch(ctx, instance, _with_trigger_correlation(_DialFailedEvent.with_data(failure), event))
            return
        # Setup was acknowledged: the far end is ringing. Not connected — that is theirs to decide.
        _ = hsm.dispatch(ctx, instance, _with_trigger_correlation(_SetupDeliveredEvent, event))

    @staticmethod
    async def _run_decline_call(
        ctx: hsm.Context,
        instance: "PhoneService",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, phone.DeclineRequestData)
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
        assert isinstance(data, phone.HangUpRequestData)
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
        assert isinstance(data, phone.TransferRequestData)
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

    def receive_remote_audio(self, ctx: hsm.Context, data: audio.InputData) -> collections.abc.Awaitable[None]:
        """Publish remote room PCM into PhoneService. Delivery is gated by HSM guards.

        Remote audio is delivered to phone firmware as `ServiceAudioReceived` only when a media
        call id is active and a phone event target is attached; otherwise it is dropped and
        counted on `media_snapshot().remote_audio_dropped_*`.

        The chunk arrives from the transport plane, so the trace context of the frames it was
        assembled from is stamped onto the event here: everything downstream (the delivery to
        firmware, and every HSM observation of it) reads that stamp rather than starting a new
        trace, which is what makes one sound followable from the wire to perception.
        """

        return self.dispatch(ctx, inject_context(_RemoteAudioReceivedEvent.with_data(data)))

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

    model: typing.ClassVar[hsm.Model] = mosfet.define(
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
            hsm.effect(_unbind_signaling),
            hsm.effect(_clear_phone_event_target),
            hsm.target("/PhoneService/unconnected"),
        ),
        # A far end that vanished mid-call hung up: presence transduces, topology decides. Dialing
        # overrides this from its own substates, because a callee who leaves while ringing never
        # made a call to hang up.
        hsm.transition(
            hsm.on(_PeerLeftEvent),
            hsm.guard(_has_call_peer_left),
            hsm.effect(_report_peer_hang_up),
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
            *_active_call_operation_resolution_transitions(
                emit_remote_hang_up=_emit_remote_hang_up,
                emit_call_failed=_emit_call_failed,
                on_hung_up=_clear_media_and_maybe_active_operation,
            ),
            _ignore_transfer_terminal_observation_transition(),
            hsm.transition(
                hsm.on(_DialFailedEvent),
                hsm.guard(_has_dial_failure),
                hsm.effect(_emit_dial_failed),
                hsm.effect(_clear_active_operation),
                hsm.target("/PhoneService/ready"),
            ),
            # A callee who left the room cannot answer, whether setup is still crossing the room
            # or already ringing there.
            hsm.transition(
                hsm.on(_PeerLeftEvent),
                hsm.guard(_has_dial_peer_left),
                hsm.effect(_emit_dial_failed_on_peer_loss),
                hsm.effect(_clear_active_operation),
                hsm.target("/PhoneService/ready"),
            ),
            hsm.initial(hsm.target("/PhoneService/dialing/setup")),
            # Two waits, two budgets. Getting call setup across the room is a transaction the
            # exchange bounds in seconds; how long to let it ring afterwards is the caller's
            # patience, which is firmware's to spend, not the provider's to cut short.
            hsm.state(
                "setup",
                hsm.activity(_run_dial),
                hsm.transition(
                    hsm.on(_SetupDeliveredEvent),
                    hsm.guard(_has_setup_delivered),
                    hsm.target("/PhoneService/dialing/ringing"),
                ),
                _active_call_operation_timeout_transition(
                    operation_timeout=_setup_timeout_value,
                    emit_active_call_timeout=_emit_active_call_timeout,
                ),
            ),
            hsm.state(
                "ringing",
                hsm.transition(
                    hsm.on(_SetupAcceptedEvent),
                    hsm.guard(_has_setup_accepted),
                    hsm.effect(_emit_call_connected),
                    hsm.effect(_emit_media_ready_for_connected_call),
                    hsm.effect(_clear_active_operation),
                    hsm.target("/PhoneService/ready"),
                ),
                hsm.transition(
                    hsm.on(_SetupDeclinedEvent),
                    hsm.guard(_has_setup_declined),
                    hsm.effect(_emit_setup_declined),
                    hsm.effect(_clear_active_operation),
                    hsm.target("/PhoneService/ready"),
                ),
                # Firmware stopped waiting (nobody answered, or the operator hung up mid-ring).
                # Cancel at the callee so it does not keep ringing for a call nobody is on.
                hsm.transition(
                    hsm.on(phone.NoCallEvent),
                    hsm.guard(_has_abandoned_dial),
                    hsm.effect(_cancel_dial),
                    hsm.effect(_clear_active_operation),
                    hsm.target("/PhoneService/ready"),
                ),
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
                hsm.effect(_emit_media_ready_for_connected_call),
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
        peripherals: collections.abc.Iterable[mosfet.device.Device] = (),
        operation_timeout: datetime.timedelta = _DEFAULT_OPERATION_TIMEOUT,
        answer_timeout: datetime.timedelta = _DEFAULT_ANSWER_TIMEOUT,
        transfer_timeout: datetime.timedelta = _DEFAULT_TRANSFER_TIMEOUT,
        dial_plan: signaling.DialPlan | None = None,
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
                dial_plan=dial_plan,
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

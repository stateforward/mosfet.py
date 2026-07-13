from bot.devices import audio

import asyncio
import collections.abc
import dataclasses
import datetime
import importlib.resources
import typing

import hsm
import bot.device

from bot.event_schema import validate_event_data
from bot.telemetry import observer
from bot.world import SoundData, SoundEvent, World, require_world_scope

from .events import (
    AnswerCallData,
    AnswerCallEvent,
    AnsweredEvent,
    CallConnectedData,
    CallConnectedEvent,
    CallFailedData,
    CallFailedEvent,
    CallIdData,
    CallTransferCompletedEvent,
    CallTransferFailedEvent,
    DeclineCallData,
    DeclineCallEvent,
    DialData,
    DialEvent,
    HangUpCallData,
    HangUpCallEvent,
    HungUpEvent,
    IncomingCallData,
    IncomingCallEvent,
    MediaReadyData,
    MediaReadyEvent,
    PhoneCallData,
    PhoneHungUpData,
    PhoneTransferData,
    PhoneTransferFailedData,
    RemoteHangUpData,
    RemoteHangUpEvent,
    RingingEvent,
    ServiceAnswerRequestedEvent,
    ServiceAudioData,
    ServiceAudioReceivedEvent,
    ServiceDeclineRequestedEvent,
    ServiceDialRequestedEvent,
    ServiceHangUpRequestedEvent,
    ServiceMediaReadyEvent,
    ServiceTransferCompletedEvent,
    ServiceTransferFailedEvent,
    ServiceTransferRequestedEvent,
    TransferAcceptedData,
    TransferAcceptedEvent,
    TransferCallData,
    TransferCallEvent,
    TransferCompletedData,
    TransferFailedData,
    TransferStartedEvent,
    TransferTarget,
)

_DEFAULT_ANSWER_TIMEOUT = datetime.timedelta(seconds=30)
_DEFAULT_TRANSFER_TIMEOUT = datetime.timedelta(seconds=30)
_PHONE_SERVICE_ORIGINATING_EVENT_NAMES = frozenset(
    {
        IncomingCallEvent.name,
        CallConnectedEvent.name,
        CallFailedEvent.name,
        ServiceMediaReadyEvent.name,
        RemoteHangUpEvent.name,
        ServiceAudioReceivedEvent.name,
        TransferAcceptedEvent.name,
        ServiceTransferCompletedEvent.name,
        ServiceTransferFailedEvent.name,
    }
)
_PHONE_OWNER_COMMAND_EVENTS = (
    DialEvent,
    AnswerCallEvent,
    DeclineCallEvent,
    HangUpCallEvent,
    TransferCallEvent,
)
_PHONE_OWNER_COMMAND_EVENTS_BY_NAME: collections.abc.Mapping[str, hsm.Event[typing.Any]] = {
    event.name: event for event in _PHONE_OWNER_COMMAND_EVENTS
}
_PHONE_OWNER_COMMAND_EVENT_NAMES = frozenset(_PHONE_OWNER_COMMAND_EVENTS_BY_NAME)
_PHONE_OBSERVATION_EVENT_NAMES = frozenset(
    {
        RingingEvent.name,
        AnsweredEvent.name,
        MediaReadyEvent.name,
        HungUpEvent.name,
        TransferStartedEvent.name,
        CallTransferCompletedEvent.name,
        CallTransferFailedEvent.name,
    }
)
def _load_ring_sound_wav() -> bytes:
    """Load the short package-local landline ring clip for world.sound elevation."""

    return (importlib.resources.files(__package__) / "assets" / "ring.wav").read_bytes()


# Real short ring WAV (see assets/SOURCES.md). Sensory classifiers match this acoustic payload;
# Listening does not special-case phone.
RING_SOUND_WAV = _load_ring_sound_wav()
_PHONE_CALL_ID_METADATA_KEY = "bot.phone.call_id"
_RingingCommittedEvent = hsm.Event[PhoneCallData](
    name="bot.phone.ringing.committed",
    kind=hsm.CompletionEventKind,
    schema=PhoneCallData,
)
_AnsweredCommittedEvent = hsm.Event[PhoneCallData](
    name="bot.phone.answered.committed",
    kind=hsm.CompletionEventKind,
    schema=PhoneCallData,
)
_MediaReadyCommittedEvent = hsm.Event[PhoneCallData](
    name="bot.phone.media_ready.committed",
    kind=hsm.CompletionEventKind,
    schema=PhoneCallData,
)
_HungUpCommittedEvent = hsm.Event[PhoneHungUpData](
    name="bot.phone.hung_up.committed",
    kind=hsm.CompletionEventKind,
    schema=PhoneHungUpData,
)
_TransferStartedCommittedEvent = hsm.Event[PhoneTransferData](
    name="bot.phone.transfer_started.committed",
    kind=hsm.CompletionEventKind,
    schema=PhoneTransferData,
)
_TransferCompletedCommittedEvent = hsm.Event[PhoneTransferData](
    name="bot.phone.transfer_completed.committed",
    kind=hsm.CompletionEventKind,
    schema=PhoneTransferData,
)
_TransferFailedCommittedEvent = hsm.Event[PhoneTransferFailedData](
    name="bot.phone.transfer_failed.committed",
    kind=hsm.CompletionEventKind,
    schema=PhoneTransferFailedData,
)


def _completed_phone_service_event() -> asyncio.Future[None]:
    future = asyncio.get_running_loop().create_future()
    future.set_result(None)
    return future


@dataclasses.dataclass
class _PhoneObservationService:
    owner: "Phone"
    service: "PhoneService"
    speaker: audio.Speaker
    target: hsm.Instance | None = dataclasses.field(default=None, init=False)

    async def attach(self, world: World, target: hsm.Instance) -> None:
        await self.service.attach(world, target)
        self.target = target

    async def detach(self, world: World, target: hsm.Instance) -> None:
        await self.service.detach(world, target)
        if self.target is target:
            self.target = None

    def speaker_ready_in_world(self, ctx: hsm.Context) -> bool:
        """True when the phone-owned speaker is running in the same world as ``ctx``."""

        if not self.speaker.state():
            return False
        speaker_context = self.speaker.context()
        return speaker_context.value(hsm.Keys.Instances) is World.from_context(ctx).value(hsm.Keys.Instances)

    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        if event.name == audio.OutputEvent.name:
            data = event.data
            assert isinstance(data, audio.AudioOutputData)
            _ = self.speaker.dispatch_audio_output_to_world(ctx, data, metadata=event.metadata)
            # Offer speaker audio to the provider for call uplink (providers suppress remote-delivery echoes).
            target = hsm.id(self.service) if isinstance(self.service, hsm.Instance) else ""
            self.service.publish(
                ctx,
                dataclasses.replace(
                    event,
                    source=hsm.id(self.target) if self.target is not None else hsm.id(self.owner),
                    target=target,
                    metadata=dict(event.metadata),
                ),
            )
            return
        target = hsm.id(self.service) if isinstance(self.service, hsm.Instance) else ""
        self.service.publish(
            ctx,
            dataclasses.replace(
                event,
                source=hsm.id(self.target) if self.target is not None else hsm.id(self.owner),
                target=target,
                metadata=dict(event.metadata),
            ),
        )
        _broadcast_observation(self.owner, ctx, event)


class PhoneService(typing.Protocol):
    """Service that receives provider requests and committed public phone events."""

    def attach(self, world: World, target: hsm.Instance) -> collections.abc.Awaitable[None]:
        """Attach this phone service to a phone-owned firmware instance."""
        ...

    def detach(self, world: World, target: hsm.Instance) -> collections.abc.Awaitable[None]:
        """Detach this phone service from a phone-owned firmware instance."""
        ...

    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        """Publish one phone event emitted by firmware."""


class PhoneEventRecorder:
    """In-memory phone service for tests and embedders without a provider yet."""

    _events: list[hsm.Event[typing.Any]]
    _target: hsm.Instance | None

    def __init__(self) -> None:
        self._events = []
        self._target = None

    @property
    def events(self) -> tuple[hsm.Event[typing.Any], ...]:
        return tuple(self._events)

    async def attach(self, world: World, target: hsm.Instance) -> None:
        require_world_scope(world, target, participant="Phone service target")
        self._target = target

    async def detach(self, world: World, target: hsm.Instance) -> None:
        require_world_scope(world, target, participant="Phone service target")
        if self._target is target:
            self._target = None

    def receive(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[None]:
        """Emit one provider-originating event into the attached phone firmware."""

        if self._target is None or event.name not in _PHONE_SERVICE_ORIGINATING_EVENT_NAMES:
            return _completed_phone_service_event()
        return self._target.dispatch(ctx, event)

    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        # Speaker uplink offers are for live providers; the in-memory recorder ignores them.
        if event.name == audio.OutputEvent.name:
            return
        self._events.append(event)


def _require_positive_timeout(name: str, value: datetime.timedelta) -> None:
    if value <= datetime.timedelta():
        raise ValueError(f"{name} must be a positive duration.")


def _world_observation_event(owner: "Phone", event: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
    """Map phone observations that bots experience as input energy into world stimuli.

    Ringing is heard as ``world.sound`` with ``source`` = the phone instance id. Device-plane
    ``phone.ringing`` still flows on the service/firmware path; bots do not receive it as a
    raw cognitive stimulus.
    """

    if event.name != RingingEvent.name:
        return dataclasses.replace(
            event,
            source=hsm.id(owner),
            metadata=dict(event.metadata),
        )
    data = event.data
    assert isinstance(data, PhoneCallData)
    metadata = dict(event.metadata)
    metadata[_PHONE_CALL_ID_METADATA_KEY] = data.call_id
    return dataclasses.replace(
        SoundEvent.with_data(
            SoundData(
                audio=RING_SOUND_WAV,
                media_type="audio/wav",
                sample_rate_hz=16_000,
                channels=1,
                kind="ring",
            )
        ),
        source=hsm.id(owner),
        metadata=metadata,
    )


def _broadcast_observation(owner: "Phone", ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
    if event.name not in _PHONE_OBSERVATION_EVENT_NAMES:
        return
    _ = hsm.dispatch_all(World.from_context(ctx), _world_observation_event(owner, event))


class PhoneFirmware(hsm.Instance):
    """Phone-owned firmware state for a single active call."""

    _service: PhoneService
    _answer_timeout: datetime.timedelta
    _transfer_timeout: datetime.timedelta
    _closed_call_ids: frozenset[str]
    _current_call_id: str | None
    _current_transfer_id: str | None
    _current_transfer_target: TransferTarget | None

    def __init__(
        self,
        *,
        service: PhoneService | None = None,
        answer_timeout: datetime.timedelta = _DEFAULT_ANSWER_TIMEOUT,
        transfer_timeout: datetime.timedelta = _DEFAULT_TRANSFER_TIMEOUT,
    ) -> None:
        super().__init__()
        _require_positive_timeout("answer_timeout", answer_timeout)
        _require_positive_timeout("transfer_timeout", transfer_timeout)
        self._service = service if service is not None else PhoneEventRecorder()
        self._answer_timeout = answer_timeout
        self._transfer_timeout = transfer_timeout
        self._closed_call_ids = frozenset()
        self._current_call_id = None
        self._current_transfer_id = None
        self._current_transfer_target = None

    def event_recorder(self) -> PhoneEventRecorder:
        service = self._service
        if isinstance(service, _PhoneObservationService):
            service = service.service
        assert isinstance(service, PhoneEventRecorder)
        return service

    @staticmethod
    def _publish(
        ctx: hsm.Context,
        instance: "PhoneFirmware",
        trigger: hsm.Event,
        event: hsm.Event[typing.Any],
    ) -> None:
        instance._service.publish(ctx, dataclasses.replace(event, metadata=dict(trigger.metadata)))

    @staticmethod
    def _queue_committed(
        ctx: hsm.Context,
        instance: "PhoneFirmware",
        trigger: hsm.Event,
        event: hsm.Event[typing.Any],
    ) -> None:
        _ = hsm.dispatch(ctx, instance, dataclasses.replace(event, metadata=dict(trigger.metadata)))

    @staticmethod
    def _is_new_incoming_call(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, IncomingCallData) and data.call_id not in instance._closed_call_ids

    @staticmethod
    def _is_new_dial(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, DialData) and data.call_id not in instance._closed_call_ids

    @staticmethod
    def _matches_current_answer_command(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, AnswerCallData) and instance._current_call_id == data.call_id

    @staticmethod
    def _matches_current_call_connected(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, CallConnectedData) and instance._current_call_id == data.call_id

    @staticmethod
    def _matches_current_call_failed(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, CallFailedData) and instance._current_call_id == data.call_id

    @staticmethod
    def _matches_current_decline_command(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, DeclineCallData) and instance._current_call_id == data.call_id

    @staticmethod
    def _matches_current_hang_up_command(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, HangUpCallData) and instance._current_call_id == data.call_id

    @staticmethod
    def _matches_current_incoming_call(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, IncomingCallData) and instance._current_call_id == data.call_id

    @staticmethod
    def _matches_current_media_ready(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, MediaReadyData) and instance._current_call_id == data.call_id

    @staticmethod
    def _matches_current_service_audio(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        data = event.data
        if not isinstance(data, ServiceAudioData) or instance._current_call_id != data.call_id:
            return False
        service = instance._service
        if not isinstance(service, _PhoneObservationService):
            return False
        return service.speaker_ready_in_world(ctx)

    @staticmethod
    def _matches_current_remote_hang_up(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, RemoteHangUpData) and instance._current_call_id == data.call_id

    @staticmethod
    def _matches_current_transfer_command(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, TransferCallData) and instance._current_call_id == data.call_id

    @staticmethod
    def _matches_current_transfer_accepted(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, TransferAcceptedData)
            and instance._current_call_id == data.call_id
            and instance._current_transfer_id == data.transfer_id
            and instance._current_transfer_target == data.target
        )

    @staticmethod
    def _matches_current_transfer_completed(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, TransferCompletedData)
            and instance._current_call_id == data.call_id
            and instance._current_transfer_id == data.transfer_id
            and instance._current_transfer_target == data.target
        )

    @staticmethod
    def _matches_current_transfer_failed(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, TransferFailedData)
            and instance._current_call_id == data.call_id
            and instance._current_transfer_id == data.transfer_id
            and instance._current_transfer_target == data.target
        )

    @staticmethod
    def _answer_operation_timeout(
        ctx: hsm.Context,
        instance: "PhoneFirmware",
        event: hsm.Event,
    ) -> datetime.timedelta:
        del ctx, event
        return instance._answer_timeout

    @staticmethod
    def _transfer_operation_timeout(
        ctx: hsm.Context,
        instance: "PhoneFirmware",
        event: hsm.Event,
    ) -> datetime.timedelta:
        del ctx, event
        return instance._transfer_timeout

    @staticmethod
    def _publish_answer_requested(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, AnswerCallData)
        PhoneFirmware._publish(ctx, instance, event, ServiceAnswerRequestedEvent.with_data(data))

    @staticmethod
    def _publish_dial_requested(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, DialData)
        PhoneFirmware._publish(ctx, instance, event, ServiceDialRequestedEvent.with_data(data))

    @staticmethod
    def _publish_decline_requested(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, DeclineCallData)
        PhoneFirmware._publish(ctx, instance, event, ServiceDeclineRequestedEvent.with_data(data))

    @staticmethod
    def _publish_hang_up_requested(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, HangUpCallData)
        PhoneFirmware._publish(ctx, instance, event, ServiceHangUpRequestedEvent.with_data(data))

    @staticmethod
    def _publish_transfer_requested(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, TransferCallData)
        PhoneFirmware._publish(ctx, instance, event, ServiceTransferRequestedEvent.with_data(data))

    @staticmethod
    def _publish_ringing(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, CallIdData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _RingingCommittedEvent.with_data(PhoneCallData(call_id=data.call_id)),
        )

    @staticmethod
    def _publish_answered(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, CallIdData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _AnsweredCommittedEvent.with_data(PhoneCallData(call_id=data.call_id)),
        )

    @staticmethod
    def _publish_media_ready(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, CallIdData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _MediaReadyCommittedEvent.with_data(PhoneCallData(call_id=data.call_id)),
        )

    @staticmethod
    def _publish_speaker_audio(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, ServiceAudioData)
        output = audio.AudioOutputData(
            audio=data.audio,
            media_type=data.media_type,
            sample_rate_hz=data.sample_rate_hz,
            channels=data.channels,
        )
        PhoneFirmware._publish(ctx, instance, event, audio.OutputEvent.with_data(output))

    @staticmethod
    def _publish_declined(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, CallIdData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _HungUpCommittedEvent.with_data(PhoneHungUpData(call_id=data.call_id, outcome="declined")),
        )

    @staticmethod
    def _publish_local_hang_up(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, CallIdData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _HungUpCommittedEvent.with_data(PhoneHungUpData(call_id=data.call_id, outcome="local_hang_up")),
        )

    @staticmethod
    def _publish_remote_hang_up(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, CallIdData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _HungUpCommittedEvent.with_data(PhoneHungUpData(call_id=data.call_id, outcome="remote_hang_up")),
        )

    @staticmethod
    def _publish_failed_hang_up(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, CallIdData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _HungUpCommittedEvent.with_data(PhoneHungUpData(call_id=data.call_id, outcome="failed")),
        )

    @staticmethod
    def _publish_answer_timeout(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        call_id = instance._current_call_id
        assert call_id is not None
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _HungUpCommittedEvent.with_data(PhoneHungUpData(call_id=call_id, outcome="failed")),
        )

    @staticmethod
    def _publish_transferred_hang_up(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, CallIdData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _HungUpCommittedEvent.with_data(PhoneHungUpData(call_id=data.call_id, outcome="transferred")),
        )

    @staticmethod
    def _publish_transfer_started(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, TransferCallData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _TransferStartedCommittedEvent.with_data(
                PhoneTransferData(call_id=data.call_id, transfer_id=data.transfer_id, target=data.target)
            ),
        )

    @staticmethod
    def _publish_transfer_completed(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, TransferCompletedData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _TransferCompletedCommittedEvent.with_data(
                PhoneTransferData(call_id=data.call_id, transfer_id=data.transfer_id, target=data.target)
            ),
        )

    @staticmethod
    def _publish_transfer_failed(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, TransferFailedData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _TransferFailedCommittedEvent.with_data(
                PhoneTransferFailedData(
                    call_id=data.call_id,
                    transfer_id=data.transfer_id,
                    target=data.target,
                    failure_kind=data.failure_kind,
                )
            ),
        )

    @staticmethod
    def _publish_transfer_timeout(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        call_id = instance._current_call_id
        transfer_id = instance._current_transfer_id
        target = instance._current_transfer_target
        assert call_id is not None
        assert transfer_id is not None
        assert target is not None
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _TransferFailedCommittedEvent.with_data(
                PhoneTransferFailedData(
                    call_id=call_id,
                    transfer_id=transfer_id,
                    target=target,
                    failure_kind="timeout",
                )
            ),
        )

    @staticmethod
    def _publish_committed_ringing(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, PhoneCallData)
        PhoneFirmware._publish(ctx, instance, event, RingingEvent.with_data(data))

    @staticmethod
    def _publish_committed_answered(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, PhoneCallData)
        PhoneFirmware._publish(ctx, instance, event, AnsweredEvent.with_data(data))

    @staticmethod
    def _publish_committed_media_ready(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, PhoneCallData)
        PhoneFirmware._publish(ctx, instance, event, MediaReadyEvent.with_data(data))

    @staticmethod
    def _publish_committed_hung_up(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, PhoneHungUpData)
        PhoneFirmware._publish(ctx, instance, event, HungUpEvent.with_data(data))

    @staticmethod
    def _publish_committed_transfer_started(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, PhoneTransferData)
        PhoneFirmware._publish(ctx, instance, event, TransferStartedEvent.with_data(data))

    @staticmethod
    def _publish_committed_transfer_completed(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, PhoneTransferData)
        PhoneFirmware._publish(ctx, instance, event, CallTransferCompletedEvent.with_data(data))

    @staticmethod
    def _publish_committed_transfer_failed(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, PhoneTransferFailedData)
        PhoneFirmware._publish(ctx, instance, event, CallTransferFailedEvent.with_data(data))

    @staticmethod
    def _set_current_call(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        del ctx
        data = event.data
        assert isinstance(data, CallIdData)
        instance._current_call_id = data.call_id

    @staticmethod
    def _set_current_transfer_target(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        del ctx
        data = event.data
        assert isinstance(data, TransferCallData)
        instance._current_transfer_id = data.transfer_id
        instance._current_transfer_target = data.target

    @staticmethod
    def _clear_current_call(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        del ctx, event
        instance._current_call_id = None

    @staticmethod
    def _clear_current_transfer_target(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        del ctx, event
        instance._current_transfer_id = None
        instance._current_transfer_target = None

    @staticmethod
    def _remember_closed_call(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        del ctx
        data = event.data
        assert isinstance(data, CallIdData)
        instance._closed_call_ids = frozenset((*instance._closed_call_ids, data.call_id))

    @staticmethod
    def _remember_current_call_closed(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        del ctx, event
        call_id = instance._current_call_id
        assert call_id is not None
        instance._closed_call_ids = frozenset((*instance._closed_call_ids, call_id))

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "Phone",
        hsm.initial(hsm.target("/Phone/hung_up")),
        hsm.state(
            "hung_up",
            hsm.transition(hsm.on(_HungUpCommittedEvent), hsm.effect(_publish_committed_hung_up)),
            hsm.transition(
                hsm.on(_TransferCompletedCommittedEvent),
                hsm.effect(_publish_committed_transfer_completed),
            ),
            hsm.transition(
                hsm.on(IncomingCallEvent),
                hsm.guard(_is_new_incoming_call),
                hsm.effect(_set_current_call),
                hsm.target("/Phone/ringing"),
            ),
            hsm.transition(
                hsm.on(DialEvent),
                hsm.guard(_is_new_dial),
                hsm.effect(_set_current_call, _publish_dial_requested),
                hsm.target("/Phone/dialing"),
            ),
        ),
        hsm.state(
            "dialing",
            hsm.transition(
                hsm.on(CallConnectedEvent),
                hsm.guard(_matches_current_call_connected),
                hsm.effect(_publish_answered),
                hsm.target("/Phone/answered"),
            ),
            hsm.transition(
                hsm.on(HangUpCallEvent),
                hsm.guard(_matches_current_hang_up_command),
                hsm.effect(
                    _publish_hang_up_requested,
                    _publish_local_hang_up,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(CallFailedEvent),
                hsm.guard(_matches_current_call_failed),
                hsm.effect(
                    _publish_failed_hang_up,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(RemoteHangUpEvent),
                hsm.guard(_matches_current_remote_hang_up),
                hsm.effect(
                    _publish_remote_hang_up,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.after(_answer_operation_timeout),
                hsm.effect(
                    _publish_answer_timeout,
                    _remember_current_call_closed,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
        ),
        hsm.state(
            "ringing",
            hsm.entry(_publish_ringing),
            hsm.transition(hsm.on(_RingingCommittedEvent), hsm.effect(_publish_committed_ringing)),
            hsm.transition(
                hsm.on(IncomingCallEvent),
                hsm.guard(_matches_current_incoming_call),
            ),
            hsm.transition(
                hsm.on(AnswerCallEvent),
                hsm.guard(_matches_current_answer_command),
                hsm.effect(_publish_answer_requested),
                hsm.target("/Phone/answering"),
            ),
            hsm.transition(
                hsm.on(DeclineCallEvent),
                hsm.guard(_matches_current_decline_command),
                hsm.effect(
                    _publish_decline_requested,
                    _publish_declined,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(CallFailedEvent),
                hsm.guard(_matches_current_call_failed),
                hsm.effect(
                    _publish_failed_hang_up,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(RemoteHangUpEvent),
                hsm.guard(_matches_current_remote_hang_up),
                hsm.effect(
                    _publish_remote_hang_up,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
        ),
        hsm.state(
            "answering",
            hsm.transition(
                hsm.on(IncomingCallEvent),
                hsm.guard(_matches_current_incoming_call),
            ),
            hsm.transition(
                hsm.on(CallConnectedEvent),
                hsm.guard(_matches_current_call_connected),
                hsm.effect(_publish_answered),
                hsm.target("/Phone/answered"),
            ),
            hsm.transition(
                hsm.on(DeclineCallEvent),
                hsm.guard(_matches_current_decline_command),
                hsm.effect(
                    _publish_decline_requested,
                    _publish_declined,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(HangUpCallEvent),
                hsm.guard(_matches_current_hang_up_command),
                hsm.effect(
                    _publish_hang_up_requested,
                    _publish_local_hang_up,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(CallFailedEvent),
                hsm.guard(_matches_current_call_failed),
                hsm.effect(
                    _publish_failed_hang_up,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(RemoteHangUpEvent),
                hsm.guard(_matches_current_remote_hang_up),
                hsm.effect(
                    _publish_remote_hang_up,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.after(_answer_operation_timeout),
                hsm.effect(
                    _publish_answer_timeout,
                    _remember_current_call_closed,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
        ),
        hsm.state(
            "answered",
            hsm.initial(hsm.target("/Phone/answered/media_connecting")),
            hsm.transition(hsm.on(_AnsweredCommittedEvent), hsm.effect(_publish_committed_answered)),
            hsm.transition(hsm.on(_MediaReadyCommittedEvent), hsm.effect(_publish_committed_media_ready)),
            hsm.transition(
                hsm.on(_TransferFailedCommittedEvent),
                hsm.effect(_publish_committed_transfer_failed),
            ),
            hsm.transition(
                hsm.on(IncomingCallEvent),
                hsm.guard(_matches_current_incoming_call),
            ),
            hsm.transition(
                hsm.on(HangUpCallEvent),
                hsm.guard(_matches_current_hang_up_command),
                hsm.effect(
                    _publish_hang_up_requested,
                    _publish_local_hang_up,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(CallFailedEvent),
                hsm.guard(_matches_current_call_failed),
                hsm.effect(
                    _publish_failed_hang_up,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(RemoteHangUpEvent),
                hsm.guard(_matches_current_remote_hang_up),
                hsm.effect(
                    _publish_remote_hang_up,
                    _remember_closed_call,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.state(
                "media_connecting",
                hsm.transition(
                    hsm.on(ServiceMediaReadyEvent),
                    hsm.guard(_matches_current_media_ready),
                    hsm.effect(_publish_media_ready),
                    hsm.target("/Phone/answered/media_ready"),
                ),
            ),
            hsm.state(
                "media_ready",
                hsm.transition(
                    hsm.on(ServiceMediaReadyEvent),
                    hsm.guard(_matches_current_media_ready),
                ),
                hsm.transition(
                    hsm.on(ServiceAudioReceivedEvent),
                    hsm.guard(_matches_current_service_audio),
                    hsm.effect(_publish_speaker_audio),
                ),
                hsm.transition(
                    hsm.on(TransferCallEvent),
                    hsm.guard(_matches_current_transfer_command),
                    hsm.effect(
                        _publish_transfer_requested,
                        _set_current_transfer_target,
                        _publish_transfer_started,
                    ),
                    hsm.target("/Phone/transferring"),
                ),
            ),
        ),
        hsm.state(
            "transferring",
            hsm.transition(
                hsm.on(_TransferStartedCommittedEvent),
                hsm.effect(_publish_committed_transfer_started),
            ),
            hsm.transition(
                hsm.on(IncomingCallEvent),
                hsm.guard(_matches_current_incoming_call),
            ),
            hsm.transition(
                hsm.on(ServiceTransferFailedEvent),
                hsm.guard(_matches_current_transfer_failed),
                hsm.effect(_publish_transfer_failed, _clear_current_transfer_target),
                hsm.target("/Phone/answered/media_ready"),
            ),
            hsm.transition(
                hsm.on(TransferAcceptedEvent),
                hsm.guard(_matches_current_transfer_accepted),
            ),
            hsm.transition(
                hsm.on(ServiceAudioReceivedEvent),
                hsm.guard(_matches_current_service_audio),
                hsm.effect(_publish_speaker_audio),
            ),
            hsm.transition(
                hsm.on(ServiceTransferCompletedEvent),
                hsm.guard(_matches_current_transfer_completed),
                hsm.effect(
                    _publish_transferred_hang_up,
                    _publish_transfer_completed,
                    _remember_closed_call,
                    _clear_current_transfer_target,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(HangUpCallEvent),
                hsm.guard(_matches_current_hang_up_command),
                hsm.effect(
                    _publish_hang_up_requested,
                    _publish_local_hang_up,
                    _remember_closed_call,
                    _clear_current_transfer_target,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(CallFailedEvent),
                hsm.guard(_matches_current_call_failed),
                hsm.effect(
                    _publish_failed_hang_up,
                    _remember_closed_call,
                    _clear_current_transfer_target,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(RemoteHangUpEvent),
                hsm.guard(_matches_current_remote_hang_up),
                hsm.effect(
                    _publish_remote_hang_up,
                    _remember_closed_call,
                    _clear_current_transfer_target,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.after(_transfer_operation_timeout),
                hsm.effect(_publish_transfer_timeout, _clear_current_transfer_target),
                hsm.target("/Phone/answered/media_ready"),
            ),
        ),
        hsm.observe(observer),
    )


class Phone(bot.device.Device):
    """Phone device with privately owned audio peripherals and event-driven call control."""

    _microphone: audio.Microphone
    _speaker: audio.Speaker
    _service: PhoneService
    _firmware_instance: PhoneFirmware
    firmware_model: typing.ClassVar[hsm.Model] = PhoneFirmware.model

    def __init__(
        self,
        bots: collections.abc.Iterable[hsm.Instance] = (),
        *,
        microphone: audio.Microphone | None = None,
        speaker: audio.Speaker | None = None,
        peripherals: collections.abc.Iterable[bot.device.Device] = (),
        service: PhoneService | None = None,
        answer_timeout: datetime.timedelta = _DEFAULT_ANSWER_TIMEOUT,
        transfer_timeout: datetime.timedelta = _DEFAULT_TRANSFER_TIMEOUT,
    ) -> None:
        resolved_microphone = microphone if microphone is not None else audio.Microphone()
        resolved_speaker = speaker if speaker is not None else audio.Speaker()
        super().__init__(bots=bots, peripherals=(resolved_microphone, resolved_speaker, *tuple(peripherals)))
        self._microphone = resolved_microphone
        self._speaker = resolved_speaker
        observation_service = _PhoneObservationService(
            owner=self,
            service=service if service is not None else PhoneEventRecorder(),
            speaker=resolved_speaker,
        )
        self._service = observation_service
        self._firmware_instance = PhoneFirmware(
            service=observation_service,
            answer_timeout=answer_timeout,
            transfer_timeout=transfer_timeout,
        )

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name in self.model.events:
            return super().dispatch(ctx, event)
        if event.name in _PHONE_SERVICE_ORIGINATING_EVENT_NAMES:
            return _completed_phone_service_event()
        command_event = _PHONE_OWNER_COMMAND_EVENTS_BY_NAME.get(event.name)
        if command_event is not None and self._firmware is not None:
            try:
                data = validate_event_data(command_event, event.data or {})
            except ValueError:
                return _completed_phone_service_event()
            return self._firmware.dispatch(
                ctx,
                dataclasses.replace(event, data=data, kind=command_event.kind, schema=command_event.schema),
            )
        return super().dispatch(ctx, event)

    @typing.override
    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        await super()._after_firmware_started(ctx, event)
        if self._firmware is None:
            return
        # Service/media outlive firmware-init activity; attach under device lifetime context (HSM-CONTEXT-001).
        await self._service.attach(World.from_context(self.context()), self._firmware)

    @typing.override
    def _create_firmware_instance(self, ctx: hsm.Context, event: hsm.Event) -> hsm.Instance:
        del ctx, event
        return self._firmware_instance

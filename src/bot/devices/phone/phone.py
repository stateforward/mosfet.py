from bot.devices import audio

import asyncio
import collections.abc
import dataclasses
import datetime
import importlib.resources
import logging
import typing

import hsm
import bot.device

from bot import lifecycle
from bot.event import validate_event_data
from bot.protocols import attachment
from bot.telemetry import observer
from bot.environment import SoundEvent, Environment, require_environment_scope, space

# Flat symbol imports (not `from . import display`): both Phone and PhoneFirmware accept a
# `display` constructor parameter, which would shadow a `display` module import.
from .display import CallerIdData, CallerIdEvent, Display
from .events import (
    AnswerCallData,
    AnswerCallEvent,
    AnswerRequestData,
    AnsweredEvent,
    Caller,
    CallConnectedData,
    CallConnectedEvent,
    CallFailedData,
    CallFailedEvent,
    CallIdData,
    CallTransferCompletedEvent,
    CallTransferFailedEvent,
    DeclineCallData,
    DeclineCallEvent,
    DeclineRequestData,
    DialData,
    DialEvent,
    DialFailedData,
    HangUpCallData,
    HangUpCallEvent,
    HangUpRequestData,
    HungUpEvent,
    IncomingCallData,
    IncomingCallEvent,
    MediaReadyData,
    MediaReadyEvent,
    NoCallData,
    NoCallEvent,
    PhoneCallData,
    PhoneHungUpData,
    PhoneSoundData,
    PhoneTransferData,
    PhoneTransferFailedData,
    RemoteHangUpData,
    RemoteHangUpEvent,
    RingingData,
    RingingEvent,
    ServiceAnswerRequestedEvent,
    ServiceAudioData,
    ServiceAudioReceivedEvent,
    ServiceDeclineRequestedEvent,
    ServiceDialFailedEvent,
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
    TransferRequestData,
    TransferStartedEvent,
    TransferTarget,
)

RINGER_DB = 80.0
"""How loud a handset ringer is, in dB SPL measured one metre away.

Device-intrinsic like a speaker's output level: it is what this hardware does, while how far the
ring carries is the environment's to work out from where the phone is.

Known interaction, deliberately not engineered around: at this level a ring clears a close-talk
mouthpiece threshold from 15 cm away (about 96 dB against 70), so a ringing phone would carry its
own ring up the wire. It cannot today, because a ringing phone is not a connected one and the
uplink transition only exists in ``/Phone/answered/media_ready``. If call-waiting is ever modelled
— a second call ringing while the first is connected — this is the interaction to handle, and it
is known rather than overlooked.
"""

CALL_PROGRESS_DB = 60.0
"""How loud a call-progress tone is out of the earpiece, in dB SPL measured one metre away.

Device-intrinsic like :data:`RINGER_DB`, and deliberately 20 dB below it, because the two are
different transducers doing different jobs. A ringer is for the room — it has to fetch someone who
is not holding the handset. A call-progress tone is for the one person already holding it, so the
earpiece only has to reach an ear 15 cm away. At that distance this lands near 76 dB SPL, which is
about what a handset receiver does with a tone, and across a room it is nearly nothing — which is
right: nobody but the caller hears their own busy tone.
"""

MOUTH_OFFSET_M = 0.15
"""Distance from a handset's earpiece to its mouthpiece, in metres.

Device-intrinsic: it is how a handset is shaped, and it holds wherever the handset is held. Where
the handset *is* stays with the wiring, which composes this offset onto whatever the robot's own
position is — a device cannot know that.
"""

_DEFAULT_ANSWER_TIMEOUT = datetime.timedelta(seconds=30)
_DEFAULT_TRANSFER_TIMEOUT = datetime.timedelta(seconds=30)

_LOG = logging.getLogger(__name__)


def _load_sound_wav(name: str) -> bytes:
    """Load one package-local acoustic clip for environment.sound elevation."""

    return (importlib.resources.files(__package__) / "assets" / name).read_bytes()


# Real acoustic WAVs (see assets/SOURCES.md). Sensory classifiers match these acoustic payloads;
# Listening does not special-case phone.
RING_SOUND_WAV = _load_sound_wav("ring.wav")
BUSY_TONE_WAV = _load_sound_wav("busy.wav")
"""Busy tone: what the exchange puts in your ear when the line you dialled refused or is engaged."""

REORDER_TONE_WAV = _load_sound_wav("reorder.wav")
"""Reorder tone (fast busy): what the network sends when it could not complete the call at all."""

_RingingCommittedEvent = hsm.Event[RingingData](
    name="bot.phone.ringing.committed",
    kind=hsm.CompletionEventKind,
    schema=RingingData,
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
    # Injected by the phone that owns this service: elevating a committed observation into an environment
    # stimulus is the phone's behaviour, not the transport's, and the phone is what knows where it
    # is standing.
    elevate: collections.abc.Callable[[hsm.Context, hsm.Event[typing.Any]], None]
    target: hsm.Instance | None = dataclasses.field(default=None, init=False)

    async def attach(self, environment: Environment, target: hsm.Instance) -> None:
        await self.service.attach(environment, target)
        self.target = target

    async def detach(self, environment: Environment, target: hsm.Instance) -> None:
        await self.service.detach(environment, target)
        if self.target is target:
            self.target = None

    def is_attached(self) -> bool:
        """True when this service still holds its attach target.

        Routing audio to peripherals is firmware work; the service only reports whether the
        phone is attached, never which transducer a payload belongs to.
        """

        return self.target is not None

    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        # Transport only: forward to the provider service. Direction is decided by which firmware
        # transition fired, never by sniffing the payload type here — a shared channel that infers
        # direction from `type(data)` is what let receiver audio take the uplink path (echo).
        service_id = hsm.id(self.service) if isinstance(self.service, hsm.Instance) else ""
        self.service.publish(
            ctx,
            dataclasses.replace(
                event,
                source=hsm.id(self.target) if self.target is not None else hsm.id(self.owner),
                target=service_id,
                metadata=dict(event.metadata),
            ),
        )
        self.elevate(ctx, event)


class PhoneService(typing.Protocol):
    """Service that receives provider requests and committed public phone events."""

    def attach(self, environment: Environment, target: hsm.Instance) -> collections.abc.Awaitable[None]:
        """Attach this phone service to a phone-owned firmware instance."""
        ...

    def detach(self, environment: Environment, target: hsm.Instance) -> collections.abc.Awaitable[None]:
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

    async def attach(self, environment: Environment, target: hsm.Instance) -> None:
        require_environment_scope(environment, target, participant="Phone service target")
        self._target = target

    async def detach(self, environment: Environment, target: hsm.Instance) -> None:
        require_environment_scope(environment, target, participant="Phone service target")
        if self._target is target:
            self._target = None

    def receive(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[None]:
        """Emit one provider-originating event into the attached phone firmware."""

        if self._target is None:
            return _completed_phone_service_event()
        # Delivery is the gate; firmware topology ignores unmatched triggers.
        return self._target.dispatch(ctx, event)

    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        # Ignore local speaker uplink offers only (exact AudioOutputData; ServiceAudioData records).
        if type(event.data) is audio.AudioOutputData:
            return
        self._events.append(event)


def _require_positive_timeout(name: str, value: datetime.timedelta) -> None:
    if value <= datetime.timedelta():
        raise ValueError(f"{name} must be a positive duration.")


def _environment_observation_event(owner: "Phone", event: hsm.Event[typing.Any]) -> hsm.Event[typing.Any] | None:
    """Map phone observations that bots experience as input energy into environment stimuli.

    Ringing is heard as ``environment.sound`` with ``source`` = the phone instance id. Device-plane
    ``phone.ringing`` still flows on the service/firmware path; bots do not receive it as a
    raw cognitive stimulus. Ring elevation selects on :class:`RingingData` payload type.
    What carries across is the caller — who is calling, which a ringing handset shows — not the
    session handle, which has no counterpart on a handset and nothing outside firmware can use.
    ``kind`` is the explicit provenance label ``phone.ringing`` (not bare ``ring``, which is
    ambiguous for models).

    ``None`` means this observation is inaudible and there is nothing to put in the environment.
    Silence is a real answer here, not a gap: several ways of ending up with no call are ones a
    real handset marks with no sound whatsoever, and manufacturing a tone for them would be
    putting a noise in the earpiece that no telephone has ever made.

    Only sound goes here. A call connecting, ending, or being transferred makes no noise and has
    no visual body, so it has no representable form in an environment that carries acoustics and
    optics — and it is nobody else's business either: broadcasting it would tell every citizen in
    the room about a call they cannot perceive. Those facts reach the bot holding this phone
    through :meth:`Device._report` instead, which is the handset in its hand rather than the room.
    """

    data = event.data
    if isinstance(data, RingingData):
        return dataclasses.replace(
            SoundEvent.with_data(
                PhoneSoundData(
                    audio=RING_SOUND_WAV,
                    media_type="audio/wav",
                    sample_rate_hz=16_000,
                    channels=1,
                    kind="phone.ringing",
                    caller=data.caller,
                    amplitude_db=RINGER_DB,
                )
            ),
            source=hsm.id(owner),
            metadata=dict(event.metadata),
        )
    if isinstance(data, NoCallData):
        # A call-progress tone, when there is one. Which tone is the exchange's verdict carried
        # down the line, not something the handset works out — the handset only reproduces it.
        if data.reason != "dial_failed":
            # nothing_to_answer: pressing answer with nothing ringing does essentially nothing
            # audible on a real handset. dial_abandoned: you hung up, so you hear nothing, because
            # you hung up. dial_not_answered: there is no "they did not answer" tone and never was
            # — a real caller hears ringback the whole time and then gives up, so what marks this
            # case is ringback *stopping*, which is a sound the phone has yet to be able to make.
            return None
        # call_declined is the one failure a caller can name by ear: somebody refused, or the line
        # is engaged. Busy tone. Every other failure kind is the network saying it could not
        # complete the call, which is reorder — and a caller genuinely cannot tell "no route"
        # from "congestion" from "the switch broke" apart either, so collapsing them is what the
        # real object does rather than a shortcut around it.
        busy = data.failure_kind == "call_declined"
        return dataclasses.replace(
            SoundEvent.with_data(
                PhoneSoundData(
                    audio=BUSY_TONE_WAV if busy else REORDER_TONE_WAV,
                    media_type="audio/wav",
                    sample_rate_hz=16_000,
                    channels=1,
                    kind="phone.busy" if busy else "phone.reorder",
                    amplitude_db=CALL_PROGRESS_DB,
                )
            ),
            source=hsm.id(owner),
            metadata=dict(event.metadata),
        )
    return None


class PhoneFirmware(hsm.Instance):
    """Phone-owned firmware state for a single active call."""

    _service: PhoneService
    # Firmware owns transducer routing: the speaker transmits service audio into the environment
    # (receiver), the microphone carries local speech to the service (mouthpiece), the display
    # shows who the call is with. The service knows about none of them.
    _speaker: audio.Speaker
    _microphone: audio.Microphone
    _display: Display
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
        speaker: audio.Speaker | None = None,
        microphone: audio.Microphone | None = None,
        display: Display | None = None,
        answer_timeout: datetime.timedelta = _DEFAULT_ANSWER_TIMEOUT,
        transfer_timeout: datetime.timedelta = _DEFAULT_TRANSFER_TIMEOUT,
    ) -> None:
        super().__init__()
        _require_positive_timeout("answer_timeout", answer_timeout)
        _require_positive_timeout("transfer_timeout", transfer_timeout)
        self._service = service if service is not None else PhoneEventRecorder()
        self._speaker = speaker if speaker is not None else audio.Speaker()
        self._microphone = microphone if microphone is not None else audio.Microphone()
        self._display = display if display is not None else Display()
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
        # Preserve request envelope id end-to-end for HSM completion correlation; metadata is telemetry only.
        instance._service.publish(
            ctx,
            dataclasses.replace(
                event,
                id=trigger.id if trigger.id else event.id,
                metadata=dict(trigger.metadata),
            ),
        )

    @staticmethod
    def _queue_committed(
        ctx: hsm.Context,
        instance: "PhoneFirmware",
        trigger: hsm.Event,
        event: hsm.Event[typing.Any],
    ) -> None:
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                event,
                id=trigger.id if trigger.id else event.id,
                metadata=dict(trigger.metadata),
            ),
        )

    @staticmethod
    def _dispatch_to_peripheral(
        ctx: hsm.Context,
        instance: "PhoneFirmware",
        peripheral: hsm.Instance,
        event: hsm.Event[typing.Any],
        trigger: hsm.Event,
    ) -> None:
        """Deliver one typed event to a firmware-driven peripheral, envelope stamped from ``trigger``.

        Firmware is the controller for every transducer it drives — the speaker, the microphone's
        uplink is published rather than dispatched, the display — so this is the one place that
        stamps source/target/metadata for that outbound wire, rather than repeating the same
        ``dataclasses.replace`` at each call site.
        """

        _ = peripheral.dispatch(
            ctx,
            dataclasses.replace(
                event,
                # Correlation only, and only when there is an id to correlate with: the live
                # transition already requires both machines started, so an unstarted one here
                # means a direct call, not a delivery decision.
                source=hsm.id(instance) if lifecycle.is_started(instance) else "",
                target=hsm.id(peripheral) if lifecycle.is_started(peripheral) else "",
                metadata=dict(trigger.metadata),
            ),
        )

    @staticmethod
    def _is_new_incoming_call(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, IncomingCallData) and data.call_id not in instance._closed_call_ids

    @staticmethod
    def _is_live_dial_observation(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        """A service observation about the dial attempt in progress.

        A dialing handset has no call id to match on — the exchange has not assigned one yet, and
        it arrives with the connect. Only one attempt is ever outstanding, so any call-scoped
        observation that is not left over from an already-closed call is about that attempt.
        """

        del ctx
        data = event.data
        return isinstance(data, CallIdData) and data.call_id not in instance._closed_call_ids

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
        if isinstance(service, _PhoneObservationService) and not service.is_attached():
            return False
        # Elevation stamps source=hsm.id(speaker) on environment.sound, so the speaker must be started
        # in the same environment Instances map as ctx. Liveness and scope only — never state().
        if not lifecycle.is_started(instance._speaker):
            return False
        speaker_context = instance._speaker.context()
        return speaker_context.value(hsm.Keys.Instances) is Environment.from_context(ctx).value(hsm.Keys.Instances)

    @staticmethod
    def _matches_current_remote_hang_up(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> bool:
        del ctx
        data = event.data
        return isinstance(data, RemoteHangUpData) and instance._current_call_id == data.call_id

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
    def _current_call(instance: "PhoneFirmware") -> str:
        """The call firmware is on. Only reachable from states that have one."""

        call_id = instance._current_call_id
        assert call_id is not None
        return call_id

    @staticmethod
    def _publish_answer_requested(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        # The command names no call; firmware stamps the line it is answering for the exchange.
        PhoneFirmware._publish(
            ctx,
            instance,
            event,
            ServiceAnswerRequestedEvent.with_data(AnswerRequestData(call_id=PhoneFirmware._current_call(instance))),
        )

    @staticmethod
    def _publish_dial_requested(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, DialData)
        PhoneFirmware._publish(ctx, instance, event, ServiceDialRequestedEvent.with_data(data))

    @staticmethod
    def _publish_decline_requested(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        PhoneFirmware._publish(
            ctx,
            instance,
            event,
            ServiceDeclineRequestedEvent.with_data(DeclineRequestData(call_id=PhoneFirmware._current_call(instance))),
        )

    @staticmethod
    def _publish_hang_up_requested(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        PhoneFirmware._publish(
            ctx,
            instance,
            event,
            ServiceHangUpRequestedEvent.with_data(HangUpRequestData(call_id=PhoneFirmware._current_call(instance))),
        )

    @staticmethod
    def _publish_transfer_requested(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, TransferCallData)
        PhoneFirmware._publish(
            ctx,
            instance,
            event,
            ServiceTransferRequestedEvent.with_data(
                TransferRequestData(
                    call_id=PhoneFirmware._current_call(instance),
                    transfer_id=data.transfer_id,
                    target=data.target,
                )
            ),
        )

    @staticmethod
    def _publish_ringing(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        data = event.data
        assert isinstance(data, IncomingCallData)
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _RingingCommittedEvent.with_data(RingingData(caller=data.caller)),
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
    def _send_microphone_audio(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        """Mouthpiece: carry locally captured audio up the wire.

        Only reachable from the answered/media_ready state, so the microphone is live exactly
        while a call is connected and environment sound is ignored otherwise. This is the only path
        to the service; the speaker never reaches it.
        """

        data = event.data
        assert isinstance(data, audio.AudioInputData)
        uplink = audio.AudioOutputData(
            audio=data.audio,
            media_type=data.media_type,
            sample_rate_hz=data.sample_rate_hz,
            channels=data.channels,
        )
        PhoneFirmware._publish(ctx, instance, event, audio.OutputEvent.with_data(uplink))

    @staticmethod
    def _receive_service_audio(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        """Receiver: transmit far-end audio out of the phone speaker into the environment.

        ``data`` is passed through as the ``ServiceAudioData`` it already is. Flattening it to
        ``AudioOutputData`` used to erase where it came from, which is how receiver audio ended
        up back on the wire.

        No call *here* reaches the service — but the loop this closes does. Far-end audio put
        into the environment is heard by this phone's own microphone, which carries it back up the
        wire as uplink. That echo is an audibility problem, out of scope for the transducer
        ownership work and tracked for Change B; do not read this docstring as saying the
        receiver path cannot reach the service.
        """

        data = event.data
        assert isinstance(data, ServiceAudioData)
        PhoneFirmware._dispatch_to_peripheral(
            ctx, instance, instance._speaker, audio.OutputEvent.with_data(data), event
        )

    @staticmethod
    def _publish_declined(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _HungUpCommittedEvent.with_data(
                PhoneHungUpData(call_id=PhoneFirmware._current_call(instance), outcome="declined")
            ),
        )

    @staticmethod
    def _publish_local_hang_up(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        PhoneFirmware._queue_committed(
            ctx,
            instance,
            event,
            _HungUpCommittedEvent.with_data(
                PhoneHungUpData(call_id=PhoneFirmware._current_call(instance), outcome="local_hang_up")
            ),
        )

    @staticmethod
    def _publish_nothing_to_answer(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        PhoneFirmware._publish(ctx, instance, event, NoCallEvent.with_data(NoCallData(reason="nothing_to_answer")))

    @staticmethod
    def _publish_dial_not_answered(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        PhoneFirmware._publish(ctx, instance, event, NoCallEvent.with_data(NoCallData(reason="dial_not_answered")))

    @staticmethod
    def _publish_dial_abandoned(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        PhoneFirmware._publish(ctx, instance, event, NoCallEvent.with_data(NoCallData(reason="dial_abandoned")))

    @staticmethod
    def _publish_dial_failed(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        # The service's verdict travels with the fact it explains. Firmware does not get to
        # decide why a dial failed and cannot reconstruct it later, so dropping it here is what
        # left the phone unable to say whether the line refused or the network gave up.
        data = event.data
        assert isinstance(data, DialFailedData)
        PhoneFirmware._publish(
            ctx,
            instance,
            event,
            NoCallEvent.with_data(NoCallData(reason="dial_failed", failure_kind=data.failure_kind)),
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
                PhoneTransferData(
                    call_id=PhoneFirmware._current_call(instance),
                    transfer_id=data.transfer_id,
                    target=data.target,
                )
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
        assert isinstance(data, RingingData)
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
    def _drive_display(
        ctx: hsm.Context,
        instance: "PhoneFirmware",
        trigger: hsm.Event,
        caller_id: Caller | None,
    ) -> None:
        """Put a caller id (or clear one) on the display; log rather than silently drop when it is not live.

        Firmware is the controller that wires and drives its own display, dispatching a typed
        event rather than reaching into the display's attribute directly. The production path
        (``Phone``) starts every peripheral before firmware exists, so a live display is the
        supported contract. Firmware built and driven standalone with no started display — e.g.
        constructing ``PhoneFirmware()`` directly, outside a ``Phone`` — has nowhere to put a
        caller id; this makes that visible instead of dispatching to a peripheral that can never
        receive it.
        """

        if not lifecycle.is_started(instance._display):
            _LOG.warning("phone firmware display is not started; caller id drive dropped")
            return
        PhoneFirmware._dispatch_to_peripheral(
            ctx, instance, instance._display, CallerIdEvent.with_data(CallerIdData(caller_id=caller_id)), trigger
        )

    @staticmethod
    def _show_caller_id(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        """Put a caller ID on the display: the caller on a ring, the provider-stamped party on a connect.

        Set from what the payload itself declares — never looked up anywhere. A connect that names
        nobody leaves the display alone: the ring may already have shown the caller, and only
        entering hung_up clears the screen, so a caller-ID-only provider never regresses shown to
        blank.
        """

        data = event.data
        if isinstance(data, IncomingCallData):
            caller = data.caller
        elif isinstance(data, CallConnectedData) and data.party is not None:
            caller = data.party
        else:
            return
        PhoneFirmware._drive_display(ctx, instance, event, caller)

    @staticmethod
    def _clear_caller_id(ctx: hsm.Context, instance: "PhoneFirmware", event: hsm.Event) -> None:
        """A hung-up phone shows nobody on the line; entering hung_up clears the display, including at start."""

        PhoneFirmware._drive_display(ctx, instance, event, None)

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

    model: typing.ClassVar[hsm.Model] = bot.define(
        "Phone",
        hsm.initial(hsm.target("/Phone/hung_up")),
        hsm.state(
            "hung_up",
            hsm.entry(_clear_caller_id),
            hsm.transition(hsm.on(_HungUpCommittedEvent), hsm.effect(_publish_committed_hung_up)),
            hsm.transition(
                hsm.on(_TransferCompletedCommittedEvent),
                hsm.effect(_publish_committed_transfer_completed),
            ),
            hsm.transition(
                hsm.on(IncomingCallEvent),
                hsm.guard(_is_new_incoming_call),
                hsm.effect(_set_current_call, _show_caller_id),
                hsm.target("/Phone/ringing"),
            ),
            # Redial is always legal. A handset does not refuse a number because you just called
            # it, and with no id on the command there is nothing to recognise as a repeat.
            hsm.transition(
                hsm.on(DialEvent),
                hsm.effect(_publish_dial_requested),
                hsm.target("/Phone/dialing"),
            ),
            # Answering with nothing ringing does nothing, but it must not do it silently.
            hsm.transition(
                hsm.on(AnswerCallEvent),
                hsm.effect(_publish_nothing_to_answer),
            ),
        ),
        # No current call here: the exchange assigns one on connect, so every exit that is not a
        # connect ends an attempt that never became a call and reports phone.no_call.
        hsm.state(
            "dialing",
            hsm.transition(
                hsm.on(CallConnectedEvent),
                hsm.guard(_is_live_dial_observation),
                hsm.effect(_set_current_call, _show_caller_id, _publish_answered),
                hsm.target("/Phone/answered"),
            ),
            hsm.transition(
                hsm.on(HangUpCallEvent),
                hsm.effect(_publish_dial_abandoned),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(ServiceDialFailedEvent),
                hsm.effect(_publish_dial_failed),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.after(_answer_operation_timeout),
                hsm.effect(_publish_dial_not_answered),
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
                hsm.effect(_publish_answer_requested),
                hsm.target("/Phone/answering"),
            ),
            hsm.transition(
                hsm.on(DeclineCallEvent),
                hsm.effect(
                    _publish_decline_requested,
                    _publish_declined,
                    _remember_current_call_closed,
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
                hsm.effect(_show_caller_id, _publish_answered),
                hsm.target("/Phone/answered"),
            ),
            hsm.transition(
                hsm.on(DeclineCallEvent),
                hsm.effect(
                    _publish_decline_requested,
                    _publish_declined,
                    _remember_current_call_closed,
                    _clear_current_call,
                ),
                hsm.target("/Phone/hung_up"),
            ),
            hsm.transition(
                hsm.on(HangUpCallEvent),
                hsm.effect(
                    _publish_hang_up_requested,
                    _publish_local_hang_up,
                    _remember_current_call_closed,
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
                hsm.effect(
                    _publish_hang_up_requested,
                    _publish_local_hang_up,
                    _remember_current_call_closed,
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
                # Microphone is live only here: the mouthpiece carries local audio up the wire
                # while a call is connected, and nothing is captured before answer.
                hsm.transition(
                    hsm.on(audio.InputEvent),
                    hsm.effect(_send_microphone_audio),
                ),
                hsm.transition(
                    hsm.on(ServiceMediaReadyEvent),
                    hsm.guard(_matches_current_media_ready),
                ),
                hsm.transition(
                    hsm.on(ServiceAudioReceivedEvent),
                    hsm.guard(_matches_current_service_audio),
                    hsm.effect(_receive_service_audio),
                ),
                hsm.transition(
                    hsm.on(TransferCallEvent),
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
                hsm.effect(_receive_service_audio),
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
                hsm.effect(
                    _publish_hang_up_requested,
                    _publish_local_hang_up,
                    _remember_current_call_closed,
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
    """Phone device with privately owned audio/display peripherals and event-driven call control."""

    _microphone: audio.Microphone
    _speaker: audio.Speaker
    _display: Display
    _service: PhoneService
    _firmware_instance: PhoneFirmware
    firmware_model: typing.ClassVar[hsm.Model] = PhoneFirmware.model

    def __init__(
        self,
        *,
        microphone: audio.Microphone | None = None,
        speaker: audio.Speaker | None = None,
        display: Display | None = None,
        peripherals: collections.abc.Iterable[bot.device.Device] = (),
        placement: space.Placement | None = None,
        service: PhoneService | None = None,
        answer_timeout: datetime.timedelta = _DEFAULT_ANSWER_TIMEOUT,
        transfer_timeout: datetime.timedelta = _DEFAULT_TRANSFER_TIMEOUT,
    ) -> None:
        resolved_microphone = microphone if microphone is not None else audio.Microphone()
        resolved_speaker = speaker if speaker is not None else audio.Speaker()
        resolved_display = display if display is not None else Display()
        # Where the handset is. Its transducers carry their own placements: the earpiece is at the
        # ear and the mouthpiece MOUTH_OFFSET_M away, which is the whole point of them being
        # separate devices. The display shows caller ID rather than producing acoustic energy, so
        # it carries no placement of its own.
        super().__init__(
            peripherals=(resolved_microphone, resolved_speaker, resolved_display, *tuple(peripherals)),
            placement=placement,
        )
        self._microphone = resolved_microphone
        self._speaker = resolved_speaker
        self._display = resolved_display
        observation_service = _PhoneObservationService(
            owner=self,
            service=service if service is not None else PhoneEventRecorder(),
            elevate=self._observe,
        )
        self._service = observation_service
        self._firmware_instance = PhoneFirmware(
            service=observation_service,
            speaker=resolved_speaker,
            microphone=resolved_microphone,
            display=resolved_display,
            answer_timeout=answer_timeout,
            transfer_timeout=transfer_timeout,
        )

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Release the service this phone acquired, then power down.

        Paired with the acquire in ``_after_firmware_started``, and ordered before it: the
        service is attached *to* the firmware, so it has to be let go while that firmware is
        still there — the same reason firmware goes down before the transducers it holds.
        Held past teardown it keeps a stopped machine alive, and a replacement phone can never
        attach to the same service.
        """

        firmware = self._firmware
        if firmware is not None:
            await self._service.detach(Environment.from_context(self.context()), firmware)
        await super().stop(ctx)

    def _observe(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        """Surface a committed observation on both of the paths a real handset has.

        Committed public payloads only — not service-request payloads (MediaReadyData / DialData
        and friends), which must not re-enter the phone shell as environment sound. Committed
        media-ready is PhoneCallData (MediaReadyEvent); MediaReadyData is service-side only.

        Two paths, because a handset genuinely has two. What it makes a *noise* about goes into
        the environment, where anyone standing nearby hears it — a ring, a busy tone, reorder.
        What it merely *becomes* goes to whoever is holding it: the call is up, the other end
        hung up, nobody picked up. That second path is the one this phone had no way to take
        before, which is why a bot could sit through a connected call never having been told a
        call existed.

        Exactly one path per observation, and which one is decided by whether it made a sound.
        A ringing phone is not two events. You do not hear a ring *and* separately feel that
        your phone is ringing — you notice your phone ringing, once, and the ear is how. So a
        sound goes to the room and is perceived by hearing it, and only what the room cannot
        carry travels the nerve. Sending both would buy the bot two turns for one happening.

        Both say what happened and neither says what to do about it.
        """

        if not isinstance(
            event.data,
            RingingData | PhoneCallData | PhoneHungUpData | PhoneTransferData | PhoneTransferFailedData | NoCallData,
        ):
            return
        stimulus = _environment_observation_event(self, event)
        if stimulus is None:
            # Nothing audible happened, which does not mean nothing happened. A call coming up,
            # a line going dead, nobody ever picking up: silent to the room, plain to the hand.
            self._report(ctx, event)
            return
        placement = self._placement
        _ = Environment.from_context(ctx).broadcast(
            stimulus,
            origin=None if placement is None else placement.position,
        )

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        """Firmware-only for owner call-control payloads; shell for lifecycle/other.

        Owner commands enter firmware only — not dual-delivered and not name-routed. Typed
        command payloads pass through; dict/None payloads whose event schema is a command
        model are coerced via ``validate_event_data`` (JSON/API ingress). Device shell
        lifecycle and other events stay on the shell HSM. Service observations enter
        firmware via the service attach target, not through this shell ingress.
        """

        async def _deliver() -> None:
            firmware = self._firmware
            command = Phone._coerce_owner_command(event)
            if command is not None:
                if firmware is not None:
                    await firmware.dispatch(ctx, command)
                return
            # Device shell lifecycle / other events.
            await hsm.Instance.dispatch(self, ctx, event)

        return asyncio.Task(_deliver(), loop=asyncio.get_running_loop(), eager_start=True)

    @staticmethod
    def _coerce_owner_command(event: hsm.Event) -> hsm.Event | None:
        """Return a firmware-bound owner command event, or None when not a command ingress.

        Already-typed command payloads pass through. Unvalidated dict/None payloads are
        coerced only when the event's declared schema is an owner command model (schema
        identity, not event.name). Invalid payloads are dropped (no shell fall-through).
        """

        command_schemas = (DialData, AnswerCallData, DeclineCallData, HangUpCallData, TransferCallData)
        data = event.data
        if isinstance(data, command_schemas):
            return event
        schema = event.schema
        if (
            schema is not DialData
            and schema is not AnswerCallData
            and schema is not DeclineCallData
            and schema is not HangUpCallData
            and schema is not TransferCallData
        ):
            return None
        try:
            validated = validate_event_data(event, {} if data is None else data)
        except ValueError:
            return None
        if not isinstance(validated, command_schemas):
            return None
        return dataclasses.replace(event, data=validated)

    @typing.override
    async def _after_firmware_started(self, ctx: hsm.Context, event: hsm.Event) -> None:
        await super()._after_firmware_started(ctx, event)
        if self._firmware is None:
            return
        # Service/media outlive firmware-init activity; attach under device lifetime context (HSM-CONTEXT-001).
        environment = Environment.from_context(self.context())
        await self._service.attach(environment, self._firmware)
        # Firmware is the controller, so it wires itself to its own transducers. Wiring, not
        # gating: this attach happens once at bring-up and nothing in src ever detaches, so the
        # microphone transduces on EVERY environment.sound for the phone's whole life — a per-broadcast
        # hot path that runs whether or not a call is up — and firmware discards what arrives
        # outside /Phone/answered/media_ready. Call state is gated by that transition's scope, not
        # by the attachment. Anything changing what the mouthpiece costs when idle changes it here.
        wired = attachment.AttachEvent.with_data(attachment.AttachData(actor=self._firmware))
        await self._microphone.attach(environment, wired)
        await self._speaker.attach(environment, wired)
        # The display is attached the same way, for the same reason: firmware wires every
        # peripheral it drives uniformly. Unlike the mouthpiece, the display's response is not
        # gated on this attachment — firmware clears it on its own first entry into hung_up,
        # before this attach has run — so the wire being present is an ownership fact, not a
        # precondition for showing a caller ID.
        await self._display.attach(environment, wired)

    @typing.override
    def _create_firmware_instance(self, ctx: hsm.Context, event: hsm.Event) -> hsm.Instance:
        del ctx, event
        return self._firmware_instance

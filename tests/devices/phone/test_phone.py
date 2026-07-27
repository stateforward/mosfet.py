from bot.devices import audio as audio_device

import asyncio
import collections.abc
import dataclasses
import datetime
import typing

import hsm
from bot.abilities import processing
import bot.devices.phone as phone_contracts
import bot.devices.phone.phone as phone_module

from bot.device import Device
from bot.devices import phone as phone_device
from bot.protocols import attachment

from bot.environment import SoundData, SoundEvent, Environment, space
from tests.hsm_instance_state import (
    device_peripherals,
    device_firmware,
    device_bots,
    phone_closed_call_ids,
    phone_current_call_id,
    phone_current_transfer_id,
    phone_current_transfer_target,
    phone_firmware,
    phone_microphone,
    phone_speaker,
)
from tests.hsm_model import transition_map
from tests.type_helpers import invalid_value

DIAL_NUMBER = "5550142"
"""A number to dial. Digits, from the fictional 555-01xx range, the way a real number is."""

def _phone_firmware(phone: phone_device.Phone) -> phone_device.PhoneFirmware:
    firmware = phone_firmware(phone)
    assert isinstance(firmware, phone_device.PhoneFirmware)
    return firmware

async def _wait_until(predicate: collections.abc.Callable[[], bool], *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("Timed out waiting for phone firmware condition.")

def _event_names(recorder: phone_device.PhoneEventRecorder) -> list[str]:
    return [event.name for event in recorder.events]

_SERVICE_ORIGINATING_EVENT_NAMES = frozenset(
    {
        phone_device.IncomingCallEvent.name,
        phone_device.CallConnectedEvent.name,
        phone_device.CallFailedEvent.name,
        phone_device.ServiceDialFailedEvent.name,
        phone_device.ServiceMediaReadyEvent.name,
        phone_device.RemoteHangUpEvent.name,
        phone_device.ServiceAudioReceivedEvent.name,
        phone_device.TransferAcceptedEvent.name,
        phone_device.ServiceTransferCompletedEvent.name,
        phone_device.ServiceTransferFailedEvent.name,
    }
)

@dataclasses.dataclass
class AttachablePhoneService:
    events: list[hsm.Event[typing.Any]] = dataclasses.field(default_factory=list)
    target: hsm.Instance | None = None

    async def attach(self, environment: Environment, target: hsm.Instance) -> None:
        assert target.context().value(hsm.Keys.Instances) is environment.value(hsm.Keys.Instances)
        self.target = target

    async def detach(self, environment: Environment, target: hsm.Instance) -> None:
        assert target.context().value(hsm.Keys.Instances) is environment.value(hsm.Keys.Instances)
        if self.target is target:
            self.target = None

    async def receive(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        if self.target is None or event.name not in _SERVICE_ORIGINATING_EVENT_NAMES:
            return
        await self.target.dispatch(ctx, event)

    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        self.events.append(event)

def _record_phone_observation(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event) -> None:
    del ctx, instance, event

class PhoneObservationRecorder(hsm.Instance):
    events: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name in {audio_device.OutputEvent.name, SoundEvent.name, phone_device.RingingEvent.name}:
            self.events.append(event)
        return super().dispatch(ctx, event)

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "PhoneObservationRecorder",
        hsm.initial(hsm.target("listening")),
        hsm.state(
            "listening",
            hsm.transition(
                hsm.on(SoundEvent, phone_device.RingingEvent),
                hsm.effect(_record_phone_observation),
            ),
        ),
    )

async def _emit_service_event(phone: phone_device.Phone, event: hsm.Event[typing.Any]) -> None:
    await _phone_firmware(phone).event_recorder().receive(phone.context(), event)

async def _answer_call(phone: phone_device.Phone, call_id: str) -> None:
    await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
    await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id=call_id)))

def test_phone_has_no_speech_specific_operator_requirements() -> None:
    phone = phone_device.Phone()

    assert isinstance(phone, Device)
    assert device_bots(phone) == ()
    assert phone_device.Phone.required_bot_abilities == ()

def test_phone_owns_private_microphone_and_speaker_peripherals() -> None:
    phone = phone_device.Phone()

    assert isinstance(phone_microphone(phone), audio_device.Microphone)
    assert isinstance(phone_speaker(phone), audio_device.Speaker)
    assert device_peripherals(phone) == (phone_microphone(phone), phone_speaker(phone))
    assert not hasattr(phone, "microphone")
    assert not hasattr(phone, "speaker")
    assert not hasattr(phone, "firmware")
    assert not hasattr(phone, "peripherals")

def test_phone_accepts_injected_audio_peripherals() -> None:
    microphone = audio_device.Microphone()
    speaker = audio_device.Speaker()
    extra_peripheral = Device()
    phone = phone_device.Phone(microphone=microphone, speaker=speaker, peripherals=(extra_peripheral,))

    assert phone_microphone(phone) is microphone
    assert phone_speaker(phone) is speaker
    assert device_peripherals(phone) == (microphone, speaker, extra_peripheral)

def test_phone_uses_phone_firmware_instance() -> None:
    phone = phone_device.Phone()

    assert phone.firmware_model is phone_device.PhoneFirmware.model
    assert not hasattr(phone, "operation_events")
    assert not hasattr(phone_module, "_PHONE_FIRMWARE_STATES")
    assert not hasattr(phone_module, "_PHONE_STATES")
    assert isinstance(_phone_firmware(phone), phone_device.PhoneFirmware)
    assert phone_closed_call_ids(_phone_firmware(phone)) == frozenset()
    assert phone_current_call_id(_phone_firmware(phone)) is None
    assert phone_current_transfer_target(_phone_firmware(phone)) is None

def test_phone_processing_operations_follow_merged_firmware_snapshot() -> None:
    def operation_names(phone: phone_device.Phone) -> list[str]:
        event_map = dict(phone.model.events)
        event_map.update(phone.firmware_model.events)
        names: list[str] = []
        for transition in hsm.take_snapshot(phone.context(), phone).Transitions:
            for event_name in transition.events:
                event = event_map.get(event_name)
                if event is not None and event.kind == processing.EventKind:
                    names.append(event.name)
        return names

    async def run() -> None:
        phone = phone_device.Phone()
        transfer_target = phone_device.TransferTarget(kind="address", value="helpdesk@example.com")

        _ = await hsm.started(None, phone, phone.model)

        snapshot_event_names = {
            event_name for transition in hsm.take_snapshot(None, phone).Transitions for event_name in transition.events
        }
        assert phone_device.IncomingCallEvent.name in snapshot_event_names
        # Idle offers dial, and answer — which reports phone.no_call rather than dropping a stray
        # answer in silence. The button exists on an idle handset; pressing it does nothing.
        assert operation_names(phone) == [phone_device.DialEvent.name, phone_device.AnswerCallEvent.name]

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))

        assert operation_names(phone) == [
            phone_device.AnswerCallEvent.name,
            phone_device.DeclineCallEvent.name,
        ]

        await _answer_call(phone, "call-123")

        assert operation_names(phone) == [phone_device.HangUpCallEvent.name]

        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")))

        assert operation_names(phone) == [
            phone_device.TransferCallEvent.name,
            phone_device.HangUpCallEvent.name,
        ]

        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(transfer_id="transfer-123", target=transfer_target)
            ),
        )

        assert operation_names(phone) == [phone_device.HangUpCallEvent.name]

    asyncio.run(run())

def test_phone_rejects_non_positive_timeouts() -> None:
    try:
        _ = phone_device.Phone(answer_timeout=datetime.timedelta(seconds=0))
    except ValueError:
        pass
    else:
        raise AssertionError("Phone should reject a non-positive answer timeout.")

    try:
        _ = phone_device.Phone(transfer_timeout=datetime.timedelta(seconds=-1))
    except ValueError:
        pass
    else:
        raise AssertionError("Phone should reject a non-positive transfer timeout.")

def test_phone_accepts_injected_service() -> None:
    recorder = phone_device.PhoneEventRecorder()
    phone = phone_device.Phone(service=recorder)

    assert _phone_firmware(phone).event_recorder() is recorder

def test_phone_connects_service_originating_events() -> None:
    async def run() -> None:
        service = AttachablePhoneService()
        phone = phone_device.Phone(service=service)

        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert service.target is device_firmware(phone)
        await service.receive(phone.context(), phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _wait_until(lambda: device_firmware(phone) is not None and device_firmware(phone).state() == "/Phone/ringing")

        assert service.events[-1].name == phone_device.RingingEvent.name

    asyncio.run(run())

def test_phone_service_events_enter_through_attached_service_target() -> None:
    async def run() -> None:
        phone = phone_device.Phone()

        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))

        assert phone.state() == "/Device/detached"
        assert device_firmware(phone).state() == "/Phone/ringing"
        assert phone_current_call_id(_phone_firmware(phone)) == "call-123"

    asyncio.run(run())

def test_phone_dispatch_does_not_forward_service_originating_events() -> None:
    async def run() -> None:
        phone = phone_device.Phone()

        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None

        await phone.dispatch(
            phone.context(),
            phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )

        assert phone.state() == "/Device/detached"
        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(_phone_firmware(phone)) is None
        assert _phone_firmware(phone).event_recorder().events == ()

    asyncio.run(run())

def test_phone_service_ingress_does_not_forward_non_service_events() -> None:
    async def run() -> None:
        service = AttachablePhoneService()
        phone = phone_device.Phone(service=service)

        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert service.target is device_firmware(phone)

        await service.receive(
            phone.context(),
            attachment.AttachEvent.with_data(attachment.AttachData(actor=hsm.Instance())),
        )
        await asyncio.sleep(0)

        assert phone.state() == "/Device/detached"
        assert device_bots(phone) == ()

        await service.receive(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await asyncio.sleep(0)

        assert device_firmware(phone) is not None
        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert service.events == []

    asyncio.run(run())

def test_phone_exports_service_contract_without_event_sink_alias() -> None:
    assert phone_contracts.PhoneService is phone_device.PhoneService
    assert "PhoneService" in phone_contracts.__all__
    assert "PHONE_SERVICE_CONNECTED" not in phone_contracts.__all__
    assert "PhoneServiceConnectionData" not in phone_contracts.__all__
    assert not hasattr(phone_contracts, "PHONE_SERVICE_CONNECTED")
    assert not hasattr(phone_contracts, "PhoneServiceConnectionData")
    assert not hasattr(phone_contracts, "PhoneEventSink")
    assert "PhoneEventSink" not in phone_contracts.__all__

def test_phone_firmware_model_tracks_call_lifecycle_and_transfer() -> None:
    model = phone_device.Phone.firmware_model

    assert model.qualified_name == "/Phone"
    assert model.initial == "/Phone/.initial"
    assert "/Phone/hung_up" in model.members
    assert "/Phone/dialing" in model.members
    assert "/Phone/ringing" in model.members
    assert "/Phone/answering" in model.members
    assert "/Phone/answered" in model.members
    assert "/Phone/answered/media_connecting" in model.members
    assert "/Phone/answered/media_ready" in model.members
    assert "/Phone/transferring" in model.members
    transitions = transition_map(model)
    assert "phone.dial" in transitions["/Phone/hung_up"]
    assert "phone.service.incoming_call" in transitions["/Phone/hung_up"]
    assert "phone.service.call_connected" in transitions["/Phone/dialing"]
    assert "phone.hang_up_call" in transitions["/Phone/dialing"]
    # A dial that fails never became a call, so dialing hears a dial failure, not a call failure.
    assert "phone.service.dial_failed" in transitions["/Phone/dialing"]
    assert "phone.service.call_failed" not in transitions["/Phone/dialing"]
    # Answering with nothing ringing is reported rather than dropped.
    assert "phone.answer_call" in transitions["/Phone/hung_up"]
    assert "phone.service.incoming_call" in transitions["/Phone/ringing"]
    assert "phone.answer_call" in transitions["/Phone/ringing"]
    assert "phone.decline_call" in transitions["/Phone/ringing"]
    assert "phone.service.remote_hang_up" in transitions["/Phone/ringing"]
    assert "phone.service.call_connected" in transitions["/Phone/answering"]
    assert "phone.decline_call" in transitions["/Phone/answering"]
    assert "phone.hang_up_call" in transitions["/Phone/answering"]
    assert "phone.hang_up_call" in transitions["/Phone/answered"]
    assert "phone.service.remote_hang_up" in transitions["/Phone/answered"]
    assert "phone.service.call_failed" in transitions["/Phone/answered"]
    assert "phone.service.media_ready" in transitions["/Phone/answered/media_connecting"]
    assert "phone.transfer_call" in transitions["/Phone/answered/media_ready"]
    assert "phone.service.audio_received" in transitions["/Phone/answered/media_ready"]
    assert "phone.service.transfer_completed" in transitions["/Phone/transferring"]
    assert "phone.service.transfer_failed" in transitions["/Phone/transferring"]
    assert "phone.service.transfer_accepted" in transitions["/Phone/transferring"]
    assert "phone.service.audio_received" in transitions["/Phone/transferring"]
    assert "phone.service.audio_received" not in transitions["/Phone/hung_up"]
    assert "phone.service.audio_received" not in transitions["/Phone/ringing"]
    assert "phone.service.audio_received" not in transitions["/Phone/answering"]
    assert "phone.service.audio_received" not in transitions["/Phone/dialing"]

def test_phone_device_start_initializes_firmware_and_routes_service_events() -> None:
    async def run() -> None:
        phone = phone_device.Phone()

        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")

        assert phone.state() == "/Device/detached"
        assert device_firmware(phone) is not None
        assert device_firmware(phone).state() == "/Phone/hung_up"
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))

        assert phone.state() == "/Device/detached"
        assert device_firmware(phone).state() == "/Phone/ringing"
        assert phone_current_call_id(firmware) == "call-123"
        assert _event_names(firmware.event_recorder()) == [phone_device.RingingEvent.name]

    asyncio.run(run())

def test_phone_dial_requests_provider_and_commits_connected_call() -> None:
    async def run() -> None:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )

        assert device_firmware(phone).state() == "/Phone/dialing"
        # Dialing holds no call: the exchange has not assigned one yet.
        assert phone_current_call_id(firmware) is None
        assert _event_names(firmware.event_recorder()) == [phone_device.ServiceDialRequestedEvent.name]

        await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-123")))

        assert device_firmware(phone).state() == "/Phone/answered/media_connecting"
        assert phone_current_call_id(firmware) == "call-123"
        assert _event_names(firmware.event_recorder()) == [
            phone_device.ServiceDialRequestedEvent.name,
            phone_device.AnsweredEvent.name,
        ]

    asyncio.run(run())

def test_phone_broadcasts_committed_ringing_observation_in_current_environment() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], str]:
        metadata = {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}
        environment = Environment()
        phone = phone_device.Phone()
        inside = PhoneObservationRecorder()
        outside = PhoneObservationRecorder()

        _ = await hsm.started(environment, phone, phone.model)
        _ = await hsm.started(environment, inside, inside.model, hsm.Config(id="inside"))
        environment.join(inside)
        _ = await hsm.started(None, outside, outside.model, hsm.Config(id="outside"))
        phone_id = hsm.id(phone)

        await _emit_service_event(
            phone,
            dataclasses.replace(
                phone_device.IncomingCallEvent.with_data(
                    phone_device.IncomingCallData(call_id="call-123", caller="Front desk")
                ),
                metadata=metadata,
            ),
        )
        await _wait_until(lambda: bool(inside.events))

        return inside.events, outside.events, phone_id

    inside_events, outside_events, phone_id = asyncio.run(run())

    assert len(inside_events) == 1
    assert inside_events[0].name == "environment.sound"
    assert inside_events[0].source == phone_id
    assert inside_events[0].target == "inside"
    sound = inside_events[0].data
    assert getattr(sound, "kind", None) == "phone.ringing"
    assert getattr(sound, "media_type", None) == "audio/wav"
    assert getattr(sound, "sample_rate_hz", None) == 16_000
    assert getattr(sound, "channels", None) == 1
    assert getattr(sound, "audio", b"").startswith(b"RIFF")
    assert getattr(sound, "audio", b"") == phone_device.RING_SOUND_WAV
    assert getattr(sound, "caller", "") == "Front desk"
    assert inside_events[0].metadata.get("traceparent") == "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"
    assert outside_events == []

async def _dialing_phone_in_environment(
    environment: Environment,
) -> tuple[phone_device.Phone, phone_device.PhoneFirmware, PhoneObservationRecorder]:
    """A started phone mid-dial, with an environment citizen standing where it can hear it."""

    phone = phone_device.Phone(answer_timeout=datetime.timedelta(milliseconds=1))
    listener = PhoneObservationRecorder()
    _ = await hsm.started(environment, phone, phone.model)
    _ = await hsm.started(environment, listener, listener.model, hsm.Config(id="listener"))
    environment.join(listener)
    await _wait_until(lambda: phone.state() == "/Device/detached")
    firmware = _phone_firmware(phone)
    await phone.dispatch(
        phone.context(),
        phone_device.DialEvent.with_data(
            phone_device.DialData(number=DIAL_NUMBER)
        ),
    )
    return phone, firmware, listener

def test_phone_dial_failure_is_heard_as_the_call_progress_tone_the_exchange_would_send() -> None:
    """A failed dial reaches the bot's ears, and busy and reorder stay distinguishable.

    The exchange's verdict is what picks the tone: a line that refused or is engaged gives busy,
    and every other way the network fails to complete a call gives reorder — which is all a real
    caller gets to hear, and all this pins.
    """

    async def run(failure_kind: phone_device.FailureKind) -> tuple[list[hsm.Event[typing.Any]], list[str], str]:
        environment = Environment()
        phone, firmware, listener = await _dialing_phone_in_environment(environment)
        await _emit_service_event(
            phone,
            phone_device.ServiceDialFailedEvent.with_data(phone_device.DialFailedData(failure_kind=failure_kind)),
        )
        await _wait_until(lambda: bool(listener.events))
        return listener.events, _event_names(firmware.event_recorder()), hsm.id(phone)

    for failure_kind, expected_kind, expected_audio in (
        ("call_declined", "phone.busy", phone_device.BUSY_TONE_WAV),
        ("remote_unavailable", "phone.reorder", phone_device.REORDER_TONE_WAV),
        ("provider_unavailable", "phone.reorder", phone_device.REORDER_TONE_WAV),
        ("signaling_failed", "phone.reorder", phone_device.REORDER_TONE_WAV),
        ("timeout", "phone.reorder", phone_device.REORDER_TONE_WAV),
        ("unknown", "phone.reorder", phone_device.REORDER_TONE_WAV),
    ):
        heard, published, phone_id = asyncio.run(run(typing.cast(phone_device.FailureKind, failure_kind)))

        assert len(heard) == 1
        assert heard[0].name == SoundEvent.name
        assert heard[0].source == phone_id
        tone = heard[0].data
        assert isinstance(tone, phone_device.PhoneSoundData)
        assert tone.kind == expected_kind
        assert tone.audio == expected_audio
        assert tone.audio.startswith(b"RIFF")
        assert tone.media_type == "audio/wav"
        assert tone.sample_rate_hz == 16_000
        assert tone.channels == 1
        # An earpiece tone, not the ringer: for the one person holding the handset.
        assert tone.amplitude_db == phone_device.CALL_PROGRESS_DB
        assert phone_device.CALL_PROGRESS_DB < phone_device.RINGER_DB
        # A tone identifies nobody; only a ringing phone shows you who is calling.
        assert tone.caller is None
        # The device plane is unchanged: the environment gets the sound, the service gets the fact.
        assert published == [
            phone_device.ServiceDialRequestedEvent.name,
            phone_device.NoCallEvent.name,
        ]

def test_phone_no_call_carries_the_service_verdict_that_chose_the_tone() -> None:
    async def run() -> phone_device.NoCallData:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        await _wait_until(lambda: phone.state() == "/Device/detached")
        firmware = _phone_firmware(phone)
        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _emit_service_event(
            phone,
            phone_device.ServiceDialFailedEvent.with_data(phone_device.DialFailedData(failure_kind="call_declined")),
        )
        published = firmware.event_recorder().events[-1].data
        assert isinstance(published, phone_device.NoCallData)
        return published

    no_call = asyncio.run(run())

    assert no_call.reason == "dial_failed"
    assert no_call.failure_kind == "call_declined"
    assert no_call == phone_device.NoCallData(reason="dial_failed", failure_kind="call_declined")

def test_phone_makes_no_sound_for_the_ways_a_handset_makes_none() -> None:
    """Three ways to end up with no call that a real handset marks with silence.

    ``nothing_to_answer``: a handset with nothing ringing does essentially nothing audible when
    you press answer. ``dial_abandoned``: you hung up, so you hear nothing, because you hung up.
    ``dial_not_answered``: there is no "they did not answer" tone — a real caller hears ringback
    the whole time and then gives up, so what marks this case is ringback stopping rather than any
    sound starting. Each still reports ``phone.no_call`` on the device plane; silence in the
    environment is what the object does, not a dropped notification.
    """

    async def answer_with_nothing_ringing(phone: phone_device.Phone) -> None:
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))

    async def hang_up_mid_dial(phone: phone_device.Phone) -> None:
        await phone.dispatch(phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData()))

    async def wait_out_the_ring(phone: phone_device.Phone) -> None:
        del phone
        await asyncio.sleep(0.05)

    async def run(
        act: collections.abc.Callable[[phone_device.Phone], collections.abc.Awaitable[None]],
        dial_first: bool,
    ) -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]]]:
        environment = Environment()
        if dial_first:
            phone, firmware, listener = await _dialing_phone_in_environment(environment)
        else:
            phone = phone_device.Phone()
            listener = PhoneObservationRecorder()
            _ = await hsm.started(environment, phone, phone.model)
            _ = await hsm.started(environment, listener, listener.model, hsm.Config(id="listener"))
            environment.join(listener)
            await _wait_until(lambda: phone.state() == "/Device/detached")
            firmware = _phone_firmware(phone)
        await act(phone)
        await _wait_until(
            lambda: phone_device.NoCallEvent.name in _event_names(firmware.event_recorder()),
        )
        # Give any broadcast that was going to happen every chance to land.
        for _ in range(10):
            await asyncio.sleep(0)
        return listener.events, list(firmware.event_recorder().events)

    for act, dial_first, expected_reason in (
        (answer_with_nothing_ringing, False, "nothing_to_answer"),
        (hang_up_mid_dial, True, "dial_abandoned"),
        (wait_out_the_ring, True, "dial_not_answered"),
    ):
        heard, published = asyncio.run(run(act, dial_first))

        assert heard == []
        no_call = published[-1].data
        assert isinstance(no_call, phone_device.NoCallData)
        assert no_call.reason == expected_reason
        # Nothing decided these but the phone and its operator, so there is no service verdict.
        assert no_call.failure_kind is None

def test_phone_committed_observations_have_priority_over_queued_external_events() -> None:
    async def run() -> tuple[str, list[str]]:
        phone = phone_device.Phone()

        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)
        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))

        connected = asyncio.ensure_future(
            _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-123")))
        )
        await asyncio.sleep(0)
        hang_up = asyncio.ensure_future(
            phone.dispatch(
                phone.context(),
                phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData()),
            )
        )
        _ = await asyncio.gather(connected, hang_up)
        await _wait_until(lambda: firmware.state() == "/Phone/hung_up")

        return firmware.state(), _event_names(firmware.event_recorder())

    state, event_names = asyncio.run(run())

    assert state == "/Phone/hung_up"
    assert event_names == [
        phone_device.RingingEvent.name,
        phone_device.ServiceAnswerRequestedEvent.name,
        phone_device.AnsweredEvent.name,
        phone_device.ServiceHangUpRequestedEvent.name,
        phone_device.HungUpEvent.name,
    ]

def test_phone_rejects_malformed_incoming_call_before_effects() -> None:
    async def run() -> None:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)
        malformed = phone_device.IncomingCallEvent.with_data(invalid_value(phone_device.IncomingCallData, object()))

        await _emit_service_event(phone, malformed)

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert firmware.event_recorder().events == ()

    asyncio.run(run())

def test_phone_rejects_malformed_payloads_before_event_specific_effects() -> None:
    async def run() -> None:
        phone = phone_device.Phone()
        transfer_target = phone_device.TransferTarget(kind="address", value="helpdesk@example.com")
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _emit_service_event(
            phone,
            phone_device.IncomingCallEvent.with_data(invalid_value(phone_device.IncomingCallData, object())),
        )
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(invalid_value(phone_device.AnswerCallData, phone_device.IncomingCallData(call_id="call-123"))),
        )

        assert device_firmware(phone).state() == "/Phone/ringing"
        assert _event_names(firmware.event_recorder()) == [phone_device.RingingEvent.name]

        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _emit_service_event(
            phone,
            phone_device.CallConnectedEvent.with_data(invalid_value(phone_device.CallConnectedData, phone_device.AnswerCallData())),
        )

        assert device_firmware(phone).state() == "/Phone/answering"
        assert _event_names(firmware.event_recorder()) == [phone_device.RingingEvent.name, phone_device.ServiceAnswerRequestedEvent.name]

        await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-123")))
        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")))
        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(transfer_id="transfer-123", target=transfer_target)
            ),
        )
        await _emit_service_event(
            phone,
            phone_device.ServiceTransferCompletedEvent.with_data(
                invalid_value(
                    phone_device.TransferCompletedData,
                    phone_device.TransferFailedData(
                        call_id="call-123",
                        transfer_id="transfer-123",
                        target=transfer_target,
                        failure_kind="transfer_rejected",
                    ),
                )
            ),
        )

        assert device_firmware(phone).state() == "/Phone/transferring"
        assert _event_names(firmware.event_recorder())[-2:] == [
            phone_device.ServiceTransferRequestedEvent.name,
            phone_device.TransferStartedEvent.name,
        ]

    asyncio.run(run())

def test_phone_emitted_events_preserve_trigger_metadata() -> None:
    async def run() -> None:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)
        metadata = {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}

        await _emit_service_event(
            phone,
            dataclasses.replace(
                phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
                metadata=metadata,
            ),
        )
        await phone.dispatch(
            phone.context(),
            dataclasses.replace(
                phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
                metadata=metadata,
            ),
        )

        assert firmware.event_recorder().events[0].name == phone_device.RingingEvent.name
        assert firmware.event_recorder().events[0].metadata == metadata
        assert firmware.event_recorder().events[1].name == phone_device.ServiceAnswerRequestedEvent.name
        assert firmware.event_recorder().events[1].metadata == metadata

    asyncio.run(run())

def test_phone_firmware_rejects_stale_service_events_by_call_id() -> None:
    async def run() -> None:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-stale")))
        await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-stale")))

        assert device_firmware(phone).state() == "/Phone/ringing"
        assert phone_current_call_id(firmware) == "call-123"
        assert _event_names(firmware.event_recorder()) == [phone_device.RingingEvent.name]

        await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-123")))
        assert device_firmware(phone).state() == "/Phone/ringing"
        assert _event_names(firmware.event_recorder()) == [phone_device.RingingEvent.name]

        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _emit_service_event(phone, phone_device.RemoteHangUpEvent.with_data(phone_device.RemoteHangUpData(call_id="call-stale")))

        assert device_firmware(phone).state() == "/Phone/answering"
        assert phone_current_call_id(firmware) == "call-123"
        assert _event_names(firmware.event_recorder()) == [phone_device.RingingEvent.name, phone_device.ServiceAnswerRequestedEvent.name]

        await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-123")))

        assert device_firmware(phone).state() == "/Phone/answered/media_connecting"
        assert phone_current_call_id(firmware) == "call-123"
        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.AnsweredEvent.name,
        ]

        await _emit_service_event(phone, phone_device.RemoteHangUpEvent.with_data(phone_device.RemoteHangUpData(call_id="call-123")))

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert "call-123" in phone_closed_call_ids(firmware)
        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.AnsweredEvent.name,
            phone_device.HungUpEvent.name,
        ]

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.AnsweredEvent.name,
            phone_device.HungUpEvent.name,
        ]

    asyncio.run(run())

def test_phone_firmware_declines_and_hangs_up_the_call_it_holds() -> None:
    async def run() -> None:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        # Answering again while already answering is not a second answer.
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))

        assert device_firmware(phone).state() == "/Phone/answering"
        assert phone_current_call_id(firmware) == "call-123"
        assert _event_names(firmware.event_recorder()) == [phone_device.RingingEvent.name, phone_device.ServiceAnswerRequestedEvent.name]

        await phone.dispatch(phone.context(), phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData()))

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.ServiceDeclineRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-456")))
        await _answer_call(phone, "call-456")

        assert device_firmware(phone).state() == "/Phone/answered/media_connecting"
        assert phone_current_call_id(firmware) == "call-456"

        await phone.dispatch(phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData()))

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert _event_names(firmware.event_recorder())[-2:] == [
            phone_device.ServiceHangUpRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

    asyncio.run(run())

def test_phone_media_ready_publishes_committed_event() -> None:
    async def run() -> None:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _answer_call(phone, "call-123")
        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-stale")))

        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.AnsweredEvent.name,
        ]

        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")))

        assert _event_names(firmware.event_recorder())[-1] == phone_device.MediaReadyEvent.name
        assert device_firmware(phone).state() == "/Phone/answered/media_ready"

    asyncio.run(run())

def test_phone_service_audio_routes_through_speaker_to_environment_observers() -> None:
    async def run() -> tuple[tuple[hsm.Event[typing.Any], ...], list[str], str]:
        metadata = {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}
        environment = Environment()
        phone = phone_device.Phone()
        observer = PhoneObservationRecorder()
        current_audio = phone_device.ServiceAudioData(
            call_id="call-123",
            audio=b"playback-audio",
            media_type="audio/pcm",
            sample_rate_hz=48_000,
            channels=1,
        )
        stale_audio = phone_device.ServiceAudioData(
            call_id="call-stale",
            audio=b"stale-audio",
            media_type="audio/pcm",
            sample_rate_hz=48_000,
            channels=1,
        )

        speaker = phone_speaker(phone)
        # The phone powers its own speaker; starting it here would be a second start.
        _ = await hsm.started(environment, phone, phone.model)
        _ = await hsm.started(environment, observer, observer.model, hsm.Config(id="observer"))
        environment.join(observer)
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _emit_service_event(phone, phone_device.ServiceAudioReceivedEvent.with_data(current_audio))
        await _answer_call(phone, "call-123")
        await _emit_service_event(phone, phone_device.ServiceAudioReceivedEvent.with_data(current_audio))
        await _emit_service_event(phone, phone_device.ServiceAudioReceivedEvent.with_data(stale_audio))

        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.AnsweredEvent.name,
        ]
        assert len(observer.events) == 1
        assert observer.events[0].name == SoundEvent.name
        ring_data = observer.events[0].data
        assert isinstance(ring_data, phone_device.PhoneSoundData)
        assert ring_data.kind == "phone.ringing"
        # A withheld caller still rings.
        assert ring_data.caller is None
        observer.events.clear()
        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")))
        await _wait_until(lambda: _event_names(firmware.event_recorder())[-1] == phone_device.MediaReadyEvent.name)
        observer.events.clear()

        await _emit_service_event(
            phone,
            dataclasses.replace(phone_device.ServiceAudioReceivedEvent.with_data(current_audio), metadata=metadata),
        )
        await _wait_until(lambda: len(observer.events) == 1)

        return tuple(observer.events), _event_names(firmware.event_recorder()), hsm.id(speaker)

    events, service_event_names, speaker_id = asyncio.run(run())

    assert service_event_names == [
        phone_device.RingingEvent.name,
        phone_device.ServiceAnswerRequestedEvent.name,
        phone_device.AnsweredEvent.name,
        phone_device.MediaReadyEvent.name,
    ]
    assert len(events) == 1
    event = events[0]
    assert event.name == SoundEvent.name
    assert event.data.audio == b"playback-audio"
    assert event.data.media_type == "audio/pcm"
    assert event.data.sample_rate_hz == 48_000
    assert event.data.channels == 1
    assert event.source == speaker_id
    assert event.target == "observer"
    assert event.metadata == {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}

def test_phone_service_audio_direct_start_does_not_accept_unstarted_speaker_audio() -> None:
    async def run() -> tuple[tuple[hsm.Event[typing.Any], ...], list[str]]:
        environment = Environment()
        phone = phone_device.Phone()
        observer = PhoneObservationRecorder()
        audio = phone_device.ServiceAudioData(
            call_id="call-123",
            audio=b"playback-audio",
            media_type="audio/pcm",
            sample_rate_hz=48_000,
            channels=1,
        )

        _ = await hsm.started(environment, phone, phone.model)
        _ = await hsm.started(environment, observer, observer.model, hsm.Config(id="observer"))
        environment.join(observer)
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _answer_call(phone, "call-123")
        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")))
        observer.events.clear()

        await _emit_service_event(phone, phone_device.ServiceAudioReceivedEvent.with_data(audio))
        await asyncio.sleep(0)

        return tuple(observer.events), _event_names(firmware.event_recorder())

    events, service_event_names = asyncio.run(run())

    assert events == ()
    assert service_event_names == [
        phone_device.RingingEvent.name,
        phone_device.ServiceAnswerRequestedEvent.name,
        phone_device.AnsweredEvent.name,
        phone_device.MediaReadyEvent.name,
    ]

def test_phone_service_audio_routes_while_transfer_in_progress() -> None:
    async def run() -> tuple[
        tuple[hsm.Event[typing.Any], ...],
        list[str],
        str,
        str,
        str | None,
        phone_device.TransferTarget | None,
    ]:
        metadata = {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}
        environment = Environment()
        phone = phone_device.Phone()
        observer = PhoneObservationRecorder()
        transfer_target = phone_device.TransferTarget(kind="address", value="helpdesk@example.com")
        current_audio = phone_device.ServiceAudioData(
            call_id="call-123",
            audio=b"transfer-audio",
            media_type="audio/pcm",
            sample_rate_hz=48_000,
            channels=1,
        )
        stale_audio = phone_device.ServiceAudioData(
            call_id="call-stale",
            audio=b"stale-transfer-audio",
            media_type="audio/pcm",
            sample_rate_hz=48_000,
            channels=1,
        )

        speaker = phone_speaker(phone)
        # The phone powers its own speaker; starting it here would be a second start.
        _ = await hsm.started(environment, phone, phone.model)
        _ = await hsm.started(environment, observer, observer.model, hsm.Config(id="observer"))
        environment.join(observer)
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _answer_call(phone, "call-123")
        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")))
        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(transfer_id="transfer-123", target=transfer_target)
            ),
        )
        assert device_firmware(phone).state() == "/Phone/transferring"
        observer.events.clear()

        await _emit_service_event(phone, phone_device.ServiceAudioReceivedEvent.with_data(stale_audio))
        await asyncio.sleep(0)
        assert observer.events == []
        assert device_firmware(phone).state() == "/Phone/transferring"
        assert phone_current_transfer_id(firmware) == "transfer-123"
        assert phone_current_transfer_target(firmware) == transfer_target

        await _emit_service_event(
            phone,
            dataclasses.replace(phone_device.ServiceAudioReceivedEvent.with_data(current_audio), metadata=metadata),
        )
        await _wait_until(lambda: len(observer.events) == 1)

        return (
            tuple(observer.events),
            _event_names(firmware.event_recorder()),
            device_firmware(phone).state(),
            hsm.id(speaker),
            phone_current_transfer_id(firmware),
            phone_current_transfer_target(firmware),
        )

    events, service_event_names, state, speaker_id, transfer_id, transfer_target = asyncio.run(run())

    assert state == "/Phone/transferring"
    assert transfer_id == "transfer-123"
    assert transfer_target == phone_device.TransferTarget(kind="address", value="helpdesk@example.com")
    assert service_event_names == [
        phone_device.RingingEvent.name,
        phone_device.ServiceAnswerRequestedEvent.name,
        phone_device.AnsweredEvent.name,
        phone_device.MediaReadyEvent.name,
        phone_device.ServiceTransferRequestedEvent.name,
        phone_device.TransferStartedEvent.name,
    ]
    assert len(events) == 1
    event = events[0]
    assert event.name == SoundEvent.name
    assert event.data.audio == b"transfer-audio"
    assert event.data.media_type == "audio/pcm"
    assert event.data.sample_rate_hz == 48_000
    assert event.data.channels == 1
    assert event.source == speaker_id
    assert event.target == "observer"
    assert event.metadata == {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}

def test_phone_answering_timeout_commits_failed_hangup_and_rejects_late_connect() -> None:
    async def run() -> None:
        phone = phone_device.Phone(answer_timeout=datetime.timedelta(milliseconds=1))
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))

        assert device_firmware(phone).state() == "/Phone/answering"
        assert _event_names(firmware.event_recorder()) == [phone_device.RingingEvent.name, phone_device.ServiceAnswerRequestedEvent.name]

        await _wait_until(lambda: device_firmware(phone) is not None and device_firmware(phone).state() == "/Phone/hung_up")

        assert phone_current_call_id(firmware) is None
        assert "call-123" in phone_closed_call_ids(firmware)
        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

        await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-123")))

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

    asyncio.run(run())

def test_phone_dialing_timeout_reports_no_call_and_rejects_a_late_connect() -> None:
    async def run() -> None:
        phone = phone_device.Phone(answer_timeout=datetime.timedelta(milliseconds=1))
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )

        assert device_firmware(phone).state() == "/Phone/dialing"
        assert _event_names(firmware.event_recorder()) == [phone_device.ServiceDialRequestedEvent.name]

        await _wait_until(lambda: device_firmware(phone) is not None and device_firmware(phone).state() == "/Phone/hung_up")

        # No call was ever assigned, so there is no hang-up to commit and nothing to close.
        assert phone_current_call_id(firmware) is None
        assert phone_closed_call_ids(firmware) == frozenset()
        assert _event_names(firmware.event_recorder()) == [
            phone_device.ServiceDialRequestedEvent.name,
            phone_device.NoCallEvent.name,
        ]
        no_call = firmware.event_recorder().events[-1].data
        assert isinstance(no_call, phone_device.NoCallData)
        assert no_call.reason == "dial_not_answered"

        # A connect arriving after the attempt is over is rejected by state, not by identity.
        await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-123")))

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert _event_names(firmware.event_recorder()) == [
            phone_device.ServiceDialRequestedEvent.name,
            phone_device.NoCallEvent.name,
        ]

        # Redial is always legal: a handset does not refuse a number it just called.
        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )

        assert device_firmware(phone).state() == "/Phone/dialing"

    asyncio.run(run())

def test_phone_call_failed_ends_current_call_in_each_active_phase() -> None:
    async def started_phone() -> tuple[phone_device.Phone, phone_device.PhoneFirmware]:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        return phone, _phone_firmware(phone)

    async def run() -> None:
        phone, firmware = await started_phone()
        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _emit_service_event(
            phone,
            phone_device.ServiceDialFailedEvent.with_data(phone_device.DialFailedData(failure_kind="signaling_failed")),
        )

        assert device_firmware(phone) is not None
        assert device_firmware(phone).state() == "/Phone/hung_up"
        # Nothing was ever assigned, so there is no call to close.
        assert phone_current_call_id(firmware) is None
        assert phone_closed_call_ids(firmware) == frozenset()
        assert _event_names(firmware.event_recorder()) == [
            phone_device.ServiceDialRequestedEvent.name,
            phone_device.NoCallEvent.name,
        ]

        phone, firmware = await started_phone()
        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="ringing-call")))
        await _emit_service_event(
            phone,
            phone_device.CallFailedEvent.with_data(phone_device.CallFailedData(call_id="ringing-call", failure_kind="signaling_failed")),
        )

        assert device_firmware(phone) is not None
        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert "ringing-call" in phone_closed_call_ids(firmware)
        assert _event_names(firmware.event_recorder()) == [phone_device.RingingEvent.name, phone_device.HungUpEvent.name]

        phone, firmware = await started_phone()
        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="answering-call")))
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _emit_service_event(
            phone,
            phone_device.CallFailedEvent.with_data(phone_device.CallFailedData(call_id="answering-call", failure_kind="media_unavailable")),
        )

        assert device_firmware(phone) is not None
        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert "answering-call" in phone_closed_call_ids(firmware)
        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

        phone, firmware = await started_phone()
        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="answered-call")))
        await _answer_call(phone, "answered-call")
        await _emit_service_event(
            phone,
            phone_device.CallFailedEvent.with_data(phone_device.CallFailedData(call_id="answered-call", failure_kind="remote_unavailable")),
        )

        assert device_firmware(phone) is not None
        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert "answered-call" in phone_closed_call_ids(firmware)
        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.AnsweredEvent.name,
            phone_device.HungUpEvent.name,
        ]

        phone, firmware = await started_phone()
        transfer_target = phone_device.TransferTarget(kind="address", value="helpdesk@example.com")
        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="transfer-call")))
        await _answer_call(phone, "transfer-call")
        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="transfer-call")))
        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(transfer_id="transfer-123", target=transfer_target)
            ),
        )
        await _emit_service_event(
            phone,
            phone_device.CallFailedEvent.with_data(phone_device.CallFailedData(call_id="transfer-call", failure_kind="provider_unavailable")),
        )

        assert device_firmware(phone) is not None
        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert phone_current_transfer_id(firmware) is None
        assert phone_current_transfer_target(firmware) is None
        assert "transfer-call" in phone_closed_call_ids(firmware)
        assert _event_names(firmware.event_recorder()) == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.AnsweredEvent.name,
            phone_device.MediaReadyEvent.name,
            phone_device.ServiceTransferRequestedEvent.name,
            phone_device.TransferStartedEvent.name,
            phone_device.HungUpEvent.name,
        ]

    asyncio.run(run())

def test_phone_firmware_owns_transfer_state_and_rejects_stale_transfer_events() -> None:
    async def run() -> None:
        phone = phone_device.Phone()
        transfer_target = phone_device.TransferTarget(kind="address", value="helpdesk@example.com")
        stale_transfer_target = phone_device.TransferTarget(kind="address", value="old-helpdesk@example.com")
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _answer_call(phone, "call-123")
        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(transfer_id="transfer-stale", target=transfer_target)
            ),
        )

        assert device_firmware(phone).state() == "/Phone/answered/media_connecting"
        assert phone_current_call_id(firmware) == "call-123"
        assert phone_current_transfer_id(firmware) is None
        assert phone_current_transfer_target(firmware) is None

        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")))
        assert device_firmware(phone).state() == "/Phone/answered/media_ready"
        assert _event_names(firmware.event_recorder())[-1] == phone_device.MediaReadyEvent.name

        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(transfer_id="transfer-123", target=transfer_target)
            ),
        )

        assert device_firmware(phone).state() == "/Phone/transferring"
        assert phone_current_call_id(firmware) == "call-123"
        assert phone_current_transfer_id(firmware) == "transfer-123"
        assert phone_current_transfer_target(firmware) == transfer_target
        assert _event_names(firmware.event_recorder())[-2:] == [
            phone_device.ServiceTransferRequestedEvent.name,
            phone_device.TransferStartedEvent.name,
        ]

        await _emit_service_event(
            phone,
            phone_device.ServiceTransferFailedEvent.with_data(
                phone_device.TransferFailedData(
                    call_id="call-stale",
                    transfer_id="transfer-123",
                    target=transfer_target,
                    failure_kind="transfer_rejected",
                )
            ),
        )
        await _emit_service_event(
            phone,
            phone_device.TransferAcceptedEvent.with_data(
                phone_device.TransferAcceptedData(call_id="call-stale", transfer_id="transfer-123", target=transfer_target)
            ),
        )
        await _emit_service_event(
            phone,
            phone_device.ServiceTransferCompletedEvent.with_data(
                phone_device.TransferCompletedData(call_id="call-123", transfer_id="transfer-123", target=stale_transfer_target)
            ),
        )
        await _emit_service_event(
            phone,
            phone_device.ServiceTransferFailedEvent.with_data(
                phone_device.TransferFailedData(
                    call_id="call-123",
                    transfer_id="transfer-123",
                    target=stale_transfer_target,
                    failure_kind="transfer_rejected",
                )
            ),
        )

        assert device_firmware(phone).state() == "/Phone/transferring"
        assert phone_current_call_id(firmware) == "call-123"
        assert phone_current_transfer_id(firmware) == "transfer-123"
        assert phone_current_transfer_target(firmware) == transfer_target

        await _emit_service_event(
            phone,
            phone_device.TransferAcceptedEvent.with_data(
                phone_device.TransferAcceptedData(call_id="call-123", transfer_id="transfer-123", target=transfer_target)
            ),
        )

        assert device_firmware(phone).state() == "/Phone/transferring"
        assert phone_current_transfer_id(firmware) == "transfer-123"
        assert phone_current_transfer_target(firmware) == transfer_target

        await _emit_service_event(
            phone,
            phone_device.ServiceTransferFailedEvent.with_data(
                phone_device.TransferFailedData(
                    call_id="call-123",
                    transfer_id="transfer-123",
                    target=transfer_target,
                    failure_kind="transfer_rejected",
                )
            ),
        )

        assert device_firmware(phone).state() == "/Phone/answered/media_ready"
        assert phone_current_call_id(firmware) == "call-123"
        assert phone_current_transfer_id(firmware) is None
        assert phone_current_transfer_target(firmware) is None
        assert _event_names(firmware.event_recorder())[-1] == phone_device.CallTransferFailedEvent.name

        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(transfer_id="transfer-456", target=transfer_target)
            ),
        )
        await _emit_service_event(
            phone,
            phone_device.ServiceTransferCompletedEvent.with_data(
                phone_device.TransferCompletedData(call_id="call-123", transfer_id="transfer-456", target=transfer_target)
            ),
        )

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert phone_current_transfer_id(firmware) is None
        assert phone_current_transfer_target(firmware) is None
        assert _event_names(firmware.event_recorder())[-2:] == [
            phone_device.CallTransferCompletedEvent.name,
            phone_device.HungUpEvent.name,
        ]

    asyncio.run(run())

def test_phone_transfer_timeout_returns_to_answered_and_publishes_failure() -> None:
    async def run() -> None:
        phone = phone_device.Phone(transfer_timeout=datetime.timedelta(milliseconds=1))
        transfer_target = phone_device.TransferTarget(kind="address", value="helpdesk@example.com")
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _answer_call(phone, "call-123")
        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")))
        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(transfer_id="transfer-1", target=transfer_target)
            ),
        )
        await _wait_until(
            lambda: device_firmware(phone) is not None and device_firmware(phone).state() == "/Phone/answered/media_ready"
        )

        assert phone_current_call_id(firmware) == "call-123"
        assert phone_current_transfer_id(firmware) is None
        assert phone_current_transfer_target(firmware) is None
        assert _event_names(firmware.event_recorder())[-1] == phone_device.CallTransferFailedEvent.name

        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(transfer_id="transfer-2", target=transfer_target)
            ),
        )
        await _emit_service_event(
            phone,
            phone_device.ServiceTransferCompletedEvent.with_data(
                phone_device.TransferCompletedData(call_id="call-123", transfer_id="transfer-1", target=transfer_target)
            ),
        )

        assert device_firmware(phone).state() == "/Phone/transferring"
        assert phone_current_transfer_id(firmware) == "transfer-2"
        assert phone_current_transfer_target(firmware) == transfer_target
        assert _event_names(firmware.event_recorder())[-2:] == [
            phone_device.ServiceTransferRequestedEvent.name,
            phone_device.TransferStartedEvent.name,
        ]

    asyncio.run(run())

def test_public_phone_events_do_not_route_back_into_phone_firmware() -> None:
    async def run() -> None:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        # Bring-up now wires firmware to the transducers, so the shell settles a few turns later.
        await _wait_until(lambda: phone.state() == "/Device/detached")
        assert device_firmware(phone) is not None

        await phone.dispatch(
            phone.context(),
            phone_device.RingingEvent.with_data(phone_device.RingingData(call_id="call-123")),
        )

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(_phone_firmware(phone)) is None

    asyncio.run(run())


def test_phone_dispatch_coerces_dict_command_payload_to_firmware() -> None:
    """Owner DialEvent with raw dict data is coerced and reaches firmware (JSON/API ingress)."""

    async def run() -> str:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        raw = dataclasses.replace(
            phone_device.DialEvent,
            data={"number": DIAL_NUMBER},
        )
        await phone.dispatch(phone.context(), raw)
        await _wait_until(lambda: (device_firmware(phone) or phone).state() == "/Phone/dialing")
        return device_firmware(phone).state() or ""

    assert asyncio.run(run()).endswith("/dialing")


def test_phone_dispatch_drops_invalid_dict_command_without_shell_fallthrough() -> None:
    """Invalid owner-command dict does not reach firmware and is not shell-admitted."""

    async def run() -> str | None:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        before = phone_current_call_id(_phone_firmware(phone))
        raw = dataclasses.replace(phone_device.DialEvent, data={"not": "a dial"})
        await phone.dispatch(phone.context(), raw)
        await asyncio.sleep(0.05)
        assert phone_current_call_id(_phone_firmware(phone)) == before
        return device_firmware(phone).state()

    assert asyncio.run(run()) == "/Phone/hung_up"


def test_phone_event_recorder_ignores_local_audio_output_not_service_audio() -> None:
    """Recorder skips exact AudioOutputData uplink offers; ServiceAudioData still records."""

    recorder = phone_device.PhoneEventRecorder()
    local = audio_device.OutputEvent.with_data(
        audio_device.AudioOutputData(audio=b"local", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
    )
    service = phone_device.ServiceAudioReceivedEvent.with_data(
        phone_device.ServiceAudioData(
            call_id="call-1",
            audio=b"remote",
            media_type="audio/pcm",
            sample_rate_hz=16_000,
            channels=1,
        )
    )
    recorder.publish(hsm.Context(), local)
    recorder.publish(hsm.Context(), service)
    assert [event.name for event in recorder.events] == [phone_device.ServiceAudioReceivedEvent.name]
    assert isinstance(recorder.events[0].data, phone_device.ServiceAudioData)


def test_receiver_audio_reaches_the_speaker_and_never_the_service() -> None:
    """Regression: far-end audio must never take the uplink path back to the caller.

    Previously firmware flattened ServiceAudioData into an exact AudioOutputData and published
    it through the service, whose direction was inferred from `type(data)`. The flattening
    erased the provenance the guard depended on, so the receiver fed the wire and the caller
    heard themselves. Direction is now which transition fired, and the service never sees it.
    """

    async def run() -> tuple[list[bytes], list[str]]:
        class TrackingSpeaker(audio_device.Speaker):
            elevated: list[audio_device.AudioOutputData]

            def __init__(self) -> None:
                super().__init__()
                self.elevated = []

            @typing.override
            def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[None]:
                del ctx
                data = event.data
                if isinstance(data, audio_device.AudioOutputData):
                    self.elevated.append(data)
                return asyncio.ensure_future(asyncio.sleep(0))

        class TrackingService:
            events: list[hsm.Event[typing.Any]]

            def __init__(self) -> None:
                self.events = []

            async def attach(self, environment: Environment, target: hsm.Instance) -> None:
                del environment, target

            async def detach(self, environment: Environment, target: hsm.Instance) -> None:
                del environment, target

            def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
                del ctx
                self.events.append(event)

        speaker = TrackingSpeaker()
        inner = TrackingService()
        firmware = phone_device.PhoneFirmware(service=inner, speaker=speaker)
        ctx = hsm.Context()
        service_audio = phone_device.ServiceAudioReceivedEvent.with_data(
            phone_device.ServiceAudioData(
                call_id="call-1",
                audio=b"remote",
                media_type="audio/pcm",
                sample_rate_hz=16_000,
                channels=1,
            )
        )
        phone_device.PhoneFirmware._receive_service_audio(ctx, firmware, service_audio)
        await asyncio.sleep(0)
        return [bytes(item.audio) for item in speaker.elevated], [event.name for event in inner.events]

    elevated, published = asyncio.run(run())
    assert elevated == [b"remote"], "far-end audio must reach the speaker"
    assert published == [], f"receiver audio must never reach the service; got {published!r}"


def test_receiver_audio_keeps_its_service_type() -> None:
    """The speaker is handed ServiceAudioData, not a flattened AudioOutputData."""

    captured: list[audio_device.AudioOutputData] = []

    class CapturingSpeaker(audio_device.Speaker):
        @typing.override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[None]:
            del ctx
            data = event.data
            if isinstance(data, audio_device.AudioOutputData):
                captured.append(data)
            return asyncio.ensure_future(asyncio.sleep(0))

    async def run() -> None:
        firmware = phone_device.PhoneFirmware(service=phone_device.PhoneEventRecorder(), speaker=CapturingSpeaker())
        event = phone_device.ServiceAudioReceivedEvent.with_data(
            phone_device.ServiceAudioData(call_id="call-1", audio=b"remote", media_type="audio/pcm")
        )
        phone_device.PhoneFirmware._receive_service_audio(hsm.Context(), firmware, event)
        await asyncio.sleep(0)

    asyncio.run(run())
    assert len(captured) == 1
    assert isinstance(captured[0], phone_device.ServiceAudioData), (
        f"provenance was flattened to {type(captured[0]).__name__}"
    )


def test_microphone_audio_uplinks_only_while_media_ready() -> None:
    """Mouthpiece carries local audio up the wire, and only while a call is connected.

    Outbound had no coverage at all before: the speaker's uplink could be removed without a
    single test failing. This pins the direction that replaced it.
    """

    class TrackingService:
        events: list[hsm.Event[typing.Any]]

        def __init__(self) -> None:
            self.events = []

        async def attach(self, environment: Environment, target: hsm.Instance) -> None:
            del environment, target

        async def detach(self, environment: Environment, target: hsm.Instance) -> None:
            del environment, target

        def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
            del ctx
            self.events.append(event)

    inner = TrackingService()
    firmware = phone_device.PhoneFirmware(service=inner, speaker=audio_device.Speaker())
    captured = audio_device.InputEvent.with_data(
        audio_device.AudioInputData(audio=b"local speech", media_type="audio/pcm", sample_rate_hz=24_000, channels=1)
    )
    phone_device.PhoneFirmware._send_microphone_audio(hsm.Context(), firmware, captured)

    uplinked = [event for event in inner.events if event.name == audio_device.OutputEvent.name]
    assert len(uplinked) == 1, f"microphone audio must uplink; got {[e.name for e in inner.events]!r}"
    payload = uplinked[0].data
    assert isinstance(payload, audio_device.AudioOutputData)
    assert bytes(payload.audio) == b"local speech"
    # Exact type: the LiveKit provider only uplinks exact AudioOutputData.
    assert type(payload) is audio_device.AudioOutputData


def _uplinks(service: AttachablePhoneService) -> list[hsm.Event[typing.Any]]:
    return [event for event in service.events if event.name == audio_device.OutputEvent.name]


def test_each_phone_uplinks_only_what_its_own_microphone_hears() -> None:
    """A microphone feeds the controller attached to it, not every phone sharing the environment.

    Two phones hear one spoken chunk. Each mouthpiece carries its own capture up its own wire,
    so each service sees exactly one uplink. Attachment membership is asserted alongside the
    count: a transducer fans out to everything attached to it, so a double-attach would restore
    the old duplicate while still reading as a plain count regression.
    """

    async def run() -> tuple[
        tuple[int, tuple[hsm.Instance, ...], tuple[hsm.Instance, ...]],
        tuple[int, tuple[hsm.Instance, ...], tuple[hsm.Instance, ...]],
    ]:
        environment = Environment()
        first_service = AttachablePhoneService()
        second_service = AttachablePhoneService()
        first = phone_device.Phone(service=first_service)
        second = phone_device.Phone(service=second_service)

        # Starting the phones is enough: each powers its own transducers.
        for phone in (first, second):
            _ = await hsm.started(environment, phone, typing.cast(hsm.Model, phone_device.Phone.model))
        await _wait_until(lambda: device_firmware(first) is not None and device_firmware(second) is not None)

        for phone, service, call_id in ((first, first_service, "call-first"), (second, second_service, "call-second")):
            await service.receive(
                phone.context(),
                phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id=call_id)),
            )
            await phone.dispatch(
                phone.context(),
                phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
            )
            await service.receive(
                phone.context(),
                phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id=call_id)),
            )
            await service.receive(
                phone.context(),
                phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id=call_id)),
            )
        await _wait_until(
            lambda: _phone_firmware(first).state() == "/Phone/answered/media_ready"
            and _phone_firmware(second).state() == "/Phone/answered/media_ready"
        )
        first_service.events.clear()
        second_service.events.clear()

        await environment.broadcast(
            SoundEvent.with_data(
                SoundData(audio=b"spoken-chunk", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
            )
        )
        await _wait_until(lambda: bool(_uplinks(first_service)) and bool(_uplinks(second_service)))
        await asyncio.sleep(0)

        return (
            (len(_uplinks(first_service)), device_bots(phone_microphone(first)), (_phone_firmware(first),)),
            (len(_uplinks(second_service)), device_bots(phone_microphone(second)), (_phone_firmware(second),)),
        )

    (first_uplinks, first_attached, first_own_firmware), (
        second_uplinks,
        second_attached,
        second_own_firmware,
    ) = asyncio.run(run())

    assert first_uplinks == 1
    assert second_uplinks == 1
    # Exact membership, not just the count: a transducer fans out to everything attached to it,
    # so a second attachment would restore the duplicate while still reading as a count bug.
    assert first_attached == first_own_firmware
    assert second_attached == second_own_firmware


_MOUTHPIECE_THRESHOLD_DB = 70.0
_EARPIECE_DB = 25.0
_VOICE_DB = 60.0
_EARS_THRESHOLD_DB = 20.0


class Ears(hsm.Instance):
    """A robot's own hearing, placed where the robot is."""

    heard: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.heard = []

    @staticmethod
    def _hear(ctx: hsm.Context, instance: "Ears", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.heard.append(event)

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "Ears",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(SoundEvent), hsm.effect(_hear))),
    )


async def _handset_on_a_call(
    environment: Environment,
    service: AttachablePhoneService,
) -> tuple[phone_device.Phone, audio_device.Speaker, Ears]:
    """A placed handset at the robot's ear, its voice at its mouth, on a connected call."""

    ear = space.Position(x=0.0, y=0.0)
    mouth = space.Position(x=phone_device.MOUTH_OFFSET_M, y=0.0)
    phone = phone_device.Phone(
        service=service,
        speaker=audio_device.Speaker(placement=space.Placement(position=ear), amplitude_db=_EARPIECE_DB),
        microphone=audio_device.Microphone(
            placement=space.Placement(position=mouth, threshold_db=_MOUTHPIECE_THRESHOLD_DB)
        ),
        placement=space.Placement(position=ear),
    )
    voice = audio_device.Speaker(placement=space.Placement(position=mouth), amplitude_db=_VOICE_DB)
    ears = Ears()

    _ = await hsm.started(environment, phone, typing.cast(hsm.Model, phone_device.Phone.model))
    _ = await hsm.started(environment, voice, typing.cast(hsm.Model, audio_device.Speaker.model))
    _ = await hsm.started(environment, ears, Ears.model, hsm.Config(id="ears"))
    environment.join(ears, placement=space.Placement(position=ear, threshold_db=_EARS_THRESHOLD_DB))
    # Speaking is this speaker's controller in production; a transducer needs one to emit.
    controller = hsm.Instance()
    _ = await hsm.started(environment, controller, hsm.define("Controller", hsm.initial(hsm.target("s")), hsm.state("s")))
    await voice.attach(environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=controller)))

    await _wait_until(lambda: device_firmware(phone) is not None)
    await service.receive(
        phone.context(), phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-1"))
    )
    await phone.dispatch(
        phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData())
    )
    await service.receive(
        phone.context(), phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-1"))
    )
    await service.receive(
        phone.context(), phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-1"))
    )
    await _wait_until(lambda: _phone_firmware(phone).state() == "/Phone/answered/media_ready")
    return phone, voice, ears


def test_far_end_audio_does_not_come_back_up_the_wire() -> None:
    """The echo: what the caller hears must not be carried back to the caller.

    The earpiece plays at the robot's ear. By the time that reaches the mouthpiece 15 cm away it
    is 28.5 dB under the close-talk threshold, so the mouthpiece does not pick it up. Nothing
    suppresses it by name or by source — it is simply too quiet where the mouthpiece is.
    """

    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        service = AttachablePhoneService()
        phone, _, _ = await _handset_on_a_call(environment, service)
        service.events.clear()

        await service.receive(
            phone.context(),
            phone_device.ServiceAudioReceivedEvent.with_data(
                phone_device.ServiceAudioData(
                    call_id="call-1", audio=b"far-end", media_type="audio/pcm", sample_rate_hz=16_000, channels=1
                )
            ),
        )
        await asyncio.sleep(0)

        return _uplinks(service)

    assert asyncio.run(run()) == []


def test_the_robot_hears_its_own_earpiece() -> None:
    """The same earpiece that is inaudible at the mouthpiece is loud at the ear: 45 dB over."""

    async def run() -> int:
        environment = Environment()
        service = AttachablePhoneService()
        phone, _, ears = await _handset_on_a_call(environment, service)
        ears.heard.clear()

        await service.receive(
            phone.context(),
            phone_device.ServiceAudioReceivedEvent.with_data(
                phone_device.ServiceAudioData(
                    call_id="call-1", audio=b"far-end", media_type="audio/pcm", sample_rate_hz=16_000, channels=1
                )
            ),
        )
        await _wait_until(lambda: bool(ears.heard))

        return len(ears.heard)

    assert asyncio.run(run()) == 1


def test_the_robots_own_voice_goes_up_the_wire() -> None:
    """The mouthpiece is not deaf, it is selective: the voice at the mouth clears it by 30 dB."""

    async def run() -> int:
        environment = Environment()
        service = AttachablePhoneService()
        _, voice, _ = await _handset_on_a_call(environment, service)
        service.events.clear()

        await voice.dispatch(
            Environment.from_context(voice.context()),
            audio_device.OutputEvent.with_data(
                audio_device.AudioOutputData(
                    audio=b"local speech", media_type="audio/pcm", sample_rate_hz=16_000, channels=1
                )
            ),
        )
        await _wait_until(lambda: bool(_uplinks(service)))

        return len(_uplinks(service))

    assert asyncio.run(run()) == 1


def test_a_ringing_phone_is_heard_nearby_and_not_across_the_room() -> None:
    """A ring is a sound something makes at a place, not an announcement to the whole environment.

    The ringer declares its level and the phone says where it is; the environment does the rest. Before
    this, a ring reached every participant at any distance — the same "reaches everyone regardless
    of geometry" shape that let one handset's earpiece arrive at another's mouthpiece, just off the
    uplink path. Neither was ever a leak: without positions, sound crossed over unconditionally.
    Standing close enough to overhear a call is what proximity does, and the model now says so.
    """

    async def run() -> tuple[int, int]:
        environment = Environment()
        service = AttachablePhoneService()
        here = space.Position(x=0.0, y=0.0)
        phone = phone_device.Phone(
            service=service,
            placement=space.Placement(position=here),
        )
        nearby = Ears()
        across_the_room = Ears()

        _ = await hsm.started(environment, phone, typing.cast(hsm.Model, phone_device.Phone.model))
        _ = await hsm.started(environment, nearby, Ears.model, hsm.Config(id="nearby"))
        _ = await hsm.started(environment, across_the_room, Ears.model, hsm.Config(id="across-the-room"))
        environment.join(nearby, placement=space.Placement(position=space.Position(x=2.0, y=0.0), threshold_db=20.0))
        environment.join(
            across_the_room,
            placement=space.Placement(position=space.Position(x=5_000.0, y=0.0), threshold_db=20.0),
        )
        await _wait_until(lambda: device_firmware(phone) is not None)
        nearby.heard.clear()
        across_the_room.heard.clear()

        await service.receive(
            phone.context(),
            phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-1")),
        )
        await _wait_until(lambda: bool(nearby.heard))

        return len(nearby.heard), len(across_the_room.heard)

    heard_nearby, heard_across_the_room = asyncio.run(run())

    assert heard_nearby == 1
    assert heard_across_the_room == 0


def test_stopping_a_phone_releases_the_service_it_acquired() -> None:
    """The phone acquires its service at bring-up, so the phone releases it at teardown.

    Held past teardown, the service keeps a strong reference to firmware that has stopped, which
    is what stops a replacement phone from ever attaching to the same service.
    """

    async def run() -> tuple[bool, bool]:
        environment = Environment()
        service = AttachablePhoneService()
        phone = phone_device.Phone(service=service)

        _ = await hsm.started(environment, phone, typing.cast(hsm.Model, phone_device.Phone.model))
        await _wait_until(lambda: service.target is not None)
        attached = service.target is not None

        await phone.stop(environment)

        return attached, service.target is None

    attached, released = asyncio.run(run())

    assert attached
    assert released


def test_stopping_a_phone_twice_still_releases_once() -> None:
    """Bot activation cleanup stops the whole configured set, started or not."""

    async def run() -> bool:
        environment = Environment()
        service = AttachablePhoneService()
        phone = phone_device.Phone(service=service)

        _ = await hsm.started(environment, phone, typing.cast(hsm.Model, phone_device.Phone.model))
        await _wait_until(lambda: service.target is not None)

        await phone.stop(environment)
        await phone.stop(environment)

        return service.target is None

    assert asyncio.run(run())

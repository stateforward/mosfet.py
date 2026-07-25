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

from bot.world import SoundEvent, World
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

    async def attach(self, world: World, target: hsm.Instance) -> None:
        assert target.context().value(hsm.Keys.Instances) is world.value(hsm.Keys.Instances)
        self.target = target

    async def detach(self, world: World, target: hsm.Instance) -> None:
        assert target.context().value(hsm.Keys.Instances) is world.value(hsm.Keys.Instances)
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
    await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id=call_id)))
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
        assert operation_names(phone) == [phone_device.DialEvent.name]

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
                phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=transfer_target)
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
        assert service.target is device_firmware(phone)
        await service.receive(phone.context(), phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _wait_until(lambda: device_firmware(phone) is not None and device_firmware(phone).state() == "/Phone/ringing")

        assert service.events[-1].name == phone_device.RingingEvent.name

    asyncio.run(run())

def test_phone_service_events_enter_through_attached_service_target() -> None:
    async def run() -> None:
        phone = phone_device.Phone()

        _ = await hsm.started(None, phone, phone.model)
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
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
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
    assert "phone.service.call_failed" in transitions["/Phone/dialing"]
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
        target = phone_device.TransferTarget(kind="address", value="sip:helpdesk@example.com")
        _ = await hsm.started(None, phone, phone.model)
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(call_id="call-123", target=target)),
        )

        assert device_firmware(phone).state() == "/Phone/dialing"
        assert phone_current_call_id(firmware) == "call-123"
        assert _event_names(firmware.event_recorder()) == [phone_device.ServiceDialRequestedEvent.name]

        await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-stale")))

        assert device_firmware(phone).state() == "/Phone/dialing"
        assert _event_names(firmware.event_recorder()) == [phone_device.ServiceDialRequestedEvent.name]

        await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-123")))

        assert device_firmware(phone).state() == "/Phone/answered/media_connecting"
        assert phone_current_call_id(firmware) == "call-123"
        assert _event_names(firmware.event_recorder()) == [
            phone_device.ServiceDialRequestedEvent.name,
            phone_device.AnsweredEvent.name,
        ]

    asyncio.run(run())

def test_phone_broadcasts_committed_ringing_observation_in_current_world() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]], str]:
        metadata = {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}
        world = World()
        phone = phone_device.Phone()
        inside = PhoneObservationRecorder()
        outside = PhoneObservationRecorder()

        _ = await hsm.started(world, phone, phone.model)
        _ = await hsm.started(world, inside, inside.model, hsm.Config(id="inside"))
        _ = await hsm.started(None, outside, outside.model, hsm.Config(id="outside"))
        phone_id = hsm.id(phone)

        await _emit_service_event(
            phone,
            dataclasses.replace(
                phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
                metadata=metadata,
            ),
        )
        await _wait_until(lambda: bool(inside.events))

        return inside.events, outside.events, phone_id

    inside_events, outside_events, phone_id = asyncio.run(run())

    assert len(inside_events) == 1
    assert inside_events[0].name == "world.sound"
    assert inside_events[0].source == phone_id
    assert inside_events[0].target == "inside"
    sound = inside_events[0].data
    assert getattr(sound, "kind", None) == "phone.ringing"
    assert getattr(sound, "media_type", None) == "audio/wav"
    assert getattr(sound, "sample_rate_hz", None) == 16_000
    assert getattr(sound, "channels", None) == 1
    assert getattr(sound, "audio", b"").startswith(b"RIFF")
    assert getattr(sound, "audio", b"") == phone_device.RING_SOUND_WAV
    assert inside_events[0].id == "call-123"
    assert inside_events[0].metadata.get("traceparent") == "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"
    assert outside_events == []

def test_phone_committed_observations_have_priority_over_queued_external_events() -> None:
    async def run() -> tuple[str, list[str]]:
        phone = phone_device.Phone()

        _ = await hsm.started(None, phone, phone.model)
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)
        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")))

        connected = asyncio.ensure_future(
            _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-123")))
        )
        await asyncio.sleep(0)
        hang_up = asyncio.ensure_future(
            phone.dispatch(
                phone.context(),
                phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData(call_id="call-123")),
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

        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")))
        await _emit_service_event(
            phone,
            phone_device.CallConnectedEvent.with_data(invalid_value(phone_device.CallConnectedData, phone_device.AnswerCallData(call_id="call-123"))),
        )

        assert device_firmware(phone).state() == "/Phone/answering"
        assert _event_names(firmware.event_recorder()) == [phone_device.RingingEvent.name, phone_device.ServiceAnswerRequestedEvent.name]

        await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-123")))
        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")))
        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=transfer_target)
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
                phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
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

        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")))
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

def test_phone_firmware_handles_decline_and_hang_up_commands_for_current_call() -> None:
    async def run() -> None:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-stale")))
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")))
        await phone.dispatch(phone.context(), phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData(call_id="call-stale")))

        assert device_firmware(phone).state() == "/Phone/answering"
        assert phone_current_call_id(firmware) == "call-123"
        assert _event_names(firmware.event_recorder()) == [phone_device.RingingEvent.name, phone_device.ServiceAnswerRequestedEvent.name]

        await phone.dispatch(phone.context(), phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData(call_id="call-123")))

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
        await phone.dispatch(phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData(call_id="call-stale")))

        assert device_firmware(phone).state() == "/Phone/answered/media_connecting"
        assert phone_current_call_id(firmware) == "call-456"

        await phone.dispatch(phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData(call_id="call-456")))

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

def test_phone_service_audio_routes_through_speaker_to_world_observers() -> None:
    async def run() -> tuple[tuple[hsm.Event[typing.Any], ...], list[str], str]:
        metadata = {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}
        world = World()
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
        _ = await hsm.started(world, phone, phone.model)
        _ = await hsm.started(world, speaker, speaker.model, hsm.Config(id="phone-speaker"))
        _ = await hsm.started(world, observer, observer.model, hsm.Config(id="observer"))
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
        assert ring_data.call_id == "call-123"
        assert observer.events[0].id == "call-123"
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
        world = World()
        phone = phone_device.Phone()
        observer = PhoneObservationRecorder()
        audio = phone_device.ServiceAudioData(
            call_id="call-123",
            audio=b"playback-audio",
            media_type="audio/pcm",
            sample_rate_hz=48_000,
            channels=1,
        )

        _ = await hsm.started(world, phone, phone.model)
        _ = await hsm.started(world, observer, observer.model, hsm.Config(id="observer"))
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
        world = World()
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
        _ = await hsm.started(world, phone, phone.model)
        _ = await hsm.started(world, speaker, speaker.model, hsm.Config(id="phone-speaker"))
        _ = await hsm.started(world, observer, observer.model, hsm.Config(id="observer"))
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _answer_call(phone, "call-123")
        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")))
        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=transfer_target)
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
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")))
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")))

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

def test_phone_dialing_timeout_commits_failed_hangup_and_rejects_late_connect() -> None:
    async def run() -> None:
        phone = phone_device.Phone(answer_timeout=datetime.timedelta(milliseconds=1))
        target = phone_device.TransferTarget(kind="address", value="sip:helpdesk@example.com")
        _ = await hsm.started(None, phone, phone.model)
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(call_id="call-123", target=target)),
        )
        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(call_id="call-123", target=target)),
        )

        assert device_firmware(phone).state() == "/Phone/dialing"
        assert _event_names(firmware.event_recorder()) == [phone_device.ServiceDialRequestedEvent.name]

        await _wait_until(lambda: device_firmware(phone) is not None and device_firmware(phone).state() == "/Phone/hung_up")

        assert phone_current_call_id(firmware) is None
        assert "call-123" in phone_closed_call_ids(firmware)
        assert _event_names(firmware.event_recorder()) == [
            phone_device.ServiceDialRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

        await _emit_service_event(phone, phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-123")))

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert _event_names(firmware.event_recorder()) == [
            phone_device.ServiceDialRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(call_id="call-123", target=target)),
        )

        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert _event_names(firmware.event_recorder()) == [
            phone_device.ServiceDialRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

    asyncio.run(run())

def test_phone_call_failed_ends_current_call_in_each_active_phase() -> None:
    async def started_phone() -> tuple[phone_device.Phone, phone_device.PhoneFirmware]:
        phone = phone_device.Phone()
        _ = await hsm.started(None, phone, phone.model)
        assert device_firmware(phone) is not None
        return phone, _phone_firmware(phone)

    async def run() -> None:
        phone, firmware = await started_phone()
        target = phone_device.TransferTarget(kind="address", value="sip:helpdesk@example.com")
        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(call_id="dialing-call", target=target)),
        )
        await _emit_service_event(
            phone,
            phone_device.CallFailedEvent.with_data(phone_device.CallFailedData(call_id="dialing-call", failure_kind="signaling_failed")),
        )

        assert device_firmware(phone) is not None
        assert device_firmware(phone).state() == "/Phone/hung_up"
        assert phone_current_call_id(firmware) is None
        assert "dialing-call" in phone_closed_call_ids(firmware)
        assert _event_names(firmware.event_recorder()) == [phone_device.ServiceDialRequestedEvent.name, phone_device.HungUpEvent.name]

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
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="answering-call")))
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
                phone_device.TransferCallData(call_id="transfer-call", transfer_id="transfer-123", target=transfer_target)
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
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _answer_call(phone, "call-123")
        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(call_id="call-stale", transfer_id="transfer-stale", target=transfer_target)
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
                phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=transfer_target)
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
                phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-456", target=transfer_target)
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
        assert device_firmware(phone) is not None
        firmware = _phone_firmware(phone)

        await _emit_service_event(phone, phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")))
        await _answer_call(phone, "call-123")
        await _emit_service_event(phone, phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")))
        await phone.dispatch(
            phone.context(),
            phone_device.TransferCallEvent.with_data(
                phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-1", target=transfer_target)
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
                phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-2", target=transfer_target)
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
            data={
                "call_id": "call-dict",
                "target": {"kind": "address", "value": "sip:desk@example.com"},
            },
        )
        await phone.dispatch(phone.context(), raw)
        await _wait_until(lambda: phone_current_call_id(_phone_firmware(phone)) == "call-dict")
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
            def dispatch_audio_output_to_world(
                self,
                ctx: hsm.Context,
                data: audio_device.AudioOutputData,
                *,
                metadata: collections.abc.Mapping[str, object] | None = None,
            ) -> collections.abc.Awaitable[None]:
                del ctx, metadata
                self.elevated.append(data)
                return asyncio.ensure_future(asyncio.sleep(0))

        class TrackingService:
            events: list[hsm.Event[typing.Any]]

            def __init__(self) -> None:
                self.events = []

            async def attach(self, world: World, target: hsm.Instance) -> None:
                del world, target

            async def detach(self, world: World, target: hsm.Instance) -> None:
                del world, target

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
        def dispatch_audio_output_to_world(
            self,
            ctx: hsm.Context,
            data: audio_device.AudioOutputData,
            *,
            metadata: collections.abc.Mapping[str, object] | None = None,
        ) -> collections.abc.Awaitable[None]:
            del ctx, metadata
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

        async def attach(self, world: World, target: hsm.Instance) -> None:
            del world, target

        async def detach(self, world: World, target: hsm.Instance) -> None:
            del world, target

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

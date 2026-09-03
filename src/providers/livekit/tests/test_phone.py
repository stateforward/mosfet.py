from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import logging
import typing

import hsm
import bot
import pytest
from livekit import rtc

import bot.providers.livekit.phone as phone_module
from bot.devices import audio as audio_device
from bot.devices import phone as phone_device
from bot.providers.livekit import signaling

from bot.providers.livekit import (
    MediaSnapshot,
    ServiceCallFailedEvent,
    ServiceIncomingCallEvent,
    ServiceMediaReadyEvent,
    ServiceRemoteHangUpEvent,
    ServiceTransferCompletedEvent,
    ServiceTransferFailedEvent,
    Phone as LiveKitPhone,
    PhoneService,
    PhoneServiceError,
)
from bot.providers.livekit.room_audio import (
    AudioStreamFactory,
    LocalAudioTrackFactory,
    RoomHandle,
)
from bot.environment import Environment
from tests.hsm_instance_state import device_firmware, phone_display
from tests.livekit_room_fakes import (
    FakeLocalParticipant,
    FakeRemoteParticipant,
    FakeRemoteTrack,
    FakeRoom,
    FakeSfu,
    fake_local_track_factory,
    fake_pcm_stream,
    rtc_pcm_frame,
)

_FORWARDED_PHONE_EVENT_NAMES = frozenset(
    {
        phone_device.CallConnectedEvent.name,
        phone_device.CallFailedEvent.name,
        phone_device.ServiceDialFailedEvent.name,
        phone_device.IncomingCallEvent.name,
        phone_device.ServiceMediaReadyEvent.name,
        phone_device.ServiceAudioReceivedEvent.name,
        phone_device.RemoteHangUpEvent.name,
        phone_device.TransferAcceptedEvent.name,
        phone_device.ServiceTransferCompletedEvent.name,
        phone_device.ServiceTransferFailedEvent.name,
    }
)


ALICE_NUMBER = "5550141"
DIAL_NUMBER = "5550142"
"""The number the tests dial. Fictional 555-01xx, so nothing here resembles a real subscriber."""

ABSENT_NUMBER = "5550143"
"""A number whose participant is not in the room: nobody home."""

UNLISTED_NUMBER = "5550199"
"""A number used with an optional alias plan that deliberately omits it (plan miss)."""

# On this provider the participant identity is the phone number (normalized digits).
ALICE_IDENTITY = ALICE_NUMBER
"""The phone under test: LiveKit identity equals its line number."""

BOB_IDENTITY = DIAL_NUMBER
"""Far-end line identity: dialing DIAL_NUMBER addresses this participant."""

CALLER_IDENTITY = "5550100"
"""Inbound far end that dials this phone; answer/decline/bye go back to this identity."""

ALIAS_IDENTITY = "alias-bob"
"""Optional MappingDialPlan target: number remapped away from identity=number for alias tests."""


async def await_value[T](value: collections.abc.Awaitable[T]) -> T:
    return await value


@dataclasses.dataclass
class FakePhoneService(PhoneService):
    dial_requests: list[phone_device.DialData]
    answer_requests: list[phone_device.AnswerRequestData]
    decline_requests: list[phone_device.DeclineRequestData]
    hang_up_requests: list[phone_device.HangUpRequestData]
    transfer_requests: list[phone_device.TransferRequestData]
    blocked_operations: frozenset[str]
    failed_operations: frozenset[str]
    fail_answer: bool

    def __init__(
        self,
        *,
        operation_timeout: datetime.timedelta | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        room: RoomHandle | None = None,
        stream_factory: AudioStreamFactory | None = None,
        local_track_factory: LocalAudioTrackFactory | None = None,
        blocked_operations: frozenset[str] | None = None,
        failed_operations: frozenset[str] | None = None,
        fail_answer: bool = False,
    ) -> None:
        super().__init__(
            operation_timeout=operation_timeout or datetime.timedelta(seconds=1),
            loop=loop,
            room=room,
            stream_factory=stream_factory,
            local_track_factory=local_track_factory,
        )
        self.dial_requests = []
        self.answer_requests = []
        self.decline_requests = []
        self.hang_up_requests = []
        self.transfer_requests = []
        self.blocked_operations = blocked_operations if blocked_operations is not None else frozenset()
        self.failed_operations = failed_operations if failed_operations is not None else frozenset()
        self.fail_answer = fail_answer

    async def _block_if_requested(self, operation: str) -> None:
        if operation in self.blocked_operations:
            _ = await asyncio.Event().wait()

    def _fail_if_requested(self, operation: str) -> None:
        if operation == "answer" and self.fail_answer:
            raise PhoneServiceError("answer failed", failure_kind="signaling_failed")
        if operation in self.failed_operations:
            raise PhoneServiceError(f"{operation} failed", failure_kind="signaling_failed")

    @typing.override
    async def dial(self, request: phone_device.DialData) -> None:
        self.dial_requests.append(request)
        self._fail_if_requested("dial")
        await self._block_if_requested("dial")
        # Setup delivered: the far end is ringing. Whether it becomes a call is theirs to say.

    @typing.override
    async def answer_call(self, request: phone_device.AnswerRequestData) -> None:
        self.answer_requests.append(request)
        self._fail_if_requested("answer")
        await self._block_if_requested("answer")

    @typing.override
    async def decline_call(self, request: phone_device.DeclineRequestData) -> None:
        self.decline_requests.append(request)
        self._fail_if_requested("decline")
        await self._block_if_requested("decline")

    @typing.override
    async def hang_up_call(self, request: phone_device.HangUpRequestData) -> None:
        self.hang_up_requests.append(request)
        self._fail_if_requested("hang_up")
        await self._block_if_requested("hang_up")

    @typing.override
    async def transfer_call(self, request: phone_device.TransferRequestData) -> None:
        self.transfer_requests.append(request)
        self._fail_if_requested("transfer")
        await self._block_if_requested("transfer")


class RecordingPhoneEventTarget(hsm.Instance):
    target: hsm.Instance
    events: list[hsm.Event[typing.Any]]

    def __init__(self, *, target: hsm.Instance, events: list[hsm.Event[typing.Any]]) -> None:
        super().__init__()
        self.target = target
        self.events = events

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[None]:
        if event.name in _FORWARDED_PHONE_EVENT_NAMES:
            self.events.append(event)
        return self.target.dispatch(ctx, event)

    model: typing.ClassVar[hsm.Model] = bot.define(
        "RecordingPhoneEventTarget",
        hsm.initial(hsm.target("active")),
        hsm.state("active"),
    )


@dataclasses.dataclass
class RecordingPhoneService:
    service: PhoneService
    events: list[hsm.Event[typing.Any]] = dataclasses.field(default_factory=list)
    forwarded_events: list[hsm.Event[typing.Any]] | None = None
    forwarding_target: RecordingPhoneEventTarget | None = None

    async def attach(self, environment: Environment, target: hsm.Instance) -> None:
        if self.forwarded_events is None:
            await self.service.attach(environment, target)
            return
        forwarding_target = RecordingPhoneEventTarget(target=target, events=self.forwarded_events)
        _ = await bot.started(environment, forwarding_target, forwarding_target.model)
        self.forwarding_target = forwarding_target
        await self.service.attach(environment, forwarding_target)

    async def detach(self, environment: Environment, target: hsm.Instance) -> None:
        attached_target: hsm.Instance = self.forwarding_target if self.forwarding_target is not None else target
        await self.service.detach(environment, attached_target)
        self.forwarding_target = None

    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        self.events.append(event)
        published_event = dataclasses.replace(event, target=hsm.id(self.service))
        if self.forwarding_target is None:
            self.service.publish(ctx, published_event)
            return
        self.service.publish(ctx, dataclasses.replace(published_event, source=hsm.id(self.forwarding_target)))


async def _start_livekit_phone_service(
    *,
    service: FakePhoneService | None = None,
    operation_timeout: datetime.timedelta | None = None,
    forwarded_events: list[hsm.Event[typing.Any]] | None = None,
) -> tuple[phone_device.Phone, FakePhoneService, RecordingPhoneService]:
    resolved = (
        service
        if service is not None
        else FakePhoneService(
            operation_timeout=operation_timeout or datetime.timedelta(seconds=1),
        )
    )
    if operation_timeout is not None:
        resolved._operation_timeout = operation_timeout
        # Call setup gets the same budget in tests: both are "how long before the provider gives up".
        resolved._setup_timeout = operation_timeout
    recording_service = RecordingPhoneService(resolved, forwarded_events=forwarded_events)
    phone = phone_device.Phone(service=recording_service)
    _ = await bot.started(None, phone, typing.cast(hsm.Model, phone.model))
    await _wait_until(lambda: resolved.state() == "/PhoneService/ready")
    return phone, resolved, recording_service


def _endpoint_in_room(sfu: FakeSfu, identity: str) -> FakeLocalParticipant:
    """Another phone already in the room: it answers to ``identity`` and acks the four methods.

    Enough of a far end for the caller's half of a call to be exercised on its own. A real one is
    a second :class:`PhoneService`, which is what the two-phone test wires up.
    """

    endpoint = FakeLocalParticipant(identity=identity, sfu=sfu)
    for method in signaling.Methods:
        _ = endpoint.register_rpc_method(method, lambda data: str(getattr(data, "payload", "")))
    return endpoint


async def _start_phone_on_room(
    identity: str,
    sfu: FakeSfu,
    *,
    setup_timeout: datetime.timedelta = datetime.timedelta(seconds=1),
    dial_plan: signaling.DialPlan | None = None,
    forwarded_events: list[hsm.Event[typing.Any]] | None = None,
    connect_room: bool = True,
) -> tuple[phone_device.Phone, PhoneService, FakeRoom, RecordingPhoneService]:
    """A real PhoneService answering to ``identity`` on ``sfu``, connected, call-setup wire live.

    Default ``dial_plan=None``: setup is addressed to the dialled number (identity = number).
    Pass a MappingDialPlan only when testing optional alias remaps.

    ``connect_room=False`` leaves the handset plugged into nothing: it has a room configured and
    no line on it yet, which is the only way to hold a call and its media apart in time.
    """

    room = FakeRoom(local_participant=FakeLocalParticipant(identity=identity, sfu=sfu))
    service = PhoneService(
        operation_timeout=datetime.timedelta(seconds=1),
        setup_timeout=setup_timeout,
        room=room,
        stream_factory=fake_pcm_stream(rtc_pcm_frame(b"\x01\x00")),
        local_track_factory=fake_local_track_factory,
        dial_plan=dial_plan,
    )
    recording_service = RecordingPhoneService(service, forwarded_events=forwarded_events)
    phone = phone_device.Phone(service=recording_service)
    _ = await bot.started(None, phone, typing.cast(hsm.Model, phone.model))
    await _wait_until(lambda: service.state() == "/PhoneService/ready")
    if connect_room:
        await service.connect_room(url="wss://livekit.example.com", token="token")
        await _wait_until(lambda: signaling.SetupMethod in room.local_participant.rpc_handlers)
    return phone, service, room, recording_service


async def _start_signalling_livekit_phone(
    *,
    setup_timeout: datetime.timedelta = datetime.timedelta(seconds=1),
    forwarded_events: list[hsm.Event[typing.Any]] | None = None,
) -> tuple[phone_device.Phone, PhoneService, FakeRoom, RecordingPhoneService]:
    """One phone under test, in a room that already holds the endpoints these tests talk to.

    Both are there because a message only reaches an endpoint that is: the line the dialled number
    is registered against, and the caller that dials this phone and their answer, decline, and bye
    go back to.
    """

    sfu = FakeSfu()
    started = await _start_phone_on_room(
        ALICE_IDENTITY,
        sfu,
        setup_timeout=setup_timeout,
        forwarded_events=forwarded_events,
    )
    _ = _endpoint_in_room(sfu, BOB_IDENTITY)
    _ = _endpoint_in_room(sfu, CALLER_IDENTITY)
    return started


def _dialed_call_id(room: FakeRoom) -> str:
    """The call id the caller minted, read off the setup message it actually sent."""

    setup = room.local_participant.rpc_calls[0]
    assert setup.method == signaling.SetupMethod
    return signaling.MessageData.model_validate_json(setup.payload).call_id


async def _wait_until(predicate: collections.abc.Callable[[], bool], *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("Timed out waiting for LiveKit phone service condition.")


def _phone_events(recording_service: RecordingPhoneService) -> tuple[hsm.Event[typing.Any], ...]:
    return tuple(recording_service.events)


def _is_answered(phone: phone_device.Phone) -> bool:
    firmware = device_firmware(phone)
    state = None if firmware is None else firmware.state()
    return isinstance(state, str) and state.startswith("/Phone/answered")


async def _mark_media_ready(phone: phone_device.Phone, service: PhoneService, call_id: str = "call-123") -> None:
    await service.dispatch(
        service.context(), ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id=call_id))
    )
    await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/answered/media_ready")


def _require_firmware(phone: phone_device.Phone) -> hsm.Instance:
    firmware = device_firmware(phone)
    assert firmware is not None
    return firmware


async def _assert_stale_provider_observations_are_consumed(
    service: PhoneService,
    recording_service: RecordingPhoneService,
) -> None:
    target = phone_device.TransferTarget(kind="address", value="sip:stale@example.com")
    committed_events = _phone_events(recording_service)
    committed_forwarded_events = (
        tuple(recording_service.forwarded_events) if recording_service.forwarded_events is not None else None
    )

    await service.dispatch(
        service.context(),
        ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="wrong-call")),
    )
    await service.dispatch(
        service.context(),
        ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="wrong-call")),
    )
    await service.dispatch(
        service.context(),
        ServiceRemoteHangUpEvent.with_data(phone_device.RemoteHangUpData(call_id="wrong-call")),
    )
    await service.dispatch(
        service.context(),
        ServiceCallFailedEvent.with_data(
            phone_device.CallFailedData(call_id="wrong-call", failure_kind="signaling_failed")
        ),
    )
    await service.dispatch(
        service.context(),
        ServiceTransferCompletedEvent.with_data(
            phone_device.TransferCompletedData(call_id="wrong-call", transfer_id="stale-transfer", target=target)
        ),
    )
    await service.dispatch(
        service.context(),
        ServiceTransferFailedEvent.with_data(
            phone_device.TransferFailedData(
                call_id="wrong-call",
                transfer_id="stale-transfer",
                target=target,
                failure_kind="transfer_rejected",
            )
        ),
    )
    service.publish(
        service.context(),
        phone_device.HungUpEvent.with_data(phone_device.HungUpData(call_id="wrong-call", outcome="remote_hang_up")),
    )
    service.publish(
        service.context(),
        phone_device.CallTransferCompletedEvent.with_data(
            phone_device.TransferData(call_id="wrong-call", transfer_id="stale-transfer", target=target)
        ),
    )
    service.publish(
        service.context(),
        phone_device.CallTransferFailedEvent.with_data(
            phone_device.CallTransferFailedData(
                call_id="wrong-call",
                transfer_id="stale-transfer",
                target=target,
                failure_kind="transfer_rejected",
            )
        ),
    )

    await asyncio.sleep(0)

    assert _phone_events(recording_service) == committed_events
    if committed_forwarded_events is not None:
        assert tuple(recording_service.forwarded_events or ()) == committed_forwarded_events


def test_livekit_phone_service_model_tracks_provider_request_operations() -> None:
    model = PhoneService.model

    assert model.qualified_name == "/PhoneService"
    assert model.initial == "/PhoneService/.initial"
    assert "/PhoneService/unconnected" in model.members
    assert "/PhoneService/ready" in model.members
    assert "/PhoneService/dialing" in model.members
    assert "/PhoneService/answering" in model.members
    assert "/PhoneService/declining" in model.members
    assert "/PhoneService/hanging_up" in model.members
    assert "/PhoneService/transferring" in model.members
    assert phone_device.ServiceDialRequestedEvent.name in model.events
    assert phone_device.ServiceAnswerRequestedEvent.name in model.events
    assert phone_device.ServiceTransferRequestedEvent.name in model.events
    assert ServiceIncomingCallEvent.name in model.events
    assert ServiceMediaReadyEvent.name in model.events
    assert ServiceTransferFailedEvent.name in model.events
    assert phone_module._RemoteAudioReceivedEvent.name in model.events


def test_livekit_phone_service_instance_does_not_store_pending_operation_correlation() -> None:
    state_fields = set(vars(PhoneService()))

    assert "active_call_id" not in state_fields
    assert "active_transfer_id" not in state_fields
    assert "active_transfer_target" not in state_fields
    assert "_attach_waiters" not in state_fields


def test_phone_service_default_call_control_is_unavailable() -> None:
    async def run() -> None:
        service = PhoneService()
        target = phone_device.TransferTarget(kind="address", value="sip:helpdesk@example.com")
        for coro in (
            service.dial(phone_device.DialData(number=DIAL_NUMBER)),
            service.answer_call(phone_device.AnswerRequestData(call_id="call-123")),
            service.decline_call(phone_device.DeclineRequestData(call_id="call-123")),
            service.hang_up_call(phone_device.HangUpRequestData(call_id="call-123")),
            service.transfer_call(
                phone_device.TransferRequestData(call_id="call-123", transfer_id="transfer-123", target=target)
            ),
        ):
            try:
                await coro
            except PhoneServiceError as error:
                assert error.failure_kind == "provider_unavailable"
            else:
                raise AssertionError("PhoneService should report missing LiveKit call control.")

    asyncio.run(run())


def test_livekit_phone_dials_the_number_it_was_given() -> None:
    """A dialled number addresses that participant identity and connects on accept.

    Identity is the number: setup's destination is the normalized digits. The acked setup leaves
    the caller *dialing* — the connect arrives later, from the callee, because answering was
    theirs to decide.
    """

    async def run() -> None:
        phone, service, room, recording_service = await _start_signalling_livekit_phone()
        participant = room.local_participant

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _wait_until(lambda: service.state() == "/PhoneService/dialing/ringing")

        setup = participant.rpc_calls[0]
        assert setup.destination_identity == DIAL_NUMBER
        assert setup.destination_identity == BOB_IDENTITY
        assert setup.method == signaling.SetupMethod
        call_id = _dialed_call_id(room)
        assert call_id.startswith("livekit:")
        # Acked setup means ringing, not connected.
        assert _require_firmware(phone).state() == "/Phone/dialing"

        _ = participant.invoke(signaling.AcceptMethod, caller_identity=BOB_IDENTITY, call_id=call_id)
        await _wait_until(lambda: _is_answered(phone))

        answered = [
            event for event in _phone_events(recording_service) if event.name == phone_device.AnsweredEvent.name
        ]
        # Both phones name the call the caller minted; nobody invented a second handle.
        assert [event.data for event in answered] == [phone_device.CallData(call_id=call_id)]

    asyncio.run(run())


def test_livekit_phone_reports_a_declined_dial_as_a_refusal_not_an_absence() -> None:
    """Somebody answered and said no. Reporting that as remote_unavailable would be a lie."""

    async def run() -> None:
        forwarded: list[hsm.Event[typing.Any]] = []
        phone, service, room, _recording = await _start_signalling_livekit_phone(forwarded_events=forwarded)
        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _wait_until(lambda: service.state() == "/PhoneService/dialing/ringing")

        _ = room.local_participant.invoke(
            signaling.DeclineMethod,
            caller_identity=BOB_IDENTITY,
            call_id=_dialed_call_id(room),
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        failures = [event for event in forwarded if event.name == phone_device.ServiceDialFailedEvent.name]
        assert len(failures) == 1
        assert failures[0].data == phone_device.DialFailedData(failure_kind="call_declined")

    asyncio.run(run())


def test_a_number_reaches_the_same_phone_however_it_was_written_down() -> None:
    """Every written form normalizes to the same digit identity on the wire.

    DialData strips punctuation before setup is addressed; without a dial plan the destination
    identity is those digits. Hyphens the STT invented must not look like a wrong number.
    """

    async def run() -> None:
        for written in ("5550142", "555-0142", "555 0142", "(555) 0142", "555.0142"):
            sfu = FakeSfu()
            phone, service, room, _recording = await _start_phone_on_room(
                ALICE_IDENTITY,
                sfu,
                dial_plan=None,
            )
            _ = _endpoint_in_room(sfu, BOB_IDENTITY)

            await phone.dispatch(
                phone.context(),
                phone_device.DialEvent.with_data(phone_device.DialData(number=written)),
            )
            await _wait_until(lambda: service.state() == "/PhoneService/dialing/ringing")

            assert room.local_participant.rpc_calls[0].destination_identity == DIAL_NUMBER, written

    asyncio.run(run())


def test_livekit_phone_reports_a_line_that_is_not_there_as_remote_unavailable() -> None:
    """Setup addresses the dialled number; the room has no such participant. Nobody home.

    RECIPIENT_NOT_FOUND from the SFU is the far end being absent — a fact the room establishes.
    """

    async def run() -> None:
        forwarded: list[hsm.Event[typing.Any]] = []
        phone, _service, room, _recording = await _start_signalling_livekit_phone(forwarded_events=forwarded)

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=ABSENT_NUMBER)),
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert room.local_participant.rpc_calls[0].destination_identity == ABSENT_NUMBER
        failures = [event for event in forwarded if event.name == phone_device.ServiceDialFailedEvent.name]
        assert failures[-1].data == phone_device.DialFailedData(failure_kind="remote_unavailable")

    asyncio.run(run())


def test_an_optional_dial_plan_miss_is_a_wrong_number_with_nothing_on_the_wire() -> None:
    """When a MappingDialPlan is present and omits the number, the exchange answers and setup is not sent."""

    async def run() -> None:
        forwarded: list[hsm.Event[typing.Any]] = []
        sfu = FakeSfu()
        phone, _service, room, _recording = await _start_phone_on_room(
            ALICE_IDENTITY,
            sfu,
            dial_plan=signaling.MappingDialPlan({DIAL_NUMBER: BOB_IDENTITY}),
            forwarded_events=forwarded,
        )
        _ = _endpoint_in_room(sfu, BOB_IDENTITY)

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=UNLISTED_NUMBER)),
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert room.local_participant.rpc_calls == []
        failures = [event for event in forwarded if event.name == phone_device.ServiceDialFailedEvent.name]
        assert failures[-1].data == phone_device.DialFailedData(failure_kind="remote_unavailable")

    asyncio.run(run())


def test_dial_with_no_plan_addresses_the_number_as_identity() -> None:
    """Default operation: dial_plan=None means destination_identity is the dialled number."""

    async def run() -> None:
        sfu = FakeSfu()
        phone, service, room, _recording = await _start_phone_on_room(
            ALICE_IDENTITY,
            sfu,
            dial_plan=None,
        )
        _ = _endpoint_in_room(sfu, BOB_IDENTITY)

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _wait_until(lambda: service.state() == "/PhoneService/dialing/ringing")

        assert room.local_participant.rpc_calls[0].destination_identity == DIAL_NUMBER

    asyncio.run(run())


def test_optional_dial_plan_can_remap_a_number_to_a_different_identity() -> None:
    """MappingDialPlan remains for rare aliases: number → non-number identity."""

    async def run() -> None:
        sfu = FakeSfu()
        phone, service, room, _recording = await _start_phone_on_room(
            ALICE_IDENTITY,
            sfu,
            dial_plan=signaling.MappingDialPlan({DIAL_NUMBER: ALIAS_IDENTITY}),
        )
        _ = _endpoint_in_room(sfu, ALIAS_IDENTITY)

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _wait_until(lambda: service.state() == "/PhoneService/dialing/ringing")

        assert room.local_participant.rpc_calls[0].destination_identity == ALIAS_IDENTITY

    asyncio.run(run())


def test_a_dial_plan_only_registers_numbers_a_handset_could_dial() -> None:
    """The plan is checked against the same rule a dialled number is, so the two cannot disagree."""

    plan = signaling.MappingDialPlan({"(555) 555-0142": ALIAS_IDENTITY})

    # Written separators are stripped on the way into the plan exactly as they are on the keypad.
    assert plan.endpoint("5555550142") == ALIAS_IDENTITY
    assert plan.endpoint(UNLISTED_NUMBER) is None
    for not_a_number in ("phone-bot-bob", "", "reception"):
        with pytest.raises(ValueError):
            _ = signaling.MappingDialPlan({not_a_number: ALIAS_IDENTITY})
    with pytest.raises(ValueError):
        _ = signaling.MappingDialPlan({DIAL_NUMBER: ""})


def test_livekit_phone_answer_tells_the_caller_so_their_phone_stops_ringing() -> None:
    async def run() -> None:
        phone, _service, room, _recording = await _start_signalling_livekit_phone()
        participant = room.local_participant

        _ = participant.invoke(signaling.SetupMethod, caller_identity=CALLER_IDENTITY, call_id="livekit:human-1")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _is_answered(phone))

        accepted = participant.rpc_calls[-1]
        assert accepted.destination_identity == CALLER_IDENTITY
        assert accepted.method == signaling.AcceptMethod
        assert signaling.MessageData.model_validate_json(accepted.payload).call_id == "livekit:human-1"

    asyncio.run(run())


def test_livekit_phone_decline_tells_the_caller_they_were_refused() -> None:
    async def run() -> None:
        phone, _service, room, _recording = await _start_signalling_livekit_phone()
        participant = room.local_participant

        _ = participant.invoke(signaling.SetupMethod, caller_identity=CALLER_IDENTITY, call_id="livekit:human-1")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")
        await phone.dispatch(
            phone.context(),
            phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData()),
        )
        await _wait_until(lambda: bool(participant.rpc_calls))

        declined = participant.rpc_calls[-1]
        assert declined.destination_identity == CALLER_IDENTITY
        assert declined.method == signaling.DeclineMethod

    asyncio.run(run())


def test_livekit_phone_hang_up_is_signalled_because_the_bot_stays_in_the_room() -> None:
    """The bug presence could not see: a bot that hangs up does not leave, so only a BYE reports it."""

    async def run() -> None:
        phone, _service, room, _recording = await _start_signalling_livekit_phone()
        participant = room.local_participant

        _ = participant.invoke(signaling.SetupMethod, caller_identity=CALLER_IDENTITY, call_id="livekit:human-1")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _is_answered(phone))
        await phone.dispatch(
            phone.context(),
            phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData()),
        )
        await _wait_until(lambda: any(call.method == signaling.ByeMethod for call in participant.rpc_calls))

        goodbye = participant.rpc_calls[-1]
        assert goodbye.destination_identity == CALLER_IDENTITY
        assert signaling.MessageData.model_validate_json(goodbye.payload).call_id == "livekit:human-1"
        assert room.disconnected is False

    asyncio.run(run())


def test_livekit_phone_bye_from_the_far_end_ends_the_call() -> None:
    async def run() -> None:
        phone, _service, room, _recording = await _start_signalling_livekit_phone()
        participant = room.local_participant

        _ = participant.invoke(signaling.SetupMethod, caller_identity=CALLER_IDENTITY, call_id="livekit:human-1")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _is_answered(phone))

        _ = participant.invoke(signaling.ByeMethod, caller_identity=CALLER_IDENTITY, call_id="livekit:human-1")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

    asyncio.run(run())


def test_a_far_end_that_leaves_the_room_mid_call_is_a_dead_line() -> None:
    """Departure is the only way to notice a peer that crashed rather than hanging up."""

    async def run() -> None:
        phone, _service, room, recording_service = await _start_signalling_livekit_phone()
        participant = room.local_participant

        _ = participant.invoke(signaling.SetupMethod, caller_identity=CALLER_IDENTITY, call_id="livekit:human-1")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _is_answered(phone))

        room.emit("participant_disconnected", FakeRemoteParticipant(identity=CALLER_IDENTITY))
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert any(
            event.name == phone_device.HungUpEvent.name
            and isinstance(event.data, phone_device.HungUpData)
            and event.data.outcome == "remote_hang_up"
            for event in _phone_events(recording_service)
        )

    asyncio.run(run())


def test_a_stranger_leaving_the_room_is_not_a_hang_up() -> None:
    """A departure only means something when it is the far end of this phone's call."""

    async def run() -> None:
        phone, _service, room, _recording = await _start_signalling_livekit_phone()
        participant = room.local_participant

        _ = participant.invoke(signaling.SetupMethod, caller_identity=CALLER_IDENTITY, call_id="livekit:human-1")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _is_answered(phone))

        room.emit("participant_disconnected", FakeRemoteParticipant(identity="observer"))
        await asyncio.sleep(0.01)

        assert _is_answered(phone)

    asyncio.run(run())


def test_a_callee_that_leaves_while_ringing_cannot_answer() -> None:
    async def run() -> None:
        forwarded: list[hsm.Event[typing.Any]] = []
        phone, service, room, _recording = await _start_signalling_livekit_phone(forwarded_events=forwarded)

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _wait_until(lambda: service.state() == "/PhoneService/dialing/ringing")

        room.emit("participant_disconnected", FakeRemoteParticipant(identity=BOB_IDENTITY))
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        failures = [event for event in forwarded if event.name == phone_device.ServiceDialFailedEvent.name]
        assert failures[-1].data == phone_device.DialFailedData(failure_kind="remote_unavailable")

    asyncio.run(run())


def test_livekit_phone_service_default_gateway_emits_provider_unavailable_failure() -> None:
    async def run() -> None:
        service = PhoneService(operation_timeout=datetime.timedelta(seconds=1))
        recording_service = RecordingPhoneService(service)
        phone = phone_device.Phone(service=recording_service)
        _ = await bot.started(None, phone, typing.cast(hsm.Model, phone.model))
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)

        await service.incoming_call(service.context(), phone_device.IncomingCallData(call_id="call-123"))
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(
            lambda: any(
                event.name == phone_device.HungUpEvent.name
                and isinstance(event.data, phone_device.HungUpData)
                and event.data.outcome == "failed"
                for event in _phone_events(recording_service)
            )
        )

        assert any(
            event.name == phone_device.HungUpEvent.name
            and isinstance(event.data, phone_device.HungUpData)
            and event.data.call_id == "call-123"
            and event.data.outcome == "failed"
            for event in _phone_events(recording_service)
        )

    asyncio.run(run())


def test_livekit_phone_service_default_gateway_fails_outbound_dial() -> None:
    async def run() -> None:
        service = PhoneService(operation_timeout=datetime.timedelta(seconds=1))
        recording_service = RecordingPhoneService(service)
        phone = phone_device.Phone(service=recording_service)
        _ = await bot.started(None, phone, typing.cast(hsm.Model, phone.model))
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _wait_until(
            lambda: any(
                event.name == phone_device.NoCallEvent.name
                and isinstance(event.data, phone_device.NoCallData)
                and event.data.reason == "dial_failed"
                for event in _phone_events(recording_service)
            )
        )

        # A dial the gateway cannot place never becomes a call, so there is no call to hang up.
        assert any(
            event.name == phone_device.NoCallEvent.name
            and isinstance(event.data, phone_device.NoCallData)
            and event.data.reason == "dial_failed"
            for event in _phone_events(recording_service)
        )
        assert not any(event.name == phone_device.HungUpEvent.name for event in _phone_events(recording_service))

    asyncio.run(run())


def test_livekit_phone_service_public_exports_include_phone_surface() -> None:
    import bot.providers.livekit as livekit

    assert livekit.PhoneService is PhoneService
    assert livekit.PhoneServiceError is PhoneServiceError
    assert livekit.ServiceIncomingCallEvent is ServiceIncomingCallEvent
    assert livekit.ServiceRemoteHangUpEvent is ServiceRemoteHangUpEvent
    assert livekit.ServiceTransferCompletedEvent is ServiceTransferCompletedEvent
    assert "PhoneService" in livekit.__all__
    assert "PhoneService" in livekit.__all__
    assert "PhoneGateway" not in livekit.__all__
    assert "create_phone_gateway" not in livekit.__all__
    assert not hasattr(livekit, "PhoneGateway")
    assert not hasattr(livekit, "create_phone_gateway")
    assert "PhoneEventDispatcher" not in livekit.__all__
    assert not hasattr(livekit, "PhoneEventDispatcher")
    assert not any(name.startswith("LiveKit") for name in livekit.__all__)
    assert not any(name.startswith("LIVEKIT_") for name in livekit.__all__)


def test_livekit_phone_service_stays_unconnected_without_attachment() -> None:
    async def run() -> None:
        service = FakePhoneService()
        _ = await bot.started(None, service, service.model)
        await service.incoming_call(service.context(), phone_device.IncomingCallData(call_id="call-123"))
        await asyncio.sleep(0)

        assert service.state() == "/PhoneService/unconnected"

    asyncio.run(run())


def test_livekit_phone_service_attach_starts_fresh_service() -> None:
    async def run() -> None:
        environment = Environment()
        service = FakePhoneService()
        target = hsm.Instance()
        target_model = bot.define(
            "PhoneServiceTarget",
            hsm.initial(hsm.target("active")),
            hsm.state("active"),
        )

        _ = await bot.started(environment, target, target_model, hsm.Config(id="phone-service-target"))
        await service.attach(environment, target)

        assert service.state() == "/PhoneService/ready"

    asyncio.run(run())


def test_livekit_phone_service_is_connected_by_phone_service_di() -> None:
    async def run() -> None:
        service = FakePhoneService()
        recording_service = RecordingPhoneService(service)
        phone = phone_device.Phone(service=recording_service)

        _ = await bot.started(None, phone, typing.cast(hsm.Model, phone.model))
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)
        await service.incoming_call(
            service.context(),
            phone_device.IncomingCallData(call_id="call-123", caller="Front desk"),
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")

        assert _phone_events(recording_service)[-1].name == phone_device.RingingEvent.name

    asyncio.run(run())


def test_livekit_phone_rejects_forged_service_originating_phone_event() -> None:
    async def run() -> None:
        service = FakePhoneService()
        phone = phone_device.Phone(service=service)

        _ = await bot.started(None, phone, typing.cast(hsm.Model, phone.model))
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)
        assert _require_firmware(phone).state() == "/Phone/hung_up"
        forged = dataclasses.replace(
            phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="forged-call")),
            source="not-the-attached-service",
            target=hsm.id(phone),
        )
        await phone.dispatch(phone.context(), forged)
        await asyncio.sleep(0)

        assert _require_firmware(phone).state() == "/Phone/hung_up"

    asyncio.run(run())


def test_livekit_phone_service_ingress_is_topology_not_envelope_admission() -> None:
    """publish dispatches into HSM; envelope target/source are not re-checked at the door.

    Delivery is the gate. Payload-typed transitions fire when data matches; envelope fields
    are for correlation/telemetry, not imperative admission.
    """

    async def run() -> None:
        phone, service, _recording_service = await _start_livekit_phone_service()
        assert device_firmware(phone) is not None

        # Unstamped answer request: typed payload matches → transition runs.
        service.publish(
            service.context(),
            phone_device.ServiceAnswerRequestedEvent.with_data(phone_device.AnswerRequestData(call_id="call-123")),
        )
        await _wait_until(lambda: service.answer_requests == [phone_device.AnswerRequestData(call_id="call-123")])
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        # Mistargeted envelope still delivers via publish→dispatch; HSM does not re-check target.
        service.publish(
            service.context(),
            dataclasses.replace(
                phone_device.ServiceAnswerRequestedEvent.with_data(phone_device.AnswerRequestData(call_id="call-123")),
                source=hsm.id(phone),
                target="not-this-service",
            ),
        )
        await _wait_until(
            lambda: service.answer_requests
            == [
                phone_device.AnswerRequestData(call_id="call-123"),
                phone_device.AnswerRequestData(call_id="call-123"),
            ]
        )
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        # Hung-up without an active request: provider-terminal hang-up guard fails (call correlation).
        service.publish(
            service.context(),
            phone_device.HungUpEvent.with_data(
                phone_device.HungUpData(call_id="no-active-op", outcome="remote_hang_up")
            ),
        )
        await asyncio.sleep(0)
        assert service.state() == "/PhoneService/ready"

        # Transfer terminals without active transfer: correlation guard fails → no state change.
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        service.publish(
            service.context(),
            phone_device.CallTransferCompletedEvent.with_data(
                phone_device.TransferData(call_id="call-123", transfer_id="transfer-123", target=target)
            ),
        )
        service.publish(
            service.context(),
            phone_device.CallTransferFailedEvent.with_data(
                phone_device.CallTransferFailedData(
                    call_id="call-123",
                    transfer_id="transfer-123",
                    target=target,
                    failure_kind="transfer_rejected",
                )
            ),
        )
        await asyncio.sleep(0)
        assert service.state() == "/PhoneService/ready"

    asyncio.run(run())


def test_livekit_phone_service_detaches_from_phone_service_target() -> None:
    async def run() -> None:
        service = FakePhoneService()
        recording_service = RecordingPhoneService(service)
        phone = phone_device.Phone(service=recording_service)

        _ = await bot.started(None, phone, typing.cast(hsm.Model, phone.model))
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)
        assert device_firmware(phone) is not None

        await service.detach(Environment.from_context(phone.context()), _require_firmware(phone))
        await _wait_until(lambda: service.state() == "/PhoneService/unconnected", timeout=0.05)
        await service.incoming_call(service.context(), phone_device.IncomingCallData(call_id="call-123"))
        await asyncio.sleep(0)

        assert _require_firmware(phone).state() == "/Phone/hung_up"

    asyncio.run(run())


def test_livekit_phone_service_keeps_first_phone_attachment_when_second_phone_attaches() -> None:
    async def run() -> None:
        service = FakePhoneService()
        first = phone_device.Phone(service=service)
        second = phone_device.Phone(service=service)

        _ = await bot.started(None, first, typing.cast(hsm.Model, first.model), hsm.Config(id="first-phone"))
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)
        assert device_firmware(first) is not None

        other_target = RecordingPhoneEventTarget(target=_require_firmware(first), events=[])
        _ = await bot.started(first.context(), other_target, other_target.model, hsm.Config(id="other-target"))
        with pytest.raises(PhoneServiceError, match="already attached"):
            await service.attach(Environment.from_context(first.context()), other_target)
        await asyncio.sleep(0)

        assert service.state() == "/PhoneService/ready"

        _ = await bot.started(
            first.context(), second, typing.cast(hsm.Model, second.model), hsm.Config(id="second-phone")
        )
        await _wait_until(lambda: second.state() == "/Device/failed")

        await second.dispatch(
            second.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await asyncio.sleep(0)
        assert second.state() == "/Device/failed"
        assert service.dial_requests == []

        await service.incoming_call(service.context(), phone_device.IncomingCallData(call_id="call-123"))
        await _wait_until(lambda: _require_firmware(first).state() == "/Phone/ringing")
        assert device_firmware(second) is None
        await asyncio.sleep(0)
        assert second.state() == "/Device/failed"

    asyncio.run(run())


def test_livekit_phone_service_conflicting_attach_event_dispatches_rejection() -> None:
    async def run() -> None:
        recorded_events: list[hsm.Event[typing.Any]] = []

        class RecordingPhoneServiceInstance(PhoneService):
            @typing.override
            def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[None]:
                recorded_events.append(event)
                return super().dispatch(ctx, event)

        service = RecordingPhoneServiceInstance()
        environment = Environment()
        terminal_model = bot.define(
            "PhoneServiceTerminal",
            hsm.initial(hsm.target("active")),
            hsm.state("active"),
        )
        first_terminal = hsm.Instance()
        second_terminal = hsm.Instance()
        first_events: list[hsm.Event[typing.Any]] = []
        second_events: list[hsm.Event[typing.Any]] = []
        first_target = RecordingPhoneEventTarget(target=first_terminal, events=first_events)
        second_target = RecordingPhoneEventTarget(target=second_terminal, events=second_events)
        attachment_data = typing.cast(
            collections.abc.Callable[..., object], getattr(phone_module, "_PhoneServiceAttachmentData")
        )
        attached_event = typing.cast(hsm.Event[typing.Any], getattr(phone_module, "_ServiceAttachedEvent"))
        rejected_event = typing.cast(hsm.Event[typing.Any], getattr(phone_module, "_ServiceAttachmentRejectedEvent"))

        _ = await bot.started(environment, first_terminal, terminal_model, hsm.Config(id="first-terminal"))
        _ = await bot.started(environment, second_terminal, terminal_model, hsm.Config(id="second-terminal"))
        _ = await bot.started(environment, first_target, first_target.model, hsm.Config(id="first-target"))
        _ = await bot.started(environment, second_target, second_target.model, hsm.Config(id="second-target"))
        await service.attach(environment, first_target)

        recorded_events.clear()
        await service.dispatch(service.context(), attached_event.with_data(attachment_data(target=second_target)))
        assert [event.name for event in recorded_events] == [attached_event.name, rejected_event.name]

        recorded_events.clear()
        await service.incoming_call(service.context(), phone_device.IncomingCallData(call_id="call-123"))
        await _wait_until(lambda: bool(first_events))

        assert first_events[-1].name == phone_device.IncomingCallEvent.name
        assert second_events == []
        assert service.state() == "/PhoneService/ready"

    asyncio.run(run())


def test_livekit_phone_sugar_builds_core_phone_from_livekit_room_credentials() -> None:
    phone = LiveKitPhone(url="wss://livekit.example.com", token="livekit-jwt", track_name="alice-audio")

    assert isinstance(phone, phone_device.Phone)
    assert type(phone) is LiveKitPhone
    assert not hasattr(phone, "service")
    assert not hasattr(phone, "room_audio")
    assert not hasattr(phone, "url")
    assert not hasattr(phone, "token")


def test_livekit_phone_sugar_requires_url_and_token_together() -> None:
    with pytest.raises(ValueError, match="url and token must both be provided or both omitted"):
        _ = LiveKitPhone(url="wss://livekit.example.com")
    with pytest.raises(ValueError, match="url and token must both be provided or both omitted"):
        _ = LiveKitPhone(token="livekit-jwt")
    with pytest.raises(ValueError, match="track_name must be a non-empty string"):
        _ = LiveKitPhone(url="wss://livekit.example.com", token="livekit-jwt", track_name="")


def test_livekit_phone_service_answers_current_phone_call_through_gateway() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123", caller="Front desk")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: service.answer_requests == [phone_device.AnswerRequestData(call_id="call-123")])
        await _wait_until(lambda: _is_answered(phone))

        assert device_firmware(phone) is not None
        assert _is_answered(phone)
        assert [event.name for event in _phone_events(recording_service)] == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.AnsweredEvent.name,
        ]
        assert _phone_events(recording_service)[-1].data == phone_device.CallData(call_id="call-123")

    asyncio.run(run())


def test_livekit_phone_service_declines_current_phone_call_through_gateway() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData()))
        await _wait_until(lambda: service.decline_requests == [phone_device.DeclineRequestData(call_id="call-123")])
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        assert _require_firmware(phone).state() == "/Phone/hung_up"
        assert [event.name for event in _phone_events(recording_service)][-2:] == [
            phone_device.ServiceDeclineRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

    asyncio.run(run())


def test_livekit_phone_service_hangs_up_current_phone_call_through_gateway() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await phone.dispatch(phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData()))
        await _wait_until(lambda: service.hang_up_requests == [phone_device.HangUpRequestData(call_id="call-123")])
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        assert _require_firmware(phone).state() == "/Phone/hung_up"
        assert [event.name for event in _phone_events(recording_service)][-2:] == [
            phone_device.ServiceHangUpRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

    asyncio.run(run())


def test_dialling_one_phones_number_makes_that_phone_ring() -> None:
    """Two real phones in one room, and one call carried between them over the wire.

    Nothing here rings Bob's phone but Alice dialling his number: the setup Alice sends is what
    Bob's service receives, and the accept Bob sends is what connects Alice. Both halves run as
    one loop, so a break anywhere in dial → ring → answer → connect → bye shows up here.

    Bob rings and stays ringing until Bob's phone is told to answer. Whether to answer is Bob's,
    and no part of this test may make that decision on the topology's behalf.
    """

    async def run() -> None:
        sfu = FakeSfu()
        alice_phone, _alice_service, alice_room, alice_recording = await _start_phone_on_room(ALICE_IDENTITY, sfu)
        bob_phone, _bob_service, _bob_room, bob_recording = await _start_phone_on_room(BOB_IDENTITY, sfu)

        await alice_phone.dispatch(
            alice_phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _wait_until(lambda: _require_firmware(bob_phone).state() == "/Phone/ringing")

        # Bob's phone shows the caller the SFU authenticated, not a name the message claimed.
        ringing = [event for event in _phone_events(bob_recording) if event.name == phone_device.RingingEvent.name]
        assert ringing[-1].data == phone_device.RingingData(caller=ALICE_IDENTITY)
        # Alice is dialling, not connected: acked setup only means the far end is ringing.
        assert _require_firmware(alice_phone).state() == "/Phone/dialing"

        call_id = _dialed_call_id(alice_room)
        await bob_phone.dispatch(
            bob_phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _is_answered(alice_phone))
        await _wait_until(lambda: _is_answered(bob_phone))

        # One call, one name for it, minted by the caller and adopted by the callee.
        for recording in (alice_recording, bob_recording):
            answered = [event for event in _phone_events(recording) if event.name == phone_device.AnsweredEvent.name]
            assert answered[-1].data == phone_device.CallData(call_id=call_id)

        await bob_phone.dispatch(
            bob_phone.context(),
            phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData()),
        )
        await _wait_until(lambda: _require_firmware(bob_phone).state() == "/Phone/hung_up")
        # Bob stays in the room after hanging up, so only the bye can tell Alice the call ended.
        await _wait_until(lambda: _require_firmware(alice_phone).state() == "/Phone/hung_up")

        assert any(
            event.name == phone_device.HungUpEvent.name
            and isinstance(event.data, phone_device.HungUpData)
            and event.data.outcome == "remote_hang_up"
            for event in _phone_events(alice_recording)
        )

    asyncio.run(run())


def test_an_accepted_dial_opens_the_callers_line_too() -> None:
    """A caller whose call was answered can speak on it, not just hear that it connected.

    Answering brings the callee's media up; an accept has to do the same for the caller, or the
    call is one-way by construction. The mouthpiece is live only in ``/Phone/answered/media_ready``
    (``bot/devices/phone/phone.py`` — the ``audio.InputEvent`` transition), so a caller left in
    ``media_connecting`` uplinks nothing however much it tries to talk. That is what a two-bot run
    over a real SFU measured: the caller sat in ``media_connecting`` through fifty-nine speaking
    turns and the callee decoded none of them.
    """

    async def run() -> None:
        sfu = FakeSfu()
        alice_phone, _alice_service, alice_room, alice_recording = await _start_phone_on_room(ALICE_IDENTITY, sfu)
        bob_phone, _bob_service, _bob_room, _bob_recording = await _start_phone_on_room(BOB_IDENTITY, sfu)

        await alice_phone.dispatch(
            alice_phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _wait_until(lambda: _require_firmware(bob_phone).state() == "/Phone/ringing")

        call_id = _dialed_call_id(alice_room)
        await bob_phone.dispatch(
            bob_phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )

        # Both ends of one call, both able to talk on it.
        await _wait_until(lambda: _require_firmware(bob_phone).state() == "/Phone/answered/media_ready")
        await _wait_until(lambda: _require_firmware(alice_phone).state() == "/Phone/answered/media_ready")

        # And the caller says so once — a line does not open twice for one call.
        media_ready = [
            event for event in _phone_events(alice_recording) if event.name == phone_device.MediaReadyEvent.name
        ]
        assert [event.data for event in media_ready] == [phone_device.CallData(call_id=call_id)]

    asyncio.run(run())


def _display_caller_id(attributes: collections.abc.Mapping[str, object]) -> object:
    """Caller ID off a started display snapshot, whatever bring-up renamed its model.

    A Display started standalone snapshots under ``/Device/caller_id``; the same display
    powered as a phone peripheral is redefined ``DeviceDisplay`` (``Device.start``), so the
    key is ``/DeviceDisplay/caller_id``. Match the leaf the way core
    ``tests/devices/phone/test_phone.py`` does rather than pinning one bring-up's prefix.
    """

    for key, value in attributes.items():
        if key == "caller_id" or str(key).endswith("/caller_id"):
            return value
    return "unset"


def test_a_connected_call_names_who_is_on_the_line_on_the_display() -> None:
    """Both ends know who they are talking to, readable off their own display snapshots.

    The callee learns the caller the SFU authenticated; the caller learns whoever the dial plan
    resolved the number to. The provider stamps it on the connect payload and firmware drives the
    display with it, which is where cognition reads live snapshots — never by asking around.
    """

    def shown_caller(phone: phone_device.Phone) -> object:
        return _display_caller_id(phone_display(phone).take_snapshot().Attributes or {})

    async def run() -> None:
        sfu = FakeSfu()
        alice_phone, _alice_service, alice_room, _alice_recording = await _start_phone_on_room(ALICE_IDENTITY, sfu)
        bob_phone, _bob_service, _bob_room, _bob_recording = await _start_phone_on_room(BOB_IDENTITY, sfu)

        await alice_phone.dispatch(
            alice_phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _wait_until(lambda: _require_firmware(bob_phone).state() == "/Phone/ringing")

        # Bob knows who is calling before anyone answers: the ring carried the caller.
        assert shown_caller(bob_phone) == ALICE_IDENTITY

        await bob_phone.dispatch(
            bob_phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _require_firmware(alice_phone).state() == "/Phone/answered/media_ready")
        await _wait_until(lambda: _require_firmware(bob_phone).state() == "/Phone/answered/media_ready")

        # Connected, each phone shows the other: Alice from her dial plan, Bob from the caller ID.
        assert shown_caller(alice_phone) == BOB_IDENTITY
        assert shown_caller(bob_phone) == ALICE_IDENTITY

    asyncio.run(run())


def test_a_call_with_a_withheld_caller_shows_nobody() -> None:
    """The display stays None when the provider never learned the far end — never invented.

    A withheld caller ID still connects: the call is no less real for being anonymous, and
    "unknown" is what a handset shows, not a made-up name.
    """

    async def run() -> None:
        phone, service, _room, _recording = await _start_signalling_livekit_phone()

        await service.incoming_call(service.context(), phone_device.IncomingCallData(call_id="call-123"))
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/answered/media_ready")

        attributes = phone_display(phone).take_snapshot().Attributes or {}
        assert _display_caller_id(attributes) is None

    asyncio.run(run())


def test_a_caller_cannot_open_its_line_on_a_phone_that_is_only_ringing() -> None:
    """Acked setup means ringing. Whether the call connects is the callee's to decide, not the wire's.

    The caller holds in ``/Phone/dialing`` for as long as the callee is deciding, and no part of
    bringing media up may reach ``media_ready`` ahead of an accept.
    """

    async def run() -> None:
        sfu = FakeSfu()
        alice_phone, _alice_service, _alice_room, alice_recording = await _start_phone_on_room(ALICE_IDENTITY, sfu)
        bob_phone, _bob_service, _bob_room, _bob_recording = await _start_phone_on_room(BOB_IDENTITY, sfu)

        await alice_phone.dispatch(
            alice_phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
        )
        await _wait_until(lambda: _require_firmware(bob_phone).state() == "/Phone/ringing")

        # Bob's phone rings on, and nobody answers it for him.
        await asyncio.sleep(0.05)

        assert _require_firmware(alice_phone).state() == "/Phone/dialing"
        assert _require_firmware(bob_phone).state() == "/Phone/ringing"
        assert not [
            event for event in _phone_events(alice_recording) if event.name == phone_device.MediaReadyEvent.name
        ]

    asyncio.run(run())


def test_a_call_connected_before_the_line_is_up_opens_when_it_comes_up() -> None:
    """Media that lands after the call connects still opens the line, rather than being missed.

    The room track and the call are two facts arriving in either order, and media is ready once
    both hold. An accept cannot arrive in this order — the call-setup wire only goes live once the
    local track has landed, so a caller's accept always finds one there — so the order is pinned on
    the leg that can produce it: a call answered on a phone whose room audio comes up afterwards.
    """

    async def run() -> None:
        phone, service, _room, recording_service = await _start_phone_on_room(
            ALICE_IDENTITY,
            FakeSfu(),
            connect_room=False,
        )

        await service.incoming_call(service.context(), phone_device.IncomingCallData(call_id="call-123"))
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))

        # Connected, and with nothing to talk over yet.
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/answered/media_connecting")

        await service.connect_room(url="wss://livekit.example.com", token="token")

        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/answered/media_ready")
        media_ready = [
            event for event in _phone_events(recording_service) if event.name == phone_device.MediaReadyEvent.name
        ]
        assert [event.data for event in media_ready] == [phone_device.CallData(call_id="call-123")]

    asyncio.run(run())


def test_livekit_phone_service_maps_provider_media_ready_to_phone_firmware() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await service.dispatch(
            service.context(),
            ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")),
        )

        assert _phone_events(recording_service)[-1].name == phone_device.MediaReadyEvent.name
        assert _phone_events(recording_service)[-1].data == phone_device.CallData(call_id="call-123")

    asyncio.run(run())


def test_livekit_phone_service_maps_provider_remote_hang_up_to_phone_firmware() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await service.dispatch(
            service.context(),
            ServiceRemoteHangUpEvent.with_data(phone_device.RemoteHangUpData(call_id="call-123")),
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name

    asyncio.run(run())


def test_livekit_phone_service_maps_provider_call_failure_to_phone_firmware() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await service.dispatch(
            service.context(),
            ServiceCallFailedEvent.with_data(
                phone_device.CallFailedData(call_id="call-123", failure_kind="signaling_failed")
            ),
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name

    asyncio.run(run())


def test_livekit_phone_service_transfer_request_is_accepted_by_gateway() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        request = phone_device.TransferCallData(transfer_id="transfer-123", target=target)
        stamped = phone_device.TransferRequestData(call_id="call-123", transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: service.transfer_requests == [stamped])

        assert _require_firmware(phone).state() == "/Phone/transferring"
        assert [event.name for event in _phone_events(recording_service)][-2:] == [
            phone_device.ServiceTransferRequestedEvent.name,
            phone_device.TransferStartedEvent.name,
        ]

    asyncio.run(run())


def test_livekit_phone_service_maps_provider_transfer_completion_to_phone_firmware() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        request = phone_device.TransferCallData(transfer_id="transfer-123", target=target)
        completion = phone_device.TransferCompletedData(call_id="call-123", transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/transferring")
        await service.dispatch(service.context(), ServiceTransferCompletedEvent.with_data(completion))
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert [event.name for event in _phone_events(recording_service)][-2:] == [
            phone_device.CallTransferCompletedEvent.name,
            phone_device.HungUpEvent.name,
        ]

    asyncio.run(run())


def test_livekit_phone_service_resolves_in_flight_answer_when_remote_hangs_up() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"answer"}))
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=20)
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        await service.dispatch(
            service.context(),
            ServiceRemoteHangUpEvent.with_data(phone_device.RemoteHangUpData(call_id="call-123")),
        )
        await _wait_until(lambda: service.state() == "/PhoneService/ready")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert _require_firmware(phone).state() == "/Phone/hung_up"
        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name
        await asyncio.sleep(0.03)
        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name

    asyncio.run(run())


def test_livekit_phone_service_consumes_stale_provider_observations_while_answering() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"answer"}))
        forwarded_events: list[hsm.Event[typing.Any]] = []
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway,
            operation_timeout=datetime.timedelta(milliseconds=200),
            forwarded_events=forwarded_events,
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        await _assert_stale_provider_observations_are_consumed(service, recording_service)

        assert service.state() == "/PhoneService/answering"
        assert gateway.answer_requests == [phone_device.AnswerRequestData(call_id="call-123")]
        assert _require_firmware(phone).state() == "/Phone/answering"

    asyncio.run(run())


def test_livekit_phone_service_declines_gateway_when_phone_declines_during_answer() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"answer", "decline"}))
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=200)
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        await phone.dispatch(phone.context(), phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData()))
        await _wait_until(lambda: gateway.decline_requests == [phone_device.DeclineRequestData(call_id="call-123")])
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        assert service.state() == "/PhoneService/ready"
        assert gateway.answer_requests == [phone_device.AnswerRequestData(call_id="call-123")]
        assert _require_firmware(phone).state() == "/Phone/hung_up"
        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name

    asyncio.run(run())


def test_livekit_phone_service_resolves_in_flight_transfer_when_provider_fails_transfer() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"transfer"}))
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=20)
        )
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        request = phone_device.TransferCallData(transfer_id="transfer-123", target=target)
        failure = phone_device.TransferFailedData(
            call_id="call-123",
            transfer_id="transfer-123",
            target=target,
            failure_kind="transfer_rejected",
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: service.state() == "/PhoneService/transferring")
        await service.dispatch(service.context(), ServiceTransferFailedEvent.with_data(failure))
        await _wait_until(lambda: service.state() == "/PhoneService/ready")
        await _wait_until(lambda: _is_answered(phone))

        assert device_firmware(phone) is not None
        assert _is_answered(phone)
        assert _phone_events(recording_service)[-1].data == phone_device.CallTransferFailedData(
            call_id="call-123",
            transfer_id="transfer-123",
            target=target,
            failure_kind="transfer_rejected",
        )
        await asyncio.sleep(0.03)
        assert _phone_events(recording_service)[-1].data == phone_device.CallTransferFailedData(
            call_id="call-123",
            transfer_id="transfer-123",
            target=target,
            failure_kind="transfer_rejected",
        )

    asyncio.run(run())


def test_livekit_phone_service_keeps_in_flight_transfer_for_stale_transfer_failure() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"transfer"}))
        forwarded_events: list[hsm.Event[typing.Any]] = []
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway,
            operation_timeout=datetime.timedelta(milliseconds=200),
            forwarded_events=forwarded_events,
        )
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        stale_target = phone_device.TransferTarget(kind="address", value="sip:other@example.com")
        request = phone_device.TransferCallData(transfer_id="transfer-123", target=target)
        stamped = phone_device.TransferRequestData(call_id="call-123", transfer_id="transfer-123", target=target)
        failure = phone_device.TransferFailedData(
            call_id="call-123",
            transfer_id="transfer-stale",
            target=stale_target,
            failure_kind="transfer_rejected",
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: service.state() == "/PhoneService/transferring")
        committed_count = len(_phone_events(recording_service))
        await service.dispatch(service.context(), ServiceTransferFailedEvent.with_data(failure))
        await service.dispatch(
            service.context(),
            ServiceTransferCompletedEvent.with_data(
                phone_device.TransferCompletedData(
                    call_id="call-123", transfer_id="transfer-stale", target=stale_target
                )
            ),
        )
        await _assert_stale_provider_observations_are_consumed(service, recording_service)

        assert service.state() == "/PhoneService/transferring"
        assert gateway.transfer_requests == [stamped]
        assert len(_phone_events(recording_service)) == committed_count
        assert _require_firmware(phone).state() == "/Phone/transferring"

    asyncio.run(run())


def test_livekit_phone_service_ignores_stale_private_gateway_results() -> None:
    async def run_dial() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"dial"}))
        phone, service, _recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=200)
        )
        setup_delivered = typing.cast(hsm.Event[None], getattr(phone_module, "_SetupDeliveredEvent"))
        dial_failed = typing.cast(hsm.Event[phone_device.CallFailedData], getattr(phone_module, "_DialFailedEvent"))

        await phone.dispatch(
            phone.context(), phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER))
        )
        await _wait_until(lambda: service.state() == "/PhoneService/dialing/setup")
        await service.dispatch(
            service.context(),
            dataclasses.replace(setup_delivered, id="stale-operation"),
        )
        await service.dispatch(
            service.context(),
            dataclasses.replace(
                dial_failed.with_data(phone_device.CallFailedData(call_id="call-123", failure_kind="signaling_failed")),
                id="stale-operation",
            ),
        )
        await asyncio.sleep(0)

        assert service.state() == "/PhoneService/dialing/setup"
        assert _require_firmware(phone).state() == "/Phone/dialing"

    async def run_answer() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"answer"}))
        phone, service, _recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=200)
        )
        answer_completed = typing.cast(
            hsm.Event[phone_device.CallConnectedData], getattr(phone_module, "_AnswerCompletedEvent")
        )
        answer_failed = typing.cast(hsm.Event[phone_device.CallFailedData], getattr(phone_module, "_AnswerFailedEvent"))

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        await service.dispatch(
            service.context(), answer_completed.with_data(phone_device.CallConnectedData(call_id="call-123"))
        )
        await service.dispatch(
            service.context(),
            dataclasses.replace(
                answer_failed.with_data(
                    phone_device.CallFailedData(call_id="call-123", failure_kind="signaling_failed")
                ),
                id="stale-operation",
            ),
        )
        await asyncio.sleep(0)

        assert service.state() == "/PhoneService/answering"
        assert _require_firmware(phone).state() == "/Phone/answering"

    async def run_transfer() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"transfer"}))
        phone, service, _recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=200)
        )
        transfer_accepted = typing.cast(
            hsm.Event[phone_device.TransferAcceptedData],
            getattr(phone_module, "_TransferAcceptedEvent"),
        )
        transfer_failed = typing.cast(
            hsm.Event[phone_device.TransferFailedData],
            getattr(phone_module, "_TransferRequestFailedEvent"),
        )
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        request = phone_device.TransferCallData(transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: service.state() == "/PhoneService/transferring")
        await service.dispatch(
            service.context(),
            transfer_accepted.with_data(
                phone_device.TransferAcceptedData(call_id="call-123", transfer_id="transfer-123", target=target)
            ),
        )
        await service.dispatch(
            service.context(),
            dataclasses.replace(
                transfer_failed.with_data(
                    phone_device.TransferFailedData(
                        call_id="call-123",
                        transfer_id="transfer-123",
                        target=target,
                        failure_kind="signaling_failed",
                    )
                ),
                id="stale-operation",
            ),
        )
        await asyncio.sleep(0)

        assert service.state() == "/PhoneService/transferring"
        assert _require_firmware(phone).state() == "/Phone/transferring"

    asyncio.run(run_dial())
    asyncio.run(run_answer())
    asyncio.run(run_transfer())


def test_livekit_phone_service_rejects_stale_transfer_terminal_after_acceptance() -> None:
    async def run() -> None:
        forwarded_events: list[hsm.Event[typing.Any]] = []
        phone, service, _recording_service = await _start_livekit_phone_service(
            forwarded_events=forwarded_events,
        )
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        stale_target = phone_device.TransferTarget(kind="address", value="sip:other@example.com")
        request = phone_device.TransferCallData(transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: service.state() == "/PhoneService/ready")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/transferring")
        committed_forwarded_events = tuple(forwarded_events)

        await service.dispatch(
            service.context(),
            ServiceTransferCompletedEvent.with_data(
                phone_device.TransferCompletedData(
                    call_id="call-123", transfer_id="transfer-stale", target=stale_target
                )
            ),
        )
        await service.dispatch(
            service.context(),
            ServiceTransferFailedEvent.with_data(
                phone_device.TransferFailedData(
                    call_id="call-123",
                    transfer_id="transfer-stale",
                    target=stale_target,
                    failure_kind="transfer_rejected",
                )
            ),
        )
        await asyncio.sleep(0)

        assert tuple(forwarded_events) == committed_forwarded_events
        assert service.state() == "/PhoneService/ready"
        assert _require_firmware(phone).state() == "/Phone/transferring"

    asyncio.run(run())


def test_livekit_phone_service_keeps_accepted_transfer_for_uncorrelated_transferred_hang_up() -> None:
    async def run() -> None:
        forwarded_events: list[hsm.Event[typing.Any]] = []
        phone, service, recording_service = await _start_livekit_phone_service(
            forwarded_events=forwarded_events,
        )
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        stale_target = phone_device.TransferTarget(kind="address", value="sip:other@example.com")
        request = phone_device.TransferCallData(transfer_id="transfer-123", target=target)
        stale_completion = phone_device.TransferCompletedData(
            call_id="call-123",
            transfer_id="transfer-stale",
            target=stale_target,
        )
        completion = phone_device.TransferCompletedData(call_id="call-123", transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: service.state() == "/PhoneService/ready")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/transferring")
        committed_forwarded_events = tuple(forwarded_events)
        committed_phone_events = _phone_events(recording_service)

        service.publish(
            service.context(),
            phone_device.HungUpEvent.with_data(phone_device.HungUpData(call_id="call-123", outcome="transferred")),
        )
        await service.dispatch(service.context(), ServiceTransferCompletedEvent.with_data(stale_completion))
        await asyncio.sleep(0)

        assert tuple(forwarded_events) == committed_forwarded_events
        assert _phone_events(recording_service) == committed_phone_events
        assert service.state() == "/PhoneService/ready"
        assert _require_firmware(phone).state() == "/Phone/transferring"
        await service.dispatch(service.context(), ServiceTransferCompletedEvent.with_data(completion))
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert forwarded_events[-1].name == phone_device.ServiceTransferCompletedEvent.name
        assert forwarded_events[-1].data == phone_device.TransferCompletedData(
            call_id="call-123",
            transfer_id="transfer-123",
            target=target,
        )
        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name

    asyncio.run(run())


def test_livekit_phone_service_hangs_up_gateway_when_phone_hangs_up_during_answer() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"answer", "hang_up"}))
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=200)
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        await phone.dispatch(phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData()))
        await _wait_until(lambda: gateway.hang_up_requests == [phone_device.HangUpRequestData(call_id="call-123")])
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        assert service.state() == "/PhoneService/ready"
        assert gateway.answer_requests == [phone_device.AnswerRequestData(call_id="call-123")]
        assert _require_firmware(phone).state() == "/Phone/hung_up"
        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name

        await asyncio.sleep(0.03)
        assert gateway.hang_up_requests == [phone_device.HangUpRequestData(call_id="call-123")]
        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name

    asyncio.run(run())


def test_livekit_phone_service_hangs_up_gateway_when_phone_hangs_up_during_transfer() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"hang_up", "transfer"}))
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=200)
        )
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        request = phone_device.TransferCallData(transfer_id="transfer-123", target=target)
        stamped = phone_device.TransferRequestData(call_id="call-123", transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: service.state() == "/PhoneService/transferring")
        await phone.dispatch(phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData()))
        await _wait_until(lambda: gateway.hang_up_requests == [phone_device.HangUpRequestData(call_id="call-123")])
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        assert service.state() == "/PhoneService/ready"
        assert gateway.transfer_requests == [stamped]
        assert _require_firmware(phone).state() == "/Phone/hung_up"
        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name

    asyncio.run(run())


def test_livekit_phone_service_consumes_stale_provider_observations_while_hanging_up() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"hang_up"}))
        forwarded_events: list[hsm.Event[typing.Any]] = []
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway,
            operation_timeout=datetime.timedelta(milliseconds=200),
            forwarded_events=forwarded_events,
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        assert recording_service.forwarding_target is not None
        service.publish(
            service.context(),
            dataclasses.replace(
                phone_device.ServiceHangUpRequestedEvent.with_data(phone_device.HangUpRequestData(call_id="call-123")),
                source=hsm.id(recording_service.forwarding_target),
                target=hsm.id(service),
            ),
        )
        await _wait_until(lambda: service.state() == "/PhoneService/hanging_up")
        await _assert_stale_provider_observations_are_consumed(service, recording_service)

        assert service.state() == "/PhoneService/hanging_up"
        assert gateway.hang_up_requests == [phone_device.HangUpRequestData(call_id="call-123")]
        assert device_firmware(phone) is not None
        assert _is_answered(phone)

    asyncio.run(run())


def test_livekit_phone_service_does_not_swallow_new_call_request_while_busy() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"answer"}))
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=200)
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-1")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-2")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await asyncio.sleep(0)

        assert service.state() == "/PhoneService/answering"
        assert gateway.answer_requests == [phone_device.AnswerRequestData(call_id="call-1")]
        assert phone_device.RingingEvent.name not in [event.name for event in _phone_events(recording_service)[2:]]

    asyncio.run(run())


def test_livekit_phone_service_maps_provider_transfer_failure_to_phone_firmware() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        request = phone_device.TransferCallData(transfer_id="transfer-123", target=target)
        failure = phone_device.TransferFailedData(
            call_id="call-123",
            transfer_id="transfer-123",
            target=target,
            failure_kind="transfer_rejected",
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/transferring")
        await service.dispatch(service.context(), ServiceTransferFailedEvent.with_data(failure))
        await _wait_until(lambda: _is_answered(phone))

        assert device_firmware(phone) is not None
        assert _is_answered(phone)
        assert _phone_events(recording_service)[-1].name == phone_device.CallTransferFailedEvent.name
        assert _phone_events(recording_service)[-1].data == phone_device.CallTransferFailedData(
            call_id="call-123",
            transfer_id="transfer-123",
            target=target,
            failure_kind="transfer_rejected",
        )

    asyncio.run(run())


def test_livekit_phone_service_maps_gateway_answer_failure_to_phone_failure() -> None:
    async def run() -> None:
        gateway = FakePhoneService(fail_answer=True)
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway,
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert _phone_events(recording_service)[-1].name == "phone.hung_up"

    asyncio.run(run())


def test_livekit_phone_service_preserves_operation_metadata_on_gateway_failures() -> None:
    async def assert_failure_metadata(operation: str) -> None:
        gateway = FakePhoneService(failed_operations=frozenset({operation}))
        forwarded_events: list[hsm.Event[typing.Any]] = []
        phone, service, _recording_service = await _start_livekit_phone_service(
            service=gateway,
            forwarded_events=forwarded_events,
        )
        metadata = {"traceparent": f"00-{operation}"}

        if operation == "dial":
            command = dataclasses.replace(
                phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER)),
                metadata=metadata,
            )
            await phone.dispatch(phone.context(), command)
            # A dial that fails has no call to report failed against.
            expected_name = phone_device.ServiceDialFailedEvent.name
        elif operation == "answer":
            await service.dispatch(
                service.context(),
                ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
            )
            command = dataclasses.replace(
                phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
                metadata=metadata,
            )
            await phone.dispatch(phone.context(), command)
            expected_name = phone_device.CallFailedEvent.name
        elif operation == "decline":
            await service.dispatch(
                service.context(),
                ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
            )
            command = dataclasses.replace(
                phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData()),
                metadata=metadata,
            )
            await phone.dispatch(phone.context(), command)
            expected_name = phone_device.CallFailedEvent.name
        elif operation == "hang_up":
            await service.dispatch(
                service.context(),
                ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
            )
            await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
            await _wait_until(lambda: _is_answered(phone))
            command = dataclasses.replace(
                phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData()),
                metadata=metadata,
            )
            await phone.dispatch(phone.context(), command)
            expected_name = phone_device.CallFailedEvent.name
        else:
            await service.dispatch(
                service.context(),
                ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
            )
            await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
            await _wait_until(lambda: _is_answered(phone))
            await _mark_media_ready(phone, service)
            target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
            command = dataclasses.replace(
                phone_device.TransferCallEvent.with_data(
                    phone_device.TransferCallData(transfer_id="transfer-123", target=target)
                ),
                metadata=metadata,
            )
            await phone.dispatch(phone.context(), command)
            expected_name = phone_device.ServiceTransferFailedEvent.name

        await _wait_until(
            lambda: any(event.name == expected_name and event.metadata == metadata for event in forwarded_events)
        )

    async def run() -> None:
        for operation in ("dial", "answer", "decline", "hang_up", "transfer"):
            await assert_failure_metadata(operation)

    asyncio.run(run())


def test_livekit_phone_service_times_out_blocked_answer_operation() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"answer"}))
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=5)
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert service.state() == "/PhoneService/ready"
        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name

    asyncio.run(run())


def test_livekit_phone_service_times_out_blocked_dial_operation() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"dial"}))
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=5)
        )
        await phone.dispatch(
            phone.context(), phone_device.DialEvent.with_data(phone_device.DialData(number=DIAL_NUMBER))
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert service.state() == "/PhoneService/ready"
        assert gateway.dial_requests == [phone_device.DialData(number=DIAL_NUMBER)]
        assert _phone_events(recording_service)[-1].name == phone_device.NoCallEvent.name
        # The provider's verdict survives the trip: firmware reports which way the dial failed,
        # not just that it did.
        assert _phone_events(recording_service)[-1].data == phone_device.NoCallData(
            reason="dial_failed", failure_kind="timeout"
        )

    asyncio.run(run())


def test_livekit_phone_service_times_out_blocked_transfer_operation() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"transfer"}))
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=5)
        )
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        request = phone_device.TransferCallData(transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: _is_answered(phone))

        assert service.state() == "/PhoneService/ready"
        assert _phone_events(recording_service)[-1].name == phone_device.CallTransferFailedEvent.name
        assert _phone_events(recording_service)[-1].data == phone_device.CallTransferFailedData(
            call_id="call-123",
            transfer_id="transfer-123",
            target=target,
            failure_kind="timeout",
        )

    asyncio.run(run())


def test_livekit_phone_service_clears_timed_out_terminal_operations() -> None:
    async def run_decline() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"decline"}))
        phone, service, _recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=5)
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData()))
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

    async def run_hang_up() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"hang_up"}))
        phone, service, _recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=5)
        )

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()))
        await _wait_until(lambda: _is_answered(phone))
        await phone.dispatch(phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData()))
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

    asyncio.run(run_decline())
    asyncio.run(run_hang_up())


def test_livekit_phone_service_ignores_malformed_provider_observations() -> None:
    async def run() -> None:
        _phone, service, recording_service = await _start_livekit_phone_service()
        malformed = ServiceCallFailedEvent.with_data(object())

        await service.dispatch(service.context(), malformed)

        assert _phone_events(recording_service) == ()

    asyncio.run(run())


def _pcm_chunk(payload: bytes = b"\x01\x00\x02\x00") -> audio_device.InputData:
    return audio_device.InputData(
        audio=payload,
        media_type="audio/pcm",
        sample_rate_hz=48_000,
        channels=1,
    )


def test_livekit_phone_service_drops_remote_audio_without_media_call_id() -> None:
    """Remote PCM without an active media call id is dropped observably by HSM guards."""

    async def run() -> None:
        _phone, service, _recording_service = await _start_livekit_phone_service()
        chunk = _pcm_chunk(b"\xaa\xbb")

        await service.receive_remote_audio(service.context(), chunk)

        snapshot = service.media_snapshot()
        assert snapshot.remote_audio_chunks == 0
        assert snapshot.remote_audio_bytes == 0
        assert snapshot.remote_audio_dropped_chunks == 1
        assert snapshot.remote_audio_dropped_bytes == 2

    asyncio.run(run())


def test_livekit_phone_service_clears_media_on_any_hung_up_including_transferred() -> None:
    """HungUp with any outcome (including transferred) clears media session even if op already retired."""

    async def run() -> str | None:
        phone, service, _recording = await _start_livekit_phone_service()
        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        assert service._media_call_id == "call-123"
        # Simulate active-op already cleared (e.g. transfer completion path) then hang-up arrives.
        service.publish(
            service.context(),
            phone_device.HungUpEvent.with_data(phone_device.HungUpData(call_id="call-123", outcome="transferred")),
        )
        await asyncio.sleep(0)
        return service._media_call_id

    assert asyncio.run(run()) is None


def test_livekit_phone_service_clears_media_on_hung_up_while_answering() -> None:
    """Active-op states must clear _media_call_id on terminal HungUp (not only ready)."""

    async def run() -> tuple[str | None, str | None]:
        phone, service, _recording = await _start_livekit_phone_service()
        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        assert service._media_call_id == "call-123"
        # Enter answering with an active AnswerCall op — do not wait for answer completion.
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        assert service._media_call_id == "call-123"
        service.publish(
            service.context(),
            phone_device.HungUpEvent.with_data(phone_device.HungUpData(call_id="call-123", outcome="remote_hang_up")),
        )
        await _wait_until(lambda: service.state() == "/PhoneService/ready")
        return service.state(), service._media_call_id

    state, media_call_id = asyncio.run(run())
    assert state == "/PhoneService/ready"
    assert media_call_id is None


def test_livekit_phone_service_drops_remote_audio_without_phone_target() -> None:
    """Remote PCM with a media call id but no attached phone target is dropped."""

    async def run() -> None:
        phone, service, _recording_service = await _start_livekit_phone_service()
        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await service.detach(Environment.from_context(phone.context()), _require_firmware(phone))
        await _wait_until(lambda: service.state() == "/PhoneService/unconnected")
        chunk = _pcm_chunk(b"\x01\x02\x03")

        await service.receive_remote_audio(service.context(), chunk)

        snapshot = service.media_snapshot()
        assert snapshot.remote_audio_chunks == 0
        assert snapshot.remote_audio_dropped_chunks == 1
        assert snapshot.remote_audio_dropped_bytes == 3

    asyncio.run(run())


def test_livekit_phone_service_delivers_remote_audio_when_media_ready() -> None:
    """Remote PCM with media call id + attached phone is delivered as ServiceAudioReceived."""

    async def run() -> None:
        phone, service, _recording_service = await _start_livekit_phone_service()
        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service, call_id="call-123")
        chunk = _pcm_chunk(b"\x01\x00\x02\x00")

        await service.receive_remote_audio(service.context(), chunk)

        snapshot = service.media_snapshot()
        assert snapshot.remote_audio_chunks == 1
        assert snapshot.remote_audio_bytes == 4
        assert snapshot.remote_audio_dropped_chunks == 0
        assert snapshot.remote_audio_dropped_bytes == 0

    asyncio.run(run())


def test_livekit_phone_service_drops_remote_audio_after_remote_hang_up() -> None:
    """Hang-up clears media call id so subsequent remote PCM is dropped by guards."""

    async def run() -> None:
        phone, service, _recording_service = await _start_livekit_phone_service()
        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service, call_id="call-123")
        await service.dispatch(
            service.context(),
            ServiceRemoteHangUpEvent.with_data(phone_device.RemoteHangUpData(call_id="call-123")),
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")
        chunk = _pcm_chunk(b"\x10\x20")

        await service.receive_remote_audio(service.context(), chunk)

        snapshot = service.media_snapshot()
        assert snapshot.remote_audio_chunks == 0
        assert snapshot.remote_audio_dropped_chunks == 1
        assert snapshot.remote_audio_dropped_bytes == 2

    asyncio.run(run())


async def _start_phone_with_fake_room(
    *,
    payload: bytes = b"\x01\x00\x02\x00",
    forwarded_events: list[hsm.Event[typing.Any]] | None = None,
) -> tuple[phone_device.Phone, FakePhoneService, FakeRoom, RecordingPhoneService]:
    """Phone + FakeRoom wired so connect_room streams remote PCM into PhoneService guards."""

    room = FakeRoom(tracks_to_emit_on_connect=[FakeRemoteTrack()])
    service = FakePhoneService(
        room=room,
        stream_factory=fake_pcm_stream(rtc_pcm_frame(payload)),
        local_track_factory=fake_local_track_factory,
    )
    phone, resolved, recording_service = await _start_livekit_phone_service(
        service=service,
        forwarded_events=forwarded_events,
    )
    return phone, resolved, room, recording_service


def test_livekit_phone_service_call_setup_rings_phone_without_auto_answer() -> None:
    """Addressed call setup → incoming_call → phone rings; agent must answer.

    Joining the room is being reachable, not being called. This phone rings because somebody
    dialled it, and the caller ID it shows is the identity the SFU authenticated — never a name
    the message claimed for itself.
    """

    async def run() -> None:
        phone, resolved, room, _recording = await _start_signalling_livekit_phone()

        _ = room.local_participant.invoke(
            signaling.SetupMethod,
            caller_identity=CALLER_IDENTITY,
            call_id="livekit:human-1",
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")

        assert resolved._media_call_id == "livekit:human-1"
        assert resolved._call_peer_identity == CALLER_IDENTITY
        assert room.connected == [("wss://livekit.example.com", "token")]
        # Do not answer here — answering is bot/operator policy.
        assert _require_firmware(phone).state() == "/Phone/ringing"

    asyncio.run(run())


def test_livekit_phone_service_does_not_ring_for_a_participant_that_merely_joins() -> None:
    """Arrival is not a call. Two robots in a room are neighbours until one of them dials."""

    async def run() -> None:
        phone, _service, room, _recording = await _start_signalling_livekit_phone()

        room.emit("participant_connected", FakeRemoteParticipant(identity=CALLER_IDENTITY))
        await asyncio.sleep(0.01)

        assert _require_firmware(phone).state() == "/Phone/hung_up"

    asyncio.run(run())


def test_livekit_phone_service_room_media_answer_is_local_and_reaches_media_ready() -> None:
    """Room-media PhoneService answers locally and marks media ready when track is live."""

    async def run() -> None:
        room = FakeRoom()
        service = PhoneService(
            operation_timeout=datetime.timedelta(seconds=1),
            room=room,
            stream_factory=fake_pcm_stream(rtc_pcm_frame(b"\x01\x00")),
            local_track_factory=fake_local_track_factory,
        )
        recording_service = RecordingPhoneService(service)
        phone = phone_device.Phone(service=recording_service)
        _ = await bot.started(None, phone, typing.cast(hsm.Model, phone.model))
        await _wait_until(lambda: service.state() == "/PhoneService/ready")
        await service.connect_room(url="wss://livekit.example.com", token="token")
        await _wait_until(lambda: service.media_snapshot().local_track_sid == "TR_local")

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="livekit:human")),
        )
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/answered/media_ready")

        assert service._media_call_id == "livekit:human"

    asyncio.run(run())


def test_livekit_phone_service_room_path_drops_remote_audio_without_media_call() -> None:
    """FakeRoom remote track → bridge → PhoneService drop path (no active media call)."""

    async def run() -> None:
        payload = b"\xaa\xbb\xcc\xdd"
        _phone, service, room, _recording = await _start_phone_with_fake_room(payload=payload)

        await service.connect_room(url="wss://livekit.example.com", token="token", track_name="bot-audio")
        await _wait_until(lambda: service.media_snapshot().remote_audio_dropped_chunks == 1)

        snapshot = service.media_snapshot()
        assert room.connected == [("wss://livekit.example.com", "token")]
        assert snapshot.local_track_sid == "TR_local"
        assert snapshot.remote_audio_chunks == 0
        assert snapshot.remote_audio_bytes == 0
        assert snapshot.remote_audio_dropped_chunks == 1
        assert snapshot.remote_audio_dropped_bytes == len(payload)

    asyncio.run(run())


def test_livekit_phone_service_room_path_delivers_remote_audio_when_media_ready() -> None:
    """FakeRoom remote track → bridge → PhoneService deliver path after media is ready."""

    async def run() -> None:
        payload = b"\x01\x00\x02\x00\x03\x00"
        phone, service, room, _recording = await _start_phone_with_fake_room(payload=payload)
        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service, call_id="call-123")

        await service.connect_room(url="wss://livekit.example.com", token="token", track_name="bot-audio")
        await _wait_until(lambda: service.media_snapshot().remote_audio_chunks == 1)

        snapshot = service.media_snapshot()
        assert room.connected == [("wss://livekit.example.com", "token")]
        assert snapshot.local_track_sid == "TR_local"
        assert snapshot.remote_audio_chunks == 1
        assert snapshot.remote_audio_bytes == len(payload)
        assert snapshot.remote_audio_dropped_chunks == 0
        assert snapshot.remote_audio_dropped_bytes == 0

    asyncio.run(run())


def test_livekit_phone_service_room_path_forwards_service_audio_to_phone_firmware() -> None:
    """Delivered remote PCM arrives on the phone firmware target as ServiceAudioReceived."""

    async def run() -> None:
        payload = b"\x10\x20\x30\x40"
        forwarded: list[hsm.Event[typing.Any]] = []
        phone, service, _room, _recording = await _start_phone_with_fake_room(
            payload=payload,
            forwarded_events=forwarded,
        )
        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service, call_id="call-123")

        await service.connect_room(url="wss://livekit.example.com", token="token")
        await _wait_until(lambda: any(event.name == phone_device.ServiceAudioReceivedEvent.name for event in forwarded))

        audio_events = [event for event in forwarded if event.name == phone_device.ServiceAudioReceivedEvent.name]
        assert len(audio_events) == 1
        assert isinstance(audio_events[0].data, phone_device.ServiceAudioData)
        assert audio_events[0].data.call_id == "call-123"
        assert audio_events[0].data.audio == payload
        assert audio_events[0].data.media_type == "audio/pcm"
        assert service.media_snapshot().remote_audio_chunks == 1

    asyncio.run(run())


def test_local_audio_uplink_publish_failure_is_surfaced(caplog: pytest.LogCaptureFixture) -> None:
    """A publish that fails must say so, rather than dying in an orphaned task.

    This is the defect a WAV-typed uplink tripped: the encoder rejected the chunk, the exception
    was never retrieved, and the only symptom was a mute bot. The failure is now a typed outcome
    the service reports on, so "the far end hears nothing" has a reason attached to it.
    """

    async def run() -> None:
        phone, service, _ = await _start_livekit_phone_service()
        del phone

        await service.dispatch(
            service.context(),
            audio_device.OutputEvent.with_data(
                audio_device.OutputData(
                    audio=b"\x01\x00\x02\x00",
                    media_type="audio/wav",
                    sample_rate_hz=24_000,
                    channels=1,
                )
            ),
        )
        await _wait_until(lambda: any(record.levelname == "ERROR" for record in caplog.records))
        captured.append(service.media_snapshot())

    captured: list[MediaSnapshot] = []
    with caplog.at_level(logging.ERROR, logger="bot.providers.livekit.phone"):
        asyncio.run(run())
    snapshot = captured[0] if captured else None

    failures = [record for record in caplog.records if "local audio uplink publish failed" in record.getMessage()]
    assert len(failures) == 1
    # The media type is in the message because it is almost always the reason.
    assert "audio/wav" in failures[0].getMessage()
    # And it is observable without reading logs, the way a dropped inbound chunk already is.
    assert snapshot is not None
    assert snapshot.local_audio_failed_chunks == 1
    assert snapshot.local_audio_failed_bytes == 4
    assert snapshot.remote_audio_dropped_chunks == 0


def test_a_replacement_phone_can_attach_to_the_same_service() -> None:
    """Swapping the phone on a live service must work: this is what the leak actually broke.

    The service refuses a second target while it still holds the first. Nothing released the
    first, so a replacement phone could never boot — its firmware initialization failed and the
    device landed in ``/Device/failed``.
    """

    async def run() -> tuple[str, str]:
        environment = Environment()
        service = PhoneService()

        first = phone_device.Phone(service=service)
        _ = await bot.started(environment, first, typing.cast(hsm.Model, phone_device.Phone.model))
        await _wait_until(lambda: first.state() == "/Device/detached")
        await first.stop(environment)

        second = phone_device.Phone(service=service)
        _ = await bot.started(environment, second, typing.cast(hsm.Model, phone_device.Phone.model))
        await _wait_until(lambda: second.state() == "/Device/detached", timeout=2.0)

        return first.state(), second.state()

    first_state, second_state = asyncio.run(run())

    assert first_state == ""
    assert second_state == "/Device/detached"


@pytest.mark.parametrize("uplink_sample_rate_hz", [24_000, 48_000])
def test_the_audio_source_is_built_at_the_configured_uplink_rate(uplink_sample_rate_hz: int) -> None:
    """The rate the robot speaks at is declared, not defaulted.

    A LiveKit audio source is fixed at construction and refuses a frame at any other rate, so a
    24 kHz voice published into a 48 kHz source is a mute call with an ``InvalidState`` error.
    Wiring owns the number; the provider carries what it is told.
    """

    async def run() -> tuple[int, int]:
        service = PhoneService(uplink_sample_rate_hz=uplink_sample_rate_hz, loop=asyncio.get_running_loop())
        bridge, _ = service._ensure_media()
        source = typing.cast(rtc.AudioSource, bridge.source_writer.source)
        return int(source.sample_rate), int(source.num_channels)

    sample_rate, channels = asyncio.run(run())

    assert sample_rate == uplink_sample_rate_hz
    assert channels == 1

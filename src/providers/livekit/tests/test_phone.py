from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import logging
import typing

import hsm
import pytest

import bot.providers.livekit.phone as phone_module
from bot.devices import audio as audio_device
from bot.devices import phone as phone_device

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
from bot.world import World
from tests.hsm_instance_state import device_firmware
from tests.livekit_room_fakes import (
    FakeRemoteParticipant,
    FakeRemoteTrack,
    FakeRoom,
    fake_local_track_factory,
    fake_pcm_stream,
    rtc_pcm_frame,
)

_FORWARDED_PHONE_EVENT_NAMES = frozenset(
    {
        phone_device.CallConnectedEvent.name,
        phone_device.CallFailedEvent.name,
        phone_device.IncomingCallEvent.name,
        phone_device.ServiceMediaReadyEvent.name,
        phone_device.ServiceAudioReceivedEvent.name,
        phone_device.RemoteHangUpEvent.name,
        phone_device.TransferAcceptedEvent.name,
        phone_device.ServiceTransferCompletedEvent.name,
        phone_device.ServiceTransferFailedEvent.name,
    }
)


async def await_value[T](value: collections.abc.Awaitable[T]) -> T:
    return await value


@dataclasses.dataclass
class FakePhoneService(PhoneService):
    dial_requests: list[phone_device.DialData]
    answer_requests: list[phone_device.AnswerCallData]
    decline_requests: list[phone_device.DeclineCallData]
    hang_up_requests: list[phone_device.HangUpCallData]
    transfer_requests: list[phone_device.TransferCallData]
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
    async def dial(self, request: phone_device.DialData) -> phone_device.CallConnectedData:
        self.dial_requests.append(request)
        self._fail_if_requested("dial")
        await self._block_if_requested("dial")
        return phone_device.CallConnectedData(call_id=request.call_id)

    @typing.override
    async def answer_call(self, request: phone_device.AnswerCallData) -> None:
        self.answer_requests.append(request)
        self._fail_if_requested("answer")
        await self._block_if_requested("answer")

    @typing.override
    async def decline_call(self, request: phone_device.DeclineCallData) -> None:
        self.decline_requests.append(request)
        self._fail_if_requested("decline")
        await self._block_if_requested("decline")

    @typing.override
    async def hang_up_call(self, request: phone_device.HangUpCallData) -> None:
        self.hang_up_requests.append(request)
        self._fail_if_requested("hang_up")
        await self._block_if_requested("hang_up")

    @typing.override
    async def transfer_call(self, request: phone_device.TransferCallData) -> None:
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

    model: typing.ClassVar[hsm.Model] = hsm.define(
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

    async def attach(self, world: World, target: hsm.Instance) -> None:
        if self.forwarded_events is None:
            await self.service.attach(world, target)
            return
        forwarding_target = RecordingPhoneEventTarget(target=target, events=self.forwarded_events)
        _ = await hsm.started(world, forwarding_target, forwarding_target.model)
        self.forwarding_target = forwarding_target
        await self.service.attach(world, forwarding_target)

    async def detach(self, world: World, target: hsm.Instance) -> None:
        attached_target: hsm.Instance = self.forwarding_target if self.forwarding_target is not None else target
        await self.service.detach(world, attached_target)
        self.forwarding_target = None

    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        self.events.append(event)
        published_event = dataclasses.replace(event, target=hsm.id(self.service))
        if self.forwarding_target is None:
            self.service.publish(ctx, published_event)
            return
        self.service.publish(ctx, dataclasses.replace(published_event, source=hsm.id(self.forwarding_target)))


@dataclasses.dataclass(frozen=True)
class LinkedLiveKitPhoneEndpoint:
    name: str
    phone: phone_device.Phone
    service: PhoneService
    recording_service: RecordingPhoneService


@dataclasses.dataclass
class InMemoryLiveKitPhoneLink:
    services: dict[str, PhoneService] = dataclasses.field(default_factory=dict)
    peers: dict[str, str] = dataclasses.field(default_factory=dict)
    answered: dict[str, set[str]] = dataclasses.field(default_factory=dict)

    async def connect(
        self, call_id: str, first: LinkedLiveKitPhoneEndpoint, second: LinkedLiveKitPhoneEndpoint
    ) -> None:
        self.peers[first.name] = second.name
        self.peers[second.name] = first.name
        await first.service.incoming_call(
            first.service.context(),
            phone_device.IncomingCallData(call_id=call_id, display_hint=second.name),
        )
        await second.service.incoming_call(
            second.service.context(),
            phone_device.IncomingCallData(call_id=call_id, display_hint=first.name),
        )

    async def answer(self, endpoint: str, request: phone_device.AnswerCallData) -> None:
        self.answered.setdefault(request.call_id, set()).add(endpoint)

    async def hang_up(self, endpoint: str, request: phone_device.HangUpCallData) -> None:
        peer = self.peers[endpoint]
        await self.services[peer].remote_hang_up(
            self.services[peer].context(),
            phone_device.RemoteHangUpData(call_id=request.call_id),
        )

    async def media_ready(self, call_id: str) -> None:
        for service in self.services.values():
            await service.media_ready(service.context(), phone_device.MediaReadyData(call_id=call_id))


class LinkedPhoneService(PhoneService):
    link: InMemoryLiveKitPhoneLink
    endpoint: str

    def __init__(
        self,
        *,
        link: InMemoryLiveKitPhoneLink,
        endpoint: str,
        operation_timeout: datetime.timedelta | None = None,
    ) -> None:
        super().__init__(operation_timeout=operation_timeout or datetime.timedelta(seconds=1))
        self.link = link
        self.endpoint = endpoint

    @typing.override
    async def dial(self, request: phone_device.DialData) -> phone_device.CallConnectedData:
        del request
        raise PhoneServiceError("in-memory link does not originate calls", failure_kind="provider_unavailable")

    @typing.override
    async def answer_call(self, request: phone_device.AnswerCallData) -> None:
        await self.link.answer(self.endpoint, request)

    @typing.override
    async def decline_call(self, request: phone_device.DeclineCallData) -> None:
        await self.link.hang_up(self.endpoint, phone_device.HangUpCallData(call_id=request.call_id))

    @typing.override
    async def hang_up_call(self, request: phone_device.HangUpCallData) -> None:
        await self.link.hang_up(self.endpoint, request)

    @typing.override
    async def transfer_call(self, request: phone_device.TransferCallData) -> None:
        raise PhoneServiceError(
            f"In-memory linked phone service cannot transfer call {request.call_id}.",
            failure_kind="transfer_rejected",
        )


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
    recording_service = RecordingPhoneService(resolved, forwarded_events=forwarded_events)
    phone = phone_device.Phone(service=recording_service)
    _ = await hsm.started(None, phone, phone.model)
    await _wait_until(lambda: resolved.state() == "/PhoneService/ready")
    return phone, resolved, recording_service


async def _start_linked_phone_endpoint(name: str, link: InMemoryLiveKitPhoneLink) -> LinkedLiveKitPhoneEndpoint:
    service = LinkedPhoneService(link=link, endpoint=name)
    recording_service = RecordingPhoneService(service)
    phone = phone_device.Phone(service=recording_service)
    _ = await hsm.started(None, phone, phone.model)
    await _wait_until(lambda: service.state() == "/PhoneService/ready")
    endpoint = LinkedLiveKitPhoneEndpoint(name=name, phone=phone, service=service, recording_service=recording_service)
    link.services[name] = service
    return endpoint


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
        phone_device.HungUpEvent.with_data(
            phone_device.PhoneHungUpData(call_id="wrong-call", outcome="remote_hang_up")
        ),
    )
    service.publish(
        service.context(),
        phone_device.CallTransferCompletedEvent.with_data(
            phone_device.PhoneTransferData(call_id="wrong-call", transfer_id="stale-transfer", target=target)
        ),
    )
    service.publish(
        service.context(),
        phone_device.CallTransferFailedEvent.with_data(
            phone_device.PhoneTransferFailedData(
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
            service.dial(phone_device.DialData(call_id="call-123", target=target)),
            service.answer_call(phone_device.AnswerCallData(call_id="call-123")),
            service.decline_call(phone_device.DeclineCallData(call_id="call-123")),
            service.hang_up_call(phone_device.HangUpCallData(call_id="call-123")),
            service.transfer_call(
                phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)
            ),
        ):
            try:
                await coro
            except PhoneServiceError as error:
                assert error.failure_kind == "provider_unavailable"
            else:
                raise AssertionError("PhoneService should report missing LiveKit call control.")

    asyncio.run(run())


def test_livekit_phone_service_dials_outbound_call_through_gateway() -> None:
    async def run() -> None:
        service = FakePhoneService()
        recording_service = RecordingPhoneService(service)
        phone = phone_device.Phone(service=recording_service)
        target = phone_device.TransferTarget(kind="address", value="sip:support@example.com")

        _ = await hsm.started(None, phone, phone.model)
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)
        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(call_id="call-123", target=target)),
        )
        await _wait_until(lambda: bool(service.dial_requests), timeout=0.05)
        await _wait_until(lambda: _is_answered(phone))

        assert service.dial_requests == [phone_device.DialData(call_id="call-123", target=target)]
        emitted = _phone_events(recording_service)[-2:]
        assert [event.name for event in emitted] == [
            phone_device.ServiceDialRequestedEvent.name,
            phone_device.AnsweredEvent.name,
        ]
        assert emitted[0].data == phone_device.DialData(call_id="call-123", target=target)
        assert emitted[1].data == phone_device.PhoneCallData(call_id="call-123")

    asyncio.run(run())


def test_livekit_phone_service_default_gateway_emits_provider_unavailable_failure() -> None:
    async def run() -> None:
        service = PhoneService(operation_timeout=datetime.timedelta(seconds=1))
        recording_service = RecordingPhoneService(service)
        phone = phone_device.Phone(service=recording_service)
        _ = await hsm.started(None, phone, phone.model)
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)

        await service.incoming_call(service.context(), phone_device.IncomingCallData(call_id="call-123"))
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(
            lambda: any(
                event.name == phone_device.HungUpEvent.name
                and isinstance(event.data, phone_device.PhoneHungUpData)
                and event.data.outcome == "failed"
                for event in _phone_events(recording_service)
            )
        )

        assert any(
            event.name == phone_device.HungUpEvent.name
            and isinstance(event.data, phone_device.PhoneHungUpData)
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
        target = phone_device.TransferTarget(kind="address", value="sip:support@example.com")
        _ = await hsm.started(None, phone, phone.model)
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)

        await phone.dispatch(
            phone.context(),
            phone_device.DialEvent.with_data(phone_device.DialData(call_id="call-123", target=target)),
        )
        await _wait_until(
            lambda: any(
                event.name == phone_device.HungUpEvent.name
                and isinstance(event.data, phone_device.PhoneHungUpData)
                and event.data.outcome == "failed"
                for event in _phone_events(recording_service)
            )
        )

        assert any(
            event.name == phone_device.HungUpEvent.name
            and isinstance(event.data, phone_device.PhoneHungUpData)
            and event.data.call_id == "call-123"
            and event.data.outcome == "failed"
            for event in _phone_events(recording_service)
        )

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
        _ = await hsm.started(None, service, service.model)
        await service.incoming_call(service.context(), phone_device.IncomingCallData(call_id="call-123"))
        await asyncio.sleep(0)

        assert service.state() == "/PhoneService/unconnected"

    asyncio.run(run())


def test_livekit_phone_service_attach_starts_fresh_service() -> None:
    async def run() -> None:
        world = World()
        service = FakePhoneService()
        target = hsm.Instance()
        target_model = hsm.define(
            "PhoneServiceTarget",
            hsm.initial(hsm.target("active")),
            hsm.state("active"),
        )

        _ = await hsm.started(world, target, target_model, hsm.Config(id="phone-service-target"))
        await service.attach(world, target)

        assert service.state() == "/PhoneService/ready"

    asyncio.run(run())


def test_livekit_phone_service_is_connected_by_phone_service_di() -> None:
    async def run() -> None:
        service = FakePhoneService()
        recording_service = RecordingPhoneService(service)
        phone = phone_device.Phone(service=recording_service)

        _ = await hsm.started(None, phone, phone.model)
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)
        await service.incoming_call(
            service.context(),
            phone_device.IncomingCallData(call_id="call-123", display_hint="Front desk"),
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")

        assert _phone_events(recording_service)[-1].name == phone_device.RingingEvent.name

    asyncio.run(run())


def test_livekit_phone_rejects_forged_service_originating_phone_event() -> None:
    async def run() -> None:
        service = FakePhoneService()
        phone = phone_device.Phone(service=service)

        _ = await hsm.started(None, phone, phone.model)
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
            phone_device.ServiceAnswerRequestedEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
        )
        await _wait_until(lambda: service.answer_requests == [phone_device.AnswerCallData(call_id="call-123")])
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        # Mistargeted envelope still delivers via publish→dispatch; HSM does not re-check target.
        service.publish(
            service.context(),
            dataclasses.replace(
                phone_device.ServiceAnswerRequestedEvent.with_data(phone_device.AnswerCallData(call_id="call-456")),
                source=hsm.id(phone),
                target="not-this-service",
            ),
        )
        await _wait_until(
            lambda: service.answer_requests
            == [
                phone_device.AnswerCallData(call_id="call-123"),
                phone_device.AnswerCallData(call_id="call-456"),
            ]
        )
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        # Hung-up without an active request: provider-terminal hang-up guard fails (call correlation).
        service.publish(
            service.context(),
            phone_device.HungUpEvent.with_data(
                phone_device.PhoneHungUpData(call_id="no-active-op", outcome="remote_hang_up")
            ),
        )
        await asyncio.sleep(0)
        assert service.state() == "/PhoneService/ready"

        # Transfer terminals without active transfer: correlation guard fails → no state change.
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        service.publish(
            service.context(),
            phone_device.CallTransferCompletedEvent.with_data(
                phone_device.PhoneTransferData(call_id="call-123", transfer_id="transfer-123", target=target)
            ),
        )
        service.publish(
            service.context(),
            phone_device.CallTransferFailedEvent.with_data(
                phone_device.PhoneTransferFailedData(
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

        _ = await hsm.started(None, phone, phone.model)
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)
        assert device_firmware(phone) is not None

        await service.detach(World.from_context(phone.context()), _require_firmware(phone))
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

        _ = await hsm.started(None, first, first.model, hsm.Config(id="first-phone"))
        await _wait_until(lambda: service.state() == "/PhoneService/ready", timeout=0.05)
        assert device_firmware(first) is not None

        other_target = RecordingPhoneEventTarget(target=_require_firmware(first), events=[])
        _ = await hsm.started(first.context(), other_target, other_target.model, hsm.Config(id="other-target"))
        with pytest.raises(PhoneServiceError, match="already attached"):
            await service.attach(World.from_context(first.context()), other_target)
        await asyncio.sleep(0)

        assert service.state() == "/PhoneService/ready"

        _ = await hsm.started(first.context(), second, second.model, hsm.Config(id="second-phone"))
        await _wait_until(lambda: second.state() == "/Device/failed")

        await second.dispatch(
            second.context(),
            phone_device.DialEvent.with_data(
                phone_device.DialData(
                    call_id="second-call",
                    target=phone_device.TransferTarget(kind="address", value="sip:second@example.com"),
                )
            ),
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
        world = World()
        terminal_model = hsm.define(
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

        _ = await hsm.started(world, first_terminal, terminal_model, hsm.Config(id="first-terminal"))
        _ = await hsm.started(world, second_terminal, terminal_model, hsm.Config(id="second-terminal"))
        _ = await hsm.started(world, first_target, first_target.model, hsm.Config(id="first-target"))
        _ = await hsm.started(world, second_target, second_target.model, hsm.Config(id="second-target"))
        await service.attach(world, first_target)

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
            ServiceIncomingCallEvent.with_data(
                phone_device.IncomingCallData(call_id="call-123", display_hint="Front desk")
            ),
        )
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: service.answer_requests == [phone_device.AnswerCallData(call_id="call-123")])
        await _wait_until(lambda: _is_answered(phone))

        assert device_firmware(phone) is not None
        assert _is_answered(phone)
        assert [event.name for event in _phone_events(recording_service)] == [
            phone_device.RingingEvent.name,
            phone_device.ServiceAnswerRequestedEvent.name,
            phone_device.AnsweredEvent.name,
        ]
        assert _phone_events(recording_service)[-1].data == phone_device.PhoneCallData(call_id="call-123")

    asyncio.run(run())


def test_livekit_phone_service_declines_current_phone_call_through_gateway() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(), phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData(call_id="call-123"))
        )
        await _wait_until(lambda: service.decline_requests == [phone_device.DeclineCallData(call_id="call-123")])
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: _is_answered(phone))
        await phone.dispatch(
            phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData(call_id="call-123"))
        )
        await _wait_until(lambda: service.hang_up_requests == [phone_device.HangUpCallData(call_id="call-123")])
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        assert _require_firmware(phone).state() == "/Phone/hung_up"
        assert [event.name for event in _phone_events(recording_service)][-2:] == [
            phone_device.ServiceHangUpRequestedEvent.name,
            phone_device.HungUpEvent.name,
        ]

    asyncio.run(run())


def test_livekit_phone_service_connects_two_real_phone_devices() -> None:
    async def run() -> None:
        link = InMemoryLiveKitPhoneLink()
        first = await _start_linked_phone_endpoint("front-desk", link)
        second = await _start_linked_phone_endpoint("operator", link)

        await link.connect("call-123", first, second)
        await _wait_until(lambda: _require_firmware(first.phone).state() == "/Phone/ringing")
        await _wait_until(lambda: _require_firmware(second.phone).state() == "/Phone/ringing")

        await first.phone.dispatch(
            first.phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
        )
        await second.phone.dispatch(
            second.phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
        )
        await _wait_until(lambda: _is_answered(first.phone))
        await _wait_until(lambda: _is_answered(second.phone))

        assert link.answered == {"call-123": {"front-desk", "operator"}}

        await link.media_ready("call-123")
        await _wait_until(lambda: _phone_events(first.recording_service)[-1].name == phone_device.MediaReadyEvent.name)
        await _wait_until(lambda: _phone_events(second.recording_service)[-1].name == phone_device.MediaReadyEvent.name)

        await first.phone.dispatch(
            first.phone.context(),
            phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData(call_id="call-123")),
        )
        await _wait_until(lambda: _require_firmware(first.phone).state() == "/Phone/hung_up")
        await _wait_until(lambda: _require_firmware(second.phone).state() == "/Phone/hung_up")

        assert _phone_events(first.recording_service)[-1].name == phone_device.HungUpEvent.name
        assert _phone_events(second.recording_service)[-1].name == phone_device.HungUpEvent.name

    asyncio.run(run())


def test_livekit_phone_service_maps_provider_media_ready_to_phone_firmware() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: _is_answered(phone))
        await service.dispatch(
            service.context(),
            ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-123")),
        )

        assert _phone_events(recording_service)[-1].name == phone_device.MediaReadyEvent.name
        assert _phone_events(recording_service)[-1].data == phone_device.PhoneCallData(call_id="call-123")

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
        request = phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: service.transfer_requests == [request])

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
        request = phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)
        completion = phone_device.TransferCompletedData(call_id="call-123", transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        await _assert_stale_provider_observations_are_consumed(service, recording_service)

        assert service.state() == "/PhoneService/answering"
        assert gateway.answer_requests == [phone_device.AnswerCallData(call_id="call-123")]
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        await phone.dispatch(
            phone.context(), phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData(call_id="call-123"))
        )
        await _wait_until(lambda: gateway.decline_requests == [phone_device.DeclineCallData(call_id="call-123")])
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        assert service.state() == "/PhoneService/ready"
        assert gateway.answer_requests == [phone_device.AnswerCallData(call_id="call-123")]
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
        request = phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: service.state() == "/PhoneService/transferring")
        await service.dispatch(service.context(), ServiceTransferFailedEvent.with_data(failure))
        await _wait_until(lambda: service.state() == "/PhoneService/ready")
        await _wait_until(lambda: _is_answered(phone))

        assert device_firmware(phone) is not None
        assert _is_answered(phone)
        assert _phone_events(recording_service)[-1].data == phone_device.PhoneTransferFailedData(
            call_id="call-123",
            transfer_id="transfer-123",
            target=target,
            failure_kind="transfer_rejected",
        )
        await asyncio.sleep(0.03)
        assert _phone_events(recording_service)[-1].data == phone_device.PhoneTransferFailedData(
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
        request = phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
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
        assert gateway.transfer_requests == [request]
        assert len(_phone_events(recording_service)) == committed_count
        assert _require_firmware(phone).state() == "/Phone/transferring"

    asyncio.run(run())


def test_livekit_phone_service_ignores_stale_private_gateway_results() -> None:
    async def run_dial() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"dial"}))
        phone, service, _recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=200)
        )
        dial_completed = typing.cast(
            hsm.Event[phone_device.CallConnectedData], getattr(phone_module, "_DialCompletedEvent")
        )
        dial_failed = typing.cast(hsm.Event[phone_device.CallFailedData], getattr(phone_module, "_DialFailedEvent"))
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")

        await phone.dispatch(
            phone.context(), phone_device.DialEvent.with_data(phone_device.DialData(call_id="call-123", target=target))
        )
        await _wait_until(lambda: service.state() == "/PhoneService/dialing")
        await service.dispatch(
            service.context(), dial_completed.with_data(phone_device.CallConnectedData(call_id="call-123"))
        )
        await service.dispatch(
            service.context(),
            dataclasses.replace(
                dial_failed.with_data(phone_device.CallFailedData(call_id="call-123", failure_kind="signaling_failed")),
                id="stale-operation",
            ),
        )
        await asyncio.sleep(0)

        assert service.state() == "/PhoneService/dialing"
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
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
        request = phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
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
        request = phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
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
        request = phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: service.state() == "/PhoneService/ready")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/transferring")
        committed_forwarded_events = tuple(forwarded_events)
        committed_phone_events = _phone_events(recording_service)

        service.publish(
            service.context(),
            phone_device.HungUpEvent.with_data(phone_device.PhoneHungUpData(call_id="call-123", outcome="transferred")),
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        await phone.dispatch(
            phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData(call_id="call-123"))
        )
        await _wait_until(lambda: gateway.hang_up_requests == [phone_device.HangUpCallData(call_id="call-123")])
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        assert service.state() == "/PhoneService/ready"
        assert gateway.answer_requests == [phone_device.AnswerCallData(call_id="call-123")]
        assert _require_firmware(phone).state() == "/Phone/hung_up"
        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name

        await asyncio.sleep(0.03)
        assert gateway.hang_up_requests == [phone_device.HangUpCallData(call_id="call-123")]
        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name

    asyncio.run(run())


def test_livekit_phone_service_hangs_up_gateway_when_phone_hangs_up_during_transfer() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"hang_up", "transfer"}))
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=200)
        )
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        request = phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: service.state() == "/PhoneService/transferring")
        await phone.dispatch(
            phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData(call_id="call-123"))
        )
        await _wait_until(lambda: gateway.hang_up_requests == [phone_device.HangUpCallData(call_id="call-123")])
        await _wait_until(lambda: service.state() == "/PhoneService/ready")

        assert service.state() == "/PhoneService/ready"
        assert gateway.transfer_requests == [request]
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: _is_answered(phone))
        assert recording_service.forwarding_target is not None
        service.publish(
            service.context(),
            dataclasses.replace(
                phone_device.ServiceHangUpRequestedEvent.with_data(phone_device.HangUpCallData(call_id="call-123")),
                source=hsm.id(recording_service.forwarding_target),
                target=hsm.id(service),
            ),
        )
        await _wait_until(lambda: service.state() == "/PhoneService/hanging_up")
        await _assert_stale_provider_observations_are_consumed(service, recording_service)

        assert service.state() == "/PhoneService/hanging_up"
        assert gateway.hang_up_requests == [phone_device.HangUpCallData(call_id="call-123")]
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-1"))
        )
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-2")),
        )
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-2"))
        )
        await asyncio.sleep(0)

        assert service.state() == "/PhoneService/answering"
        assert gateway.answer_requests == [phone_device.AnswerCallData(call_id="call-1")]
        assert phone_device.RingingEvent.name not in [event.name for event in _phone_events(recording_service)[2:]]

    asyncio.run(run())


def test_livekit_phone_service_maps_provider_transfer_failure_to_phone_firmware() -> None:
    async def run() -> None:
        phone, service, recording_service = await _start_livekit_phone_service()
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        request = phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/transferring")
        await service.dispatch(service.context(), ServiceTransferFailedEvent.with_data(failure))
        await _wait_until(lambda: _is_answered(phone))

        assert device_firmware(phone) is not None
        assert _is_answered(phone)
        assert _phone_events(recording_service)[-1].name == phone_device.CallTransferFailedEvent.name
        assert _phone_events(recording_service)[-1].data == phone_device.PhoneTransferFailedData(
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
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
            target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
            command = dataclasses.replace(
                phone_device.DialEvent.with_data(phone_device.DialData(call_id="call-123", target=target)),
                metadata=metadata,
            )
            await phone.dispatch(phone.context(), command)
            expected_name = phone_device.CallFailedEvent.name
        elif operation == "answer":
            await service.dispatch(
                service.context(),
                ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
            )
            command = dataclasses.replace(
                phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
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
                phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData(call_id="call-123")),
                metadata=metadata,
            )
            await phone.dispatch(phone.context(), command)
            expected_name = phone_device.CallFailedEvent.name
        elif operation == "hang_up":
            await service.dispatch(
                service.context(),
                ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
            )
            await phone.dispatch(
                phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
            )
            await _wait_until(lambda: _is_answered(phone))
            command = dataclasses.replace(
                phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData(call_id="call-123")),
                metadata=metadata,
            )
            await phone.dispatch(phone.context(), command)
            expected_name = phone_device.CallFailedEvent.name
        else:
            await service.dispatch(
                service.context(),
                ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
            )
            await phone.dispatch(
                phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
            )
            await _wait_until(lambda: _is_answered(phone))
            await _mark_media_ready(phone, service)
            target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
            command = dataclasses.replace(
                phone_device.TransferCallEvent.with_data(
                    phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
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
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")

        await phone.dispatch(
            phone.context(), phone_device.DialEvent.with_data(phone_device.DialData(call_id="call-123", target=target))
        )
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/hung_up")

        assert service.state() == "/PhoneService/ready"
        assert gateway.dial_requests == [phone_device.DialData(call_id="call-123", target=target)]
        assert _phone_events(recording_service)[-1].name == phone_device.HungUpEvent.name
        assert isinstance(_phone_events(recording_service)[-1].data, phone_device.PhoneHungUpData)
        assert _phone_events(recording_service)[-1].data == phone_device.PhoneHungUpData(
            call_id="call-123", outcome="failed"
        )

    asyncio.run(run())


def test_livekit_phone_service_times_out_blocked_transfer_operation() -> None:
    async def run() -> None:
        gateway = FakePhoneService(blocked_operations=frozenset({"transfer"}))
        phone, service, recording_service = await _start_livekit_phone_service(
            service=gateway, operation_timeout=datetime.timedelta(milliseconds=5)
        )
        target = phone_device.TransferTarget(kind="address", value="sip:operator@example.com")
        request = phone_device.TransferCallData(call_id="call-123", transfer_id="transfer-123", target=target)

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-123")),
        )
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        await phone.dispatch(phone.context(), phone_device.TransferCallEvent.with_data(request))
        await _wait_until(lambda: _is_answered(phone))

        assert service.state() == "/PhoneService/ready"
        assert _phone_events(recording_service)[-1].name == phone_device.CallTransferFailedEvent.name
        assert _phone_events(recording_service)[-1].data == phone_device.PhoneTransferFailedData(
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
        await phone.dispatch(
            phone.context(), phone_device.DeclineCallEvent.with_data(phone_device.DeclineCallData(call_id="call-123"))
        )
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
        await phone.dispatch(
            phone.context(), phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123"))
        )
        await _wait_until(lambda: _is_answered(phone))
        await phone.dispatch(
            phone.context(), phone_device.HangUpCallEvent.with_data(phone_device.HangUpCallData(call_id="call-123"))
        )
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


def _pcm_chunk(payload: bytes = b"\x01\x00\x02\x00") -> audio_device.AudioInputData:
    return audio_device.AudioInputData(
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
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
        )
        await _wait_until(lambda: _is_answered(phone))
        await _mark_media_ready(phone, service)
        assert service._media_call_id == "call-123"
        # Simulate active-op already cleared (e.g. transfer completion path) then hang-up arrives.
        service.publish(
            service.context(),
            phone_device.HungUpEvent.with_data(
                phone_device.PhoneHungUpData(call_id="call-123", outcome="transferred")
            ),
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
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
        )
        await _wait_until(lambda: service.state() == "/PhoneService/answering")
        assert service._media_call_id == "call-123"
        service.publish(
            service.context(),
            phone_device.HungUpEvent.with_data(
                phone_device.PhoneHungUpData(call_id="call-123", outcome="remote_hang_up")
            ),
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
        await service.detach(World.from_context(phone.context()), _require_firmware(phone))
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
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
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
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
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


def test_livekit_phone_service_remote_participant_join_rings_phone_without_auto_answer() -> None:
    """Remote LiveKit participant → incoming_call → phone rings; agent must answer."""

    async def run() -> None:
        room = FakeRoom(participants_to_emit_on_connect=[FakeRemoteParticipant(identity="human")])
        service = FakePhoneService(
            room=room,
            stream_factory=fake_pcm_stream(rtc_pcm_frame(b"\x01\x00")),
            local_track_factory=fake_local_track_factory,
        )
        phone, resolved, _recording = await _start_livekit_phone_service(service=service)

        await resolved.connect_room(url="wss://livekit.example.com", token="token", track_name="bot-audio")
        await _wait_until(lambda: _require_firmware(phone).state() == "/Phone/ringing")

        assert resolved._media_call_id == "livekit:human"
        assert room.connected == [("wss://livekit.example.com", "token")]
        # Do not answer here — answering is bot/operator policy.
        assert _require_firmware(phone).state() == "/Phone/ringing"

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
        _ = await hsm.started(None, phone, phone.model)
        await _wait_until(lambda: service.state() == "/PhoneService/ready")
        await service.connect_room(url="wss://livekit.example.com", token="token")
        await _wait_until(lambda: service.media_snapshot().local_track_sid == "TR_local")

        await service.dispatch(
            service.context(),
            ServiceIncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="livekit:human")),
        )
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="livekit:human")),
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
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
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
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData(call_id="call-123")),
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
                audio_device.AudioOutputData(
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
        world = World()
        service = PhoneService()

        first = phone_device.Phone(service=service)
        _ = await hsm.started(world, first, typing.cast(hsm.Model, phone_device.Phone.model))
        await _wait_until(lambda: first.state() == "/Device/detached")
        await first.stop(world)

        second = phone_device.Phone(service=service)
        _ = await hsm.started(world, second, typing.cast(hsm.Model, phone_device.Phone.model))
        await _wait_until(lambda: second.state() == "/Device/detached", timeout=2.0)

        return first.state(), second.state()

    first_state, second_state = asyncio.run(run())

    assert first_state == ""
    assert second_state == "/Device/detached"

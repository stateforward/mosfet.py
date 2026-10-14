"""PhoneService lifecycle: the phone holding it starts and stops it, and attaches by message."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import importlib
import typing
import uuid

import hsm
import pytest

import mosfet.lifecycle
from mosfet import abilities
from mosfet.environment import Environment
from mosfet.protocols import attachment
from mosfet.providers.livekit.phone import PhoneService
from mosfet.providers.livekit.room_audio import RoomAudioTrackPath

_DEFINE = importlib.import_module("mosfet.define")
_TIMEOUT = datetime.timedelta(seconds=1)


class _InspectablePhoneService(PhoneService):
    def track_path(self) -> RoomAudioTrackPath:
        track_path = self._track_path
        if track_path is None:
            raise AssertionError("PhoneService has no room audio track path")
        return track_path


async def _actor(environment: Environment, name: str) -> hsm.Instance:
    return await mosfet.started(
        environment, hsm.Instance(), mosfet.define(name, hsm.initial(hsm.target("s")), hsm.state("s"))
    )


async def _attach(environment: Environment, service: PhoneService, actor: hsm.Instance) -> hsm.Event[typing.Any]:
    return await abilities.run_terminal_operation(
        environment,
        child=service,
        request=lambda reply: dataclasses.replace(
            attachment.AttachEvent.with_data(attachment.AttachData(actor=actor, reply_to=reply, timeout=_TIMEOUT)),
            id=uuid.uuid4().hex,
        ),
        terminals=(attachment.AttachCompleteEvent, attachment.AttachFailedEvent),
        timeout=_TIMEOUT,
    )


async def _detach(environment: Environment, service: PhoneService, actor: hsm.Instance) -> hsm.Event[typing.Any]:
    return await abilities.run_terminal_operation(
        environment,
        child=service,
        request=lambda reply: dataclasses.replace(
            attachment.DetachEvent.with_data(attachment.DetachData(actor=actor, reply_to=reply, timeout=_TIMEOUT)),
            id=uuid.uuid4().hex,
        ),
        terminals=(attachment.DetachedEvent, attachment.DetachFailedEvent),
        timeout=_TIMEOUT,
    )


def test_phone_service_does_not_start_itself_and_its_track_path_starts_and_stops_with_it() -> None:
    async def run() -> tuple[bool, bool, bool, bool]:
        environment = Environment()
        service = _InspectablePhoneService()
        unstarted = mosfet.lifecycle.is_started(service)
        _ = await mosfet.started(environment, service, PhoneService.model)
        track_path = service.track_path()
        track_started = mosfet.lifecycle.is_started(track_path)
        await hsm.stop(service)
        return unstarted, track_started, mosfet.lifecycle.is_started(service), mosfet.lifecycle.is_started(track_path)

    unstarted, track_started, service_after_stop, track_after_stop = asyncio.run(run())
    assert unstarted is False
    assert track_started is True
    assert service_after_stop is False
    assert track_after_stop is False


def test_phone_service_answers_attach_and_detach_by_message() -> None:
    """One firmware at a time: the attached one is answered again, another is refused as a conflict."""

    async def run() -> list[tuple[str, object]]:
        environment = Environment()
        service = PhoneService()
        _ = await mosfet.started(environment, service, PhoneService.model)
        first = await _actor(environment, "FirstFirmware")
        second = await _actor(environment, "SecondFirmware")
        answers = [
            await _attach(environment, service, first),
            await _attach(environment, service, first),
            await _attach(environment, service, second),
            await _detach(environment, service, first),
            await _detach(environment, service, first),
            await _attach(environment, service, second),
        ]
        await hsm.stop(service)
        return [
            (
                answer.name,
                answer.data.kind
                if isinstance(answer.data, attachment.FailedData)
                else getattr(answer.data, "created", getattr(answer.data, "removed", None)),
            )
            for answer in answers
        ]

    assert asyncio.run(run()) == [
        (attachment.AttachCompleteEvent.name, True),
        (attachment.AttachCompleteEvent.name, False),
        (attachment.AttachFailedEvent.name, attachment.FailureKind.CONFLICT),
        (attachment.DetachedEvent.name, True),
        (attachment.DetachedEvent.name, False),
        (attachment.AttachCompleteEvent.name, True),
    ]


def test_phone_service_restarts_after_stop_and_attaches_again() -> None:
    async def run() -> tuple[str, str]:
        environment = Environment()
        service = PhoneService()
        target = await _actor(environment, "RestartTarget")
        _ = await mosfet.started(environment, service, PhoneService.model)
        first = await _attach(environment, service, target)
        await hsm.stop(service)
        assert mosfet.lifecycle.is_started(service) is False
        _ = await mosfet.started(environment, service, PhoneService.model)
        again = await _attach(environment, service, target)
        await hsm.stop(service)
        return first.name, again.name

    assert asyncio.run(run()) == (attachment.AttachCompleteEvent.name, attachment.AttachCompleteEvent.name)


def test_phone_service_direct_stop_clears_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[dict[str, object], str]] = []

    def _record(payload: dict[str, object], url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> None:
        environment = Environment()
        service = PhoneService()
        _ = await mosfet.started(environment, service, PhoneService.model, owner="/Phone")
        _ = await _attach(environment, service, await _actor(environment, "StopTarget"))
        calls.clear()

        await service.stop(environment)

    asyncio.run(run())
    live_payloads = [
        payload for payload, url in calls if url.endswith("/v1/models/live") and payload.get("name") == "/PhoneService"
    ]
    assert live_payloads
    assert live_payloads[-1].get("live") is False
    assert live_payloads[-1].get("owner") is None


def test_phone_service_stop_clears_owner_for_an_independently_stopped_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[dict[str, object], str]] = []

    def _record(payload: dict[str, object], url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> None:
        environment = Environment()
        service = _InspectablePhoneService()
        _ = await mosfet.started(environment, service, PhoneService.model, owner="/Phone")
        track_path = service.track_path()
        await hsm.stop(track_path, environment)
        assert not mosfet.lifecycle.is_started(track_path)
        calls.clear()

        await service.stop(environment)

    asyncio.run(run())
    live_payloads = [
        payload
        for payload, url in calls
        if url.endswith("/v1/models/live") and payload.get("name") == "/RoomAudioTrackPath"
    ]
    assert live_payloads
    assert live_payloads[-1].get("live") is False
    assert live_payloads[-1].get("owner") is None


def test_phone_service_attach_and_conflict_do_not_republish_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The phone that started the service owns it; attaching firmware never changes that."""

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[dict[str, object], str]] = []

    def _record(payload: dict[str, object], url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> str:
        environment = Environment()
        service = PhoneService()
        first_target = await _actor(environment, "FirstTarget")
        second_target = await _actor(environment, "SecondTarget")
        _ = await mosfet.started(environment, service, PhoneService.model, owner="/Phone")
        calls.clear()
        _ = await _attach(environment, service, first_target)
        refused = await _attach(environment, service, second_target)
        await hsm.stop(service)
        return refused.name

    assert asyncio.run(run()) == attachment.AttachFailedEvent.name
    republished = [
        payload
        for payload, url in calls
        if url.endswith("/v1/models/live") and payload.get("name") == "/PhoneService" and payload.get("live")
    ]
    assert republished == []

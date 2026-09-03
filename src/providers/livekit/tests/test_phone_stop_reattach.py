"""PhoneService production stop then attach restarts HSM (hsm 1.3.2 stop semantics)."""

from __future__ import annotations

import asyncio
import importlib

import hsm
import pytest

import bot.lifecycle
from bot.providers.livekit.phone import PhoneService, PhoneServiceError
from bot.providers.livekit.room_audio import RoomAudioTrackPath
from bot.environment import Environment

_DEFINE = importlib.import_module("bot.define")


class _InspectablePhoneService(PhoneService):
    def track_path(self) -> RoomAudioTrackPath:
        track_path = self._track_path
        if track_path is None:
            raise AssertionError("PhoneService has no room audio track path")
        return track_path


def test_phone_service_production_stop_unstarts_machine() -> None:
    """``hsm.stop(service)`` makes ``hsm.id`` fail without any process-side hold."""

    async def run() -> tuple[bool, bool]:
        environment = Environment()
        service = PhoneService()
        target = hsm.Instance()
        await bot.started(
            environment,
            target,
            bot.define("T", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await service.attach(environment, target)
        after_attach = bot.lifecycle.is_started(service)
        await hsm.stop(service)
        after_stop = bot.lifecycle.is_started(service)
        return after_attach, after_stop

    after_attach, after_stop = asyncio.run(run())
    assert after_attach is True
    assert after_stop is False


def test_phone_service_direct_stop_clears_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[dict[str, object], str]] = []

    def _record(payload: dict[str, object], url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> None:
        environment = Environment()
        service = PhoneService()
        target = hsm.Instance()
        target_model = bot.define("StopTarget", hsm.initial(hsm.target("s")), hsm.state("s"))
        await bot.started(environment, target, target_model)
        await service.attach(environment, target)
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
        target = hsm.Instance()
        target_model = bot.define("StoppedTrackTarget", hsm.initial(hsm.target("s")), hsm.state("s"))
        await bot.started(environment, target, target_model)
        await service.attach(environment, target)
        track_path = service.track_path()
        await hsm.stop(track_path, environment)
        assert not bot.lifecycle.is_started(track_path)
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


def test_phone_service_attach_after_stop_restarts_machine() -> None:
    """Stop via production API, then re-attach must restart without RuntimeError."""

    async def run() -> None:
        environment = Environment()
        service = PhoneService()
        target = hsm.Instance()
        await bot.started(
            environment,
            target,
            bot.define("T", hsm.initial(hsm.target("s")), hsm.state("s")),
        )
        await service.attach(environment, target)
        assert bot.lifecycle.is_started(service) is True

        await hsm.stop(service)
        assert bot.lifecycle.is_started(service) is False

        await service.attach(environment, target)
        assert bot.lifecycle.is_started(service) is True
        await service.detach(environment, target)

    asyncio.run(run())


def test_phone_service_reattach_refreshes_owner_without_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[dict[str, object], str]] = []

    def _record(payload: dict[str, object], url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> tuple[str, str]:
        environment = Environment()
        service = PhoneService()
        target_model = bot.define("FirstTarget", hsm.initial(hsm.target("s")), hsm.state("s"))
        replacement_model = bot.define("ReplacementTarget", hsm.initial(hsm.target("s")), hsm.state("s"))
        first_target = await bot.started(environment, hsm.Instance(), target_model)
        replacement_target = await bot.started(environment, hsm.Instance(), replacement_model)
        await service.attach(environment, first_target)
        await service.detach(environment, first_target)
        detached_live_payloads = [
            payload
            for payload, url in calls
            if url.endswith("/v1/models/live") and payload.get("name") == "/PhoneService"
        ]
        assert detached_live_payloads
        detached_live_payload = detached_live_payloads[-1]
        assert "owner" in detached_live_payload
        assert detached_live_payload["owner"] is None
        calls.clear()
        before = hsm.id(service)
        await service.attach(environment, replacement_target)
        return before, hsm.id(service)

    before, after = asyncio.run(run())
    live_payloads = [payload for payload, url in calls if url.endswith("/v1/models/live")]
    assert before == after
    assert len(live_payloads) == 1
    live_payload = live_payloads[0]
    owner = live_payload.get("owner")
    assert isinstance(owner, str)
    assert owner == "/ReplacementTarget"


def test_phone_service_conflicting_attach_does_not_republish_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[dict[str, object], str]] = []

    def _record(payload: dict[str, object], url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> None:
        environment = Environment()
        service = PhoneService()
        first_model = bot.define("FirstTarget", hsm.initial(hsm.target("s")), hsm.state("s"))
        second_model = bot.define("SecondTarget", hsm.initial(hsm.target("s")), hsm.state("s"))
        first_target = await bot.started(environment, hsm.Instance(), first_model)
        second_target = await bot.started(environment, hsm.Instance(), second_model)
        await service.attach(environment, first_target)
        calls.clear()

        with pytest.raises(PhoneServiceError, match="already attached"):
            await service.attach(environment, second_target)

    asyncio.run(run())
    assert calls == []

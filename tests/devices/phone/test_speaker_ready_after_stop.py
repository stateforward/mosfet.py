"""Receiver readiness needs an attach hold plus a started, same-environment speaker."""

from __future__ import annotations

import asyncio
import typing

import hsm
import bot

from bot.devices import audio
from bot.devices.phone import phone as phone_device
from bot.devices.phone.phone import PhoneFirmware, _PhoneObservationService
from bot.environment import Environment


def _service_audio(call_id: str) -> hsm.Event[phone_device.ServiceAudioData]:
    return phone_device.ServiceAudioReceivedEvent.with_data(
        phone_device.ServiceAudioData(
            call_id=call_id,
            audio=b"remote",
            media_type="audio/pcm",
            sample_rate_hz=16_000,
            channels=1,
        )
    )


def test_receiver_requires_attach_and_started_speaker() -> None:
    """Speaker liveness is firmware's guard now; the service only reports attachment."""

    async def run() -> tuple[bool, bool, bool, bool]:
        environment = Environment()
        speaker = audio.Speaker()
        target = hsm.Instance()
        # This test is about attach/liveness gating, not elevation; the phone injects the real
        # elevation, so a no-op stands in here.
        observation = _PhoneObservationService(
            owner=phone_device.Phone(),
            service=phone_device.PhoneEventRecorder(),
            elevate=lambda ctx, event: None,
        )
        firmware = PhoneFirmware(service=observation, speaker=speaker)
        firmware._current_call_id = "call-1"
        event = _service_audio("call-1")

        unattached = PhoneFirmware._matches_current_service_audio(environment, firmware, event)
        observation.target = target
        attached_unstarted = PhoneFirmware._matches_current_service_audio(environment, firmware, event)
        _ = await bot.started(environment, speaker, typing.cast(hsm.Model, audio.Speaker.model))
        ready = PhoneFirmware._matches_current_service_audio(environment, firmware, event)
        await hsm.stop(speaker)
        after_stop = PhoneFirmware._matches_current_service_audio(environment, firmware, event)
        return unattached, attached_unstarted, ready, after_stop

    unattached, attached_unstarted, ready, after_stop = asyncio.run(run())
    assert unattached is False
    assert attached_unstarted is False
    assert ready is True
    assert after_stop is False

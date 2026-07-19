"""speaker_ready_in_world needs attach hold + started speaker (hsm.id)."""

from __future__ import annotations

import asyncio

import hsm

from bot.devices import audio
from bot.devices.phone.phone import _PhoneObservationService
from bot.world import World


def test_speaker_ready_requires_attach_and_started_speaker() -> None:
    async def run() -> tuple[bool, bool, bool, bool]:
        world = World()
        speaker = audio.Speaker()
        owner = hsm.Instance()
        target = hsm.Instance()
        service = _PhoneObservationService(owner=owner, service=object(), speaker=speaker)  # type: ignore[arg-type]
        unattached = service.speaker_ready_in_world(world.context)
        service.target = target
        attached_unstarted = service.speaker_ready_in_world(world.context)
        await hsm.started(world.context, speaker, speaker.model)
        ready = service.speaker_ready_in_world(world.context)
        await hsm.stop(speaker)
        after_stop = service.speaker_ready_in_world(world.context)
        return unattached, attached_unstarted, ready, after_stop

    unattached, attached_unstarted, ready, after_stop = asyncio.run(run())
    assert unattached is False
    assert attached_unstarted is False
    assert ready is True
    assert after_stop is False

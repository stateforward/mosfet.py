"""The receiver forwards service audio only for its current call."""

from __future__ import annotations

import asyncio
import collections.abc
import typing

import hsm
import mosfet
import mosfet.lifecycle

from mosfet.devices import audio
from mosfet.devices import phone as phone_device
from mosfet.environment import Environment
from tests.hsm_instance_state import phone_firmware


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


class _RecordingSpeaker(audio.Speaker):
    """Speaker that records receiver audio forwarded by phone firmware."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[object] = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == audio.OutputEvent.name:
            self.received.append(event.data)
        return super().dispatch(ctx, event)


def test_receiver_forwards_only_current_call_service_audio() -> None:
    """Call correlation — not transducer readiness — gates the receiver path.

    The firmware guard admits service audio by payload type and current call id
    only; transducer liveness is a delivery-time concern, so this test drives the
    whole path through public phone seams and asserts who hears what.
    """

    async def run() -> tuple[int, int, int]:
        environment = Environment()
        speaker = _RecordingSpeaker()
        phone = phone_device.Phone(speaker=speaker, service=phone_device.EventRecorder())
        firmware = phone_firmware(phone)

        async def forwarded_count(call_id: str) -> int:
            await firmware.event_recorder().receive(phone.context(), _service_audio(call_id))
            await asyncio.sleep(0)
            return len(speaker.received)

        # No call yet: service audio for an unknown call goes nowhere.
        before_call = await forwarded_count("call-1")

        _ = await mosfet.started(environment, phone, typing.cast(hsm.Model, phone_device.Phone.model))
        assert mosfet.lifecycle.is_started(speaker)
        await firmware.event_recorder().receive(
            phone.context(),
            phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id="call-1")),
        )
        await phone.dispatch(
            phone.context(),
            phone_device.AnswerCallEvent.with_data(phone_device.AnswerCallData()),
        )
        await firmware.event_recorder().receive(
            phone.context(),
            phone_device.CallConnectedEvent.with_data(phone_device.CallConnectedData(call_id="call-1")),
        )
        await firmware.event_recorder().receive(
            phone.context(),
            phone_device.ServiceMediaReadyEvent.with_data(phone_device.MediaReadyData(call_id="call-1")),
        )

        # Current call on a started same-environment speaker: audio is forwarded.
        ready = await forwarded_count("call-1")
        # A live receiver never leaks another call's audio onto this speaker.
        after_wrong_call = await forwarded_count("call-2")
        return before_call, ready, after_wrong_call

    before_call, ready, after_wrong_call = asyncio.run(run())
    assert before_call == 0
    assert ready == 1
    assert after_wrong_call == 1

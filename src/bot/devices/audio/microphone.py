from .events import InputEvent, AudioInputData, routed_audio_event

import collections.abc
import typing

import hsm

from bot.device import Device
from bot.telemetry import observer
from bot.world import SoundData, SoundEvent, World


class Microphone(Device):
    """Transducer that hears the world and emits what it captures as audio input.

    Event-driven only: nothing hands a microphone audio. It observes ``world.sound`` and
    re-emits the acoustic energy as ``devices.audio.input`` for whichever device firmware
    models a transition on it. A phone's mouthpiece is exactly this — the bot speaks into
    the world, its phone microphone picks that up, and firmware carries it up the wire.
    """

    input_event: typing.ClassVar[hsm.Event[AudioInputData]] = InputEvent

    @staticmethod
    def _capture_world_sound(ctx: hsm.Context, instance: "Microphone", event: hsm.Event[typing.Any]) -> None:
        """Re-emit heard acoustic energy as audio input for owning firmware to route."""

        data = event.data
        if not isinstance(data, SoundData):
            return
        captured = routed_audio_event(
            Microphone.input_event,
            AudioInputData(
                audio=data.audio,
                media_type=data.media_type,
                sample_rate_hz=data.sample_rate_hz,
                channels=data.channels,
            ),
            source=instance,
            target=instance,
            metadata=event.metadata,
        )
        _ = hsm.dispatch_all(World.from_context(ctx), captured)

    firmware_model: typing.ClassVar[hsm.Model] = hsm.define(
        "MicrophoneFirmware",
        hsm.initial(hsm.target("/MicrophoneFirmware/listening")),
        hsm.state(
            "listening",
            hsm.transition(
                hsm.on(SoundEvent),
                hsm.effect(_capture_world_sound),
            ),
        ),
        hsm.observe(observer),
    )

    def dispatch_audio_input(
        self,
        ctx: hsm.Context,
        target: hsm.Instance,
        data: AudioInputData,
        *,
        metadata: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.Awaitable[None]:
        event = routed_audio_event(self.input_event, data, source=self, target=target, metadata=metadata)
        return target.dispatch(ctx, event)

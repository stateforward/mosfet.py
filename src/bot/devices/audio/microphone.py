from .events import InputEvent, InputData, routed_audio_event

import collections.abc
import typing

import hsm

from bot.device import Device
from bot.environment import SoundData, SoundEvent


class Microphone(Device):
    """Transducer that converts heard acoustic energy into audio input signal.

    A microphone makes no routing decision. It hears ``environment.sound`` and hands what it captured
    to whatever is attached to it; the controller that attached — phone firmware, for a
    mouthpiece — decides what the signal is for. It has no firmware of its own: a real
    microphone is a transducer, not a computer.

    Transduction happens only while attached, exactly as an unwired microphone produces nothing.
    A phone's mouthpiece is therefore live precisely while its firmware holds the attachment.
    """

    input_event: typing.ClassVar[hsm.Event[InputData]] = InputEvent

    @staticmethod
    def _transduce(ctx: hsm.Context, instance: "Microphone", event: hsm.Event[typing.Any]) -> None:
        """Convert heard acoustic energy to signal for every attached controller."""

        data = event.data
        if not isinstance(data, SoundData):
            return
        captured = InputData(
            audio=data.audio,
            media_type=data.media_type,
            sample_rate_hz=data.sample_rate_hz,
            channels=data.channels,
        )
        for controller in instance._attachments:
            _ = instance.dispatch_audio_input(ctx, controller, captured, metadata=event.metadata)

    model: typing.ClassVar[hsm.Model | None] = hsm.redefine(
        typing.cast(hsm.Model, Device.model),
        hsm.transition(hsm.source("attached"), hsm.on(SoundEvent), hsm.effect(_transduce)),
    )

    def dispatch_audio_input(
        self,
        ctx: hsm.Context,
        target: hsm.Instance,
        data: InputData,
        *,
        metadata: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.Awaitable[None]:
        event = routed_audio_event(self.input_event, data, source=self, target=target, metadata=metadata)
        return target.dispatch(ctx, event)

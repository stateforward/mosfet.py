from .events import OutputEvent, AudioOutputData, routed_audio_event

import collections.abc
import dataclasses
import typing

import hsm

from bot.device import Device
from bot.world import SoundData, SoundEvent, World


class Speaker(Device):
    """Transducer that converts audio output signal into acoustic energy in the world.

    The mirror of :class:`~bot.devices.audio.microphone.Microphone`: a microphone converts what
    it hears into signal on the way in, a speaker converts signal into sound on the way out.
    Both are transducers with no firmware and no routing decision of their own; the difference is
    only which way the attachment points. The controller that attached decides what to play.

    Transduction happens only while attached, exactly as an unwired speaker is silent.
    """

    output_event: typing.ClassVar[hsm.Event[AudioOutputData]] = OutputEvent

    @staticmethod
    def _transduce(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        """Convert signal from an attached controller into acoustic energy in the world.

        World broadcast is the input elevation path: bots fan ``world.sound`` to input abilities
        and never treat raw device playout as cognition input. Targeted playout into another
        device still uses ``devices.audio.output`` via :meth:`dispatch_audio_output`.
        """

        data = event.data
        if not isinstance(data, AudioOutputData):
            return
        sound = dataclasses.replace(
            SoundEvent.with_data(
                SoundData(
                    audio=data.audio,
                    media_type=data.media_type,
                    sample_rate_hz=data.sample_rate_hz,
                    channels=data.channels,
                )
            ),
            source=hsm.id(instance),
            metadata=dict(event.metadata),
        )
        _ = World.from_context(ctx).broadcast(sound)

    model: typing.ClassVar[hsm.Model | None] = hsm.redefine(
        typing.cast(hsm.Model, Device.model),
        hsm.transition(hsm.source("attached"), hsm.on(OutputEvent), hsm.effect(_transduce)),
    )

    def dispatch_audio_output(
        self,
        ctx: hsm.Context,
        target: hsm.Instance,
        data: AudioOutputData,
        *,
        metadata: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.Awaitable[None]:
        event = routed_audio_event(self.output_event, data, source=self, target=target, metadata=metadata)
        return target.dispatch(ctx, event)

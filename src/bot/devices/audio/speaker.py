from .events import OutputEvent, AudioOutputData, routed_audio_event

import collections.abc
import dataclasses
import typing

import hsm

from bot.device import Device
from bot.world import SoundData, SoundEvent, World


class Speaker(Device):
    """Audio output event dispatcher for another device or audio implementation."""

    output_event: typing.ClassVar[hsm.Event[AudioOutputData]] = OutputEvent

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

    def dispatch_audio_output_to_world(
        self,
        ctx: hsm.Context,
        data: AudioOutputData,
        *,
        metadata: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.Awaitable[None]:
        """Broadcast acoustic energy into the world as ``world.sound`` for bot input abilities.

        Targeted device playout still uses ``devices.audio.output`` via
        :meth:`dispatch_audio_output`. World broadcast is the input elevation path:
        bots fan ``world.sound`` to input abilities and never treat raw device
        playout as cognition input.
        """

        sound = dataclasses.replace(
            SoundEvent.with_data(
                SoundData(
                    audio=data.audio,
                    media_type=data.media_type,
                    sample_rate_hz=data.sample_rate_hz,
                    channels=data.channels,
                )
            ),
            source=hsm.id(self),
            metadata=dict(metadata) if metadata is not None else {},
        )
        return hsm.dispatch_all(World.from_context(ctx), sound)

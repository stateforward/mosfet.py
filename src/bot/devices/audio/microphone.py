from .events import InputEvent, AudioInputData, routed_audio_event

import collections.abc
import typing

import hsm

from bot.device import Device

class Microphone(Device):
    """Audio input event dispatcher for another device or audio implementation."""

    input_event: typing.ClassVar[hsm.Event[AudioInputData]] = InputEvent

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

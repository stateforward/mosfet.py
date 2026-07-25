from .events import InputEvent, AudioInputData, routed_audio_event

import asyncio
import collections.abc
import typing

import hsm

from bot.device import Device

class Microphone(Device):
    """Audio input event dispatcher for another device or audio implementation."""

    input_event: typing.ClassVar[hsm.Event[AudioInputData]] = InputEvent
    # Where captured audio goes, wired by the owning device (a Phone points this at its own
    # firmware, giving microphone -> firmware -> service). None means nothing is listening yet.
    _uplink: hsm.Instance | None = None

    def connect_uplink(self, target: hsm.Instance) -> None:
        """Point captured audio at the owning device's firmware."""

        self._uplink = target

    def capture(
        self,
        ctx: hsm.Context,
        data: AudioInputData,
        *,
        metadata: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.Awaitable[None]:
        """Dispatch captured audio to the connected uplink, if any."""

        uplink = self._uplink
        if uplink is None:
            future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return future
        return self.dispatch_audio_input(ctx, uplink, data, metadata=metadata)

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

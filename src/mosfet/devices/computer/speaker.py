from mosfet.devices import audio

import collections.abc
import dataclasses
import typing

import hsm

from mosfet.device import Device
from mosfet.environment import space
from mosfet.protocols import transform

# The speaker stream's wire form, here only to label what is dropped while no stream is owned.
_CODEC = audio.PcmCodec()


class Speaker(Device):
    """Remote-side output port: plays what its controller drives on the remote machine's speaker.

    Unlike :class:`mosfet.devices.audio.Speaker`, this speaker is not in the room: it never
    broadcasts ``environment.sound``. While attached, ``devices.audio.output`` from its controller
    (the computer's operating system) is sent, one way, to the speaker stream this port owns. Which
    stream that is, it is told: ``protocol.transform.owned`` hands it one and
    ``protocol.transform.ended`` takes that one away. With no stream a frame is dropped and counted on
    ``bot.transform.dropped.count{reason=unowned}``; the stream answers nothing back here.

    It plays only while attached, exactly as an unwired speaker is silent.
    """

    # The stream this speaker owns: an address, dispatched to and never read. Written only by the
    # owned and ended transitions.
    _stream: hsm.Instance | None

    def __init__(
        self,
        *,
        peripherals: collections.abc.Iterable[Device] = (),
        placement: space.Placement | None = None,
    ) -> None:
        super().__init__(peripherals=peripherals, placement=placement)
        self._stream = None

    @staticmethod
    def _is_frame(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, audio.OutputData)

    @staticmethod
    def _is_streamed_frame(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, audio.OutputData) and instance._stream is not None

    @staticmethod
    def _play(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        """Send the frame, one way, to the stream this speaker owns."""

        stream = instance._stream
        assert stream is not None
        _ = hsm.dispatch(
            ctx,
            stream,
            dataclasses.replace(event, source=hsm.id(instance), target=hsm.id(stream), metadata=dict(event.metadata)),
        )

    @staticmethod
    def _drop_unowned(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        transform.telemetry.dropped(_CODEC, "unowned")

    @staticmethod
    def _is_owned(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, transform.OwnedData)

    @staticmethod
    def _own(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, transform.OwnedData)
        instance._stream = data.stream

    @staticmethod
    def _is_own_stream_ended(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> bool:
        """Only the end of the stream this speaker owns takes it away; an older stream's end is stale."""

        del ctx
        data = event.data
        return isinstance(data, transform.EndedData) and data.stream is instance._stream

    @staticmethod
    def _disown(ctx: hsm.Context, instance: "Speaker", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._stream = None

    model: typing.ClassVar[hsm.Model | None] = hsm.redefine(
        typing.cast(hsm.Model, Device.model),
        hsm.transition(hsm.on(transform.OwnedEvent), hsm.guard(_is_owned), hsm.effect(_own)),
        hsm.transition(hsm.on(transform.EndedEvent), hsm.guard(_is_own_stream_ended), hsm.effect(_disown)),
        hsm.transition(
            hsm.source("attached"), hsm.on(audio.OutputEvent), hsm.guard(_is_streamed_frame), hsm.effect(_play)
        ),
        hsm.transition(
            hsm.source("attached"), hsm.on(audio.OutputEvent), hsm.guard(_is_frame), hsm.effect(_drop_unowned)
        ),
    )

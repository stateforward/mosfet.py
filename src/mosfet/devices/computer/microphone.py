from mosfet.devices import audio

import dataclasses
import typing

import hsm

from mosfet.device import Device


class Microphone(Device):
    """Remote-side port that hands what the remote microphone captured to whatever is attached to it.

    Unlike :class:`mosfet.devices.audio.Microphone`, this microphone is not in the room: it owns the
    link's microphone stream, which delivers the remote machine's captured frames as
    ``devices.audio.input`` here, and
    the port re-addresses each one, from itself, to every controller holding it. It has no
    ``environment.sound`` transition, so sound in the computer's own room never reaches it — which
    is what keeps the room from being captured and relabelled as the remote person.

    Frames flow only while attached, exactly as an unplugged microphone captures nothing.
    """

    @staticmethod
    def _is_frame(ctx: hsm.Context, instance: "Microphone", event: hsm.Event[typing.Any]) -> bool:
        """Only a validated frame is a frame; the stream's codec validates wire data before delivering it."""

        del ctx, instance
        return isinstance(event.data, audio.InputData)

    @staticmethod
    def _transduce(ctx: hsm.Context, instance: "Microphone", event: hsm.Event[typing.Any]) -> None:
        """Re-address the frame, from this port, to every attached controller."""

        for controller in instance._attachments:
            _ = hsm.dispatch(
                ctx,
                controller,
                dataclasses.replace(
                    event, source=hsm.id(instance), target=hsm.id(controller), metadata=dict(event.metadata)
                ),
            )

    model: typing.ClassVar[hsm.Model | None] = hsm.redefine(
        typing.cast(hsm.Model, Device.model),
        hsm.transition(hsm.source("attached"), hsm.on(audio.InputEvent), hsm.guard(_is_frame), hsm.effect(_transduce)),
    )

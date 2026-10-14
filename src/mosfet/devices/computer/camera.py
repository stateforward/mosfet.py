from .events import StillData, StillEvent

import dataclasses
import typing

import hsm

from mosfet.device import Device


class Camera(Device):
    """Remote-side port that hands each still the remote camera took to whatever is attached to it.

    The camera owns the link's camera stream: its codec validates each still into ``media.Image`` and
    the stream delivers ``devices.computer.camera.still`` here; the camera re-addresses it, from itself, to every
    controller holding it, and the controller (the computer's operating system) decides what the
    still is for. It has no ``environment.*`` transition: light in the room the computer is in is not
    what the remote camera sees, so environment broadcasts reach it and go unhandled.

    Stills flow only while attached, exactly as an unplugged camera shows nothing.
    """

    @staticmethod
    def _is_still(ctx: hsm.Context, instance: "Camera", event: hsm.Event[typing.Any]) -> bool:
        """Only a validated still is a still; the stream's codec validates wire data before delivering it."""

        del ctx, instance
        return isinstance(event.data, StillData)

    @staticmethod
    def _transduce(ctx: hsm.Context, instance: "Camera", event: hsm.Event[typing.Any]) -> None:
        """Re-address the still, from this port, to every attached controller."""

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
        hsm.transition(hsm.source("attached"), hsm.on(StillEvent), hsm.guard(_is_still), hsm.effect(_transduce)),
    )

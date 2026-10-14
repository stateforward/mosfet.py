from .events import PointerData, PointerEvent

import dataclasses
import typing

import hsm

from mosfet.device import Device


class Pointer(Device):
    """Remote-side port that hands each pointer transition a person made to whatever is attached to it.

    A pointer makes no decision about what a press or a movement means. The pointer stream it owns,
    from a remote mouse, touch surface or pen, delivers ``devices.hid.pointer`` here, and the pointer
    re-addresses it, from itself, to every controller holding it; the controller (a computer's
    operating system) hit-tests and decides what it is for. It has no firmware of its own and no
    ``environment.*`` transition, so environment broadcasts reach it and go unhandled.

    Pointer transitions flow only while attached, exactly as an unplugged mouse moves nothing.
    """

    @staticmethod
    def _is_pointer(ctx: hsm.Context, instance: "Pointer", event: hsm.Event[typing.Any]) -> bool:
        """Only a validated pointer transition counts; the stream's codec validates wire data first."""

        del ctx, instance
        return isinstance(event.data, PointerData)

    @staticmethod
    def _transduce(ctx: hsm.Context, instance: "Pointer", event: hsm.Event[typing.Any]) -> None:
        """Re-address the pointer transition, from this port, to every attached controller."""

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
        hsm.transition(hsm.source("attached"), hsm.on(PointerEvent), hsm.guard(_is_pointer), hsm.effect(_transduce)),
    )

from .events import KeyData, KeyEvent

import dataclasses
import typing

import hsm

from mosfet.device import Device


class Keyboard(Device):
    """Remote-side port that hands each key a person pressed to whatever is attached to it.

    A keyboard makes no decision about what a key means. The link that carries the remote keyboard
    delivers ``devices.hid.key`` here, and the keyboard re-addresses it, from itself, to every
    controller holding it; the controller (a computer's operating system) decides what the key is
    for. It has no firmware of its own and no ``environment.*`` transition: a keystroke on a remote
    machine is not something the room hears, so environment broadcasts reach it and go unhandled.

    Keys flow only while attached, exactly as an unplugged keyboard types into nothing.
    """

    @staticmethod
    def _is_key(ctx: hsm.Context, instance: "Keyboard", event: hsm.Event[typing.Any]) -> bool:
        """Only a validated key is a key; the link validates wire data before delivering it."""

        del ctx, instance
        return isinstance(event.data, KeyData)

    @staticmethod
    def _transduce(ctx: hsm.Context, instance: "Keyboard", event: hsm.Event[typing.Any]) -> None:
        """Re-address the key, from this port, to every attached controller."""

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
        hsm.transition(hsm.source("attached"), hsm.on(KeyEvent), hsm.guard(_is_key), hsm.effect(_transduce)),
    )

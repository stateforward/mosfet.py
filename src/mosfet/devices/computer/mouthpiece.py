from .events import HeardData, HeardEvent

import dataclasses
import typing

import hsm

from mosfet.device import Device
from mosfet.environment import SoundData, SoundEvent


class Mouthpiece(Device):
    """Bot-side port: what the computer hears in the room it is in, handed to its operating system.

    The one environment-facing port a computer has, and the way a bot's voice reaches the remote
    speaker: like a handset's mouthpiece, it sits with the computer and hears ``environment.sound``
    there. Each sound it hears is handed, from itself, to every controller holding it as
    ``devices.computer.mouthpiece.heard``, carrying the received sound exactly as it arrived — the
    same object, subtype included, never re-validated or re-labelled. It makes no decision: it does
    not filter by type or by who made the sound. Whether a sound goes up the link is the operating
    system's to decide.

    It hears only while attached, exactly as an unwired mouthpiece carries nothing.
    """

    @staticmethod
    def _is_sound(ctx: hsm.Context, instance: "Mouthpiece", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, SoundData)

    @staticmethod
    def _hear(ctx: hsm.Context, instance: "Mouthpiece", event: hsm.Event[typing.Any]) -> None:
        """Hand the sound, as received, to every attached controller."""

        data = event.data
        assert isinstance(data, SoundData)
        heard = HeardData(sound=data)
        for controller in instance._attachments:
            _ = hsm.dispatch(
                ctx,
                controller,
                dataclasses.replace(
                    HeardEvent.with_data(heard),
                    id=event.id,
                    source=hsm.id(instance),
                    target=hsm.id(controller),
                    metadata=dict(event.metadata),
                ),
            )

    model: typing.ClassVar[hsm.Model | None] = hsm.redefine(
        typing.cast(hsm.Model, Device.model),
        hsm.transition(hsm.source("attached"), hsm.on(SoundEvent), hsm.guard(_is_sound), hsm.effect(_hear)),
    )

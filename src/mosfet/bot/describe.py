"""``bot.describe``: what defined bots are, as JSON, read from their models alone.

``bot.describe(*models)`` reads each model's members and constructs, starts and opens nothing. It
names what each bot is made of: its traits, its faculties, its parts by id, kind, role and the
class each was constructed from, and its firmware's commands and states. No value given to a part's
constructor, a timeout, a faculty or a firmware attribute appears in it::

    print(bot.describe(model).model_dump_json(indent=2))
"""

import typing

import hsm
import pydantic

import mosfet.device
from mosfet.abilities import memory

from . import elements, firmware

_FROZEN = pydantic.ConfigDict(frozen=True, extra="forbid")

Timeout = typing.Literal["processing", "deactivation", "attachment"]
"""A body timeout ``bot.timeouts`` can override."""


class Part(pydantic.BaseModel):
    """One device the bot holds or ability it has, as ``bot.attach``, ``bot.input`` or ``bot.output`` gave it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    id: str = pydantic.Field(
        description="The id the bot knows the part by: its key on the bot's tool menu and its address.",
        examples=["desk"],
    )
    kind: typing.Literal["device", "ability"] = pydantic.Field(
        description="`device` for a device the bot holds, `ability` for an ability it has.", examples=["device"]
    )
    roles: tuple[elements.Role, ...] = pydantic.Field(
        description=(
            "How the body takes it, in the order given: `attach` (a held device or an acquired ability), "
            "`input` (a sense organ on the stimulus fan-out) and `output` (an effector port)."
        ),
        examples=[["attach"], ["attach", "input"]],
    )
    constructor: str = pydantic.Field(description="The class the part was constructed from.", examples=["Computer"])


class FirmwareOutline(pydantic.BaseModel):
    """The bot's one firmware: the commands cognition may be offered and the states it can be in."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    commands: tuple[str, ...] = pydantic.Field(
        description=(
            "The tool events its transitions react to, in the order it names them: offered to cognition "
            "under the actor key `firmware` while a state that reacts to them is active."
        ),
        examples=[["timer.set", "timer.cancel"]],
    )
    states: tuple[str, ...] = pydantic.Field(
        description="Its states by qualified name, in the order they were written, nested states after their parent.",
        examples=[["/timer_firmware/idle", "/timer_firmware/running"]],
    )


class Outline(pydantic.BaseModel):
    """One bot: its traits, its faculties and its parts."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    name: str = pydantic.Field(description="The bot's name, as `bot.define` was given it.", examples=["greeter"])
    traits: tuple[str, ...] = pydantic.Field(
        description="The owner's traits, the seed of the bot's nature; with traits it has a persona.",
        examples=[["curious", "patient"]],
    )
    faculties: tuple[str, ...] = pydantic.Field(
        description="Its cognition's faculties: intuition, reasoning and reflection, then each one it was given.",
        examples=[["intuition", "reasoning", "reflection", "associative"]],
    )
    memory: str | None = pydantic.Field(
        description="The class of the memory its faculties share, or null for a fresh in-memory one.",
        examples=["Memory", None],
    )
    parts: tuple[Part, ...] = pydantic.Field(description="Its devices and abilities, in the order they were given.")
    sensory_register: str | None = pydantic.Field(
        description="The class of its sensory register, or null without one.", examples=["StmMemory", None]
    )
    timeouts: tuple[Timeout, ...] = pydantic.Field(
        description="The body timeouts it overrides (never their values).", examples=[["processing"]]
    )
    allowed: tuple[str, ...] = pydantic.Field(
        description="The lints it allows with `bot.allow`.", examples=[["no_devices"]]
    )
    firmware: FirmwareOutline | None = pydantic.Field(
        description="Its firmware, compiled from the hsm partials given to `bot.define`, or null without any."
    )


class Description(pydantic.BaseModel):
    """Defined bots, in the order they were described."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _FROZEN

    bots: tuple[Outline, ...] = pydantic.Field(description="Each bot described, in order.")


def describe(*models: elements.Model) -> Description:
    """What each of ``models`` is made of, read from the model alone; nothing is constructed or started."""

    return Description(bots=tuple(_outline(model) for model in models))


def _outline(model: elements.Model) -> Outline:
    traits: tuple[str, ...] = ()
    faculties: tuple[str, ...] = ()
    store: str | None = None
    parts: list[Part] = []
    sensory_register: str | None = None
    timeouts: tuple[Timeout, ...] = ()
    allowed: tuple[str, ...] = ()
    outlined: FirmwareOutline | None = None
    for member in model.members.values():
        match member:
            case elements.AttachmentElement(instance=instance, roles=roles):
                kind = "device" if isinstance(instance, mosfet.device.Device) else "ability"
                parts.append(Part(id=member.name(), kind=kind, roles=tuple(roles), constructor=type(instance).__name__))
            case elements.TraitsElement(traits=given):
                traits = tuple(str(trait) for trait in given)
            case elements.CognitionElement(memory=given):
                faculties = member.faculties()
                store = type(given).__name__ if isinstance(given, memory.Memory) else None
            case elements.RegisterElement(store=given):
                sensory_register = type(given).__name__
            case elements.TimeoutsElement():
                overrides: tuple[tuple[Timeout, object], ...] = (
                    ("processing", member.processing),
                    ("deactivation", member.deactivation),
                    ("attachment", member.attachment),
                )
                timeouts = tuple(name for name, value in overrides if value is not None)
            case elements.AllowElement(lints=given):
                allowed = tuple(str(lint) for lint in given)
            case firmware.FirmwareElement(model=compiled):
                outlined = FirmwareOutline(
                    commands=tuple(command.name for command in member.commands()),
                    states=tuple(
                        vertex.qualified_name
                        for vertex in compiled.members.values()
                        if isinstance(vertex, hsm.StateElement) and vertex is not compiled
                    ),
                )
            case _:
                pass
    return Outline(
        name=model.qualified_name[1:],
        traits=traits,
        faculties=faculties,
        memory=store,
        parts=tuple(parts),
        sensory_register=sensory_register,
        timeouts=timeouts,
        allowed=allowed,
        firmware=outlined,
    )


__all__ = ["Description", "FirmwareOutline", "Outline", "Part", "Timeout", "describe"]

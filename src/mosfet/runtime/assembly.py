"""What a bot factory returns: the named bots one process runs, and the shared actors they use."""

import collections.abc
import dataclasses
import re
import types
import typing

import hsm
import mosfet
import pydantic

from mosfet.environment import space

from . import errors

MAX_BOTS = 32
"""Most bots one process supervises. Bot names are telemetry attributes, so this bounds their cardinality."""

_BOT_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
_BOT_NAME_MAX_LENGTH = 64

BotName = typing.Annotated[
    str,
    pydantic.Field(
        pattern=_BOT_NAME_PATTERN,
        max_length=_BOT_NAME_MAX_LENGTH,
        description="Author-chosen name of one supervised bot: lower snake case, at most 64 characters.",
        examples=["builder"],
    ),
]


@dataclasses.dataclass(frozen=True)
class Assembly:
    """The bots one ``mosfet run`` process supervises, and the root-owned actors they share.

    * ``bots``: one to :data:`MAX_BOTS` bots under author-chosen names (lower snake case, at most 64
      characters). Each name gets its own supervisor; one bot instance may appear only once.
    * ``actors``: shared actors such as a websocket server or an embedding model. The runtime starts
      each, in order, with its class ``model`` before any bot, and stops them in reverse order after
      every bot has stopped. Each actor's class must declare ``model`` as an ``hsm.Model``.
    * ``placements``: where a named bot joins the process environment; keys must be bot names.

    Mappings are copied into read-only views. A violation raises ``errors.BootError`` with kind
    ``invalid_assembly``.
    """

    bots: collections.abc.Mapping[str, "mosfet.Bot"]
    actors: tuple[hsm.Instance, ...] = ()
    placements: collections.abc.Mapping[str, space.Placement] = dataclasses.field(
        default_factory=lambda: types.MappingProxyType({})
    )

    def __post_init__(self) -> None:
        # Factories are untyped code: check what they built, not what they were annotated to build.
        bots = typing.cast(dict[str, object], dict(self.bots))
        if not 1 <= len(bots) <= MAX_BOTS:
            raise errors.BootError("invalid_assembly", f"an assembly needs 1 to {MAX_BOTS} bots, got {len(bots)}")
        seen: set[int] = set()
        for name, bot in bots.items():
            if len(name) > _BOT_NAME_MAX_LENGTH or not re.fullmatch(_BOT_NAME_PATTERN, name):
                raise errors.BootError(
                    "invalid_assembly",
                    f"bot name {name!r} must be lower snake case and at most {_BOT_NAME_MAX_LENGTH} characters",
                )
            if not isinstance(bot, mosfet.Bot):
                raise errors.BootError("invalid_assembly", f"bot {name!r} is {type(bot).__name__}, not mosfet.Bot")
            if id(bot) in seen:
                raise errors.BootError("invalid_assembly", f"bot {name!r} is already assembled under another name")
            seen.add(id(bot))
        actors = typing.cast(tuple[object, ...], tuple(self.actors))
        for actor in actors:
            if not isinstance(actor, hsm.Instance) or not isinstance(getattr(type(actor), "model", None), hsm.Model):
                raise errors.BootError(
                    "invalid_assembly",
                    f"actor {type(actor).__name__} must be an hsm.Instance whose class declares model as an hsm.Model",
                )
        placements = dict(self.placements)
        for name in placements:
            if name not in bots:
                raise errors.BootError("invalid_assembly", f"placement {name!r} does not name an assembled bot")
        object.__setattr__(self, "bots", types.MappingProxyType(bots))
        object.__setattr__(self, "actors", actors)
        object.__setattr__(self, "placements", types.MappingProxyType(placements))


__all__ = ["MAX_BOTS", "Assembly", "BotName"]

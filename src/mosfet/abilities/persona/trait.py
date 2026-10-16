"""Traits: the fixed seed of a bot's nature, given by its owner.

Traits come from composition and nowhere else. The bot never writes them, memory never stores them
as their own row, and no turn offers a way to change them. A bot grows a persona on top of its
traits; the traits themselves are its makers' to give.
"""

from __future__ import annotations

import typing

import pydantic

Trait: typing.TypeAlias = typing.Annotated[
    str,
    pydantic.StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
"""One trait in plain language, a word or a short phrase, trimmed."""


class Traits(pydantic.BaseModel):
    """The traits a bot's owner gave it, in the owner's order."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "The traits the bot's owner gave it: the fixed seed of its nature. They come from "
                "composition, never from the bot, and the bot's own description grows on top of them."
            ),
            "examples": [{"traits": ["curious", "patient with children"]}],
        },
    )

    traits: tuple[Trait, ...] = pydantic.Field(
        min_length=1,
        max_length=24,
        description="One to 24 traits, each a word or short phrase of at most 200 characters.",
        examples=[["curious", "patient with children"]],
    )


__all__ = ["Trait", "Traits"]

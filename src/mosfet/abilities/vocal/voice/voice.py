"""The voice a bot speaks in, as its owner is told it: given at inception, with what it can do with an expression."""

from __future__ import annotations

import typing
from xml.sax import saxutils

import hsm
import pydantic

from .. import speech
from . import catalog


class VoiceData(pydantic.BaseModel):
    """The voice the bot was given at inception, and what it can do with an expression."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "The voice the bot was given when it was made, fixed for life: the description it was made from, "
                "the languages it speaks, and what it can do with a part's expression."
            ),
        },
    )

    ref: catalog.Ref = pydantic.Field(description="The voice, as its source holds it.")
    origin: typing.Literal["designed", "catalog"] = pydantic.Field(
        description="Whether the voice was designed for this bot, or is one its source already held."
    )
    description: str = pydantic.Field(min_length=1, max_length=1000, description="What the voice was made from.")
    languages: tuple[speech.Locale, ...] = pydantic.Field(default=(), description="The languages it speaks.")
    expressiveness: speech.Expressiveness = pydantic.Field(
        description="What the voice can do with a part's expression: its cues, vocabularies, ranges and sounds."
    )

    def __model_repr__(self) -> str:
        """The voice block: given at inception, its languages, its description, then its expressiveness."""

        languages = f' languages="{saxutils.escape(" ".join(self.languages))}"' if self.languages else ""
        return (
            f'<voice given_at="inception"{languages}>{saxutils.escape(self.description)}'
            f"<expressiveness>{self.expressiveness.__model_repr__()}</expressiveness></voice>"
        )


OutputEvent = hsm.Event[VoiceData](
    name="bot.ability.voice.output",
    schema=VoiceData,
)
"""The bot speaks in the voice it was given: told to the owner once Speaking has constructed it."""


__all__ = ["OutputEvent", "VoiceData"]

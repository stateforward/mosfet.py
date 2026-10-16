"""The voice a bot was given at inception, stored the way every other self-fact is.

A given voice is a **self-fact**: a ``memory`` fact row about ``memory.SELF_SUBJECT`` tagged
``query_tags=self_voice``, written once per source and read with the shared self-fact helpers. It records the voice
the bot's makers gave it and the traits it was made for; nothing about it is the bot's choice, and it never changes.
"""

from __future__ import annotations

import datetime
import typing

import pydantic

from ... import memory
from ... import persona
from .. import speech
from . import catalog

# Stable query_tags value for the voice the bot was given, in bot_memory.
VOICE_QUERY = "self_voice"


class Given(pydantic.BaseModel):
    """The voice a bot was given when it was made: its ref, how it was made, and the traits it was made for."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "The voice the bot was given at inception (JSON in memory, query_tags=self_voice), fixed for life."
            ),
        },
    )

    ref: catalog.Ref = pydantic.Field(description="The voice, as its source holds it.")
    origin: typing.Literal["designed", "catalog"] = pydantic.Field(
        description="Whether the voice was designed for this bot, or is one its source already held."
    )
    description: str = pydantic.Field(
        min_length=1, max_length=1000, description="The description the voice was made from, or its catalog name."
    )
    descriptors: catalog.Descriptors | None = pydantic.Field(default=None, description="How it sounds.")
    languages: tuple[speech.Locale, ...] = pydantic.Field(default=(), description="The languages it speaks.")
    traits: tuple[persona.Trait, ...] = pydantic.Field(
        min_length=1, max_length=24, description="The owner-given traits the voice was made for."
    )
    kind: typing.Literal["fact"] = pydantic.Field(
        default="fact", description="Always `fact`: the voice is a fact about the bot.", examples=["fact"]
    )


def recalled_given(output: memory.OutputData, *, statement_index: int = 0) -> tuple[Given, ...]:
    """Every given voice in a committed memory SELECT result, oldest first.

    A row that does not decode as a :class:`Given` is skipped, so one damaged row never costs the bot its voice.
    """

    given: list[Given] = []
    for content in output.contents(statement_index=statement_index):
        try:
            given.append(Given.model_validate_json(content))
        except pydantic.ValidationError:
            continue
    return tuple(given)


def given_select_input(*, limit: int = 50) -> memory.InputData:
    """Memory apply input that SELECTs the ``limit`` most recent given voices, oldest first."""

    return memory.self_fact_select_input(VOICE_QUERY, limit=limit)


def given_insert_input(
    given: Given,
    *,
    memory_id: str | None = None,
    created_at: datetime.datetime | None = None,
) -> memory.InputData:
    """Memory apply input that INSERTs the given voice as a self-fact."""

    return memory.self_fact_insert_input(VOICE_QUERY, given, created_at=created_at, memory_id=memory_id)


__all__ = ["VOICE_QUERY", "Given", "given_insert_input", "given_select_input", "recalled_given"]

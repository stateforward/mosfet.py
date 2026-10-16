"""The description a bot writes of itself, stored the way every other self-fact is.

A description is a **self-fact**: a ``memory`` fact row about ``memory.SELF_SUBJECT`` tagged
``query_tags=self_persona``, written and read with the shared self-fact helpers. It records who the
bot takes itself to be, never something it was told to do.

Revision is append-only. Every row records the traits it was written under, so a description kept
across a change of traits still says what nature it grew from; the current description is simply
the most recent row.
"""

from __future__ import annotations

from .. import memory
from .trait import Trait

import datetime
import typing

import pydantic

# Stable query_tags value for the bot's self-written descriptions in bot_memory.
PERSONA_QUERY = "self_persona"


class Description(pydantic.BaseModel):
    """One description the bot wrote of itself, with why, and the traits it was written under.

    No example text, deliberately: an example description in a schema is a description the bot
    could adopt as its own without ever having written it.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A description the bot wrote of itself (JSON in memory, query_tags=self_persona). A bot "
                "starts with none; this row exists only because the bot revised its description."
            ),
        },
    )

    text: str = pydantic.Field(
        min_length=1,
        max_length=2000,
        description="The whole description, in the bot's own words, exactly as it wrote it.",
    )
    reason: str = pydantic.Field(
        min_length=1,
        max_length=500,
        description="Why the bot revised its description, in its own words.",
    )
    traits: tuple[Trait, ...] = pydantic.Field(
        min_length=1,
        max_length=24,
        description="The owner-given traits the bot had when it wrote this description.",
    )
    kind: typing.Literal["fact"] = pydantic.Field(
        default="fact",
        description="Always `fact`: a description is a fact about the bot, never an instruction.",
        examples=["fact"],
    )

    @pydantic.field_validator("text", "reason")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


def latest_description(output: memory.OutputData, *, statement_index: int = 0) -> Description | None:
    """Return the most recent description in a committed memory SELECT result, or None when there is none.

    A row that does not decode as a :class:`Description` is skipped, so one damaged row never costs
    the bot the description written before it.
    """

    latest: Description | None = None
    for content in output.contents(statement_index=statement_index):
        try:
            latest = Description.model_validate_json(content)
        except pydantic.ValidationError:
            continue
    return latest


def description_select_input(*, limit: int = 50) -> memory.InputData:
    """Memory apply input that SELECTs the ``limit`` most recent descriptions, oldest first."""

    return memory.self_fact_select_input(PERSONA_QUERY, limit=limit)


def description_insert_input(
    description: Description,
    *,
    memory_id: str | None = None,
    created_at: datetime.datetime | None = None,
) -> memory.InputData:
    """Memory apply input that INSERTs one description as a self-fact, stamped to keep revision order."""

    return memory.self_fact_insert_input(PERSONA_QUERY, description, created_at=created_at, memory_id=memory_id)


__all__ = [
    "PERSONA_QUERY",
    "Description",
    "description_insert_input",
    "description_select_input",
    "latest_description",
]

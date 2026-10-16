"""The name a bot answers to, stored the way every other durable fact is.

A name is a **self-fact**: a ``memory`` fact row about ``memory.SELF_SUBJECT`` tagged
``query_tags=self_name``, written and read with the shared self-fact helpers. A name is a fact
row, not an episode and not an instruction: it records something the bot *is*, never something it
was told to *do*. Nothing about a name is ever completed, released, or acted on.

Adoption is append-only. A bot that was renamed still remembers having been called something
else, so the current name is simply the most recently adopted row.
"""

from __future__ import annotations

from .. import memory

import datetime
import typing

import pydantic

# Stable query_tags value for adopted names in bot_memory.
NAME_QUERY = "self_name"


class Name(pydantic.BaseModel):
    """One name the bot adopted, exactly as it was given."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A name the bot answers to (JSON in memory, query_tags=self_name). A bot starts with "
                "none; this row exists only because the bot adopted one."
            ),
            "examples": [{"text": "Bob", "kind": "fact"}],
        },
    )

    text: str = pydantic.Field(
        min_length=1,
        description="The name as the bot adopted it, in plain language.",
        examples=["Bob"],
    )
    kind: typing.Literal["fact"] = pydantic.Field(
        default="fact",
        description="Always `fact`: a name is a fact about the bot, never an instruction.",
        examples=["fact"],
    )


def names_from_output(output: memory.OutputData, *, statement_index: int = 0) -> tuple[Name, ...]:
    """Decode adopted names from a committed memory SELECT result, oldest first."""

    names: list[Name] = []
    for content in output.contents(statement_index=statement_index):
        try:
            names.append(Name.model_validate_json(content))
        except Exception:
            continue
    return tuple(names)


def adopted_name(output: memory.OutputData, *, statement_index: int = 0) -> Name | None:
    """Return the name the bot currently answers to, or None when it has never adopted one."""

    names = names_from_output(output, statement_index=statement_index)
    return names[-1] if names else None


def name_select_input(*, limit: int = 50) -> memory.InputData:
    """Memory apply input that SELECTs the ``limit`` most recently adopted names, oldest first."""

    return memory.self_fact_select_input(NAME_QUERY, limit=limit)


def name_insert_input(
    name: Name,
    *,
    memory_id: str | None = None,
    created_at: datetime.datetime | None = None,
) -> memory.InputData:
    """Memory apply input that INSERTs one adopted name as a self-fact.

    "Which name is current" is the order of adoption, which the self-fact stamp keeps even for two
    adoptions made in the same second.
    """

    return memory.self_fact_insert_input(NAME_QUERY, name, created_at=created_at, memory_id=memory_id)


__all__ = [
    "NAME_QUERY",
    "Name",
    "adopted_name",
    "name_insert_input",
    "name_select_input",
    "names_from_output",
]

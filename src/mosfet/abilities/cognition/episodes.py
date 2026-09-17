"""Shared cognitive episode types for reflection and reasoning.

Episodes are prior turns in memory. Behavior inventory types live in ``mosfet.behavior``
(create/change/break); abilities import those rather than owning behavior vocabulary.
"""

from __future__ import annotations

import mosfet
from mosfet.behavior import BreakData, CreateData, ChangeData
from .. import memory

import typing
import uuid

import pydantic
from sqlalchemy import insert
from sqlalchemy import or_
from sqlalchemy import select

from . import types

# Stable query_tags value for cognition turn episodes in bot_memory.
COGNITIVE_EPISODE_QUERY = "cognitive_episode"


class CognitiveEpisode(pydantic.BaseModel):
    """Prior cognition turn stored for later behavior create/change/break on similar events."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Recallable cognition episode (JSON in memory, query_tags=cognitive_episode). "
                "Used when reflection chooses bot.behavior.create / change / break."
            ),
        },
    )

    focus: str | None = pydantic.Field(default=None, examples=["device-a"])
    focus_candidates: tuple[str, ...] = pydantic.Field(default=(), examples=[["device-a"]])
    stimulus_name: str | None = pydantic.Field(default=None, min_length=1, examples=["environment.sound"])
    output: types.OutputData = pydantic.Field(description="Typed cognitive output from that prior turn.")
    behavior: CreateData | ChangeData | BreakData | None = pydantic.Field(
        default=None,
        description="Behavior create/change/break payload recorded with this episode, if any.",
    )


def stimulus_name(stimulus: mosfet.InputData) -> str | None:
    """Low-cardinality stimulus label for episode recall (event name or bot.input)."""

    if isinstance(stimulus, mosfet.InputEventData):
        return mosfet.InputEvent.name
    return stimulus.name


def episodes_from_contents(contents: tuple[str, ...]) -> tuple[CognitiveEpisode, ...]:
    """Decode prior episodes from memory content strings (JSON CognitiveEpisode)."""

    episodes: list[CognitiveEpisode] = []
    for content in contents:
        try:
            episodes.append(CognitiveEpisode.model_validate_json(content))
        except Exception:
            continue
    return tuple(episodes)


def episodes_from_output(output: memory.OutputData, *, statement_index: int = 0) -> tuple[CognitiveEpisode, ...]:
    """Decode episodes from a committed memory SELECT result."""

    return episodes_from_contents(output.contents(statement_index=statement_index))


def episode_select_input(
    *,
    context_ref: str | None = None,
    limit: int = 50,
) -> memory.InputData:
    """Memory apply input that SELECTs prior cognitive episodes for behavior learning."""

    table = memory.memory_table
    clause = select(table).where(table.c.query_tags == COGNITIVE_EPISODE_QUERY)
    if context_ref is not None:
        clause = clause.where(or_(table.c.context_ref.is_(None), table.c.context_ref == context_ref))
    clause = clause.order_by(table.c.created_at).limit(limit)
    return memory.InputData(statements=memory.compile_statements(clause))


def episode_insert_input(
    episode: CognitiveEpisode,
    *,
    context_ref: str | None,
    scope: str = "short_term",
    memory_id: str | None = None,
) -> memory.InputData:
    """Memory apply input that INSERTs one cognitive episode for later behavior learning."""

    table = memory.memory_table
    clause = insert(table).values(
        memory_id=memory_id or uuid.uuid4().hex,
        scope=scope,
        context_ref=context_ref,
        subject_ref=None,
        kind="task",
        sensitivity="standard",
        retention="retain",
        content=episode.model_dump_json(),
        content_format="application/json",
        query_tags=COGNITIVE_EPISODE_QUERY,
    )
    return memory.InputData(statements=memory.compile_statements(clause))


__all__ = [
    "COGNITIVE_EPISODE_QUERY",
    "CognitiveEpisode",
    "episode_insert_input",
    "episode_select_input",
    "episodes_from_contents",
    "episodes_from_output",
    "stimulus_name",
]

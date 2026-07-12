"""SQLite-backed stateforward.bot Memory: one apply is one SQL transaction."""

from __future__ import annotations

from bot.abilities import ability
from bot.abilities import memory

import pathlib
import sqlite3
import typing

import hsm


class ProviderError(RuntimeError):
    """Raised when the SQLite memory provider cannot complete an operation."""


class SqliteMemory(memory.Memory):
    """Durable SQLite Memory ability (Statement transactions against bot_memory)."""

    default_scope: typing.ClassVar[str] = "sqlite"
    input_event: typing.ClassVar[hsm.Event[memory.InputData]] = ability.ability_input_event(
        "bot.ability.memory.sqlite.input",
        memory.InputData,
        description="SQL transaction against SQLite-backed bot memory.",
    )
    output_event: typing.ClassVar[hsm.Event[memory.OutputData]] = ability.ability_output_event(
        "bot.ability.memory.sqlite.output",
        memory.OutputData,
        description="Per-statement results after a committed SQLite memory transaction.",
    )
    submodel: typing.ClassVar[hsm.Model | None] = memory.memory_model(
        name="SqliteMemory",
        input_event=input_event,
    )

    def __init__(
        self,
        *,
        database_path: pathlib.Path | str = ":memory:",
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if connection is not None:
            super().__init__(connection=connection)
            return
        path = ":memory:" if str(database_path) == ":memory:" else str(database_path)
        if path != ":memory:":
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(database=path)


# Hard-cut alias: store means SQL ability, not encode/decode bag.
MemoryStore = SqliteMemory

__all__ = [
    "MemoryStore",
    "ProviderError",
    "SqliteMemory",
]

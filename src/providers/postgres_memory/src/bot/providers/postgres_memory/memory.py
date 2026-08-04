"""Postgres-backed stateforward.bot Memory: one apply is one SQL transaction."""

from __future__ import annotations

from bot.abilities import ability
from bot.abilities import memory

import collections.abc
import contextlib
import dataclasses
import enum
import typing

import hsm

from bot.telemetry import observer

QueryParameters: typing.TypeAlias = tuple[object, ...]
DatabaseRow: typing.TypeAlias = collections.abc.Mapping[str, object]


class ProviderError(RuntimeError):
    """Raised when the Postgres memory provider cannot complete an operation."""


class ParameterStyle(enum.StrEnum):
    """Positional parameter style for the injected database adapter."""

    FORMAT = "format"  # %s
    NUMERIC = "numeric"  # $1, $2, ...


class DatabaseConnection(typing.Protocol):
    """Transaction-scoped Postgres or PGlite connection."""

    async def execute(self, statement: str, parameters: QueryParameters = ()) -> None:
        """Execute a statement (DML/DDL)."""
        ...

    async def fetch_one(
        self,
        statement: str,
        parameters: QueryParameters = (),
    ) -> DatabaseRow | None:
        """Fetch one row as a mapping."""
        ...

    async def fetch_all(
        self,
        statement: str,
        parameters: QueryParameters = (),
    ) -> tuple[DatabaseRow, ...]:
        """Fetch all rows as mappings."""
        ...


class Database(typing.Protocol):
    """Database adapter for hosted Postgres, PGlite, or another Postgres-compatible runtime."""

    def transaction(self) -> contextlib.AbstractAsyncContextManager[DatabaseConnection]:
        """Open a transaction-scoped connection."""
        ...


_SCHEMA_STATEMENTS = (
    f"""
    CREATE TABLE IF NOT EXISTS {memory.MEMORY_TABLE} (
      memory_id TEXT PRIMARY KEY,
      scope TEXT NOT NULL DEFAULT 'memory',
      context_ref TEXT,
      subject_ref TEXT,
      kind TEXT,
      sensitivity TEXT,
      retention TEXT,
      content TEXT NOT NULL,
      content_format TEXT NOT NULL DEFAULT 'text/plain',
      query_tags TEXT,
      metadata_json TEXT NOT NULL DEFAULT '{{}}',
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    f"CREATE INDEX IF NOT EXISTS bot_memory_context_idx ON {memory.MEMORY_TABLE}(context_ref, subject_ref)",
    f"CREATE INDEX IF NOT EXISTS bot_memory_query_idx ON {memory.MEMORY_TABLE}(query_tags)",
    f"CREATE INDEX IF NOT EXISTS bot_memory_scope_idx ON {memory.MEMORY_TABLE}(scope, created_at)",
)


def _convert_parameter_style(statement: str, style: ParameterStyle) -> str:
    """Convert SQLite `?` placeholders to the adapter style."""

    if "?" not in statement:
        if style is ParameterStyle.FORMAT:
            return statement
        if style is ParameterStyle.NUMERIC and "%s" in statement:
            parts = statement.split("%s")
            converted = parts[0]
            for index, part in enumerate(parts[1:], start=1):
                converted += f"${index}{part}"
            return converted
        return statement

    parts = statement.split("?")
    if style is ParameterStyle.FORMAT:
        return "%s".join(parts)
    converted = parts[0]
    for index, part in enumerate(parts[1:], start=1):
        converted += f"${index}{part}"
    return converted


def _statement_kind(sql: str) -> str:
    head = sql.lstrip().split(None, 1)[0].upper() if sql.lstrip() else ""
    return head


async def execute_postgres_transaction(
    database: Database,
    statements: tuple[memory.Statement, ...],
    *,
    parameter_style: ParameterStyle = ParameterStyle.FORMAT,
    create_schema: bool = True,
) -> memory.OutputData:
    """Run statements in one Postgres transaction; commit on success, rollback on failure."""

    if not statements:
        raise ValueError("InputData requires at least one statement.")
    results: list[memory.StatementResult] = []
    async with database.transaction() as connection:
        if create_schema:
            for schema_sql in _SCHEMA_STATEMENTS:
                await connection.execute(schema_sql)
        for statement in statements:
            sql = _convert_parameter_style(statement.sql, parameter_style)
            kind = _statement_kind(statement.sql)
            params = tuple(statement.parameters)
            if kind == "SELECT":
                rows_raw = await connection.fetch_all(sql, params)
                rows: list[memory.Row] = []
                for row in rows_raw:
                    columns = tuple(row.keys())
                    values = tuple(
                        typing.cast(str | int | float | bool | bytes | None, row[column]) for column in columns
                    )
                    rows.append(memory.Row(columns=columns, values=values))
                results.append(memory.StatementResult(rowcount=len(rows), rows=tuple(rows)))
            else:
                await connection.execute(sql, params)
                results.append(memory.StatementResult(rowcount=1 if kind in {"INSERT", "UPDATE", "DELETE"} else 0))
    return memory.OutputData(results=tuple(results))


_PostgresApplyCompletedEvent = hsm.Event[memory.OutputData](
    name="bot.ability.memory.postgres.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=memory.OutputData,
)
_PostgresApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.memory.postgres.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


class PostgresMemory(ability.Ability[memory.InputData, memory.OutputData]):
    """Postgres Memory ability: Statement transactions against bot_memory."""

    default_scope: typing.ClassVar[str] = "postgres"
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = memory.InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = memory.OutputData
    input_event: typing.ClassVar[hsm.Event[memory.InputData]] = hsm.Event[memory.InputData](
    name="bot.ability.memory.postgres.input",
    schema=memory.InputData,

    )
    output_event: typing.ClassVar[hsm.Event[memory.OutputData]] = hsm.Event[memory.OutputData](
    name="bot.ability.memory.postgres.output",
    schema=memory.OutputData,

    )

    _database: Database
    _parameter_style: ParameterStyle
    _create_schema: bool

    def __init__(
        self,
        *,
        database: Database,
        parameter_style: ParameterStyle | str = ParameterStyle.FORMAT,
        create_schema: bool = True,
    ) -> None:
        super().__init__()
        self._database = database
        self._parameter_style = ParameterStyle(parameter_style)
        self._create_schema = create_schema

    @staticmethod
    def _has_input(ctx: hsm.Context, instance: "PostgresMemory", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, memory.InputData)

    @staticmethod
    def _has_output(ctx: hsm.Context, instance: "PostgresMemory", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, memory.OutputData)

    @staticmethod
    def _has_failure(ctx: hsm.Context, instance: "PostgresMemory", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, ability.FailureData)

    @staticmethod
    def _dispatch_output(ctx: hsm.Context, instance: "PostgresMemory", event: hsm.Event[typing.Any]) -> None:
        output = event.data
        assert isinstance(output, memory.OutputData)
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _dispatch_failure(ctx: hsm.Context, instance: "PostgresMemory", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    async def _run_transaction_activity(
        ctx: hsm.Context,
        instance: "PostgresMemory",
        event: hsm.Event[memory.InputData],
    ) -> None:
        data = event.data
        assert isinstance(data, memory.InputData)
        try:
            output = await execute_postgres_transaction(
                instance._database,
                data.statements,
                parameter_style=instance._parameter_style,
                create_schema=instance._create_schema,
            )
        except Exception as error:
            message = str(error) if str(error) else "Postgres memory transaction failed."
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _PostgresApplyFailedEvent.with_data(ability.FailureData(message=message)),
                    id=event.id or None,
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _PostgresApplyCompletedEvent.with_data(output),
                id=event.id or None,
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "PostgresMemory",
        hsm.initial(hsm.target("/PostgresMemory/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_input),
                hsm.target("/PostgresMemory/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(_run_transaction_activity),
            hsm.transition(
                hsm.on(_PostgresApplyCompletedEvent),
                hsm.guard(_has_output),
                hsm.effect(_dispatch_output),
                hsm.target("/PostgresMemory/idle"),
            ),
            hsm.transition(
                hsm.on(_PostgresApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("/PostgresMemory/idle"),
            ),
        ),
        hsm.observe(observer),
    )


MemoryStore = PostgresMemory

__all__ = [
    "Database",
    "DatabaseConnection",
    "DatabaseRow",
    "MemoryStore",
    "ParameterStyle",
    "PostgresMemory",
    "ProviderError",
    "QueryParameters",
    "execute_postgres_transaction",
]

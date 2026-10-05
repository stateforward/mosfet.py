"""SQLAlchemy Core memory store.

One ability apply runs one or more parameterized statements in a single transaction.
Success commits and returns per-statement results; any failure rolls back with no
partial durable effects.

Domain code should build SQLAlchemy Core clauses against ``schema.memory_table`` and
compile them with ``compile_statement`` / ``compile_statements``. The ability executes
crude ``Statement(sql, parameters)`` batches only.
"""

from __future__ import annotations

from .. import ability

import dataclasses
import importlib.resources
import typing
import uuid

import hsm
import mosfet
import pydantic
import sqlalchemy
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.engine import Dialect
from sqlalchemy.engine import Engine
from sqlalchemy.sql import ClauseElement


from . import schema

ParameterValue: typing.TypeAlias = str | int | float | bool | bytes | None

# Re-export logical table name for callers that only need the identifier.
MEMORY_TABLE = schema.MEMORY_TABLE


class Statement(pydantic.BaseModel):
    """One parameterized statement in a relational transaction.

    Crude wire unit after SQLAlchemy Core compilation: SQL text plus positional
    bind values. Providers may rewrite placeholder style; never interpolate caller
    strings into ``sql``.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "One parameterized SQL statement produced by compiling a SQLAlchemy Core "
                "clause (or an equivalent provider statement)."
            ),
            "examples": [
                {
                    "sql": (f"SELECT content FROM {MEMORY_TABLE} WHERE query_tags = ? ORDER BY created_at"),
                    "parameters": ["cognitive_episode"],
                },
            ],
        },
    )

    sql: str = pydantic.Field(
        min_length=1,
        description="SQL text with positional placeholders after Core compilation.",
        examples=[f"SELECT content FROM {MEMORY_TABLE} WHERE context_ref = ?"],
    )
    parameters: tuple[ParameterValue, ...] = pydantic.Field(
        default=(),
        description="Positional bind values matching placeholders in sql.",
        examples=[("active-task",)],
    )


class InputData(pydantic.BaseModel):
    """One transaction: ordered Statements (all commit, or all roll back)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Memory apply input: ordered Statements in one transaction. All commit together on "
                "success, or all roll back on any failure. Query-only, store-only, or mixed batches."
            ),
            "examples": [
                {
                    "statements": [
                        {
                            "sql": f"SELECT content FROM {MEMORY_TABLE} WHERE query_tags = ?",
                            "parameters": ["cognitive_episode"],
                        }
                    ]
                },
            ],
        },
    )

    statements: tuple[Statement, ...] = pydantic.Field(
        min_length=1,
        description="Ordered statements executed inside one transaction.",
    )


class Row(pydantic.BaseModel):
    """One result row with ordered columns."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    columns: tuple[str, ...] = pydantic.Field(description="Column names in result order.")
    values: tuple[ParameterValue, ...] = pydantic.Field(description="Cell values aligned with columns.")

    def as_mapping(self) -> dict[str, ParameterValue]:
        return dict(zip(self.columns, self.values, strict=True))

    def value(self, column: str) -> ParameterValue:
        mapping = self.as_mapping()
        if column not in mapping:
            raise KeyError(column)
        return mapping[column]


class StatementResult(pydantic.BaseModel):
    """Outcome of one statement inside a committed transaction."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    rowcount: int = pydantic.Field(description="Affected or returned row count.")
    rows: tuple[Row, ...] = pydantic.Field(
        default=(),
        description="Result rows for SELECT (empty for pure DML without RETURNING).",
    )


class OutputData(pydantic.BaseModel):
    """Results for every statement, only after the full transaction commits."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Memory apply output after a committed transaction. results align 1:1 with input statements. "
                "On rollback the ability fails and this payload is not emitted."
            ),
        },
    )

    results: tuple[StatementResult, ...] = pydantic.Field(
        min_length=1,
        description="One StatementResult per InputData.statements entry, same order.",
    )

    def contents(self, *, statement_index: int = 0) -> tuple[str, ...]:
        """Collect string content cells from a SELECT result (column name ``content`` preferred)."""

        if statement_index < 0 or statement_index >= len(self.results):
            raise IndexError("statement_index out of range for memory output results.")
        contents: list[str] = []
        for row in self.results[statement_index].rows:
            mapping = row.as_mapping()
            if "content" in mapping and isinstance(mapping["content"], str):
                contents.append(mapping["content"])
            elif len(row.values) == 1 and isinstance(row.values[0], str):
                contents.append(row.values[0])
        return tuple(contents)


class MemoryRecord(pydantic.BaseModel):
    """Row projection of bot_memory for callers that need a structured record."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )

    memory_id: str = pydantic.Field(min_length=1, description="Primary key of the memory row.")
    scope: str = pydantic.Field(min_length=1, description="Memory scope label.", examples=["short_term"])
    context_ref: str | None = pydantic.Field(default=None, min_length=1)
    subject_ref: str | None = pydantic.Field(default=None, min_length=1)
    kind: str | None = pydantic.Field(default=None)
    content: str = pydantic.Field(min_length=1, description="Primary text or JSON body.")
    content_format: str = pydantic.Field(default="text/plain", min_length=1)
    query_tags: str | None = pydantic.Field(default=None)

    @classmethod
    def from_row(cls, row: Row) -> "MemoryRecord":
        data = row.as_mapping()
        memory_id = data.get("memory_id")
        content = data.get("content")
        if not isinstance(memory_id, str) or not memory_id:
            memory_id = uuid.uuid4().hex
        if not isinstance(content, str) or not content:
            raise ValueError("MemoryRecord requires content column.")
        scope = data.get("scope")
        return cls(
            memory_id=memory_id,
            scope=scope if isinstance(scope, str) and scope else "memory",
            context_ref=_optional_str(data.get("context_ref")),
            subject_ref=_optional_str(data.get("subject_ref")),
            kind=_optional_str(data.get("kind")),
            content=content,
            content_format=_optional_str(data.get("content_format")) or "text/plain",
            query_tags=_optional_str(data.get("query_tags")),
        )


def _optional_str(value: ParameterValue) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    return str(value)


def compile_statement(
    clause: ClauseElement,
    *,
    dialect: Dialect | None = None,
) -> Statement:
    """Compile a SQLAlchemy Core clause into a crude ``Statement(sql, parameters)``."""

    from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect

    active = dialect if dialect is not None else sqlite_dialect()
    compiled = clause.compile(dialect=active, compile_kwargs={"render_postcompile": True})
    sql = str(compiled)
    positiontup = typing.cast(tuple[str, ...] | None, getattr(compiled, "positiontup", None))
    params = compiled.params or {}
    if positiontup:
        parameters = tuple(params[name] for name in positiontup)
    else:
        parameters = tuple(params.values())
    return Statement(
        sql=sql,
        parameters=typing.cast(tuple[ParameterValue, ...], parameters),
    )


def compile_statements(
    *clauses: ClauseElement,
    dialect: Dialect | None = None,
) -> tuple[Statement, ...]:
    """Compile one or more Core clauses into a transaction batch."""

    return tuple(compile_statement(clause, dialect=dialect) for clause in clauses)


def _statement_kind(sql: str) -> str:
    head = sql.lstrip().split(None, 1)[0].upper() if sql.lstrip() else ""
    return head


def _execute_transaction(
    engine: Engine,
    statements: tuple[Statement, ...],
) -> OutputData:
    """Run statements in one transaction on the ability-owned engine."""

    if not statements:
        raise ValueError("InputData requires at least one statement.")
    results: list[StatementResult] = []
    with engine.begin() as connection:
        for statement in statements:
            cursor = connection.exec_driver_sql(statement.sql, statement.parameters)
            kind = _statement_kind(statement.sql)
            if kind == "SELECT" or cursor.returns_rows:
                columns = tuple(cursor.keys())
                rows = tuple(
                    Row(
                        columns=columns,
                        values=tuple(
                            typing.cast(ParameterValue, cell)
                            for cell in typing.cast(tuple[object, ...], typing.cast(object, raw))
                        ),
                    )
                    for raw in cursor.fetchall()
                )
                results.append(StatementResult(rowcount=len(rows), rows=rows))
            else:
                results.append(
                    StatementResult(
                        rowcount=cursor.rowcount if cursor.rowcount >= 0 else 0,
                        rows=(),
                    )
                )
    return OutputData(results=tuple(results))


# Alembic version table name and the baseline revision (the create_all-era schema).
VERSION_TABLE = "alembic_version"
BASELINE_REVISION = "0001"

# Frozen: exactly the tables revision 0001 creates. Later revisions must not change this set;
# it only identifies databases that ``metadata.create_all`` created before migrations existed.
_BASELINE_TABLES = frozenset({"bot_behavior", "bot_behavior_trigger", "bot_memory", "bot_stm_memory"})


class MigrationError(RuntimeError):
    """Raised when a memory database matches no known schema revision and cannot be migrated."""


def migration_config() -> Config:
    """Programmatic Alembic config for the packaged ``migrations/`` scripts (no ``alembic.ini``).

    To run a command, set ``config.attributes["connection"]`` to an open SQLAlchemy
    connection; ``migrations/env.py`` runs online against that connection only.
    """

    config = Config()
    config.set_main_option("script_location", str(importlib.resources.files(schema) / "migrations"))
    return config


def migrate(engine: Engine) -> None:
    """Bring ``engine``'s database to the head memory schema revision (performs database I/O).

    Runs in one transaction. A database ``metadata.create_all`` created before migrations
    existed (all baseline tables, no ``alembic_version``) is stamped at ``0001`` first — the
    one-time cutover documented in ``migrations/versions/0001_initial_memory_schema.py``.

    Raises:
        MigrationError: the database has some baseline tables but no version table.
    """

    config = migration_config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        tables = set(sqlalchemy.inspect(connection).get_table_names())
        if VERSION_TABLE not in tables and tables & _BASELINE_TABLES:
            missing = sorted(_BASELINE_TABLES - tables)
            if missing:
                message = f"Unversioned memory database lacks baseline tables {missing}; not stamping it."
                raise MigrationError(message)
            command.stamp(config, BASELINE_REVISION)
        command.upgrade(config, "head")


def open_sqlite_engine(*, database: str = ":memory:", connection: typing.Any | None = None) -> Engine:
    """Composition-root factory: open a private SQLite engine migrated to the head schema.

    The ``sqlite+pysqlite`` URL default lives here — not in :class:`MemoryStore` — so the
    ability stays dialect-agnostic and the sqlite_memory provider (or any explicit root)
    owns the SQLite default. Pass the result as ``MemoryStore(engine=...)``.

    Raises:
        MigrationError: the database matches no known schema revision.
    """

    if connection is not None:
        engine = create_engine("sqlite+pysqlite://", creator=lambda: typing.cast(object, connection))
    else:
        url = "sqlite+pysqlite:///:memory:" if database == ":memory:" else f"sqlite+pysqlite:///{database}"
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
        )
    migrate(engine)
    return engine


_MemoryStoreApplyCompletedEvent = hsm.Event[OutputData](
    name="bot.ability.memory.store.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=OutputData,
)
_MemoryStoreApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.memory.store.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


class MemoryStore(ability.Ability[InputData, OutputData]):
    """Ability that executes compiled SQL statements as one transaction per apply."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.memory.store.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = hsm.Event[OutputData](
        name="bot.ability.memory.store.output",
        schema=OutputData,
    )

    _engine: Engine
    _apply_completed_event: typing.ClassVar[hsm.Event[OutputData]] = _MemoryStoreApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _MemoryStoreApplyFailedEvent

    def __init__(
        self,
        *,
        engine: Engine | None = None,
        connection: typing.Any | None = None,
        database: str = ":memory:",
    ) -> None:
        super().__init__()
        if engine is not None:
            if connection is not None or database != ":memory:":
                raise ValueError("MemoryStore takes either engine or database/connection, not both.")
            migrate(engine)
            self._engine = engine
            return
        self._engine = open_sqlite_engine(database=database, connection=connection)

    def execute(self, data: InputData) -> OutputData:
        """Run one SQL transaction on this store's engine and return per-statement results.

        This is the store's public transaction primitive (same role as ``Encoder.encode``).
        Ability apply drives the same work through HSM lifecycle events for event-driven
        coordination; nested cognition inventory/episode ops may call ``execute`` directly
        when the store is already owned and a concurrent apply would re-enter the machine.
        """

        return _execute_transaction(self._engine, data.statements)

    @staticmethod
    def _has_input(ctx: hsm.Context, instance: "MemoryStore", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, InputData)

    @staticmethod
    def _has_output(ctx: hsm.Context, instance: "MemoryStore", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, OutputData)

    @staticmethod
    def _has_failure(ctx: hsm.Context, instance: "MemoryStore", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, ability.FailureData)

    @staticmethod
    def _dispatch_output(ctx: hsm.Context, instance: "MemoryStore", event: hsm.Event[typing.Any]) -> None:
        output = event.data
        assert isinstance(output, OutputData)
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _dispatch_failure(ctx: hsm.Context, instance: "MemoryStore", event: hsm.Event[typing.Any]) -> None:
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
        instance: "MemoryStore",
        event: hsm.Event[InputData],
    ) -> None:
        data = event.data
        assert isinstance(data, InputData)
        try:
            output = instance.execute(data)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    instance._apply_failed_event.with_data(ability.FailureData(message=str(error))),
                    id=event.id or None,
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                instance._apply_completed_event.with_data(output),
                id=event.id or None,
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "MemoryStore",
        hsm.initial(hsm.target("/MemoryStore/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_input),
                hsm.target("/MemoryStore/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(_run_transaction_activity),
            hsm.transition(
                hsm.on(_MemoryStoreApplyCompletedEvent),
                hsm.guard(_has_output),
                hsm.effect(_dispatch_output),
                hsm.target("/MemoryStore/idle"),
            ),
            hsm.transition(
                hsm.on(_MemoryStoreApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("/MemoryStore/idle"),
            ),
        ),
    )


__all__ = [
    "BASELINE_REVISION",
    "MEMORY_TABLE",
    "InputData",
    "MemoryRecord",
    "MemoryStore",
    "MigrationError",
    "OutputData",
    "ParameterValue",
    "Row",
    "Statement",
    "StatementResult",
    "VERSION_TABLE",
    "compile_statement",
    "compile_statements",
    "migrate",
    "migration_config",
    "open_sqlite_engine",
]

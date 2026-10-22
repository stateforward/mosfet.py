"""SQLAlchemy Core memory store.

One ability apply runs one or more parameterized statements in a single transaction.
Success commits and returns per-statement results; any failure rolls back with no
partial durable effects.

Domain code should build SQLAlchemy Core clauses against ``schema.memory_table`` and
compile them with ``compile_statement`` / ``compile_statements``. The ability executes
crude ``Statement(sql, parameters)`` batches only.

Constructing a store does no I/O (DSL-29): it records the engine, engine handle (a callable that
opens one, called once in the opening state), database path or connection it was given. The store opens and migrates its database in its own ``opening`` state, the first time
its behavior starts, and keeps that database for its lifetime; requests sent before it is open
wait. Every open, migration and transaction runs on the store's one database worker thread, in
submission order, never on the event loop; the worker lives from ``start`` to ``stop``. Telemetry: ``bot.memory.store.open.count`` counts opens by ``bot.ability.name`` (the store
class, a closed set from code) and ``bot.outcome`` {ok, failed, timeout}.
"""

from __future__ import annotations

from .. import ability

import asyncio
import collections.abc
import concurrent.futures
import dataclasses
import datetime
import importlib.resources
import logging
import sqlite3
import typing
import uuid

import hsm
import mosfet
import pydantic
import sqlalchemy
import sqlalchemy.event
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.engine import Dialect
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql import ColumnElement
from opentelemetry import metrics

from mosfet.protocols import attachment
from mosfet.telemetry import span

from . import schema

_LOG = logging.getLogger(__name__)
_OPENS = metrics.get_meter("bot.abilities.memory").create_counter(
    "bot.memory.store.open.count",
    unit="{open}",
    description="Memory store database opens, by store class and outcome (ok, failed, timeout).",
)

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


def recent_select(*criteria: ColumnElement[bool], limit: int) -> Select[typing.Any]:
    """SELECT the ``limit`` most recent ``bot_memory`` rows matching ``criteria``, oldest first.

    Recency is ``created_at``. Rows that share a stamp read in insertion order: SQLite's implicit
    ``rowid`` (``bot_memory`` has a TEXT primary key, so it keeps one) breaks the tie, never the
    uuid ``memory_id``. The window is taken newest-first, then re-read in chronological order for
    consumers; the result carries exactly the ``bot_memory`` columns.
    """

    table = schema.memory_table
    insertion = sqlalchemy.literal_column(f"{table.name}.rowid", sqlalchemy.Integer).label("insertion")
    window = (
        select(table, insertion)
        .where(*criteria)
        .order_by(table.c.created_at.desc(), insertion.desc())
        .limit(limit)
        .subquery()
    )
    return select(*(window.c[column.name] for column in table.c)).order_by(window.c.created_at, window.c.insertion)


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


# Alembic version table name.
VERSION_TABLE = "alembic_version"


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

    Runs in one transaction. A database that holds memory tables but no ``alembic_version``
    table was not created by these migrations; it matches no revision and is refused rather
    than guessed at.

    Raises:
        MigrationError: the database has memory tables but no version table.
    """

    config = migration_config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        tables = set(sqlalchemy.inspect(connection).get_table_names())
        known = tables & set(schema.metadata.tables)
        if VERSION_TABLE not in tables and known:
            message = f"Unversioned memory database already holds {sorted(known)}; not migrating it."
            raise MigrationError(message)
        command.upgrade(config, "head")


def open_sqlite_engine(
    *,
    database: str = ":memory:",
    connection: typing.Any | None = None,
    on_connect: collections.abc.Callable[[sqlite3.Connection], None] | None = None,
) -> Engine:
    """Composition-root factory: open a private SQLite engine migrated to the head schema.

    The ``sqlite+pysqlite`` URL default lives here — not in :class:`MemoryStore` — so the
    ability stays dialect-agnostic and the sqlite_memory provider (or any explicit root)
    owns the SQLite default. Pass the result as ``MemoryStore(engine=...)``.

    An in-memory database (``":memory:"``, or a caller's ``connection``) is one shared
    connection (``StaticPool``), so every thread sees the same database. ``on_connect`` runs
    on every new DBAPI connection before the schema is migrated; a provider loads extensions
    and sets pragmas there.

    Raises:
        MigrationError: the database matches no known schema revision.
    """

    if connection is not None:
        engine = create_engine(
            "sqlite+pysqlite://",
            creator=lambda: typing.cast(object, connection),
            poolclass=StaticPool,
        )
    elif database == ":memory:":
        engine = create_engine(
            "sqlite+pysqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_engine(
            f"sqlite+pysqlite:///{database}",
            connect_args={"check_same_thread": False},
        )
    if on_connect is not None:
        hook = on_connect

        def connected(dbapi_connection: sqlite3.Connection, record: object) -> None:
            del record
            hook(dbapi_connection)

        sqlalchemy.event.listen(engine, "connect", connected)
    migrate(engine)
    return engine


class _OpenedData(pydantic.BaseModel):
    """The store's database is open; ``entered`` is the event that started the store's behavior."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    entered: hsm.Event[typing.Any] = pydantic.Field(description="The event the opening state was entered with.")


class _OpenFailedData(pydantic.BaseModel):
    """The store could not open or migrate its database."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    entered: hsm.Event[typing.Any] = pydantic.Field(description="The event the opening state was entered with.")
    message: str = pydantic.Field(description="Why the database could not be opened: the error type and message.")


_OpenedEvent = hsm.Event[_OpenedData](
    name="bot.ability.memory.store.opened",
    kind=hsm.CompletionEventKind,
    schema=_OpenedData,
)
_OpenFailedEvent = hsm.Event[_OpenFailedData](
    name="bot.ability.memory.store.open.failed",
    kind=hsm.ErrorEventKind,
    schema=_OpenFailedData,
)

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
    """Ability that executes compiled SQL statements as one transaction per apply.

    Construction only records ``engine``, or ``database`` and ``connection``; nothing is opened.
    ``engine`` is an open engine or a handle that opens one (a callable, such as a database other
    parts share opening on its first use). The store's behavior starts in ``opening``: it opens the
    database (an injected ``engine``, or the one its handle opens, is migrated; otherwise a private
    SQLite engine is opened and migrated) and goes on to ``idle``.
    The open, the migration and every transaction run on the store's one database worker (a single
    thread created by ``start`` and let go by ``stop``), in submission order, so the event loop never
    blocks on database I/O. A caller's ``connection`` is opened and used on that thread only.
    The open is bounded by the attach timeout. A store that cannot open its database, or does not
    within that bound, is ``degraded`` and answers every request failed. The open database is
    kept for the store's lifetime, so a store that detaches and attaches again opens nothing new.

    Raises:
        ValueError: if both ``engine`` and ``database``/``connection`` are given.
    """

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

    # The open database: None until the opening state has opened it, then kept for the lifetime.
    _engine: Engine | None
    # What construction was given; read only by the opening state.
    _injected: Engine | collections.abc.Callable[[], Engine] | None
    _database: str
    _connection: typing.Any | None
    # The one database worker, created by start and shut down by stop. A restart's new worker first
    # waits for the previous one to drain, so a job left running by a timeout, a detach or a stop can
    # never overlap a later one on the same connection.
    _worker: concurrent.futures.ThreadPoolExecutor | None
    # The jobs submitted to the worker and not yet done, so stop can close the database after the last.
    _jobs: set[concurrent.futures.Future[typing.Any]]
    _apply_completed_event: typing.ClassVar[hsm.Event[OutputData]] = _MemoryStoreApplyCompletedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _MemoryStoreApplyFailedEvent
    _opened_event: typing.ClassVar[hsm.Event[_OpenedData]] = _OpenedEvent
    _open_failed_event: typing.ClassVar[hsm.Event[_OpenFailedData]] = _OpenFailedEvent

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        engine: Engine | collections.abc.Callable[[], Engine] | None = None,
        connection: typing.Any | None = None,
        database: str = ":memory:",
    ) -> None:
        super().__init__(id=id, clock=clock)
        if engine is not None and (connection is not None or database != ":memory:"):
            raise ValueError("MemoryStore takes either engine or database/connection, not both.")
        self._engine = None
        self._injected = engine
        self._database = database
        self._connection = connection
        self._worker = None
        self._jobs = set()

    def _connect(self) -> Engine:
        """Open and migrate this store's database (performs database I/O; blocking).

        Raises:
            MigrationError: the database matches no known schema revision.
        """

        injected = self._injected
        if injected is not None:
            engine = injected if isinstance(injected, Engine) else injected()
            migrate(engine)
            return engine
        return open_sqlite_engine(database=self._database, connection=self._connection)

    async def _open_engine(self) -> Engine:
        """Open and migrate the database for the opening state, on the worker, so the event loop never blocks on it."""

        return await MemoryStore._run(self, self._connect)

    @typing.override
    async def start(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        """Give the store its database worker, then start it.

        A worker left by an earlier start is drained first, on the new worker, so its last job
        finishes before any job of this start runs.
        """

        previous = self._worker
        self._worker = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=type(self).__name__.lower()
        )
        if previous is not None:
            _ = self._worker.submit(previous.shutdown, wait=True)
        return await super().start(ctx, data)

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop, close the database this store opened, and let the worker exit: queued jobs are cancelled, a running one finishes.

        Whoever opens a database closes it: an injected engine, one a handle opened, or a connection is
        its owner's and is left open. A store started again opens its database again.
        """

        await super().stop(ctx)
        engine, self._engine = self._engine, None
        if engine is not None and self._injected is None and self._connection is None:
            self._close_engine(engine)
        if self._worker is not None:
            self._worker.shutdown(wait=False, cancel_futures=True)

    def _close_engine(self, engine: Engine) -> None:
        """Close the database once the worker's last job is done: queued jobs are cancelled, a running one finishes first.

        Stop never waits on that job: the engine is disposed on the worker, right after it.
        """

        worker = self._worker
        if worker is not None:
            worker.shutdown(wait=False, cancel_futures=True)
        # One worker: once the queue is cancelled, at most the job it is running is not done.
        running = next((job for job in self._jobs if not job.done()), None)
        if running is None:
            engine.dispose()
            return
        running.add_done_callback(lambda _: engine.dispose())

    @staticmethod
    async def _run[**P, T](
        instance: "MemoryStore", job: collections.abc.Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Run one blocking database job on this store's worker, in submission order.

        Cancelling the await never interrupts the job: one already running finishes (its
        transaction commits or rolls back whole) and one still queued is dropped.
        """

        worker = instance._worker
        if worker is None:
            raise RuntimeError(f"{type(instance).__name__} runs database jobs only while started.")
        bound = span.bind(job)
        submitted = worker.submit(bound, *args, **kwargs)
        instance._jobs.add(submitted)
        submitted.add_done_callback(instance._jobs.discard)
        return await asyncio.wrap_future(submitted)

    def _opened_engine(self) -> Engine:
        """The open database; only states after ``opening`` run statements, so it is always open there."""

        engine = self._engine
        if engine is None:
            raise RuntimeError(f"{type(self).__name__} has not opened its database.")
        return engine

    def _execute(self, data: InputData) -> OutputData:
        """Run one SQL transaction on this store's engine and return per-statement results.

        The machine's own transaction primitive, run by its applying activity. A store is a
        state machine and is reached by message only (:func:`transact` sends ``input_event``
        and awaits the correlated reply).
        """

        return _execute_transaction(self._opened_engine(), data.statements)

    @staticmethod
    def _count_open(instance: "MemoryStore", outcome: typing.Literal["ok", "failed", "timeout"]) -> None:
        _OPENS.add(1, {"bot.ability.name": type(instance).__name__, "bot.outcome": outcome})

    @staticmethod
    async def _open_activity(ctx: hsm.Context, instance: "MemoryStore", event: hsm.Event[typing.Any]) -> None:
        """Open the database and report opened or failed; the entering event rides along on the report."""

        try:
            # Opened once: a store that is already open keeps the database it has.
            if instance._engine is None:
                instance._engine = await instance._open_engine()
        except Exception as error:
            # Boundary: an unopenable database is the store's typed open failure, never a raise.
            message = f"{type(instance).__name__} could not open its database: {type(error).__name__}: {error}"
            _LOG.warning("memory store could not open its database: %s", type(error).__name__)
            MemoryStore._count_open(instance, "failed")
            _ = hsm.dispatch(
                ctx, instance, instance._open_failed_event.with_data(_OpenFailedData(entered=event, message=message))
            )
            return
        MemoryStore._count_open(instance, "ok")
        _ = hsm.dispatch(ctx, instance, instance._opened_event.with_data(_OpenedData(entered=event)))

    @staticmethod
    def _count_open_timeout(ctx: hsm.Context, instance: "MemoryStore", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _LOG.warning("memory store did not open its database within its attach timeout")
        MemoryStore._count_open(instance, "timeout")

    @staticmethod
    def _refuse_attach(ctx: hsm.Context, instance: "MemoryStore", event: hsm.Event[typing.Any]) -> None:
        """Fail the attach a composite store was opening for: the owner is told it cannot attach."""

        failed = event.data
        assert isinstance(failed, _OpenFailedData)
        request = failed.entered.data
        assert isinstance(request, attachment.AttachData)
        instance._dispatch_composite_attachment_failure(
            ctx,
            instance,
            request,
            failed.entered,
            attachment.FailedData(
                actor=request.actor, kind=attachment.FailureKind.INITIALIZATION, message=failed.message
            ),
        )

    @staticmethod
    async def _attach_opened(ctx: hsm.Context, instance: "MemoryStore", event: hsm.Event[typing.Any]) -> None:
        """A composite store's group attach, run once its database is open, for the attach it opened for."""

        opened = event.data
        assert isinstance(opened, _OpenedData)
        await ability.Ability._attach_composite_group(ctx, instance, opened.entered)

    @staticmethod
    def _refuse_unopened(ctx: hsm.Context, instance: "MemoryStore", event: hsm.Event[typing.Any]) -> None:
        """Answer a request to a store whose database never opened."""

        failure = ability.FailureData(message=f"unavailable: {type(instance).__name__} could not open its database.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=MemoryStore._requester(instance, event),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _requester(instance: "MemoryStore", event: hsm.Event[typing.Any]) -> str:
        """The address that asked: the reply goes back to it, never to the owner instead."""

        return event.source if event.source and event.source != hsm.id(instance) else ""

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
            target=MemoryStore._requester(instance, event),
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
            target=MemoryStore._requester(instance, event),
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
            output = await MemoryStore._run(instance, instance._execute, data)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    instance._apply_failed_event.with_data(ability.FailureData(message=str(error))),
                    id=event.id or None,
                    source=event.source,
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
                source=event.source,
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "MemoryStore",
        hsm.initial(hsm.target("/MemoryStore/opening")),
        hsm.state(
            "opening",
            hsm.defer(input_event),
            hsm.activity(_open_activity),
            hsm.transition(hsm.on(_OpenedEvent), hsm.target("/MemoryStore/idle")),
            hsm.transition(hsm.on(_OpenFailedEvent), hsm.target("/MemoryStore/degraded")),
            hsm.transition(
                hsm.after(attachment.Attachment._attachment_timeout_delay),
                hsm.effect(_count_open_timeout),
                hsm.target("/MemoryStore/degraded"),
            ),
        ),
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
            # A detach waits for the transaction on the worker, so the request in flight is answered.
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
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
        hsm.state(
            "degraded",
            hsm.transition(hsm.on(input_event), hsm.effect(_refuse_unopened)),
        ),
    )


# How long a peer waits for a store to answer one transaction before the request fails.
OPERATION_TIMEOUT = datetime.timedelta(seconds=30)


class RequestError(RuntimeError):
    """A transaction sent to a store came back failed, or was never answered.

    Raised by :func:`transact` when the store replies with its failure event (the transaction
    rolled back), when the store is not started in an addressable scope, or when no correlated
    reply arrives within the timeout.
    """


async def transact(
    ctx: hsm.Context,
    store: MemoryStore,
    data: InputData,
    *,
    metadata: collections.abc.Mapping[str, object] | None = None,
    timeout: datetime.timedelta = OPERATION_TIMEOUT,
) -> OutputData:
    """Send one transaction to ``store`` by message and await its correlated reply.

    A store is a state machine, so a peer never calls into it: this dispatches the store's
    ``input_event`` with a fresh id from a one-shot reply actor parented to ``ctx`` (the
    caller's activity context, so cancelling the activity abandons the wait) and returns the
    committed ``OutputData``. The store answers the requesting address directly, applies
    requests one at a time, and defers the rest, so peers sharing a store are serialized.

    Raises:
        RequestError: the store replied with a failure (nothing was committed), is not
            started in an addressable scope, or did not answer within ``timeout``.
    """

    request = dataclasses.replace(
        store.input_event.with_data_and_id(data, uuid.uuid4().hex),
        metadata=dict(metadata) if metadata is not None else {},
    )
    try:
        terminal = await ability.run_terminal_operation(
            ctx,
            child=store,
            request=request,
            terminals=(store.output_event, store.failed_event),
            timeout=timeout,
        )
    except TimeoutError as error:
        raise RequestError(f"{type(store).__name__} did not answer within {timeout.total_seconds():g}s.") from error
    except RuntimeError as error:
        raise RequestError(f"{type(store).__name__} cannot be reached: {error}") from error
    reply = terminal.data
    if isinstance(reply, ability.FailureData):
        raise RequestError(reply.message)
    if not isinstance(reply, OutputData):
        raise RequestError(f"{type(store).__name__} answered with {type(reply).__name__}, not OutputData.")
    return reply


__all__ = [
    "MEMORY_TABLE",
    "OPERATION_TIMEOUT",
    "InputData",
    "MemoryStore",
    "MigrationError",
    "OutputData",
    "ParameterValue",
    "RequestError",
    "Row",
    "Statement",
    "StatementResult",
    "VERSION_TABLE",
    "compile_statement",
    "compile_statements",
    "migrate",
    "migration_config",
    "open_sqlite_engine",
    "recent_select",
    "transact",
]

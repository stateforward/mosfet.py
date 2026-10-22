"""SQL-transaction MemoryStore tests (ability boundary only)."""

from mosfet.abilities import memory

import asyncio
import collections.abc
import datetime
import pathlib
import typing

import hsm
import mosfet
import pytest
from sqlalchemy.engine import Engine

from tests import metric_reader
from tests.bot.abilities.memory.memory_fixtures import disposed_engines, execute_on
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test


def _insert_content(
    *,
    content: str,
    scope: str = "memory",
    context_ref: str | None = None,
    subject_ref: str | None = None,
    kind: str | None = "task",
    query_tags: str | None = None,
    content_format: str = "text/plain",
    memory_id: str | None = None,
) -> memory.Statement:
    import uuid
    from sqlalchemy import insert

    table = memory.memory_table
    clause = insert(table).values(
        memory_id=memory_id or uuid.uuid4().hex,
        scope=scope,
        context_ref=context_ref,
        subject_ref=subject_ref,
        kind=kind,
        sensitivity="standard",
        retention="retain",
        content=content,
        content_format=content_format,
        query_tags=query_tags,
    )
    return memory.compile_statement(clause)


def _select_by_query_tags(*, query_tags: str, context_ref: str | None = None, limit: int = 50) -> memory.Statement:
    from sqlalchemy import or_, select

    table = memory.memory_table
    clause = select(table).where(table.c.query_tags == query_tags)
    if context_ref is not None:
        clause = clause.where(or_(table.c.context_ref.is_(None), table.c.context_ref == context_ref))
    clause = clause.order_by(table.c.created_at).limit(limit)
    return memory.compile_statement(clause)


def test_memory_store_ability_apply_is_transaction() -> None:
    async def run() -> memory.OutputData:
        store = memory.MemoryStore()
        await start_abilities_for_test(shared_hsm_context(), store)
        return await dispatch_ability_for_test(
            store,
            None,
            memory.InputData(
                statements=(
                    _insert_content(content="via-ability", query_tags="t"),
                    memory.Statement(
                        sql=f"SELECT content FROM {memory.MEMORY_TABLE} WHERE query_tags = ?",
                        parameters=("t",),
                    ),
                )
            ),
        )

    output = asyncio.run(run())
    assert output.contents(statement_index=1) == ("via-ability",)


def test_memory_store_ability_rolls_back_on_failure() -> None:
    async def run() -> tuple[str, ...]:
        store = memory.Memory()
        await start_abilities_for_test(shared_hsm_context(), store)
        with pytest.raises(RuntimeError):
            _ = await dispatch_ability_for_test(
                store,
                None,
                memory.InputData(
                    statements=(
                        _insert_content(content="should-not-remain", query_tags="x"),
                        memory.Statement(sql="INSERT INTO not_a_table (a) VALUES (1)", parameters=()),
                    )
                ),
            )
        check = await dispatch_ability_for_test(
            store,
            None,
            memory.InputData(
                statements=(
                    memory.Statement(
                        sql=f"SELECT content FROM {memory.MEMORY_TABLE} WHERE query_tags = ?",
                        parameters=("x",),
                    ),
                )
            ),
        )
        return check.contents()

    assert asyncio.run(run()) == ()


def test_input_requires_at_least_one_statement() -> None:
    with pytest.raises(Exception):
        _ = memory.InputData(statements=())


def test_memory_store_accepts_injected_engine() -> None:
    """The ability runs on a caller-provided engine; the SQLite default stays in the factory."""

    from mosfet.abilities.memory import store as store_module

    engine = store_module.open_sqlite_engine()

    async def run() -> memory.OutputData:
        injected = memory.MemoryStore(engine=engine)
        await start_abilities_for_test(shared_hsm_context(), injected)
        return await dispatch_ability_for_test(
            injected,
            None,
            memory.InputData(statements=(_insert_content(content="injected-engine", query_tags="task"),)),
        )

    output = asyncio.run(run())
    assert len(output.results) == 1
    selected = execute_on(engine, memory.InputData(statements=(_select_by_query_tags(query_tags="task"),)))
    assert "injected-engine" in selected.contents()
    with pytest.raises(ValueError, match="either engine or database/connection"):
        _ = memory.MemoryStore(engine=engine, database="/tmp/memory-store-injected.db")


def _schema_revision_state(engine: Engine) -> tuple[str | None, str | None, list[object]]:
    """Current DB revision, head script revision, and autogenerate diffs vs ``memory.metadata``."""

    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    from mosfet.abilities.memory import store as store_module

    head = ScriptDirectory.from_config(store_module.migration_config()).get_current_head()
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        diffs = typing.cast("list[object]", compare_metadata(context, memory.metadata))
        return context.get_current_revision(), head, diffs


def test_open_sqlite_engine_upgrades_fresh_database_to_head_without_drift() -> None:
    """Bring-up runs every revision; the migrated schema matches ``memory.metadata`` exactly."""

    from mosfet.abilities.memory import store as store_module

    engine = store_module.open_sqlite_engine()
    current, head, diffs = _schema_revision_state(engine)
    engine.dispose()
    assert head is not None
    assert current == head
    assert diffs == []


def test_open_sqlite_engine_rejects_an_unversioned_database_that_holds_memory_tables(tmp_path: pathlib.Path) -> None:
    """Memory tables without a version table were not made by these migrations and are not guessed at."""

    import contextlib
    import sqlite3

    from mosfet.abilities.memory import store as store_module

    path = tmp_path / "unversioned.db"
    with contextlib.closing(sqlite3.connect(path)) as connection, connection:
        _ = connection.execute(f"CREATE TABLE {memory.MEMORY_TABLE} (memory_id TEXT PRIMARY KEY)")
    with pytest.raises(store_module.MigrationError, match="Unversioned memory database"):
        _ = store_module.open_sqlite_engine(database=str(path))


def test_the_revisions_create_and_drop_every_memory_table() -> None:
    """Revision 0001 is the baseline (relational, behavior, register, episodic graph) and 0002 adds the turns of
    short-term memory; no data moves."""

    from alembic import command
    from alembic.script import ScriptDirectory
    from sqlalchemy import create_engine, inspect

    from mosfet.abilities.memory import store as store_module

    config = store_module.migration_config()
    assert [script.revision for script in ScriptDirectory.from_config(config).walk_revisions()] == ["0002", "0001"]
    engine = create_engine("sqlite+pysqlite://")
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
        created = set(inspect(connection).get_table_names())
        command.downgrade(config, "base")
        remaining = set(inspect(connection).get_table_names())
    engine.dispose()
    assert set(memory.metadata.tables) <= created
    assert remaining == {store_module.VERSION_TABLE}
    lifetime = {column.name for column in memory.metadata.tables["bot_behavior"].columns}
    assert "lifetime" in lifetime


def test_recent_select_returns_same_instant_rows_in_insertion_order() -> None:
    """Recall reaches the latest rows; within the window they read in the order they happened.

    ``created_at`` decides recency even when ``memory_id`` sorts the other way, and rows that share
    one ``created_at`` read in insertion order, never in ``memory_id`` (uuid) order, both when the
    window is cut and when it is re-read oldest first.
    """

    import uuid

    from sqlalchemy import insert

    table = memory.memory_table
    same_instant = "2026-01-01 00:00:03.000000"
    rows = (
        ("oldest", "2026-01-01 00:00:01.000000", uuid.UUID(int=0xF).hex),
        ("older", "2026-01-01 00:00:02.000000", uuid.UUID(int=0xE).hex),
        ("first", same_instant, uuid.UUID(int=0xC).hex),
        ("second", same_instant, uuid.UUID(int=0xA).hex),
        ("third", same_instant, uuid.UUID(int=0xB).hex),
        ("other-tag", "2026-01-01 00:00:04.000000", uuid.UUID(int=0x1).hex),
    )

    async def run() -> tuple[memory.OutputData, memory.OutputData]:
        store = memory.Memory()
        await start_abilities_for_test(shared_hsm_context(), store)
        for content, created_at, memory_id in rows:
            _ = await dispatch_ability_for_test(
                store,
                None,
                memory.InputData(
                    statements=memory.compile_statements(
                        insert(table).values(
                            memory_id=memory_id,
                            content=content,
                            query_tags="other" if content == "other-tag" else "t",
                            created_at=created_at,
                        )
                    )
                ),
            )

        async def recall(limit: int) -> memory.OutputData:
            return await dispatch_ability_for_test(
                store,
                None,
                memory.InputData(
                    statements=memory.compile_statements(memory.recent_select(table.c.query_tags == "t", limit=limit))
                ),
            )

        return await recall(4), await recall(2)

    window, cut = asyncio.run(run())

    assert window.contents() == ("older", "first", "second", "third")
    assert cut.contents() == ("second", "third")
    assert set(window.results[0].rows[0].as_mapping()) == {column.name for column in table.c}


def test_transact_reaches_the_store_by_message_and_returns_its_committed_reply() -> None:
    async def run() -> memory.OutputData:
        store = memory.Memory()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, store)
        return await memory.transact(
            ctx,
            store,
            memory.InputData(
                statements=(
                    _insert_content(content="by-message", query_tags="m"),
                    _select_by_query_tags(query_tags="m"),
                )
            ),
        )

    assert asyncio.run(run()).contents(statement_index=1) == ("by-message",)


def test_transact_raises_request_error_when_the_store_rolls_back() -> None:
    async def run() -> tuple[str, ...]:
        store = memory.Memory()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, store)
        with pytest.raises(memory.RequestError, match="not_a_table"):
            _ = await memory.transact(
                ctx,
                store,
                memory.InputData(
                    statements=(
                        _insert_content(content="rolled-back", query_tags="r"),
                        memory.Statement(sql="INSERT INTO not_a_table (a) VALUES (1)", parameters=()),
                    )
                ),
            )
        check = await memory.transact(ctx, store, memory.InputData(statements=(_select_by_query_tags(query_tags="r"),)))
        return check.contents()

    assert asyncio.run(run()) == ()


def test_transact_raises_request_error_for_a_store_that_is_not_started() -> None:
    async def run() -> None:
        with pytest.raises(memory.RequestError, match="cannot be reached"):
            _ = await memory.transact(
                shared_hsm_context(),
                memory.Memory(),
                memory.InputData(statements=(_select_by_query_tags(query_tags="none"),)),
            )

    asyncio.run(run())


def test_transact_raises_request_error_when_the_store_never_answers() -> None:
    async def run() -> None:
        store = memory.MemoryStore()
        ctx = shared_hsm_context()
        # Started but never attached: the request reaches a machine that cannot serve it yet.
        _ = await mosfet.started(ctx, store, typing.cast(hsm.Model, store.model))
        with pytest.raises(memory.RequestError, match="did not answer"):
            _ = await memory.transact(
                ctx,
                store,
                memory.InputData(statements=(_select_by_query_tags(query_tags="none"),)),
                timeout=datetime.timedelta(milliseconds=50),
            )

    asyncio.run(run())


def test_in_memory_engine_is_shared_across_threads() -> None:
    """A row written on a worker thread is visible from the caller thread: one ``:memory:`` database."""

    import concurrent.futures

    from sqlalchemy import insert, select

    from mosfet.abilities.memory import store as store_module

    engine = store_module.open_sqlite_engine()
    table = memory.memory_table

    def write() -> None:
        with engine.begin() as connection:
            _ = connection.execute(insert(table).values(memory_id="threaded", content="written on a worker"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(write).result()
    with engine.connect() as connection:
        contents = connection.execute(select(table.c.content)).scalars().all()
    engine.dispose()
    assert contents == ["written on a worker"]


def test_open_sqlite_engine_runs_the_connect_hook_before_migrating() -> None:
    """``on_connect`` sees every new DBAPI connection before the schema is brought up."""

    import sqlite3

    from mosfet.abilities.memory import store as store_module

    seen: list[list[str]] = []

    def on_connect(connection: sqlite3.Connection) -> None:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        seen.append(tables)

    engine = store_module.open_sqlite_engine(on_connect=on_connect)
    engine.dispose()
    assert seen == [[]]


def _unversioned_database(tmp_path: pathlib.Path) -> str:
    """A file holding a memory table but no version table: these migrations refuse to touch it."""

    import contextlib
    import sqlite3

    path = tmp_path / "unversioned.db"
    with contextlib.closing(sqlite3.connect(path)) as connection, connection:
        _ = connection.execute(f"CREATE TABLE {memory.MEMORY_TABLE} (memory_id TEXT PRIMARY KEY)")
    return str(path)


def _opens(points: collections.abc.Sequence[metric_reader.Point]) -> dict[object, float]:
    return {
        point.attributes["bot.outcome"]: point.value
        for point in points
        if point.attributes.get("bot.ability.name") == "MemoryStore"
    }


def test_a_store_opens_its_database_when_its_behavior_starts_not_when_constructed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DSL-29a: construction records the path; the opening state creates and migrates the file."""

    path = tmp_path / "store.db"

    async def run(reader: metric_reader.Reader) -> tuple[bool, tuple[str, ...], dict[object, float]]:
        store = memory.MemoryStore(database=str(path))
        constructed = path.exists()
        before = _opens(reader.points("bot.memory.store.open.count"))
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, store)
        output = await memory.transact(
            ctx,
            store,
            memory.InputData(
                statements=(_insert_content(content="opened", query_tags="o"), _select_by_query_tags(query_tags="o"))
            ),
        )
        after = _opens(reader.points("bot.memory.store.open.count"))
        await store.stop(ctx)
        return constructed, output.contents(statement_index=1), {k: v - before.get(k, 0.0) for k, v in after.items()}

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        constructed, contents, opens = asyncio.run(run(reader))

    assert constructed is False
    assert path.exists()
    assert contents == ("opened",)
    assert opens == {"ok": 1.0}


def test_a_store_that_cannot_open_its_database_answers_every_request_failed(tmp_path: pathlib.Path) -> None:
    """An unversioned database is refused when the store opens it, never when the store is constructed."""

    database = _unversioned_database(tmp_path)

    async def run() -> str:
        store = memory.MemoryStore(database=database)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, store)
        with pytest.raises(memory.RequestError, match="could not open its database"):
            _ = await memory.transact(ctx, store, memory.InputData(statements=(_select_by_query_tags(query_tags="x"),)))
        state = store.state()
        await store.stop(ctx)
        return state

    assert asyncio.run(run()) == "/MemoryStoreLifecycle/attached/behavior/degraded"


class _NeverOpens(memory.MemoryStore):
    """A store whose database open never finishes, so only the attach timeout ends the opening state."""

    @typing.override
    async def _open_engine(self) -> Engine:
        _ = await asyncio.Event().wait()
        raise AssertionError("never opened")


def test_a_store_that_does_not_open_within_its_attach_timeout_is_degraded() -> None:
    async def run() -> tuple[str, str]:
        store = _NeverOpens()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, store, timeout=datetime.timedelta(milliseconds=50))
        request = asyncio.ensure_future(
            memory.transact(ctx, store, memory.InputData(statements=(_select_by_query_tags(query_tags="w"),)))
        )
        await asyncio.sleep(0.01)
        waiting = store.state()
        with pytest.raises(memory.RequestError, match="could not open its database"):
            _ = await request
        state = store.state()
        await store.stop(ctx)
        return waiting, state

    waiting, state = asyncio.run(run())
    assert waiting == "/_NeverOpensLifecycle/attached/behavior/opening"
    assert state == "/_NeverOpensLifecycle/attached/behavior/degraded"


def test_a_store_closes_the_database_it_opened_when_it_stops(tmp_path: pathlib.Path) -> None:
    """D-W7-9: a stopped store leaves no database open; what it committed is there for the next to open it."""

    path = tmp_path / "store.db"

    async def run(disposed: list[Engine]) -> tuple[int, tuple[str, ...]]:
        ctx = shared_hsm_context()
        first = memory.MemoryStore(database=str(path))
        await start_abilities_for_test(ctx, first)
        _ = await memory.transact(
            ctx, first, memory.InputData(statements=(_insert_content(content="kept", query_tags="k"),))
        )
        await hsm.stop(first, ctx)
        closed = len(disposed)
        second = memory.MemoryStore(database=str(path))
        await start_abilities_for_test(ctx, second)
        found = await memory.transact(
            ctx, second, memory.InputData(statements=(_select_by_query_tags(query_tags="k"),))
        )
        await hsm.stop(second, ctx)
        return closed, found.contents()

    with disposed_engines() as disposed:
        assert asyncio.run(run(disposed)) == (1, ("kept",))
    assert len(disposed) == 2


def test_a_store_given_an_engine_handle_opens_it_when_its_behavior_starts_and_leaves_it_open() -> None:
    """D-W8-20: a handle is called in the opening state, never at construction; what it opens is its owner's."""

    from mosfet.abilities.memory import store as store_module

    opened: list[Engine] = []

    def handle() -> Engine:
        engine = store_module.open_sqlite_engine()
        opened.append(engine)
        return engine

    async def run() -> tuple[int, tuple[str, ...]]:
        given = memory.MemoryStore(engine=handle)
        constructed = len(opened)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, given)
        found = await memory.transact(
            ctx,
            given,
            memory.InputData(
                statements=(_insert_content(content="handled", query_tags="h"), _select_by_query_tags(query_tags="h"))
            ),
        )
        await hsm.stop(given, ctx)
        return constructed, found.contents(statement_index=1)

    with disposed_engines() as disposed:
        constructed, contents = asyncio.run(run())

    assert (constructed, len(opened), contents) == (0, 1, ("handled",))
    assert disposed == []
    opened[0].dispose()


def test_a_store_leaves_an_engine_it_was_given_open_when_it_stops() -> None:
    """Whoever opens a database closes it: an injected engine is its owner's to dispose."""

    from mosfet.abilities.memory import store as store_module

    engine = store_module.open_sqlite_engine()

    async def run() -> None:
        injected = memory.MemoryStore(engine=engine)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, injected)
        _ = await memory.transact(
            ctx, injected, memory.InputData(statements=(_insert_content(content="theirs", query_tags="t"),))
        )
        await hsm.stop(injected, ctx)

    with disposed_engines() as disposed:
        asyncio.run(run())

    assert disposed == []
    selected = execute_on(engine, memory.InputData(statements=(_select_by_query_tags(query_tags="t"),)))
    assert selected.contents() == ("theirs",)
    engine.dispose()


def _thread_recording_handle(threads: list[int], *, stall: float = 0.0) -> collections.abc.Callable[[], Engine]:
    """An engine handle that records the thread it opens on and every thread a statement runs on.

    ``stall`` holds each ``SELECT`` on its thread, so a store running SQL on the event loop stalls the loop.
    """

    import threading
    import time

    import sqlalchemy.event

    from mosfet.abilities.memory import store as store_module

    def handle() -> Engine:
        threads.append(threading.get_ident())
        engine = store_module.open_sqlite_engine()

        def executing(*args: object) -> None:
            statement = args[2]
            threads.append(threading.get_ident())
            if stall and isinstance(statement, str) and statement.lstrip().upper().startswith("SELECT"):
                time.sleep(stall)

        sqlalchemy.event.listen(engine, "before_cursor_execute", executing)
        return engine

    return handle


def test_a_store_opens_migrates_and_transacts_off_the_event_loop() -> None:
    """3.4: the open, the migration and every transaction run on the store's worker, never on the loop's thread."""

    import threading

    threads: list[int] = []

    async def run() -> tuple[int, float]:
        loop_thread = threading.get_ident()
        lag = 0.0

        async def ticker() -> None:
            nonlocal lag
            loop = asyncio.get_running_loop()
            while True:
                before = loop.time()
                await asyncio.sleep(0.01)
                lag = max(lag, loop.time() - before - 0.01)

        ticking = asyncio.ensure_future(ticker())
        store = memory.MemoryStore(engine=_thread_recording_handle(threads, stall=0.2))
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, store)
        _ = await memory.transact(
            ctx,
            store,
            memory.InputData(
                statements=(_insert_content(content="off", query_tags="o"), _select_by_query_tags(query_tags="o"))
            ),
        )
        await hsm.stop(store, ctx)
        _ = ticking.cancel()
        return loop_thread, lag

    loop_thread, lag = asyncio.run(run())

    assert threads
    assert loop_thread not in threads
    assert lag < 0.1


def test_a_detach_waits_for_the_transaction_on_the_worker_and_its_request_is_answered() -> None:
    """3.4: a transaction is never cut off on the worker: a detach sent while it runs waits for its reply."""

    import time

    import sqlalchemy.event

    from mosfet.abilities.memory import store as store_module
    from mosfet.protocols import attachment
    from tests.hsm_instance_state import ability_terminal_owner

    def handle() -> Engine:
        engine = store_module.open_sqlite_engine()

        def executing(*args: object) -> None:
            statement = args[2]
            if isinstance(statement, str) and "query_tags" in statement:
                time.sleep(0.3)

        sqlalchemy.event.listen(engine, "before_cursor_execute", executing)
        return engine

    async def run() -> tuple[str, str, memory.OutputData]:
        store = memory.MemoryStore(engine=handle)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, store)
        request = asyncio.ensure_future(
            memory.transact(ctx, store, memory.InputData(statements=(_select_by_query_tags(query_tags="d"),)))
        )
        while store.state() != "/MemoryStoreLifecycle/attached/behavior/applying":
            await asyncio.sleep(0.01)
        owner = ability_terminal_owner(store)
        assert owner is not None
        _ = await store.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)))
        during = store.state()
        output = await asyncio.wait_for(request, 5)
        await asyncio.sleep(0.01)
        after = store.state()
        await hsm.stop(store, ctx)
        return during, after, output

    during, after, output = asyncio.run(run())

    assert during == "/MemoryStoreLifecycle/attached/behavior/applying"
    assert after == "/MemoryStoreLifecycle/detached"
    assert output.results[0].rows == ()

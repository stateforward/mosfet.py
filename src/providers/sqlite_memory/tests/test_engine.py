"""``open_engine``: a migrated SQLite engine with the bundled vec0 loaded on every connection."""

from __future__ import annotations

import concurrent.futures
import contextlib
import pathlib
import sqlite3
import typing

import pytest
from sqlalchemy import text

from mosfet.abilities import embedding, memory
from mosfet.providers import sqlite_memory
from mosfet.providers.sqlite_memory import engine as provider_engine


def test_open_engine_loads_vec0_and_migrates_to_head() -> None:
    engine = sqlite_memory.open_engine()
    with engine.connect() as connection:
        version = connection.execute(text("SELECT vec_version()")).scalar()
        tables = {row[0] for row in connection.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))}
    engine.dispose()
    assert isinstance(version, str) and version.startswith("v0.1.")
    assert {"bot_memory", "bot_memory_node", "bot_memory_vector", "bot_memory_index_attempt"} <= tables


def test_a_file_database_uses_wal_and_a_busy_timeout(tmp_path: pathlib.Path) -> None:
    engine = sqlite_memory.open_engine(database_path=tmp_path / "nested" / "memory.sqlite3")
    with engine.connect() as connection:
        mode = connection.execute(text("PRAGMA journal_mode")).scalar()
        timeout = connection.execute(text("PRAGMA busy_timeout")).scalar()
    engine.dispose()
    assert mode == "wal"
    assert timeout == 5000


def test_extension_loading_is_locked_again_after_vec0_loads() -> None:
    connection = sqlite3.connect(":memory:")
    provider_engine.load_vec0(connection, provider_engine.extension_path())
    with pytest.raises(sqlite3.OperationalError):
        connection.load_extension("anything")
    connection.close()


def test_a_missing_artifact_fails_closed(tmp_path: pathlib.Path) -> None:
    connection = sqlite3.connect(":memory:")
    with pytest.raises(sqlite_memory.ProviderError, match="vec0 did not load"):
        provider_engine.load_vec0(connection, tmp_path / "vec0.dylib")
    connection.close()


def test_a_python_without_extension_loading_fails_closed() -> None:
    class NoExtensions:
        """A DBAPI connection double from a Python built without loadable extensions."""

    with pytest.raises(sqlite_memory.ProviderError, match="cannot load extensions"):
        provider_engine.load_vec0(
            typing.cast(sqlite3.Connection, typing.cast(object, NoExtensions())), provider_engine.extension_path()
        )


def test_migrations_leave_vec0_tables_to_the_provider() -> None:
    """The packaged env.py excludes provider vec0 tables, so autogenerate reports no drift beside them."""

    from alembic import command

    engine = sqlite_memory.open_engine()
    with engine.begin() as connection:
        sqlite_memory.VecIndex().ensure(connection, embedding.Space(model="test/embedder", dimensions=2))
        config = memory.store.migration_config()
        config.attributes["connection"] = connection
        command.check(config)
    engine.dispose()


def test_a_database_opens_nothing_until_its_first_open_and_then_opens_once(tmp_path: pathlib.Path) -> None:
    """D-W8-20: what Memory and the identity stores share is opened by whichever starts first, once."""

    path = tmp_path / "data" / "memory.sqlite3"
    database = sqlite_memory.Database(path=path)
    constructed = path.parent.exists()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        engines = [opened.result() for opened in [pool.submit(database.open) for _ in range(8)]]

    assert constructed is False
    assert all(engine is engines[0] for engine in engines)
    with engines[0].connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM alembic_version")).scalar_one() == 1
        assert connection.execute(text("SELECT vec_version()")).scalar_one()
    engines[0].dispose()


def test_a_database_is_one_file_and_one_that_cannot_open_raises_each_time_it_is_opened(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "unversioned.sqlite3"
    with contextlib.closing(sqlite3.connect(path)) as connection, connection:
        _ = connection.execute(f"CREATE TABLE {memory.MEMORY_TABLE} (memory_id TEXT PRIMARY KEY)")
    database = sqlite_memory.Database(path=path)

    for _ in range(2):
        with pytest.raises(memory.store.MigrationError):
            _ = database.open()
    with pytest.raises(ValueError, match="file"):
        _ = sqlite_memory.Database(path=":memory:")

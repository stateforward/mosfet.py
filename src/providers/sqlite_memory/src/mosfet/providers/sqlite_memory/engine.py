"""Open the SQLite engine a ``memory.Memory`` runs on, with the bundled ``vec0`` extension loaded.

``open_engine`` is core ``memory.store.open_sqlite_engine`` plus a connect hook that loads the
``vec0`` loadable this package ships (``_native/vec0.dylib`` or ``vec0.so``), re-disables
extension loading immediately so SQL can never load another one, sets ``busy_timeout``, and
switches file databases to WAL. It fails closed with ``ProviderError`` when the interpreter's
``sqlite3`` cannot load extensions (for example Apple's system Python), when the artifact is
missing, or when ``vec0`` does not answer.
"""

from __future__ import annotations

import collections.abc
import importlib.resources
import pathlib
import sqlite3
import sys

from mosfet.abilities import memory
from sqlalchemy.engine import Engine

_BUSY_TIMEOUT_MS = 5000


class ProviderError(RuntimeError):
    """The SQLite memory provider cannot open a vector-capable database here."""


def extension_path() -> pathlib.Path:
    """Where this package's bundled ``vec0`` loadable lives (performs filesystem I/O).

    Raises:
        ProviderError: if the artifact was not built into this installation.
    """

    name = {"darwin": "vec0.dylib", "win32": "vec0.dll"}.get(sys.platform, "vec0.so")
    artifact = importlib.resources.files("mosfet.providers.sqlite_memory") / "_native" / name
    if not artifact.is_file():
        raise ProviderError(f"the bundled vec0 extension {name} is missing from this installation")
    # Installed as files (never zipped), so the resource is a real path for the process lifetime.
    return pathlib.Path(str(artifact))


def load_vec0(connection: sqlite3.Connection, path: pathlib.Path) -> None:
    """Load ``vec0`` into one DBAPI connection and lock extension loading again.

    Raises:
        ProviderError: if the connection cannot load extensions, or ``vec0`` does not load or answer.
    """

    if not hasattr(connection, "enable_load_extension"):
        raise ProviderError("this Python's sqlite3 cannot load extensions; use a Python built with them")
    try:
        connection.enable_load_extension(True)
        try:
            connection.load_extension(str(path))
        finally:
            connection.enable_load_extension(False)
        _ = connection.execute("SELECT vec_version()").fetchone()
    except sqlite3.Error as error:
        raise ProviderError(f"vec0 did not load: {error}") from error


def connect_hook(path: pathlib.Path, *, file_database: bool) -> collections.abc.Callable[[sqlite3.Connection], None]:
    """The per-connection setup ``open_engine`` installs: vec0, busy timeout, and WAL for files."""

    def connected(connection: sqlite3.Connection) -> None:
        load_vec0(connection, path)
        _ = connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        if file_database:
            _ = connection.execute("PRAGMA journal_mode = WAL")

    return connected


def open_engine(*, database_path: pathlib.Path | str = ":memory:") -> Engine:
    """Open (and migrate) a vector-capable SQLite memory database; pass it as ``memory.Memory(engine=...)``.

    Creates the parent directory of a file database. Performs filesystem and database I/O.

    Raises:
        ProviderError: if extensions cannot be loaded or the bundled ``vec0`` is missing.
        memory.store.MigrationError: if the database matches no known schema revision.
    """

    database = str(database_path)
    file_database = database != ":memory:"
    if file_database:
        pathlib.Path(database).parent.mkdir(parents=True, exist_ok=True)
    hook = connect_hook(extension_path(), file_database=file_database)
    return memory.store.open_sqlite_engine(database=database, on_connect=hook)


__all__ = ["ProviderError", "connect_hook", "extension_path", "load_vec0", "open_engine"]

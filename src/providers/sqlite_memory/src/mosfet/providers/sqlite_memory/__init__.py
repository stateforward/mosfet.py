"""SQLite + sqlite-vec storage for ``memory.Memory`` and identities: a vector-capable engine, a
``vec0`` index, and an ``identification.Store``.

Usage::

    from mosfet.abilities import identification, memory
    from mosfet.providers import sqlite_memory

    database = sqlite_memory.Database(path="bot-memory.sqlite3")  # opened by the first part to start
    store = memory.Memory(engine=database.open, index=sqlite_memory.VecIndex(), embedder=embedder)
    voices = identification.Matcher(store=sqlite_memory.IdentityStore(engine=database.open), policy=policy)

``open_engine(database_path=...)`` opens one now, for a script or a test.
"""

from .engine import Database, ProviderError, open_engine
from .identity import IdentityStore
from .index import VecIndex

__version__ = "0.1.0"

__all__ = [
    "Database",
    "IdentityStore",
    "ProviderError",
    "VecIndex",
    "__version__",
    "open_engine",
]

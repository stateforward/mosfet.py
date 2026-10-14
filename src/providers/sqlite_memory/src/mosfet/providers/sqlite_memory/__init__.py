"""SQLite + sqlite-vec storage for ``memory.Memory``: a vector-capable engine and a ``vec0`` index.

Usage::

    from mosfet.abilities import memory
    from mosfet.providers import sqlite_memory

    store = memory.Memory(
        engine=sqlite_memory.open_engine(database_path="bot-memory.sqlite3"),
        index=sqlite_memory.VecIndex(),
        embedder=embedder,
    )
"""

from .engine import ProviderError, open_engine
from .index import VecIndex

__version__ = "0.1.0"

__all__ = [
    "ProviderError",
    "VecIndex",
    "__version__",
    "open_engine",
]

"""SQLite + sqlite-vec storage for ``memory.Memory`` and identities: a vector-capable engine, a
``vec0`` index, and an ``identification.Store``.

Usage::

    from mosfet.abilities import identification, memory
    from mosfet.providers import sqlite_memory

    engine = sqlite_memory.open_engine(database_path="bot-memory.sqlite3")
    store = memory.Memory(engine=engine, index=sqlite_memory.VecIndex(), embedder=embedder)
    voices = identification.Matcher(store=sqlite_memory.IdentityStore(engine=engine), policy=policy)
"""

from .engine import ProviderError, open_engine
from .identity import IdentityStore
from .index import VecIndex

__version__ = "0.1.0"

__all__ = [
    "IdentityStore",
    "ProviderError",
    "VecIndex",
    "__version__",
    "open_engine",
]

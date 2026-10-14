# mosfet-provider-sqlite-memory

SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) storage for the core `memory.Memory`.

`memory.Memory` keeps the bot's relational rows and its episodic graph (episodes, persons,
concepts, links, and one canonical vector per node) in SQLite. This package supplies what core
cannot ship without a native extension:

- `open_engine(database_path=...)`: the core SQLite engine, migrated to the head schema, with the
  bundled `vec0` loadable extension loaded on every connection (extension loading is locked again
  right after), `busy_timeout=5000`, and WAL for file databases.
- `VecIndex()`: approximate cosine nearest search over one `vec0` table per embedding space
  (`bot_memory_vec_<dims>_<hash>`), written inside Memory's own transactions. Canonical vectors
  stay in `bot_memory_vector`, so the index can be dropped and rebuilt.
- `ProviderError`: raised when the interpreter's `sqlite3` cannot load extensions (for example
  Apple's system Python), or when the bundled `vec0` is missing or does not answer.

```python
from mosfet.abilities import memory
from mosfet.providers import sqlite_memory

store = memory.Memory(
    engine=sqlite_memory.open_engine(database_path="bot-memory.sqlite3"),
    index=sqlite_memory.VecIndex(),
    embedder=embedder,  # any embedding.Embedder, or None for a bot that cannot embed
)
```

The wheel build hook copies the `vec0` loadable from the build host's `sqlite-vec` wheel into
`src/mosfet/providers/sqlite_memory/_native/` (`BOT_SQLITE_MEMORY_SKIP_NATIVE_BUILD=1` skips it,
and `open_engine` then fails closed). Migrations never manage the `bot_memory_vec_*` tables.

Run the provider tests:

```sh
uv run --package mosfet-provider-sqlite-memory --group dev python -m pytest src/providers/sqlite_memory/tests
```

# mosfet-provider-sqlite-memory

SQLite-backed stateforward.mosfet Memory provider.

One ability apply runs one or more parameterized SQL statements in a single transaction against `bot_memory`. Success commits; any failure rolls back.

```python
from bot.providers.sqlite_memory import SqliteMemory, Memory
from bot.abilities import memory

store = SqliteMemory(database_path="agent-memory.sqlite3")
# apply memory.InputData(statements=(Statement(sql=..., parameters=(...)), ...))
```

There is no separate encode/decode store path. Generation (if used) is a core sibling ability that produces content for `INSERT` parameters.

Run the provider tests:

```sh
uv run --package mosfet-provider-sqlite-memory --group dev python -m pytest src/providers/sqlite_memory/tests
```

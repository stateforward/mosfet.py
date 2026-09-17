# mosfet-provider-postgres-memory

Postgres-backed stateforward.mosfet Memory provider.

One ability apply runs one or more parameterized SQL statements in a single transaction against `bot_memory`. Success commits; any failure rolls back.

Inject a `Database` adapter (hosted Postgres, PGlite, etc.). Statement SQL uses `?` placeholders; the provider rewrites them for the adapter parameter style (`%s` or `$1`).

```python
from bot.providers.postgres_memory import PostgresMemory, Memory
from bot.abilities import memory

memory_ability = PostgresMemory(database=my_database)
# apply memory.InputData(statements=(...))
```

There is no separate encode/decode or object-store retain path on the provider surface. Multimodal storage can be expressed later as additional SQL tables/statements if needed.

Run the provider tests:

```sh
uv run --package mosfet-provider-postgres-memory --group dev python -m pytest src/providers/postgres_memory/tests
```

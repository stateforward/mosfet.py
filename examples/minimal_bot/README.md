# minimal_bot

The smallest bot: one bot, `minimal`, with no devices and a cognition (intuition, reasoning, reflection,
recall and consolidation on one memory): reasoning, reflection and consolidation on any OpenAI-compatible
endpoint, and intuition on a decision model, hosted TypeSafe or a local system_one server. Its bot file is
`minimal_bot_example/__main__.py`: it reads the endpoint and the secret, then has one `bot.define` that
constructs every part inline, and `python -m minimal_bot_example` runs it with `bot.run`:

```python
minimal = bot.define(
    "minimal",
    bot.cognition(
        openai_compat.Processor(
            client=openai_compat.ChatClient(model=model, base_url=base_url, api_key=...),
            provider="minimal_bot",
        ),
        intuition=typesafe.Processor(base_url=os.environ.get("TYPESAFE_BASE_URL") or None, api_key=...),
        memory=memory.Memory(database=str(process.data_dir / "memory.sqlite3")),
        associative=True,
        consolidation=True,
    ),
    bot.allow("no_devices"),
)
bot.run(minimal, process=process)
```

With no devices nothing reaches the bot, so it activates, stays ready, and stops cleanly; add a device
or an acquired ability to give it something to perceive.

The bot file reads:

* `OPENAI_BASE_URL`: the endpoint, for example `https://api.openai.com/v1` or a local server;
* `OPENAI_MODEL`: the model that endpoint serves;
* the `openai_api_key` secret: the Docker secret `/run/secrets/openai_api_key`, the file named by
  `OPENAI_API_KEY_FILE`, or `OPENAI_API_KEY`;
* `TYPESAFE_BASE_URL`, optional: the local system_one server intuition decides on, for example
  `http://127.0.0.1:8102`; unset, intuition decides on hosted TypeSafe;
* the `typesafe_api_key` secret, read the same way, for hosted TypeSafe only: its key. With `TYPESAFE_BASE_URL`
  set it is never read, since a local system_one server takes no key and is never sent the hosted one.

Run it from the repository root:

```bash
uv sync --project examples/minimal_bot
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 OPENAI_MODEL=... OPENAI_API_KEY=... TYPESAFE_API_KEY=... \
  examples/minimal_bot/.venv/bin/python -m minimal_bot_example --ready-file /tmp/ready
```

The process stays up until SIGTERM or SIGINT and exits `0`. A missing variable or secret exits `1`
before anything starts. Memory is kept in `<data dir>/memory.sqlite3`.
[`docs/runtime.md`](../../docs/runtime.md) has every `bot.process` option and exit code.

Test it, including the command above run as a child process. The tests write their own `bot.define` of a bot of
the same shape on scripted models (`testing.Scripted`, and `testing.Decider` for its intuition):

```bash
uv run --project examples/minimal_bot python -m pytest examples/minimal_bot/tests -q
```

## Run it under compose against the lab box

[`compose.yaml`](compose.yaml) builds the generic image
([Running in a container](../../docs/runtime.md#running-in-a-container)) for this project and runs it with a
read-only root, a tmpfs ready file, the `minimal-data` volume, `init: true`, all capabilities dropped, and
`restart: on-failure:5`. It points at the lab box's OpenAI-compatible endpoint; it does not run one.

```bash
cd examples/minimal_bot
cp .env.example .env              # set the endpoint's PORT and OPENAI_MODEL
printf '%s' "$KEY" > secrets/openai_api_key   # any non-empty placeholder if the endpoint needs no key
printf '%s' "$TYPESAFE_KEY" > secrets/typesafe_api_key   # hosted TypeSafe's key; empty with TYPESAFE_BASE_URL
chmod 0644 secrets/openai_api_key secrets/typesafe_api_key
docker compose up --build --detach
docker compose ps                 # STATUS shows (healthy) once the bot is ready
docker compose stop               # graceful stop; the container exits 0
```

Compose mounts the secret files at `/run/secrets/openai_api_key` and `/run/secrets/typesafe_api_key`, which
the bot file reads first; compose needs the second file to exist, but with `TYPESAFE_BASE_URL` set the bot never
reads it. Outside
swarm that is a bind mount that keeps the host file's owner and mode, so the file must be readable by the
container's uid `10001`; `secrets/` is git-ignored and never enters the image. A missing `.env` stops
`docker compose` before it starts anything; a missing endpoint variable or secret exits `1`, and the
restart policy gives up after five attempts.

`scripts/docker_smoke.sh --compose`, from the repository root, brings this file up against a closed local
endpoint with placeholder secrets, waits for healthy, stops it and checks exit `0`.

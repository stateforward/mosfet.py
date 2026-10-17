# builder_bot

One bot, for one person, that the person reaches through a computer and that builds and tests software on a
sandbox bench of its own. `builder_bot_example.builder` defines that bot, `builder`, with `bot.define`, and
`python -m builder_bot_example` runs it with `bot.run`:

```python
return bot.define(
    "builder",
    bot.traits(*traits),
    bot.cognition(processor, memory=store, associative=True, consolidation=True, autonomy=True, routines=clock),
    bot.attach(computer.Computer(id="desk", transport=server, peer_id=peer, person_id=person_id)),
    bot.attach(sandbox.Sandbox(host=host, capacity=capacity, spec=spec, max_output_chars=MAX_OUTPUT_CHARS)),
)
```

* **Its computer** is a `computer.Computer`, `desk`, served on this host by a `websocket.Server`. The
  person sits at its far side, `builder-terminal`: the screen shows what the bot shows, and the keyboard types
  into the field the bot focused.
* **Its bench** is a `sandbox.Sandbox` on a `docker.Host` over a rootless Docker or Podman socket. The bot may
  run commands, write and read files, and start over on a fresh machine.
* **Its mind** is the full cognition, without voice: intuition, reasoning, reflection with its persona, the
  routines it authors, and memory kept in `<data dir>/memory.sqlite3`. One local OpenAI-compatible model does
  all of its thinking.

Nothing tells it what to do. The bot gets no seeded behavior, no skill and no instruction text. It learns
about its computer and its bench only from the tools they offer on each turn: `computer.show`, and
`bot.ability.sandbox.run`, `write`, `read` and `reset`, which are offered only while the bench is ready. It also
learns from what it sees. Whether it shows anything, uses the bench or answers a line is up to the bot. Doing
nothing is a choice it may make too. Its owner sets only its traits (`careful, curious, candid` unless
`BUILDER_TRAITS` gives others) and its bench.

It does not change its own code, load code it wrote, or reach the host's files, a repository on the host, or the
network (unless the owner gives the bench an egress bridge). Files in the workspace are lost when its machine is
renewed or the process stops. Memory keeps only what the bot chose to remember.

## Set up the bench

The bench needs a rootless daemon. Rootful Docker and Docker Desktop are refused. On macOS, use a rootless
Podman machine and find its Docker-compatible API socket:

```bash
podman machine init --rootful=false   # once; then
podman machine start
podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}'
```

That path is under `/var/folders/…/T/` and can change across reboots, so it is configuration with no default.
On Linux it is `$XDG_RUNTIME_DIR/podman/podman.sock` (`systemctl --user enable --now podman.socket`).

Build bench image 2 into that daemon, from the repository root. Nothing is ever pulled for the bench. Besides
python, uv, git and pytest it has mosfet installed, so the bot can write a bot file (`from mosfet import bot`),
test it with pytest, and read `bot.define`'s diagnostics when it is wrong. Then check it. This needs no model:

```bash
podman build -t localhost/mosfet-bench:2 -f docker/bench/Dockerfile .
docker/bench/smoke.sh --socket "$BUILDER_SANDBOX_SOCKET" --python examples/builder_bot/.venv/bin/python
```

## Configure

Run these from the repository root:

```bash
uv sync --project examples/builder_bot
cp examples/builder_bot/.env.example examples/builder_bot/.env   # then fill it in
printf '%s' "$KEY" > examples/builder_bot/secrets/openai_api_key  # any placeholder for a keyless local server
head -c 32 /dev/urandom | base64 > examples/builder_bot/secrets/computer_token_secret
chmod 0600 examples/builder_bot/secrets/*
```

| Name | Kind | Default | Meaning |
|---|---|---|---|
| `OPENAI_BASE_URL`, `OPENAI_MODEL` | env, required | | The local OpenAI-compatible endpoint and the model it serves |
| `openai_api_key` | secret, required | | The endpoint's key; a placeholder for a keyless server |
| `BUILDER_SANDBOX_SOCKET` | env, required | | The rootless Docker or Podman API socket |
| `BUILDER_SANDBOX_IMAGE` | env | `localhost/mosfet-bench:2` | The one image the bench may use |
| `BUILDER_SANDBOX_EGRESS_NETWORK` | env | unset | The owner's egress bridge; unset, the bench has no network |
| `BUILDER_SANDBOX_RUNTIME` | env | `runc` | `runc`, or `runsc` for gVisor |
| `BUILDER_COMPUTER_HOST`, `BUILDER_COMPUTER_PORT` | env | `127.0.0.1`, `8765` | Where the computer is served |
| `BUILDER_COMPUTER_PEER` | env | `builder-desk` | The name the computer's far side connects as (its peer id) |
| `computer_token_secret` | secret, required | | The key the terminal's token is signed with |
| `BUILDER_PERSON_ID` | env | unset | Who uses the computer, stamped on what it observes |
| `BUILDER_TRAITS` | env | `careful,curious,candid` | The owner's traits for the bot, comma separated |

`builder_bot_example.Settings` reads the variables; an empty one is unset. A secret is read from `--secrets-dir`
(`MOSFET_SECRETS_DIR`), then from the file `<NAME>_FILE` names, then from `<NAME>`. The bench is fixed by the bot
file (`builder_bot_example.bench`). It gets the default limits: one CPU, 1 GiB without swap, 256
pids and a 512 MiB workspace at `/workspace`, which is also `HOME`. Each output stream and file read is kept to
8 Ki characters, so a turn's observations fit a local model's context.

## Run it

Start the bot, then sit down at its computer in another terminal:

```bash
set -a; . examples/builder_bot/.env; set +a
examples/builder_bot/.venv/bin/python -m builder_bot_example \
  --secrets-dir examples/builder_bot/secrets --data-dir examples/builder_bot/.mosfet --ready-file /tmp/builder.ready
```

```bash
examples/builder_bot/.venv/bin/builder-terminal --url ws://127.0.0.1:8765 --peer builder-desk \
  --secret-file examples/builder_bot/secrets/computer_token_secret
```

A plain line is typed into the field the bot focused, then Enter. A line longer than 4095 characters is not typed,
and the terminal says so. `/click <id>` presses the element with that id. `/quit` leaves. When nothing is focused,
the line goes nowhere, and the terminal says so. The process stays up until SIGTERM or SIGINT and exits `0`. A
missing or malformed setting or secret exits `1` before anything starts, and the message names it.
[`docs/runtime.md`](../../docs/runtime.md) lists every `bot.process` option and exit code.

## The live check (manual)

The automated tests never run a model. To see the builder on the real surface, its owner runs this by hand
against rootless Podman and a local model:

1. `python -m builder_bot_example`, as above, reaches ready (the ready file exists) with the Podman socket.
2. `builder-terminal` connects, and within a bound the bot is given a turn for `computer.link.up`.
   `uv run python scripts/otel_turns.py --dir <the directory the bot ran in>` prints each turn from
   `otel-spans.jsonl` and `otel-logs.jsonl`.
3. That turn offered `computer.show` and `bot.ability.sandbox.{run,write,read,reset}`. Run with
   `BOT_OTEL_CAPTURE=full` to see the offered tools and the captured turn input.
4. Every sandbox request the bot made was answered exactly once: `bot.ability.sandbox.request.count` equals
   the number of requests issued.

What the bot chose (what it showed, what it asked of the bench, and how each request ended) is a report, not
a pass condition. A bot that shows nothing has not failed the check. The question then is what it lacked.

## Security

* **The bench** is a container on a rootless daemon, with a read-only root, uid `65532`, every capability
  dropped, `no-new-privileges`, and the limits above. It has no network unless the owner sets
  `BUILDER_SANDBOX_EGRESS_NETWORK`, and it has only the image the owner built. Nothing is pulled, mounted or
  shared from the host. The workspace is an ephemeral tmpfs. No secret goes into the bench's environment.
  [`src/providers/docker/README.md`](../../src/providers/docker/README.md) has the full machine mapping.
* **Untrusted output.** Code the bot runs can print text aimed at the model. With no network, the only way out
  is the display, which the person sees.
* **The computer's transport is plaintext** `ws://` with an HMAC token. Anyone on the path can read the session
  and replay the token. It listens on loopback by default. Serving it on a LAN (`BUILDER_COMPUTER_HOST`) is the
  owner's choice, made knowing that.
* **Unbounded work.** Each observation wakes a turn, so the bot can keep running commands. The bench bounds
  that: one request at a time, a short queue, ten minutes per run, and the limits. The owner can always stop
  the process. `bot.ability.sandbox.request.count` shows how much it did.

## Test it

```bash
uv run --project examples/builder_bot python -m pytest examples/builder_bot/tests -q
```

`test_build.py` checks the settings and the bot `builder` defines, and runs `python -m builder_bot_example` as a
child process. `test_terminal.py` runs the terminal against a real loopback server and computer. `test_builder.py`
runs the bot with `testing.running`, a scripted model (`testing.Scripted`) and a scripted sandbox host. The
terminal connects, and the model's choices reach the screen, the bench and the next turn. A model that chooses
nothing changes nothing. `test_writes_bots.py` proves the bot can fix a bot file it wrote from `bot.define`'s
diagnostics alone: the scripted host runs the real `bot.define` on the written file, and the rewrite made by
applying only the `help:` in the run's stderr defines clean.

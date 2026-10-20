# Running bots (`bot.run`, `bot.process`, `mosfet.runtime`)

A bot file is a plain Python file: it defines its bots with `bot.define` ([`bot.md`](bot.md)) and runs
them from its `__main__` with `bot.run`, the way an hsm program defines and starts its machines.

```python
from mosfet import bot
from mosfet.devices import computer
from mosfet.providers import decision, openai_compat, websocket

if __name__ == "__main__":
    process = bot.process()
    server = websocket.Server(verifier=websocket.HmacTokens(process.secrets.get("computer_token_secret")))
    greeter = bot.define(
        "greeter",
        bot.cognition(
            openai_compat.Processor(client=openai_compat.ChatClient(model="qwen3-coder")),
            intuition=decision.Processor(base_url="http://127.0.0.1:8102"),
        ),
        bot.attach(computer.Computer(id="desk", transport=server)),
    )
    bot.run(greeter, actors=(server,), process=process)
```

`bot.run` runs one process's bots: it builds them from their models, keeps each one running within a
restart budget, and stops the whole process gracefully within one bound. Two decomposed state machines in
`mosfet.runtime` do the work: `runtime.Runtime` owns the process lifecycle, and one `runtime.Supervisor` per
bot owns that bot. All waiting is bounded by `hsm.after`, and all asynchronous work runs in activities that
end in typed events.

## What `bot.run` runs

`bot.run(*models, actors=(), process=None, observe=())` is synchronous and never returns: it ends the
process with its exit code.

* `models`: 1 to 32 `bot.Model`s, each given once (`B0105`), each under a name of its own (`B0104`). A
  bot's name is its supervisor's name and a telemetry attribute, so `MAX_BOTS` bounds that cardinality.
* `actors`: shared actors such as a websocket server or an embedding model. Each must be an `hsm.Instance`
  whose class declares `model` (`B0106`); the process starts each one with exactly that model before any
  bot and stops them in reverse after every bot.
* `process`: the process's options, `bot.process()` when omitted.
* `observe`: boundary observers (`bot.observe(...)`, which is `hsm.observe`) for logs and records. Each
  bot's body runs on its model redefined with them, so they watch every bot the process runs, across its
  supervisor's restarts, and never act on it.

A violation is printed to stderr like `bot.define`'s errors, with the span of the bot file's own `bot.run`
call, and the process exits `78` (`invalid_models`). Then `bot.run` creates the data dir `0700`, configures
telemetry once, and builds each bot from its model as `bot.start` does: a model whose parts already run is
`B0101`, and a part that raises while it is built is `B0102` (`constructor_raised`), both exit `78`, with the
traceback only under `MOSFET_TRACEBACK=1`. [`bot_diagnostics.md`](bot_diagnostics.md) explains each code.

Errors the bot file raises itself before it calls `bot.run` (its own settings, a secret it asks for) are
its own: they end the process with Python's exit status `1`.

## Process options (`bot.process`)

`bot.process(argv=None, environ=None)` reads the options once, at the bot file's `__main__` boundary: the
`MOSFET_*` variables first, then the flags after the bot's module (`python -m greeter --ready-file
/tmp/ready`), so a flag wins over its variable. It returns a frozen `bot.Process`. It knows only its own
options: a bot file that takes flags of its own reads them itself and passes the rest as `argv=`.

| Flag | Variable | Default | Meaning |
|---|---|---|---|
| `--name NAME` | `MOSFET_NAME` | the `__main__` package | `Process.name`, `service.name` |
| `--instance-id ID` | `MOSFET_INSTANCE_ID` | `$HOSTNAME`, else the host name | `Process.instance_id`, `service.instance.id` |
| `--data-dir PATH` | `MOSFET_DATA_DIR` | `./.mosfet` | `Process.data_dir`, created `0700` when absent |
| `--secrets-dir PATH` | `MOSFET_SECRETS_DIR` | `/run/secrets` | Directory searched first by `Process.secrets` |
| `--ready-file PATH` | `MOSFET_READY_FILE` | unset | Exists exactly while the Runtime is `running/ready` (for `HEALTHCHECK test -f`) |
| `--stop-grace SECONDS` | `MOSFET_STOP_GRACE` | `8` | `Policy.stop_grace` |
| `--bot-stop-timeout SECONDS` | `MOSFET_BOT_STOP_TIMEOUT` | `4` | `Policy.bot_stop_timeout` |
| `--bot-release-timeout SECONDS` | `MOSFET_BOT_RELEASE_TIMEOUT` | `1` | `Policy.bot_release_timeout` |
| `--start-timeout SECONDS` | `MOSFET_START_TIMEOUT` | `120` | `Policy.start_timeout` |
| `--max-restarts N` | `MOSFET_MAX_RESTARTS` | `5` | `Policy.max_restarts` |
| `--backoff-initial SECONDS` | `MOSFET_BACKOFF_INITIAL` | `1` | `Policy.backoff_initial` |
| `--backoff-max SECONDS` | `MOSFET_BACKOFF_MAX` | `60` | `Policy.backoff_max` |
| `--stable-after SECONDS` | `MOSFET_STABLE_AFTER` | `60` | `Policy.stable_after` |

The policy options are generated from the `runtime.Policy` fields, so an omitted option keeps the policy
default. An unknown option, a value that does not parse, or policy values that do not fit together (for
example `--bot-stop-timeout` plus `--bot-release-timeout` not below `--stop-grace`) is `B0107`: it is printed
to stderr naming the option or variable, never its value, and the process exits `64`. `--help` prints the
options and exits `0`. An empty variable is unset. `MOSFET_TRACEBACK=1` makes `bot.run` print the traceback
of a bot it could not build. `BOT_OTEL_*` and `OTEL_*` keep their telemetry meaning.

`Process.secrets` is a `runtime.Secrets` reader. `get(name)` returns a `pydantic.SecretStr` from the first
source that has it: the file `<secrets dir>/<name lowercased>` (the Docker `/run/secrets` convention), then
the file named by the environment variable `<NAME>_FILE`, then the environment variable `<NAME>`. A missing
secret raises `runtime.MissingSecretError`, whose message names the secret and never a value.

## Policy

`runtime.Policy` is injected, mechanical configuration. It never decides what a bot does.

| Field | Default | Meaning |
|---|---|---|
| `stop_grace` | 8 s | The single bound on the whole graceful stop |
| `bot_stop_timeout` | 4 s | How long one bot may take to answer its stop request |
| `bot_release_timeout` | 1 s | How long releasing (stopping the machine of) one bot may take |
| `start_timeout` | 120 s | One activation attempt; also bounds actor and supervisor bring-up |
| `max_restarts` | 5 | Consecutive failed attempts before the restart budget is exhausted (at most 100) |
| `backoff_initial` | 1 s | Delay before the first restart; doubles per consecutive failure |
| `backoff_max` | 60 s | Cap on that delay |
| `stable_after` | 60 s | Time running before the count of consecutive failures resets |

The validator requires positive durations, `backoff_max >= backoff_initial`, and
`bot_stop_timeout + bot_release_timeout < stop_grace`, so every bot's stop fits inside the process grace
and the shared actors still get the remainder.

## Process lifecycle (`runtime.Runtime`)

`bot.run` creates one `mosfet.Environment`, passes it to the `Runtime` with the bots it built and the
shared actors, and starts it with `mosfet.started(environment, runtime, Runtime.model)`. The Runtime's own
context is therefore a non-private child of that environment.

```
starting_actors       remove a stale ready file, then start each actor in order
                                                     -> starting_supervisors | stopping
starting_supervisors  start one Supervisor per bot   -> running | stopping
running               settling -> ready (all bots reported running; the ready file exists only here)
stopping              one stop_grace bound           -> stopped (0, 70 or 78) | forced (1)
  draining (choice)   any supervisor recorded?       -> stopping_supervisors | stopping_actors
  stopping_supervisors  ask each to stop, await each answer or its unreachable outcome
  stopping_actors       stop every supervisor (idempotent), then the actors in reverse order
```

* Actors start with `mosfet.started(runtime context, actor, type(actor).model)`; supervisors start in the
  same context, and every bot attaches to the same environment.
* A stop request during bring-up leaves at once; nothing is deferred. Everything bring-up recorded is
  stopped.
* A bot can activate, or exhaust its budget, before the last supervisor has started. Those
  `runtime.supervisor.running` and `runtime.supervisor.exhausted` reports are deferred in
  `starting_supervisors`, so readiness and exit code `70` never depend on start timing.
* Supervisors and actors are recorded **before** their start is invoked, in the Runtime's own state. Once
  invoked, hsm runs a start to completion in the started machine's own task, so a supervisor whose start a
  stop interrupted is still started: it receives `runtime.supervisor.stop` and detaches its bot. A recorded
  supervisor whose start never ran cannot be asked to stop; that undeliverable request, like a stop request
  whose delivery fails, becomes the typed outcome `runtime.supervisor.unreachable`, and the Runtime stops
  waiting for it. It never probes a supervisor's state.
* A supervisor or actor stop in `stopping_actors` that raises is logged and counted, and the remaining
  ones are still stopped, so a failing stop does not cost the grace period or change the exit code.
* Every stop in `stopping_actors` is awaited so that cancelling the activity ends the wait at once. An
  actor whose stop ignores cancellation therefore cannot hold the Runtime past `stop_grace`: the grace
  timer still forces exit `1`, and the stop that outlived it is left to the process boundary.
* `on_exit(runtime.ExitedData)` is called exactly once, by the transition into a final state.
* The ready file is touched on entry to `running/ready` and removed on its exit. A run killed without
  leaving `ready` (SIGKILL, an OOM kill, a host crash) cannot remove it, and a restarted container keeps
  its writable layer, so `starting_actors` removes any ready file left by an earlier run before it starts
  anything. The file therefore exists exactly while this run is ready, across crashes too. A failure to remove
  it is logged with its failure kind only.

### Exit codes

| Code | Reason | When |
|---|---|---|
| `0` | `stopped` | Graceful stop within the grace (signal or `runtime.StopEvent`), from any state. A bot force-released inside the grace still counts as graceful. |
| `1` | `forced` | The grace expired, or a second **signal** arrived while stopping. A programmatic stop request never forces. |
| `70` | `exhausted` | A bot used up its restart budget. Kept when a first signal arrives during that stop. |
| `78` | `boot_failed` | Actor or supervisor bring-up failed or timed out. Kept when a first signal arrives during that stop. `bot.run` also exits `78` before any Runtime exists: invalid models or actors, a bot that could not be built, a data dir it cannot create, or telemetry it cannot configure. |

`64` (a bad process option) belongs to `bot.process`, which rejects it before a Runtime exists. `1` is
also the exit status of a bot file that raises before it calls `bot.run`.

## The process boundary (`bot.run`)

* **Stdout** stays empty; diagnostics go to stderr. `bot.run` configures logging to stderr at `WARNING`.
* **Signals.** SIGTERM and SIGINT each dispatch `runtime.StopEvent` with reason `signal`; the first starts
  the graceful stop and a second one forces it. The handlers are installed before the Runtime starts, so a
  signal during bring-up is delivered once its start returns, and removed once the Runtime reports its exit.
  SIGHUP keeps its default disposition.
* **Bounded exit.** After the Runtime reports its exit, `bot.run` cancels every remaining task and waits
  at most 1 s for them and for threads running blocking calls, then flushes telemetry for at most 0.5 s. If
  anything is still running, it ends the process with `os._exit` and the Runtime's exit code; `atexit`
  handlers do not run on that path. From SIGTERM to process exit therefore takes at most `stop_grace` plus
  1.5 s, 9.5 s by default, which is under Docker's 10 s SIGKILL. Raising `stop_grace` needs a matching
  `docker run --stop-timeout`.

## Running in a container

`docker/Dockerfile` is one image recipe for any bot project with a `__main__`. The build context is the
repository root, and the `PROJECT` build argument names the project to install from its own `uv.lock`:

```bash
docker build --build-arg PROJECT=examples/minimal_bot -f docker/Dockerfile -t minimal-bot .
```

| Build argument | Default | Meaning |
|---|---|---|
| `PROJECT` | `examples/minimal_bot` | The project installed with `uv sync --locked --no-dev --no-editable` |
| `NATIVE` | `0` | `1` adds a C toolchain and cmake to the build stage, for providers that compile native code (`sqlite_memory`, `llama_cpp`) |
| `RUNTIME_APT` | empty | Debian packages the runtime needs, for example `libgl1 libglib2.0-0` for `opencv` |

The build stage installs the project into `/opt/mosfet/venv` with a pinned `uv`; path dependencies are
installed as regular packages, so the runtime image holds the virtual environment and no source tree. The
runtime image is laid out for a read-only root:

* It runs as the fixed non-root user `mosfet`, uid and gid `10001`.
* The working directory `/var/lib/mosfet` (owned by `10001`, mode `0700`) is the data volume. The data dir
  is `/var/lib/mosfet/data`, and the JSONL telemetry files are written next to it in the working directory,
  where they grow until something rotates them.
* `/run/mosfet` holds the ready file. Mount a tmpfs there (`--tmpfs
  /run/mosfet:uid=10001,gid=10001,mode=0700`) so the file is gone on every restart. Without one, the
  Runtime still removes a ready file that a killed run left behind.
* `PYTHONDONTWRITEBYTECODE=1` and `PYTHONUNBUFFERED=1`; the bytecode is compiled at build time.

The image fixes the options the layout depends on as `ENV MOSFET_READY_FILE=/run/mosfet/ready` and
`MOSFET_DATA_DIR=/var/lib/mosfet/data`, and its entrypoint is `python -m` in exec form, so the bot's own
Python is the process Docker signals:

```
/opt/mosfet/venv/bin/python -m
```

The command supplies the bot's package and any other flag (`docker run IMAGE minimal_bot_example
--stop-grace 6`); the default command is `minimal_bot_example`.
`HEALTHCHECK` runs `test -f /run/mosfet/ready` every 10 s with a 130 s start period, the default
`--start-timeout` plus margin. A container therefore turns healthy only once its Runtime is `running/ready`,
and unhealthy after three failed checks once it leaves.

Run it with `--init` (compose `init: true`) so Docker's bundled init is PID 1 and reaps any orphaned
grandchild a provider leaves; the bot's Python still receives the forwarded SIGTERM. `docker stop` sends SIGTERM,
and the process exits within `--stop-grace` plus 1.5 s, 9.5 s by default, inside Docker's default 10 s
(compose `stop_grace_period: 10s`). Raise `MOSFET_STOP_GRACE` only together with `docker run --stop-timeout`
or `stop_grace_period`.

```bash
docker run --init --read-only --tmpfs /run/mosfet:uid=10001,gid=10001,mode=0700 --tmpfs /tmp \
  -v minimal-data:/var/lib/mosfet --env OPENAI_BASE_URL=... --env OPENAI_MODEL=... \
  --env DECISION_BASE_URL=http://192.168.2.32:8102 \
  -v "$PWD/openai_api_key:/run/secrets/openai_api_key:ro" minimal-bot
```

Secrets come from `/run/secrets`, which `Process.secrets` reads first; the image holds none. A bind-mounted
secret keeps its host owner and mode, so the host file must be readable by uid `10001`. `mlx-audio`
and `mlx-vlm` are macOS-only, so a containerized bot must choose providers that are not mlx-based.
`scripts/docker_smoke.sh` builds the minimal_bot image and checks this contract: health, uid, the read-only
root, the data dir mode, the stop bound, exit codes `1` (no secret) and `64` (a bad flag), and the stale
ready file.

## Bot supervision (`runtime.Supervisor`)

```
starting     attach the bot (operation id <name>:<serial>)  -> running | recovering | stopping
running      report running; after stable_after the failure count resets
recovering   detaching -> releasing -> retry (choice)       -> backoff | exhausted
backoff      after the doubling delay                        -> starting
exhausted    report exhausted
stopping     detaching -> releasing                          -> stopped (report graceful | timed_out)
```

* Every request to the bot carries a fresh operation id, and only the bot's terminal with that id moves the
  supervisor. A late terminal from an earlier request is ignored.
* Recovery deactivates the bot and waits for its answer (bounded by `bot_stop_timeout`) before releasing
  it (bounded by `bot_release_timeout`), so a bot's own cleanup is not cut short. A release that times out
  or raises exhausts the budget, a raising one at once. A stop request during recovery waits for it and is
  then handled at once.
* A stop is bounded by `bot_stop_timeout + bot_release_timeout`. A bot that does not answer is
  force-released and reported `timed_out`; a bot whose release raises is reported `timed_out` at once.
* The restart delay doubles from `backoff_initial` and stops doubling at `backoff_max`, so every budget the
  policy allows reaches exhaustion.

The bot's own deactivation bound defaults to 5 minutes and its attachment bound to 30 seconds, both far
above the default `bot_stop_timeout`. With those defaults a bot whose teardown hangs is force-released by
its supervisor before the bot's own timers fire. A bot file that wants the bot's own timer path (its
`bot.deactivated` or `bot.degraded` answer) to win gives it `bot.timeouts(deactivation=...)` below
`bot_stop_timeout`.

## Telemetry

Meter scope `bot.runtime`. Every attribute is a closed set or a bot's name (at most 32 per process), so
cardinality stays bounded; operation ids, secrets and messages are never attributes.

| Instrument | Attributes |
|---|---|
| `bot.runtime.bot.start.count` | `bot.name`, `outcome` (`activated`, `activation_failed`, `start_raised`, `start_timeout`, `degraded`) |
| `bot.runtime.bot.restart.count` | `bot.name`, `cause` (`activation_failed`, `start_raised`, `start_timeout`, `degraded`, `release_failed`, `release_timeout`) |
| `bot.runtime.bot.backoff.duration` | `bot.name` |
| `bot.runtime.bot.exhausted.count` | `bot.name` |
| `bot.runtime.bot.stop.count` | `bot.name`, `outcome` (`graceful`, `timed_out`) |
| `bot.runtime.bot.running` | none |
| `bot.runtime.signal.count` | `signal` (`SIGTERM`, `SIGINT`) |
| `bot.runtime.stop.duration` | `outcome` (`graceful`, `forced`) |

The machines read no clock of their own: every bound is an `hsm.after` timer on the machine's injected
`hsm` clock, so a test that injects a clock decides when each bound expires. The wall-clock
`bot.runtime.stop.duration` is therefore recorded by `bot.run`, the process boundary: the time from the
first stop signal to the Runtime's exit, `forced` when the Runtime exits `forced`. A stop that no signal
started (exhaustion, bring-up failure) is not timed.

Spans: `runtime.boot` (`bot.run` building the bots from their models; `constructor_raised` or
`invalid_models` on failure), `runtime.supervisor.attempt` (one bot attach request: `bot.name`,
`bot.runtime.attempt`), `runtime.stop` (stopping the supervisors and actors: `bot.runtime.stop.reason`,
`process.exit.code`). Logs name the failure kind, bot name and counts only.

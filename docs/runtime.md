# Process runtime (`mosfet.runtime`)

`mosfet.runtime` runs one process's bots: it builds them from a factory, keeps each one running within a
restart budget, and stops the whole process gracefully within one bound. Two decomposed state machines do
the work: `runtime.Runtime` owns the process lifecycle, and one `runtime.Supervisor` per bot owns that bot.
All waiting is bounded by `hsm.after`, and all asynchronous work runs in activities that end in typed
events.

## The factory and what it returns

A process runs a factory `build(config: runtime.Config) -> runtime.Assembly`, named by a
`runtime.Target` (`module.path:attribute`). The module is imported from the installed environment;
nothing is added to `sys.path`. The runtime resolves the target and calls the factory off the event loop.

`runtime.Config` is the frozen process configuration, built once at the process boundary:

| Field | Meaning |
|---|---|
| `name` | Service name; also the OTEL `service.name` |
| `instance_id` | Process or container identity; also the OTEL `service.instance.id` |
| `data_dir` | Directory for durable state (for example the SQLite memory store) |
| `environ` | Immutable snapshot of the process environment; left out of `repr` |
| `secrets` | `runtime.Secrets` resolver |

`Secrets.get(name)` returns a `pydantic.SecretStr` from the first source that has it: the file
`<secrets dir>/<name lowercased>` (the Docker `/run/secrets` convention), then the file named by the
environment variable `<NAME>_FILE`, then the environment variable `<NAME>`. A missing secret raises
`runtime.MissingSecretError`, whose message names the secret and never a value.

`runtime.Assembly` is the factory's result:

* `bots`: 1 to 32 bots under author-chosen names (lower snake case, at most 64 characters). One bot
  instance may appear under one name only.
* `actors`: shared root-owned actors such as a websocket server or an embedding model. Each actor's class
  must declare `model` as an `hsm.Model`; the runtime starts each one with exactly that model.
* `placements`: where a named bot joins the process environment.

A violation raises `runtime.BootError` with kind `invalid_assembly`.

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

The process boundary creates one `mosfet.Environment`, passes it to the `Runtime`, and starts it with
`mosfet.started(environment, runtime, Runtime.model)`. The Runtime's own context is therefore a
non-private child of that environment.

```
booting               build the assembly            -> starting_actors | failed (78) | stopped (0, on stop)
starting_actors       start each actor in order      -> starting_supervisors | stopping
starting_supervisors  start one Supervisor per bot   -> running | stopping
running               settling -> ready (all bots reported running; the ready file exists only here)
stopping              one stop_grace bound           -> stopped (0, 70 or 78) | forced (1)
  draining (choice)   any supervisor recorded?       -> stopping_supervisors | stopping_actors
  stopping_supervisors  ask each to stop, await each answer or its unreachable outcome
  stopping_actors       stop every supervisor (idempotent), then the actors in reverse order
```

* Actors start with `mosfet.started(runtime context, actor, type(actor).model)`; supervisors start in the
  same context, and every bot attaches to the same environment.
* A stop request during boot or bring-up leaves at once; nothing is deferred. Everything bring-up recorded
  is stopped.
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

### Exit codes

| Code | Reason | When |
|---|---|---|
| `0` | `stopped` | Graceful stop within the grace (signal or `runtime.StopEvent`), from any state. A bot force-released inside the grace still counts as graceful. |
| `1` | `forced` | The grace expired, or a second **signal** arrived while stopping. A programmatic stop request never forces. |
| `70` | `exhausted` | A bot used up its restart budget. Kept when a first signal arrives during that stop. |
| `78` | `boot_failed` | Import failed, factory missing or raised, invalid assembly, missing secret, or actor or supervisor bring-up failed or timed out. Kept when a first signal arrives during that stop. |

`64` (usage error) belongs to the command-line boundary, which rejects a malformed target before a
Runtime exists.

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
its supervisor before the bot's own timers fire. A factory that wants the bot's own timer path (its
`bot.deactivated` or `bot.degraded` answer) to win should construct bots with a `deactivation_timeout`
below `bot_stop_timeout`.

## Telemetry

Meter scope `bot.runtime`. Every attribute is a closed set or an assembly bot name (at most 32 per
process), so cardinality stays bounded; operation ids, secrets and messages are never attributes.

| Instrument | Attributes |
|---|---|
| `bot.runtime.bot.start.count` | `bot.name`, `outcome` (`activated`, `activation_failed`, `start_raised`, `start_timeout`, `degraded`) |
| `bot.runtime.bot.restart.count` | `bot.name`, `cause` (`activation_failed`, `start_raised`, `start_timeout`, `degraded`, `release_failed`, `release_timeout`) |
| `bot.runtime.bot.backoff.duration` | `bot.name` |
| `bot.runtime.bot.exhausted.count` | `bot.name` |
| `bot.runtime.bot.stop.count` | `bot.name`, `outcome` (`graceful`, `timed_out`) |
| `bot.runtime.bot.running` | none |
| `bot.runtime.signal.count` | `signal` (`SIGTERM`, `SIGINT`) |

The machines read no clock of their own: every bound is an `hsm.after` timer on the machine's injected
`hsm` clock, so a test that injects a clock decides when each bound expires.

Spans: `runtime.boot` (failure kind on failure), `runtime.supervisor.attempt` (one bot attach request:
`bot.name`, `bot.runtime.attempt`), `runtime.stop` (stopping the supervisors and actors:
`bot.runtime.stop.reason`, `process.exit.code`). Logs name the failure kind, bot name and target only.

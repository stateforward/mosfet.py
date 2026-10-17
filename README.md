# stateforward.mosfet

`stateforward.mosfet` is an event-driven Python runtime for software robots: bots that
perceive an environment, decide what to do, and act. Lifecycle, coordination, retries, and
timeouts are modeled as hierarchical state machines rather than written as ad-hoc async
control flow.

This is the Python implementation. It belongs to
[stateforward/mosfet](https://github.com/stateforward/mosfet).

Install:

```bash
pip install stateforward.mosfet
```

Import:

```python
import mosfet
```

## The shape of a bot

A bot is a **body** and a **cognition**.

The body owns lifetime, environment-facing I/O, and stimulus fan-out. It hands cognition an
explicit turn. It does not interpret stimuli or decide what matters. Cognition owns
interpretation, behavior selection, and dispatch. Nothing sits in between deciding on the
bot's behalf.

```python
import mosfet
from mosfet.abilities import cognition, listening, speaking


class Assistant(mosfet.Bot):
    """Hears, thinks, answers. Devices attach at the body; cognition selects."""

    _listening: listening.Listening
    _speaking: speaking.Speaking

    def __init__(self, *, cognition: cognition.Cognition) -> None:
        super().__init__()
        ...
```

`Bot` is concrete: it is constructed with the devices it attaches, its cognition, and the
abilities it composes, so a subclass is optional. See [`examples/`](examples/README.md) for complete programs, including a phone bot
that answers a real call over LiveKit.

## Pieces

- **Abilities** are the composition unit: typed input, output, and failure events, with
  optional nested abilities. Hearing, listening, speaking, vision, memory, learning.
- **Devices** are environment-facing and bot-agnostic. A phone rings; it does not decide
  that ringing matters.
- **Providers** own transport and SDKs. Core sees provider-neutral IDs, payloads, and
  failure kinds. Eleven ship here, from LiveKit to local MLX audio.
- **Behaviors** are learned event-only programs, compiled from Starlark and stored, so a
  rule the bot was taught costs nothing to run again. A behavior lives for one turn, or is a
  persistent **routine** that keeps running on `hsm.every` / `hsm.at` schedules across turns
  and restarts; each tick it emits is an observation the bot decides on in a new turn.

Every stateful concern is modeled with
[`stateforward-hsm`](https://github.com/stateforward/hsm.py).

## Defining and running a bot

A bot file is a plain Python file. It defines the bot with `bot.define` and runs it from its
`__main__` with `bot.run`, the way an hsm program defines and starts its machines.
[`examples/builder_bot`](examples/builder_bot/README.md) is the first example: one bot, for one
person, reached through a computer, that builds and tests software on a sandbox bench of its own:

```python
from mosfet import bot
from mosfet.abilities import sandbox
from mosfet.devices import computer


def builder(*, processor, server, host, capacity, spec, peer, person_id, traits, store, clock) -> bot.Model:
    return bot.define(
        "builder",
        bot.traits(*traits),
        bot.cognition(processor, memory=store, associative=True, consolidation=True, autonomy=True, routines=clock),
        bot.attach(computer.Computer(id="desk", transport=server, peer_id=peer, person_id=person_id)),
        bot.attach(sandbox.Sandbox(host=host, capacity=capacity, spec=spec, max_output_chars=MAX_OUTPUT_CHARS)),
    )
```

Its `__main__` reads its settings and secrets, builds the parts, and runs the bot with the
websocket server its computer is served by as a shared actor:

```python
if __name__ == "__main__":
    process = bot.process()
    settings = Settings.model_validate(os.environ)
    ...
    bot.run(builder(processor=..., server=server, ...), actors=(server,), process=process)
```

```bash
uv sync --project examples/builder_bot
examples/builder_bot/.venv/bin/python -m builder_bot_example --ready-file /tmp/ready
```

A mistake in a bot file comes back the way rustc reports one, with the span, why, and the
rewrite, so the builder bot, which writes bots in a DSL outside any model's training data, can fix
its own:

```text
error[B0003]: device id `desk` is used twice
  --> greeter.py:15:5
   |
14 |     bot.attach(computer.Computer(id="desk")),
   |     ---------------------------------------- first used here
15 |     bot.attach(phone.Phone(id="desk")),
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ used again here
   |
help: give the second device its own id
   = explain: bot.explain("B0003")
```

[`docs/bot_diagnostics.md`](docs/bot_diagnostics.md) explains every code. `bot.run` stops
gracefully on SIGTERM or SIGINT, keeps stdout empty, and exits `0` (graceful stop), `1` (forced
stop), `64` (a bad process option), `70` (a bot used up its restart budget) or `78` (the process
could not boot). [`examples/minimal_bot`](examples/minimal_bot/README.md) is the smallest bot, one
device-free bot on any OpenAI-compatible endpoint:

```bash
uv sync --project examples/minimal_bot
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 OPENAI_MODEL=... OPENAI_API_KEY=... \
  examples/minimal_bot/.venv/bin/python -m minimal_bot_example --ready-file /tmp/ready
```

[`docs/runtime.md`](docs/runtime.md) has every `bot.process` option, the stop bound and the
restart policy. [Running in a container](docs/runtime.md#running-in-a-container) covers the one
image recipe, `docker/Dockerfile`, that runs any bot project with `python -m`.

## Package name, event names

The import package is `mosfet`. The events are `bot.*`, because they name what a bot did,
not what library emitted them:

```python
import mosfet

mosfet.InputEvent    # canonical event name: bot.input
```

That split is deliberate and load-bearing. Event names are the wire contract.

## Development

[`docs/development.md`](docs/development.md) has the test, lint, and type-check commands.
[`AGENTS.md`](AGENTS.md) is the engineering contract this repository is held to; it is
worth reading before a first change.

## License

MIT. See [LICENSE](LICENSE).

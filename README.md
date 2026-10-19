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
from mosfet import bot
from mosfet.devices import computer
from mosfet.providers import openai_compat, typesafe

# Reached at its desk; what it does about what is typed there is its own.
greeter = bot.define(
    "greeter",
    bot.traits("warm", "curious"),
    bot.cognition(
        openai_compat.Processor(client=openai_compat.ChatClient(model="qwen3-coder")),
        # Intuition is never a chat model: it decides on TypeSafe, hosted or a local system_one server.
        intuition=typesafe.Processor(base_url="http://127.0.0.1:8102"),
    ),
    bot.attach(computer.Computer(id="desk")),
)
```

`bot.define` returns the bot as data. `bot.start` (or `bot.run`, for a process) builds its body,
`mosfet.Bot`, around the very parts the file constructed: the devices it holds, the abilities it
composes, and one cognition, whose intuition decides on a decision model while it reasons on a language
model. The hsm states and transitions given beside them are the bot's
firmware, the machinery its owner wrote for it. [`docs/bot.md`](docs/bot.md) is the guide to the
DSL. See [`examples/`](examples/README.md) for complete programs, including a phone bot that answers
a real call over LiveKit.

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
if __name__ == "__main__":
    process = bot.process()
    settings = Settings.model_validate(os.environ)
    server = websocket.Server(verifier=websocket.HmacTokens(process.secrets.get("computer_token_secret")))
    desk = environment.Object(id="desk")
    builder = bot.define(
        "builder",
        bot.traits(*settings.traits.traits),
        bot.cognition(
            openai_compat.Processor(client=openai_compat.ChatClient(model=settings.model, base_url=settings.base_url)),
            intuition=typesafe.Processor(base_url=settings.typesafe_base_url, api_key=...),
            memory=memory.Memory(database=str(process.data_dir / "memory.sqlite3")),
            associative=True,
            consolidation=True,
            autonomy=True,
            routines=functools.partial(datetime.datetime.now, datetime.UTC),
        ),
        bot.attach(
            computer.Computer(id="computer", transport=server, peer_id=settings.peer, placement=desk.surface())
        ),
        bot.attach(
            sandbox.Sandbox(
                host=docker.Host(socket=settings.socket, images=frozenset({settings.image}), owner=process.name),
                capacity=docker.capacity(max_resources=sandbox.Resources()),
                spec=sandbox.Spec(image=settings.image),
            )
        ),
    )
    bot.run(builder, actors=(server,), process=process)
```

Its `__main__` reads its settings and secrets, then has one `bot.define` that constructs every part
inline. `bot.define` is the only function that defines a bot. Two variables come first: the websocket
server its computer is served by, because `bot.run` also starts it as a shared actor, and the desk its
computer sits on. The desk is an `environment.Object`: it does nothing and is never given to the bot,
and the bot sees it only as its computer's `on="desk"`.

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

[`docs/bot_diagnostics.md`](docs/bot_diagnostics.md) explains every code, and
[`docs/bot.md`](docs/bot.md) covers every element, starting and testing a bot, its firmware and its
diagnostics, with examples that run. `bot.run` stops
gracefully on SIGTERM or SIGINT, keeps stdout empty, and exits `0` (graceful stop), `1` (forced
stop), `64` (a bad process option), `70` (a bot used up its restart budget) or `78` (the process
could not boot). [`examples/minimal_bot`](examples/minimal_bot/README.md) is the smallest bot, one
device-free bot on any OpenAI-compatible endpoint:

```bash
uv sync --project examples/minimal_bot
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 OPENAI_MODEL=... OPENAI_API_KEY=... TYPESAFE_API_KEY=... \
  examples/minimal_bot/.venv/bin/python -m minimal_bot_example --ready-file /tmp/ready
```

[`examples/willy`](examples/willy/README.md) is one bot for a whole house, on local models: a computer in
each room, and the capacity to learn who people are, under the owner's household:

```bash
uv sync --project examples/willy
set -a; . examples/willy/.env; set +a
examples/willy/.venv/bin/python -m willy_example --secrets-dir examples/willy/secrets
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

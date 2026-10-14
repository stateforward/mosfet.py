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

`Bot` is abstract: a concrete bot declares the abilities it composes and the devices it
attaches. See [`examples/`](examples/README.md) for complete programs, including a phone bot
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

## Running a bot

`mosfet run` runs a bot factory as one supervised process. The factory takes the process
configuration and returns the bots to run:

```python
# my_bot/app.py
from mosfet import runtime

from my_bot.assistant import Assistant  # the bot class above


def build(config: runtime.Config) -> runtime.Assembly:
    """Build the bots this process runs. ``config`` carries the data dir and secrets."""

    return runtime.Assembly(bots={"assistant": Assistant(cognition=...)})
```

```bash
mosfet run my_bot.app:build --ready-file /tmp/ready
```

The module is imported from the installed environment. The process stops gracefully on SIGTERM
or SIGINT, keeps stdout empty, and exits `0` (graceful stop), `1` (forced stop), `64` (usage
error), `70` (a bot used up its restart budget) or `78` (boot or configuration failure).
[`examples/minimal_bot`](examples/minimal_bot/README.md) is the smallest runnable factory, one
device-free bot on any OpenAI-compatible endpoint:

```bash
uv sync --project examples/minimal_bot
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_MODEL=gpt-5.4-mini OPENAI_API_KEY=... \
  examples/minimal_bot/.venv/bin/mosfet run minimal_bot_example:build --ready-file /tmp/ready
```

[`docs/runtime.md`](docs/runtime.md) has every flag, the stop bound and the restart policy.

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

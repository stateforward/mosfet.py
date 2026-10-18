# Defining a bot (`mosfet.bot`)

A bot file is a plain Python file. It defines its bot with `bot.define`, the way an hsm program
defines a machine with `hsm.define`, and starts it with `bot.start` or runs it as a process with
`bot.run`. There is no subclass to write, no factory and no configuration layer: the file constructs
the bot's parts itself, with literal arguments and the collaborators it injects, and hands them to
`bot.define`.

```python
from mosfet import bot
```

Every example on this page runs: `tests/bot/dsl/test_docs.py` runs them as doctests, with the module
docstrings of `mosfet.bot`.

## A bot file

`bot.define(name, *elements)` returns a `bot.Model`: one bot, as data. Defining constructs, starts,
opens and reads nothing; the model holds the very instances the file built.

```pycon
>>> from mosfet import bot
>>> from mosfet.bot import testing
>>> from mosfet.devices import computer
>>> greeter = bot.define(
...     "greeter",
...     bot.traits("warm", "curious"),
...     bot.cognition(testing.Scripted()),
...     bot.attach(computer.Computer(id="desk")),
... )
>>> greeter.qualified_name
'/greeter'

```

`testing.Scripted()` stands in for a model here; a bot file passes a real one, such as
`openai_compat.Processor(client=openai_compat.ChatClient(model=...))`. `bot.define` is the only
function that defines a bot: a bot file writes one `bot.define(...)` and constructs every part inline
in it, from literals and the settings and secrets it read. A bot that needs values only its process
has (a secret, the data dir) defines itself under `if __name__ == "__main__":`, after reading them. A
bot's tests write their own small `bot.define`, with `testing.Scripted` and fake parts.

`bot.describe(*models)` is the JSON view of defined bots, read from their models alone. It names
what each bot is made of, and never a value given to a part:

```pycon
>>> print(bot.describe(greeter).model_dump_json(indent=2))
{
  "bots": [
    {
      "name": "greeter",
      "traits": [
        "warm",
        "curious"
      ],
      "faculties": [
        "intuition",
        "reasoning",
        "reflection"
      ],
      "memory": null,
      "parts": [
        {
          "id": "desk",
          "kind": "device",
          "roles": [
            "attach"
          ],
          "constructor": "Computer"
        }
      ],
      "sensory_register": null,
      "timeouts": [],
      "allowed": [],
      "firmware": null
    }
  ]
}

```

`bot.redefine(model, *elements)` is a new model from the old one's elements and more, as
`hsm.redefine` is. The old model is unchanged.

## The elements

| Element | What it gives the bot |
|---|---|
| `bot.traits(*traits)` | The owner's fixed seed of the bot's nature: short descriptors such as `"curious"`, never instructions. With traits, the bot has a persona that it develops itself. At most one. |
| `bot.cognition(processor, /, *, memory=None, associative=False, consolidation=False, autonomy=False, routines=None)` | Its mind: intuition, reasoning and reflection on `processor`, plus each faculty switched on, all on one memory (a fresh in-memory `memory.Memory()` when none is given). `routines` is the wall clock routines keep time on. Exactly one. |
| `bot.attach(instance)` | A device it holds, keyed by its `id`, or an ability it acquires. |
| `bot.input(ability)` | A sense organ: the body fans every environment stimulus out to its input abilities, in parallel. |
| `bot.output(ability)` | An effector port, off the stimulus fan-out. |
| `bot.register(store)` | The body's sensory register (a `memory.StmMemory`), recording every admitted stimulus. At most one. |
| `bot.timeouts(*, processing=None, deactivation=None, attachment=None)` | The body's own bounds, each positive; an omitted one keeps the body's default. At most one. |
| `bot.allow(*lints)` | Lints allowed for this bot (see [Diagnostics](#diagnostics)). |
| `bot.state`, `bot.transition`, `bot.on`, `bot.after`, ... | The hsm element functions, re-exported as the same objects: the bot's [firmware](#the-bots-firmware). |

Each element remembers where it was written, so a mistake points at the bot file's own line.

**Instances, not constructors.** `bot.attach(computer.Computer(id="desk"))` takes a constructed,
unstarted instance. Construction is side-effect free everywhere in mosfet: no file, socket, thread or
model is opened until the bot starts. One instance may be handed to another part's constructor as
well as attached (`speaking.Speaking(encoder=..., listening=(hearing,))`); that is plain dependency
injection.

**Ids.** The bot knows each part by its `id`: on its tool menu, as its focus, and in what it owns.
A held device without one is known by its kind (`phone.Phone()` is `phone`), and an ability by its
class in snake case (`sandbox.Sandbox()` is `sandbox`), so two parts of one kind need ids of their
own. `bot`, `cognition` and `firmware` are the body's own.

**Defaults.** The common case needs only the one essential collaborator. `computer.Computer(id="desk")`
with no transport is a computer with no remote side, unplugged. `websocket.Server` listens on
`127.0.0.1:8765` and needs only its `verifier`. `openai_compat.ChatClient` talks to
`http://127.0.0.1:8080/v1` and needs only its `model`. `speaking.Speaking` speaks through a fresh
`audio.Speaker()` and needs only its `encoder`. `docker.Host` derives its capacity from its runtime,
and `sandbox.Spec` gives `HOME` the workspace.

## Objects and placements

A desk, a table or a shelf is an `environment.Object`: a physical thing with an id and a place that
does nothing. It has no power, no state and no machine, so it is never given to the bot. A part sits
on it through its own `placement=`, and `surface(offset)` is where a thing set on the object is:

```pycon
>>> from mosfet import environment
>>> from mosfet.environment import space
>>> desk = environment.Object(id="desk", placement=space.Placement(position=space.Position(2.0, 1.0)))
>>> desk.surface(space.Position(0.5, 0.0))
Placement(position=Position(x=2.5, y=1.0, z=0.0), threshold_db=None, on='desk')
>>> keeper = bot.define(
...     "keeper",
...     bot.cognition(testing.Scripted()),
...     bot.attach(computer.Computer(id="computer", placement=desk.surface())),
... )
>>> [part.id for part in bot.describe(keeper).bots[0].parts]
['computer']

```

The bot's world block shows the desk only through what sits on it: the computer's own element reads
`<device ref="computer" id="…" state="…" on="desk">`. Giving the object itself to `bot.attach`,
`bot.input` or `bot.output` is [`B0019`](bot_diagnostics.md#b0019), whose help deletes that element.

## Starting, stopping and testing a bot

`bot.start(ctx, model, data=None, *, observe=())` follows `hsm.start(ctx, instance, data)`. It builds the bot's body,
`mosfet.Bot`, around the model's parts, with one cognition and, given traits, a persona, starts it in
the environment `ctx`, and resolves once the body has answered its activation. The body runs under the
bot's name. `bot.stop(instance)` deactivates it, takes it out of the environment, and stops it and
every part it started, so the model can be started again:

```pycon
>>> import asyncio
>>> from mosfet.environment import Environment
>>> async def lifetime() -> tuple[str, str]:
...     started = await bot.start(Environment(), greeter)
...     running = started.state()
...     await bot.stop(started)
...     return running, started.state()
>>> asyncio.run(lifetime())
('/greeter/active/unfocused', '')

```

A model holds the instances its file built, so it runs one bot at a time: starting it again while
it runs is `B0101`.

**Observers.** Whoever starts a bot may watch it, for logs, harness records and summaries:
`observe=` on `bot.start`, `bot.run` and `testing.running` takes `bot.Observer`s, made by
`bot.observe`, which is `hsm.observe` itself (`bot.observe(callback)` watches everything,
`bot.observe(event, ..., callback)` those events). The body runs on its model redefined with them, so
they watch that bot only, and only for that start (across its restarts, under `bot.run`). An observer
watches and never acts on the bot: it never dispatches into the bot, its devices or its cognition.
It is not part of the bot's definition, so given to `bot.define` it is `B0015`.

```pycon
>>> import typing
>>> import hsm
>>> import mosfet
>>> seen: list[str] = []
>>> def record(ctx: hsm.Context, instance: hsm.Instance, observed: hsm.Event[typing.Any]) -> None:
...     seen.append(typing.cast(dict[str, hsm.Event[typing.Any]], observed.data)["event"].name)
>>> async def watched() -> None:
...     started = await bot.start(Environment(), greeter, observe=(bot.observe(mosfet.ActivateEvent, record),))
...     await bot.stop(started)
>>> asyncio.run(watched())
>>> seen
['bot.activate']

```

`mosfet.bot.testing` is what a bot's tests need, and nothing that decides for the bot:

* `testing.Scripted(*choices)` is a model whose choices a test writes down. It records every call,
  makes each `testing.Choice(stimulus=, event=, data=)` once, only on a turn woken by that stimulus
  that offers that event, and otherwise selects nothing. `calls()`, `offered_names()`,
  `woken_by(name)` and `reflected()` read what it was given.
* `testing.running(model, *, actors=(), clock=None, observe=())` starts the bot for an `async with`
  block, after the shared actors it needs, in an environment keeping time on `clock` when one is given
  (an hsm test clock brings a bot's waits early) and watched by the `observe` observers, and stops the
  bot and then the actors when the block ends, even when it raised.
* `testing.snapshot(instance)` is the running body's own snapshot, for assertions only.

```pycon
>>> async def snapshot() -> str:
...     async with testing.running(greeter) as running:
...         return testing.snapshot(running.bot).State
>>> asyncio.run(snapshot())
'/greeter/active/unfocused'

```

A test pins what the bot was offered and what woke it. It never pins what the bot chooses: a bot that
does nothing about a stimulus has made a choice it is entitled to.

## The bot's firmware

A bot's mind is its cognition. Its firmware is the machinery its owner wrote for it, the way a
phone's call handling is the phone's: a timer, a multi-step device procedure. The hsm element
functions given at the top level of `bot.define`, mixed freely with the `bot.*` elements, compile
into the bot's **one** firmware: a machine of its own, `<name>_firmware`, separate from the body's.
A bot given no hsm elements has no firmware.

A kitchen timer, where the firmware is the timer and cognition decides whether to set it and what to
do when it rings:

```pycon
>>> import datetime
>>> import typing
>>> import hsm
>>> import pydantic
>>> from mosfet import event
>>> class SetData(pydantic.BaseModel):
...     """Start the kitchen timer."""
...
...     model_config = pydantic.ConfigDict(frozen=True)
...     minutes: int = pydantic.Field(ge=1, le=240, description="Minutes until the timer rings.", examples=[10])
>>> class RangData(pydantic.BaseModel):
...     """The timer ran out."""
...
...     model_config = pydantic.ConfigDict(frozen=True)
...     minutes: int = pydantic.Field(description="How long the timer ran.", examples=[10])
>>> SetEvent = hsm.Event[SetData](name="timer.set", kind=event.EventKind, schema=SetData)
>>> RangEvent = hsm.Event[RangData](name="timer.rang", schema=RangData)
>>> def remember(ctx: hsm.Context, instance: bot.Firmware, set_: hsm.Event[SetData]) -> None:
...     _ = instance.set("minutes", set_.data.minutes)
>>> def minutes(instance: bot.Firmware) -> int:
...     value, _ = instance.get("minutes")
...     return typing.cast(int, value)
>>> def duration(ctx: hsm.Context, instance: bot.Firmware, trigger: hsm.Event[typing.Any]) -> datetime.timedelta:
...     return datetime.timedelta(minutes=minutes(instance))
>>> def ring(ctx: hsm.Context, instance: bot.Firmware, trigger: hsm.Event[typing.Any]) -> None:
...     bot.report(ctx, instance, RangEvent.with_data(RangData(minutes=minutes(instance))), trigger=trigger)
>>> timer = bot.define(
...     "timer",
...     bot.traits("attentive"),
...     bot.cognition(testing.Scripted()),
...     bot.attribute("minutes", int, 0),
...     bot.initial(bot.target("idle")),
...     bot.state("idle", bot.transition(bot.on(SetEvent), bot.effect(remember), bot.target("../running"))),
...     bot.state("running", bot.transition(bot.after(duration), bot.effect(ring), bot.target("../idle"))),
... )
>>> bot.describe(timer).bots[0].firmware
FirmwareOutline(commands=('timer.set',), states=('/timer_firmware/idle', '/timer_firmware/running'))

```

* **Shown like a device.** Cognition sees the firmware in its world block as one
  `<firmware ref="firmware" id="..." state="...">` element with its attributes, the way it sees a
  device, and nothing more:

  ```pycon
  >>> async def world() -> str:
  ...     async with testing.running(timer) as running:
  ...         return running.environment.model_snapshot(running.bot) or ""
  >>> shown = asyncio.run(world())
  >>> '<firmware ref="firmware"' in shown, 'state="/timer_firmware/idle"' in shown
  (True, True)

  ```

* **Commands are tools.** A tool event (`kind=event.EventKind`) the firmware reacts to is a command:
  cognition is offered it under the actor key `firmware` while a state that reacts to it is active,
  and a selection goes straight to the firmware. Every field of a command's schema needs a
  description (`B0012`), and examples (the lint `command_without_examples`). A command named like an
  event one of the bot's devices or abilities offers is `B0013`.
* **It acts on its own.** `bot.send(ctx, instance, to, event, trigger=)` sends a typed event to one of
  the bot's devices or abilities by id, in its trigger's trace, on the firmware's own initiative. An
  unknown id raises `LookupError` naming the ids the bot has, and a part that is not running drops it.
  The firmware never sends to the body, cognition, memory or another bot.
* **It reports.** `bot.report(ctx, instance, observation, trigger=)` hands an observation to the body,
  the way an ability hands it a product, and it wakes cognition. What cognition does about it is its
  own choice.
* **It perceives what is attached to the bot.** Once it starts, the firmware is a second holder of
  every device the bot holds, beside the body, so it gets what each device tells its holders as the
  device's own typed event: `bot.on(computer.TextEnteredEvent)` matches. An event its machine does
  not name is ignored.
* **Its lifetime is the bot's.** The body starts it once every device and ability is attached, stops
  it first, and starts it again, fresh, with the bot.

The line between firmware and the bot's choices: the firmware may drive its devices and abilities on
its own initiative by typed events, keep observable state (`bot.attribute`), run timers, retries and
timeouts (`bot.after`), report, and offer commands. It must not select on cognition's behalf among
what cognition is offered, fake a cognition selection, call into another machine directly, or carry
persona text, instructions or decisions in prose: those are the bot's to develop.

## Diagnostics

The DSL is new to every writer, person or model, so each mistake comes back the way rustc reports
one: a stable code, the span of the writer's own call with a caret under it, why, and a `help:` with
the rewrite in the DSL's real names. `bot.define` collects every mistake in one pass and raises them
together, in one `bot.DefinitionError`:

```pycon
>>> from mosfet.devices import phone
>>> try:
...     _ = bot.define(
...         "greeter",
...         bot.attach(computer.Computer(id="desk")),
...         bot.attach(phone.Phone(id="desk")),
...     )
... except bot.DefinitionError as error:
...     [found.code for found in error.diagnostics]
['B0002', 'B0003']
>>> bot.explain("B0003").splitlines()[0]
'error[B0003]: a device or ability id is used twice'

```

Here the bot has no cognition (`B0002`), and its second device reuses the id `desk` (`B0003`), in the
order they appear in the file.
`str(error)` is the rendered text, ending in each diagnostic's `= explain: bot.explain("<code>")`;
`bot.explain(code)` says what any code or lint means, why the DSL requires it, and a wrong and a
corrected example. `Diagnostic.model_dump_json()` is the same diagnostic as JSON. A diagnostic names
ids, classes and fields, never a value: every literal outside the diagnosed span is masked.

Lints are clippy-style warnings about a bot that defines, but probably not as meant, such as a bot
with nothing it can perceive. One pass finds every error and every warning: a bot file with errors
reports its lints beside them in the one `DefinitionError` (a lint about what the bot lacks waits
until every part defines), and one without emits each lint as a `bot.LintWarning`. `bot.check(*models)`
returns them, and `bot.allow` allows one for a bot when it is deliberate:

```pycon
>>> import warnings
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore", bot.LintWarning)
...     lonely = bot.define("lonely", bot.cognition(testing.Scripted()))
>>> [found.code for found in bot.check(lonely)]
['no_devices']
>>> quiet = bot.define("lonely", bot.cognition(testing.Scripted()), bot.allow("no_devices"))
>>> bot.check(quiet)
()

```

`bot.start` and `bot.run` report the same way, with `B01xx` codes. [`bot_diagnostics.md`](bot_diagnostics.md)
lists every code and lint with its explanation; it is generated from the same table as `bot.explain`.

## Running a bot

A bot file runs its bots as one process from its `__main__`. `bot.process()` reads the process's
options once (the `MOSFET_*` variables, then the flags after the module), and `bot.run` runs and
supervises the bots until the process is told to stop, then exits with its code:

```python
if __name__ == "__main__":
    process = bot.process()
    settings = Settings.model_validate(os.environ)
    server = websocket.Server(verifier=websocket.HmacTokens(process.secrets.get("computer_token_secret")))
    greeter = bot.define(
        "greeter",
        bot.cognition(openai_compat.Processor(client=openai_compat.ChatClient(model=settings.model))),
        bot.attach(computer.Computer(id="desk", transport=server)),
    )
    bot.run(greeter, actors=(server,), process=process)
```

The file reads its own settings and secrets with plain code at that boundary and constructs the parts
with them inline, in its one `bot.define`. `actors` are machines several bots or parts share, such as
the websocket server a computer is served by: the process starts them before the bots and stops them
after, so a shared actor is the one part the file names as a variable, to give it to both. [`runtime.md`](runtime.md) has every
option, the exit codes, the stop bound, the restart policy and the container recipe that runs any bot
project with `python -m`. [`examples/builder_bot`](../examples/builder_bot/README.md) is the flagship
bot file, and [`examples/minimal_bot`](../examples/minimal_bot/README.md) the smallest.

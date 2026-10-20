# Bot diagnostics

<!-- Generated from mosfet.bot.diagnostics.EXPLANATIONS. Do not edit by hand; regenerate with
python -c 'from mosfet.bot import diagnostics; print(diagnostics.document(), end="")' > docs/bot_diagnostics.md -->

`bot.define` reports every mistake in a bot file at once, the way rustc does: each with a stable
code, the source span, notes on why, and a `help:` that shows the rewrite. Errors (`B00xx` from
`bot.define`, `B01xx` from `bot.start` and `bot.run`) stop the bot; lints (snake_case names) are
warnings. `bot.explain("<code>")` prints any entry below.

The examples assume `from mosfet import bot`, `from mosfet.devices import phone`,
`from mosfet.abilities import communication, memory, speaking`, `import hsm`, `import mosfet`,
`import datetime`, `import pydantic`, a `processor` (a `processing.Processor`), a `decider` (a
`processing.DecisionProcessor`), an `environment`, a `catalog` (a `vocal.voice.Catalog`) and `voices`
(an encoder factory `(key, description) -> encoder`).

- [`B0001`](#b0001) (error): the bot's name is missing or not lower snake case
- [`B0002`](#b0002) (error): the bot has no `bot.cognition`
- [`B0003`](#b0003) (error): a device or ability id is used twice
- [`B0004`](#b0004) (error): an element is given a class instead of an instance
- [`B0005`](#b0005) (error): an element is given an instance that is already started
- [`B0006`](#b0006) (error): an element is given something it does not take
- [`B0007`](#b0007) (error): a device or ability uses a reserved id
- [`B0008`](#b0008) (error): an element is written inside another element
- [`B0009`](#b0009) (error): an element a bot has once is given twice
- [`B0010`](#b0010) (error): the traits are not one to 24 short traits
- [`B0011`](#b0011) (error): a timeout is not a positive `datetime.timedelta`
- [`B0012`](#b0012) (error): a firmware command's field has no description
- [`B0013`](#b0013) (error): a firmware command has the name of an event an attached part offers
- [`B0014`](#b0014) (error): hsm rejects the bot's firmware
- [`B0015`](#b0015) (error): an observer is given to `bot.define`
- [`B0016`](#b0016) (error): a held device has no id
- [`B0017`](#b0017) (error): `bot.define` is given something that is not a bot element
- [`B0018`](#b0018) (error): `bot.allow` names a lint that does not exist
- [`B0019`](#b0019) (error): an object is given to the bot
- [`B0021`](#b0021) (error): `bot.cognition` is given seeds but no autonomy
- [`B0022`](#b0022) (error): `bot.voice` is given to a bot with no traits
- [`B0023`](#b0023) (error): `bot.voice` is given to a bot without exactly one `speaking.Speaking` in `bot.output`
- [`B0024`](#b0024) (error): `bot.cognition` has no intuition
- [`B0025`](#b0025) (error): `bot.cognition(intuition=)` is given a chat processor
- [`B0101`](#b0101) (error): a model is started again while its parts run
- [`B0102`](#b0102) (error): a part raised while `bot.start` built the bot
- [`B0103`](#b0103) (error): `bot.start` is given something that is not an `Environment`
- [`B0104`](#b0104) (error): two bots given to `bot.run` share a name
- [`B0105`](#b0105) (error): one model is given to `bot.run` twice
- [`B0106`](#b0106) (error): `bot.run` is given a shared actor that is not an actor
- [`B0107`](#b0107) (error): a process option is unknown or its value is bad
- [`ability_without_tools`](#ability_without_tools) (warning): an acquired ability offers cognition nothing to select
- [`command_without_examples`](#command_without_examples) (warning): a firmware command's field has no examples
- [`empty_timeouts`](#empty_timeouts) (warning): `bot.timeouts()` sets no timeout
- [`instruction_like_traits`](#instruction_like_traits) (warning): a trait reads as an instruction
- [`no_devices`](#no_devices) (warning): the bot can perceive nothing
- [`unreachable_state`](#unreachable_state) (warning): a firmware state is never entered
- [`unused_register`](#unused_register) (warning): a sensory register on a bot that perceives nothing

## B0001

**error:** the bot's name is missing or not lower snake case

`bot.define` takes the bot's name first: lower snake case (a letter, then letters, digits and underscores), at most 64 characters.

Why: The name is the bot's topology name and a telemetry attribute, so it is bounded and spelled one way. The suggestion is the name in snake case.

Wrong:

```python
model = bot.define("Greeter Bot", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

Right:

```python
model = bot.define("greeter_bot", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## B0002

**error:** the bot has no `bot.cognition`

Every bot has exactly one `bot.cognition`, the mind that perceives and decides.

Why: A body with no cognition can hold devices but can never choose anything to do with them. `bot.cognition(processor, intuition=decider)` thinks with a `processing.Processor` (`from mosfet.abilities import processing`), and its intuition decides with a `processing.DecisionProcessor`, such as `decision.Processor(...)` (`from mosfet.providers import decision`). In a test, the offline ones are `testing.Scripted()` and `testing.Decider()` (`from mosfet.bot import testing`), which make only the choices the test writes down.

Wrong:

```python
model = bot.define("greeter", bot.attach(phone.Phone()))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## B0003

**error:** a device or ability id is used twice

A bot keys its devices and the abilities it can address by `id`, so no two of them share one. A device or ability built without an `id` is known by its kind (`phone.Phone()` is `phone`), so two of one kind need ids of their own. The same instance given twice is also this error.

Why: The id is how the bot addresses a part: on its tool menu, as its focus, and in what it owns. Two parts with one id would be one name for two things.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
    bot.attach(phone.Phone()),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
    bot.attach(phone.Phone(id="handset")),
)
```

## B0004

**error:** an element is given a class instead of an instance

`bot.attach`, `bot.input`, `bot.output` and `bot.register` take a constructed instance: `bot.attach(phone.Phone(id="phone"))`, not `bot.attach(phone.Phone)`.

Why: The bot file builds its parts and passes in what they need; the bot holds the very instances it is given, so one may also be handed to another part's constructor. Constructing a part does nothing until the bot starts.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone(id="phone")))
```

## B0005

**error:** an element is given an instance that is already started

A part given to a bot is unstarted: the bot starts it, once, as part of its body.

Why: A started instance already belongs to whatever started it. Construct a new one for this bot; construction has no side effects, so that costs nothing.

Wrong:

```python
handset = await mosfet.started(environment, phone.Phone(), phone.Phone.model)
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(handset))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## B0006

**error:** an element is given something it does not take

`bot.attach` takes a device or an ability; `bot.input` and `bot.output` take an ability; `bot.register` takes a `memory.StmMemory`; `bot.cognition` takes a `processing.Processor` first, `intuition=` takes a `processing.DecisionProcessor`, `memory=` takes a `memory.Memory` and `seeds=` takes `behavior.Seed`s; `bot.voice` takes a `vocal.voice.Catalog` first, `designer=` takes a `vocal.voice.Designer`, `languages=` takes language tags and `allowed=` takes the keys of the voices of a source without a `designer=`. The help names the element that takes what was given.

Why: Each element is one part of the body: a device is held, a sense organ is an input, a processor is cognition's. Giving a part to the wrong element would wire it where it cannot work.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.input(phone.Phone()))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## B0007

**error:** a device or ability uses a reserved id

The ids `bot`, `cognition` and `firmware` are the body's own.

Why: Cognition addresses the body, its cognition and its firmware by those keys, so no part may take one.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone(id="bot")))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone(id="phone")))
```

## B0008

**error:** an element is written inside another element

Every part of a bot is an argument of `bot.define` itself. `bot.cognition` takes a processor and keyword flags, not `bot.attach` or another element.

Why: A bot is one flat list of parts; nesting would suggest an ownership the body does not have.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, bot.attach(phone.Phone()), intuition=decider))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## B0009

**error:** an element a bot has once is given twice

A bot has at most one `bot.traits`, `bot.cognition`, `bot.voice`, `bot.timeouts`, `bot.register` and `bot.allow`.

Why: Each is one thing about the bot. Merge the two into one, for example one `bot.traits` with every trait.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.traits("curious"),
    bot.traits("patient"),
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.traits("curious", "patient"),
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
)
```

## B0010

**error:** the traits are not one to 24 short traits

`bot.traits` takes one to 24 traits, each a word or a short phrase of at most 200 characters.

Why: Traits are the owner's fixed seed of who the bot is. The bot grows its own description on top of them, so they stay few and short.

Wrong:

```python
model = bot.define("greeter", bot.traits(), bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

Right:

```python
model = bot.define("greeter", bot.traits("curious"), bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## B0011

**error:** a timeout is not a positive `datetime.timedelta`

Each of `bot.timeouts(processing=, deactivation=, attachment=, step=)` is a positive `datetime.timedelta`.

Why: A timeout bounds how long the bot waits for something; zero or less would never wait at all.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
    bot.timeouts(processing=datetime.timedelta(0)),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
    bot.timeouts(processing=datetime.timedelta(seconds=30)),
)
```

## B0012

**error:** a firmware command's field has no description

The tool events the bot's firmware reacts to (`hsm.Event(..., kind=mosfet.event.EventKind)`) are its commands, offered to cognition as tools. Every field of a command's schema has a description: `pydantic.Field(description="...")`.

Why: The model choosing a tool reads its schema to know what each field means; a field without a description is a blank on the form it fills in.

Wrong:

```python
class SetData(pydantic.BaseModel):
    minutes: int


SetEvent = hsm.Event(name="timer.set", kind=mosfet.event.EventKind, schema=SetData)
model = bot.define(
    "timer",
    bot.cognition(processor, intuition=decider),
    bot.initial(bot.target("idle")),
    bot.state("idle", bot.transition(bot.on(SetEvent), bot.target("../idle"))),
)
```

Right:

```python
class SetData(pydantic.BaseModel):
    minutes: int = pydantic.Field(description="Minutes until the timer rings.", examples=[10])


SetEvent = hsm.Event(name="timer.set", kind=mosfet.event.EventKind, schema=SetData)
model = bot.define(
    "timer",
    bot.cognition(processor, intuition=decider),
    bot.initial(bot.target("idle")),
    bot.state("idle", bot.transition(bot.on(SetEvent), bot.target("../idle"))),
)
```

## B0013

**error:** a firmware command has the name of an event an attached part offers

Cognition is offered the firmware's commands under the `firmware` key, beside every tool its devices and abilities offer. A firmware command's name is none of theirs.

Why: Cognition selects a tool by its name. One name offered by the firmware and by a device would leave it unclear which of them a selection is for, so the firmware's commands are named for the bot.

Wrong:

```python
class DialData(pydantic.BaseModel):
    """Dial the saved number."""


DialEvent = hsm.Event(name="phone.dial", kind=mosfet.event.EventKind, schema=DialData)
model = bot.define(
    "dialer",
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
    bot.initial(bot.target("idle")),
    bot.state("idle", bot.transition(bot.on(DialEvent), bot.target("../idle"))),
)
```

Right:

```python
class DialData(pydantic.BaseModel):
    """Dial the saved number."""


DialEvent = hsm.Event(name="dialer.dial", kind=mosfet.event.EventKind, schema=DialData)
model = bot.define(
    "dialer",
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
    bot.initial(bot.target("idle")),
    bot.state("idle", bot.transition(bot.on(DialEvent), bot.target("../idle"))),
)
```

## B0014

**error:** hsm rejects the bot's firmware

The hsm partials given to `bot.define` (`bot.state`, `bot.transition`, `bot.initial`, ...) are the bot's firmware, one machine hsm defines. hsm found a mistake in it: the message is hsm's own, the span is where the bot file wrote it, and the help says how the DSL writes it. hsm stops at its first mistake, so a firmware reports one per pass.

Why: The firmware is a real state machine: it starts in the state its `bot.initial` names, a target is relative to the state its transition is in (`../<state>` is a sibling), and `bot.on`/`bot.after` go inside a `bot.transition`.

Wrong:

```python
model = bot.define("ticker", bot.cognition(processor, intuition=decider), bot.state("waiting"))
```

Right:

```python
model = bot.define("ticker", bot.cognition(processor, intuition=decider), bot.initial(bot.target("waiting")), bot.state("waiting"))
```

## B0015

**error:** an observer is given to `bot.define`

An observer (`hsm.observe(...)`) is not part of a bot's definition.

Why: An observer watches a bot while it runs, for logs and records, and never acts on it. It is given where the bot is started, `bot.start(..., observe=(...))`.

Wrong:

```python
def log(ctx, instance, event):
    print(event.name)


model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()), hsm.observe(log))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## B0016

**error:** a held device has no id

Every device has an id, set by `mosfet.device.Device.__init__`. A device class whose `__init__` never calls its base's has none.

Why: The id is how the bot addresses the device; a device without one cannot be held.

Wrong:

```python
class Lamp(mosfet.device.Device):
    def __init__(self):
        pass


model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(Lamp()))
```

Right:

```python
class Lamp(mosfet.device.Device):
    def __init__(self, *, id=None):
        super().__init__(id=id)


model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(Lamp()))
```

## B0017

**error:** `bot.define` is given something that is not a bot element

`bot.define` takes the bot's name, then elements: `bot.attach`, `bot.input`, `bot.output`, `bot.register`, `bot.traits`, `bot.cognition`, `bot.voice`, `bot.timeouts` and `bot.allow`, and the bot's firmware: `bot.state`, `bot.initial`, `bot.transition`, `bot.attribute`, `bot.operation`, `bot.final`, `bot.choice`, `bot.submachine_state`, `bot.entry`, `bot.exit`, `bot.activity` and `bot.defer`, with `bot.on`, `bot.after`, `bot.every`, `bot.at`, `bot.when`, `bot.target`, `bot.guard`, `bot.effect` and the rest inside them. A device, a processor, a seed, a tuple of elements or an element function given uncalled is this error, and the help says which element takes it.

Why: Each element says what a part is to the body; a bare part has no place in it until it is given to one.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), phone.Phone())
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## B0018

**error:** `bot.allow` names a lint that does not exist

Each name given to `bot.allow` is a lint's. The help names the nearest one.

Why: Allowing a lint that does not exist allows nothing, so the warning it was meant for still shows.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.allow("no_device"))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.allow("no_devices"))
```

## B0019

**error:** an object is given to the bot

An `environment.Object` (a desk, a table, a shelf) is not attached, input or output: it is placed. A part sits on it through its own placement, `placement=desk.surface()`, and the help deletes the element that gave the object to the bot.

Why: An object does nothing: it has no power, no state and no machine, so there is nothing for the bot to hold or start. It shows in the bot's world only through what sits on it, as `on=` on that part.

Wrong:

```python
desk = mosfet.environment.Object(id="desk")
model = bot.define(
    "greeter",
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone(placement=desk.surface())),
    bot.attach(desk),
)
```

Right:

```python
desk = mosfet.environment.Object(id="desk")
model = bot.define(
    "greeter",
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone(placement=desk.surface())),
)
```

## B0021

**error:** `bot.cognition` is given seeds but no autonomy

`seeds=` are native behaviors, such as `communication.speech_heard_seed()`, which takes heard speech into a conversation. Autonomy runs them, beside the behaviors the bot learns, before the bot deliberates; `bot.cognition` without `autonomy=True` has nothing to run them. The help switches autonomy on.

Why: A seed nothing runs is machinery for nothing: the speech it was given for would never reach the conversation.

Wrong:

```python
model = bot.define(
    "listener",
    bot.cognition(processor, intuition=decider, seeds=(communication.speech_heard_seed(),)),
    bot.attach(phone.Phone()),
)
```

Right:

```python
model = bot.define(
    "listener",
    bot.cognition(processor, intuition=decider, autonomy=True, seeds=(communication.speech_heard_seed(),)),
    bot.attach(phone.Phone()),
)
```

## B0022

**error:** `bot.voice` is given to a bot with no traits

The first time a bot with `bot.voice` starts, its inception conceives its voice from its traits. A bot without `bot.traits` has none to conceive it from. The traits are the owner's to give, so the help's trait is only an example.

Why: A voice is part of who the bot is, so it comes from the owner's seed of its nature, never from a default; the bot keeps the voice it is given for life.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.voice(catalog),
    bot.cognition(processor, intuition=decider),
    bot.output(speaking.Speaking(encoder=voices)),
    bot.attach(phone.Phone()),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.traits("warm", "patient"),
    bot.voice(catalog),
    bot.cognition(processor, intuition=decider),
    bot.output(speaking.Speaking(encoder=voices)),
    bot.attach(phone.Phone()),
)
```

## B0023

**error:** `bot.voice` is given to a bot without exactly one `speaking.Speaking` in `bot.output`

The voice inception gives is said by the bot's mouth: the one `speaking.Speaking` given to `bot.output`. A bot with none has nothing to speak in it, and a bot with two has no one mouth for it. The Speaking's `encoder` is a factory `(key, description) -> encoder`, so it speaks in the voice it is given.

Why: A bot has one voice, and one mouth says it; which of two Speakings would be the bot's is not the DSL's to guess.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.traits("warm", "patient"),
    bot.voice(catalog),
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.traits("warm", "patient"),
    bot.voice(catalog),
    bot.cognition(processor, intuition=decider),
    bot.output(speaking.Speaking(encoder=voices)),
    bot.attach(phone.Phone()),
)
```

## B0024

**error:** `bot.cognition` has no intuition

`bot.cognition` takes `intuition=`, the `processing.DecisionProcessor` its intuition decides on: `decision.Processor(...)` (`from mosfet.providers import decision`), on the server given as its `base_url` that speaks the decisions wire format, such as a local decide. In a test, the offline one is `testing.Decider()` (`from mosfet.bot import testing`). The suggestion writes `intuition=decision.Processor(base_url="http://127.0.0.1:8102")`, which may need your server's address and key.

Why: Intuition is the bot's fast first decision on every turn: it chooses among the offered events, each with its confidence, and hands what it cannot decide to reasoning. A decision model does that; a chat model does not, so intuition never runs on cognition's processor.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## B0025

**error:** `bot.cognition(intuition=)` is given a chat processor

`intuition=` takes a `processing.DecisionProcessor`, a decision model's processor, such as `decision.Processor(...)` (`from mosfet.providers import decision`); in a test, `testing.Decider()`. Any other `processing.Processor` is a chat model's: it is cognition's processor, given first.

Why: A chat model writes and calls tools; intuition chooses among the offered events with a confidence for each. Intuition on a chat model would be slow reasoning wearing a fast decision's name.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=processor), bot.attach(phone.Phone()))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## B0101

**error:** a model is started again while its parts run

`bot.start` builds one bot around the very parts a model holds. Starting a model again, or another model that shares one of its parts, while any of those parts runs is an error; the labels name the parts that are running.

Why: A part is one real thing: a phone is held by one body, and a memory remembers for one mind. Two bots cannot hold it at once, so each bot is defined from parts of its own, and a model runs one bot at a time: `bot.stop` stops its parts, and then it starts again.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
greeter = await bot.start(environment, model)
again = await bot.start(environment, model)
```

Right:

```python
greeter = await bot.start(environment, bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone())))
helper = await bot.start(environment, bot.define("helper", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone())))
```

## B0102

**error:** a part raised while `bot.start` built the bot

`bot.start` builds the bot's cognition, its faculties and its persona from `bot.cognition` and `bot.traits`, then its body around every part. One of them raised: the diagnostic points at the element that declared it, and its note is the cause, as `<Type>: <message>`.

Why: A mistake is reported where it was written. A part that only fails once it is built is reported at the element that declared it, with its cause, so it is fixed there.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
greeter = await bot.start(environment, model)  # building its cognition raised
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
greeter = await bot.start(environment, model)  # once what the cause names is fixed
```

## B0103

**error:** `bot.start` is given something that is not an `Environment`

`bot.start(ctx, model)` starts the bot in `ctx`, an `Environment`: the world it perceives and acts in.

Why: An environment is where a bot's stimuli come from and the scope its parts are addressed in; a plain `hsm.Context` is neither.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
greeter = await bot.start(hsm.Context(), model)
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
greeter = await bot.start(environment, model)
```

## B0104

**error:** two bots given to `bot.run` share a name

`bot.run` runs every bot it is given under its name, so the names of the models it is given are all different. The diagnostic points at the second `bot.define` that used the name.

Why: A bot's name is its supervisor's name, its topology name and its telemetry name in the process; two bots under one name could not be told apart.

Wrong:

```python
greeter = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
helper = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
bot.run(greeter, helper)
```

Right:

```python
greeter = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
helper = bot.define("helper", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
bot.run(greeter, helper)
```

## B0105

**error:** one model is given to `bot.run` twice

Each model given to `bot.run` runs as one bot, so a model is given once.

Why: A model holds the very parts its bot file built, and a part is held by one bot. A second bot needs a second model, with parts of its own.

Wrong:

```python
greeter = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
bot.run(greeter, greeter)
```

Right:

```python
greeter = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
bot.run(greeter)
```

## B0106

**error:** `bot.run` is given a shared actor that is not an actor

`actors=` holds the machines the process shares among its bots, such as a `websocket.Server`: each an `hsm.Instance` whose class declares `model`. The process starts them in order before any bot and stops them in reverse after every bot.

Why: The process starts each actor on its class's model; anything else has nothing to start. A part of one bot goes to that bot's `bot.define`, and a processor to its `bot.cognition`.

Wrong:

```python
greeter = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
bot.run(greeter, actors=(processor,))
```

Right:

```python
greeter = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
bot.run(greeter)
```

## B0107

**error:** a process option is unknown or its value is bad

`bot.process` reads the process options from the `MOSFET_*` variables, then from the flags after the bot's module, and knows only its own: `--name`, `--instance-id`, `--data-dir`, `--secrets-dir`, `--ready-file` and one per `runtime.Policy` field, such as `--stop-grace SECONDS`. The process exits 64, and the help shows how the option is written.

Why: A process that guessed at a bad option would run with bounds nobody chose. A bot file with flags of its own reads them itself and passes `bot.process` the rest as `argv=`.

Wrong:

```python
process = bot.process(argv=["--stop-grace", "soon"])
```

Right:

```python
process = bot.process(argv=["--stop-grace", "8"])
```

## ability_without_tools

**warning:** an acquired ability offers cognition nothing to select

An ability given to `bot.attach` is acquired: cognition uses it through the events it offers. One that offers none can never be used. A sense is given to `bot.input`, an effector to `bot.output`, and the sensory register to `bot.register`.

Why: An acquired ability with no tools is almost always a sense or an effector given to the wrong element.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
    bot.attach(memory.StmMemory()),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
    bot.register(memory.StmMemory()),
)
```

## command_without_examples

**warning:** a firmware command's field has no examples

A firmware command's field is described but has no `examples`: `pydantic.Field(description="...", examples=[...])`.

Why: A command is offered to cognition as a tool, and a tool's examples show the model choosing it what a value looks like.

Wrong:

```python
class SetData(pydantic.BaseModel):
    minutes: int = pydantic.Field(description="Minutes until it rings.")


SetEvent = hsm.Event(name="timer.set", kind=mosfet.event.EventKind, schema=SetData)
model = bot.define(
    "timer",
    bot.cognition(processor, intuition=decider),
    bot.initial(bot.target("idle")),
    bot.state("idle", bot.transition(bot.on(SetEvent), bot.target("../idle"))),
)
```

Right:

```python
class SetData(pydantic.BaseModel):
    minutes: int = pydantic.Field(description="Minutes until it rings.", examples=[10])


SetEvent = hsm.Event(name="timer.set", kind=mosfet.event.EventKind, schema=SetData)
model = bot.define(
    "timer",
    bot.cognition(processor, intuition=decider),
    bot.initial(bot.target("idle")),
    bot.state("idle", bot.transition(bot.on(SetEvent), bot.target("../idle"))),
)
```

## empty_timeouts

**warning:** `bot.timeouts()` sets no timeout

With no values, `bot.timeouts()` keeps every default, so it changes nothing.

Why: An element that changes nothing reads as if it did something. Remove it, or set the timeout meant.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()), bot.timeouts())
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## instruction_like_traits

**warning:** a trait reads as an instruction

A trait opens like an instruction (`always`, `never`, `you must`, `reply`, `respond`, `do not`), speaks to the bot as `you`, or runs past 12 words.

Why: Traits are who the bot is: the owner's fixed seed of its nature. How it acts is the bot's own to develop; an instruction in a trait would script a choice that is the bot's to make.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.traits("always reply in French"),
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.traits("curious", "speaks French"),
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
)
```

## no_devices

**warning:** the bot can perceive nothing

The bot has a cognition but no device and no input ability, so no stimulus ever reaches it and nothing can wake it.

Why: A bot that perceives nothing is allowed, and doing nothing is a choice it may make; but with nothing to perceive it was never given the choice. Attach what it should perceive, or allow the lint.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor, intuition=decider), bot.attach(phone.Phone()))
```

## unreachable_state

**warning:** a firmware state is never entered

No `bot.initial` targets the firmware state, and no transition out of a state the firmware enters targets it, so the firmware can never be in it.

Why: A state nothing enters is machinery that never runs, almost always a missing transition or a misspelled target. Add the transition that enters it, or remove it.

Wrong:

```python
model = bot.define(
    "ticker",
    bot.cognition(processor, intuition=decider),
    bot.initial(bot.target("waiting")),
    bot.state("waiting"),
    bot.state("ringing"),
)
```

Right:

```python
model = bot.define(
    "ticker",
    bot.cognition(processor, intuition=decider),
    bot.initial(bot.target("waiting")),
    bot.state("waiting", bot.transition(bot.on("ring"), bot.target("../ringing"))),
    bot.state("ringing"),
)
```

## unused_register

**warning:** a sensory register on a bot that perceives nothing

`bot.register` records every stimulus the body admits, and only devices and input abilities bring stimuli. With neither, it records nothing.

Why: The register is perception machinery; without a sense it is machinery for nothing.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor, intuition=decider),
    bot.register(memory.StmMemory()),
    bot.allow("no_devices"),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor, intuition=decider),
    bot.attach(phone.Phone()),
    bot.register(memory.StmMemory()),
)
```

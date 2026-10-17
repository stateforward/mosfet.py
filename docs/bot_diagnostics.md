# Bot diagnostics

<!-- Generated from mosfet.bot.diagnostics.EXPLANATIONS. Do not edit by hand; regenerate with
python -c 'from mosfet.bot import diagnostics; print(diagnostics.document(), end="")' > docs/bot_diagnostics.md -->

`bot.define` reports every mistake in a bot file at once, the way rustc does: each with a stable
code, the source span, notes on why, and a `help:` that shows the rewrite. Errors (`B00xx` from
`bot.define`, `B01xx` from `bot.start` and `bot.run`) stop the bot; lints (snake_case names) are
warnings. `bot.explain("<code>")` prints any entry below.

The examples assume `from mosfet import bot`, `from mosfet.devices import phone`,
`from mosfet.abilities import memory`, `import hsm`, `import mosfet`, `import datetime`, a `processor`
(a `processing.Processor`) and an `environment`.

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
- [`B0015`](#b0015) (error): an observer is given to `bot.define`
- [`B0016`](#b0016) (error): a held device has no id
- [`B0017`](#b0017) (error): `bot.define` is given something that is not a bot element
- [`B0018`](#b0018) (error): `bot.allow` names a lint that does not exist
- [`ability_without_tools`](#ability_without_tools) (warning): an acquired ability offers cognition nothing to select
- [`empty_timeouts`](#empty_timeouts) (warning): `bot.timeouts()` sets no timeout
- [`instruction_like_traits`](#instruction_like_traits) (warning): a trait reads as an instruction
- [`no_devices`](#no_devices) (warning): the bot can perceive nothing
- [`unused_register`](#unused_register) (warning): a sensory register on a bot that perceives nothing

## B0001

**error:** the bot's name is missing or not lower snake case

`bot.define` takes the bot's name first: lower snake case (a letter, then letters, digits and underscores), at most 64 characters.

Why: The name is the bot's topology name and a telemetry attribute, so it is bounded and spelled one way. The suggestion is the name in snake case.

Wrong:

```python
model = bot.define("Greeter Bot", bot.cognition(processor), bot.attach(phone.Phone()))
```

Right:

```python
model = bot.define("greeter_bot", bot.cognition(processor), bot.attach(phone.Phone()))
```

## B0002

**error:** the bot has no `bot.cognition`

Every bot has exactly one `bot.cognition`, the mind that perceives and decides.

Why: A body with no cognition can hold devices but can never choose anything to do with them. `bot.cognition(processor)` thinks with a `processing.Processor`.

Wrong:

```python
model = bot.define("greeter", bot.attach(phone.Phone()))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))
```

## B0003

**error:** a device or ability id is used twice

A bot keys its devices and the abilities it can address by `id`, so no two of them share one. A device or ability built without an `id` is known by its kind (`phone.Phone()` is `phone`), so two of one kind need ids of their own. The same instance given twice is also this error.

Why: The id is how the bot addresses a part: on its tool menu, as its focus, and in what it owns. Two parts with one id would be one name for two things.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor),
    bot.attach(phone.Phone()),
    bot.attach(phone.Phone()),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor),
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
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone(id="phone")))
```

## B0005

**error:** an element is given an instance that is already started

A part given to a bot is unstarted: the bot starts it, once, as part of its body.

Why: A started instance already belongs to whatever started it. Construct a new one for this bot; construction has no side effects, so that costs nothing.

Wrong:

```python
handset = await mosfet.started(environment, phone.Phone(), phone.Phone.model)
model = bot.define("greeter", bot.cognition(processor), bot.attach(handset))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))
```

## B0006

**error:** an element is given something it does not take

`bot.attach` takes a device or an ability; `bot.input` and `bot.output` take an ability; `bot.register` takes a `memory.StmMemory`; `bot.cognition` takes a `processing.Processor` first, and `memory=` takes a `memory.Memory`. The help names the element that takes what was given.

Why: Each element is one part of the body: a device is held, a sense organ is an input, a processor is cognition's. Giving a part to the wrong element would wire it where it cannot work.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor), bot.input(phone.Phone()))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))
```

## B0007

**error:** a device or ability uses a reserved id

The ids `bot`, `cognition` and `firmware` are the body's own.

Why: Cognition addresses the body, its cognition and its firmware by those keys, so no part may take one.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone(id="bot")))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone(id="phone")))
```

## B0008

**error:** an element is written inside another element

Every part of a bot is an argument of `bot.define` itself. `bot.cognition` takes a processor and keyword flags, not `bot.attach` or another element.

Why: A bot is one flat list of parts; nesting would suggest an ownership the body does not have.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor, bot.attach(phone.Phone())))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))
```

## B0009

**error:** an element a bot has once is given twice

A bot has at most one `bot.traits`, `bot.cognition`, `bot.timeouts`, `bot.register` and `bot.allow`.

Why: Each is one thing about the bot. Merge the two into one, for example one `bot.traits` with every trait.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.traits("curious"),
    bot.traits("patient"),
    bot.cognition(processor),
    bot.attach(phone.Phone()),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.traits("curious", "patient"),
    bot.cognition(processor),
    bot.attach(phone.Phone()),
)
```

## B0010

**error:** the traits are not one to 24 short traits

`bot.traits` takes one to 24 traits, each a word or a short phrase of at most 200 characters.

Why: Traits are the owner's fixed seed of who the bot is. The bot grows its own description on top of them, so they stay few and short.

Wrong:

```python
model = bot.define("greeter", bot.traits(), bot.cognition(processor), bot.attach(phone.Phone()))
```

Right:

```python
model = bot.define("greeter", bot.traits("curious"), bot.cognition(processor), bot.attach(phone.Phone()))
```

## B0011

**error:** a timeout is not a positive `datetime.timedelta`

Each of `bot.timeouts(processing=, deactivation=, attachment=)` is a positive `datetime.timedelta`.

Why: A timeout bounds how long the body waits for something; zero or less would never wait at all.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor),
    bot.attach(phone.Phone()),
    bot.timeouts(processing=datetime.timedelta(0)),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor),
    bot.attach(phone.Phone()),
    bot.timeouts(processing=datetime.timedelta(seconds=30)),
)
```

## B0015

**error:** an observer is given to `bot.define`

An observer (`hsm.observe(...)`) is not part of a bot's definition.

Why: An observer watches a bot while it runs, for logs and records, and never acts on it. It is given where the bot is started, `bot.start(..., observe=(...))`.

Wrong:

```python
def log(ctx, instance, event):
    print(event.name)


model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()), hsm.observe(log))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))
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


model = bot.define("greeter", bot.cognition(processor), bot.attach(Lamp()))
```

Right:

```python
class Lamp(mosfet.device.Device):
    def __init__(self, *, id=None):
        super().__init__(id=id)


model = bot.define("greeter", bot.cognition(processor), bot.attach(Lamp()))
```

## B0017

**error:** `bot.define` is given something that is not a bot element

`bot.define` takes the bot's name, then elements: `bot.attach`, `bot.input`, `bot.output`, `bot.register`, `bot.traits`, `bot.cognition` and `bot.timeouts`. A device, a processor or a tuple of elements given directly is this error, and the help says which element takes it.

Why: Each element says what a part is to the body; a bare part has no place in it until it is given to one.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor), phone.Phone())
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))
```

## B0018

**error:** `bot.allow` names a lint that does not exist

Each name given to `bot.allow` is a lint's. The help names the nearest one.

Why: Allowing a lint that does not exist allows nothing, so the warning it was meant for still shows.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor), bot.allow("no_device"))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor), bot.allow("no_devices"))
```

## ability_without_tools

**warning:** an acquired ability offers cognition nothing to select

An ability given to `bot.attach` is acquired: cognition uses it through the events it offers. One that offers none can never be used. A sense is given to `bot.input`, an effector to `bot.output`, and the sensory register to `bot.register`.

Why: An acquired ability with no tools is almost always a sense or an effector given to the wrong element.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor),
    bot.attach(phone.Phone()),
    bot.attach(memory.StmMemory()),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor),
    bot.attach(phone.Phone()),
    bot.register(memory.StmMemory()),
)
```

## empty_timeouts

**warning:** `bot.timeouts()` sets no timeout

With no values, `bot.timeouts()` keeps every default, so it changes nothing.

Why: An element that changes nothing reads as if it did something. Remove it, or set the timeout meant.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()), bot.timeouts())
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))
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
    bot.cognition(processor),
    bot.attach(phone.Phone()),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.traits("curious", "speaks French"),
    bot.cognition(processor),
    bot.attach(phone.Phone()),
)
```

## no_devices

**warning:** the bot can perceive nothing

The bot has a cognition but no device and no input ability, so no stimulus ever reaches it and nothing can wake it.

Why: A bot that perceives nothing is allowed, and doing nothing is a choice it may make; but with nothing to perceive it was never given the choice. Attach what it should perceive, or allow the lint.

Wrong:

```python
model = bot.define("greeter", bot.cognition(processor))
```

Right:

```python
model = bot.define("greeter", bot.cognition(processor), bot.attach(phone.Phone()))
```

## unused_register

**warning:** a sensory register on a bot that perceives nothing

`bot.register` records every stimulus the body admits, and only devices and input abilities bring stimuli. With neither, it records nothing.

Why: The register is perception machinery; without a sense it is machinery for nothing.

Wrong:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor),
    bot.register(memory.StmMemory()),
    bot.allow("no_devices"),
)
```

Right:

```python
model = bot.define(
    "greeter",
    bot.cognition(processor),
    bot.attach(phone.Phone()),
    bot.register(memory.StmMemory()),
)
```

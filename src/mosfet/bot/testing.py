"""``mosfet.bot.testing``: what a bot's tests need, and nothing that decides for the bot.

* :class:`Scripted` is a model whose choices a test writes down. It records every call it is given,
  makes each :class:`Choice` once, only on a turn woken by that choice's stimulus that offers its
  event, and otherwise selects nothing. A test pins what the bot was *offered* and what *woke* it;
  what the bot chooses is scripted only to prove a capacity, never to stand in for its judgment.
* :func:`running` starts a bot for the length of an ``async with`` block, with the shared actors its
  parts need, in an environment keeping time on the test's clock when it gives one, and stops it and
  them when the block ends, even when it raised.
* :func:`snapshot` is the running body's own snapshot, for assertions and observation only: a
  snapshot never drives a bot or a test's progression.

There is no factory seam and no configuration helper: a test imports the model, or the function that
defines it, and passes its doubles as plain arguments:

>>> import asyncio
>>> from mosfet import bot
>>> from mosfet.bot import testing
>>> from mosfet.devices import computer
>>> def greeter(*, processor: testing.Scripted) -> bot.Model:
...     return bot.define("greeter", bot.cognition(processor), bot.attach(computer.Computer(id="desk")))
>>> model = testing.Scripted()
>>> async def run() -> str:
...     async with testing.running(greeter(processor=model)) as running:
...         return testing.snapshot(running.bot).State
>>> asyncio.run(run())
'/greeter/active/unfocused'
>>> model.woken_by(computer.TextEnteredEvent.name)
()

Nothing was typed into the desk, so nothing woke the bot.
"""

import collections.abc
import contextlib
import dataclasses
import typing

import hsm

import mosfet
from mosfet.abilities import cognition, processing
from mosfet.environment import Environment
from mosfet.start import bound_model

from . import body, elements
from .start import Observer, start, stop


@dataclasses.dataclass(frozen=True, kw_only=True)
class Choice:
    """On the first model call of a turn woken by ``stimulus`` that offers ``event``, select it with ``data``.

    ``stimulus`` is the name of what woke the turn: a device observation (``device.phone.ringing``) or an
    ability's own observation event. ``data`` is the selected event's payload, as the model would write it.
    """

    stimulus: str
    event: str
    data: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=dict[str, object])


class Scripted(processing.Processor):
    """A model, scripted: it records every call, and makes each of its ``choices`` once; otherwise it selects nothing."""

    _calls: list[processing.InputData]
    _choices: tuple[Choice, ...]
    # The positions in ``_choices`` already made.
    _made: set[int]

    def __init__(self, *choices: Choice) -> None:
        self._calls = []
        self._choices = choices
        self._made = set()

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        """Record ``input``; select the first choice not yet made whose stimulus woke it and whose event it offers."""

        del drafts
        self._calls.append(input)
        turn = input.input.host_input if isinstance(input.input, cognition.reasoning.ProcessorInput) else input
        offered = _offered(input)
        woke = _stimulus(input)
        for position, choice in enumerate(self._choices):
            if position in self._made or woke != choice.stimulus or choice.event not in offered:
                continue
            self._made.add(position)
            (target, *_) = turn.actor_events[choice.event]
            return (processing.SelectedEvent(event=choice.event, target=target, data=dict(choice.data)),)
        return ()

    def calls(self) -> tuple[processing.InputData, ...]:
        """Every call this model was given, in order."""

        return tuple(self._calls)

    def offered_names(self) -> frozenset[str]:
        """The name of every event offered on any call: the capacities the bot was given."""

        return frozenset(name for call in self._calls for name in _offered(call))

    def woken_by(self, name: str) -> tuple[processing.InputData, ...]:
        """The calls on turns that ``name`` woke: what the bot perceived."""

        return tuple(call for call in self._calls if _stimulus(call) == name)

    def reflected(self) -> bool:
        """Whether the bot reflected on a turn: once it has, anything it chose is on its way."""

        return any(isinstance(call.input, cognition.reflection.SelectInput) for call in self._calls)


def _stimulus(call: processing.InputData) -> str | None:
    """The name of what woke the turn ``call`` belongs to: a device's observation, or an ability's event."""

    given = call.input
    if isinstance(given, cognition.reasoning.ProcessorInput):
        given = given.host_input.input
    if isinstance(given, mosfet.InputEventData):
        return None if given.observation is None else given.observation.event
    if isinstance(given, hsm.Event):
        return typing.cast(hsm.Event[typing.Any], given).name
    return None


def _offered(call: processing.InputData) -> frozenset[str]:
    return frozenset(schema.name for schema in call.schemas)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Run:
    """A bot running for a test: the started body, and the environment it runs in."""

    bot: body.Bot
    environment: Environment


@contextlib.asynccontextmanager
async def running(
    model: elements.Model,
    *,
    actors: collections.abc.Sequence[hsm.Instance] = (),
    clock: hsm.Clock | None = None,
    observe: collections.abc.Sequence[Observer] = (),
) -> collections.abc.AsyncGenerator[Run]:
    """Start ``model``'s bot in a new ``Environment`` for the block, after the shared ``actors`` it needs.

    The environment keeps time on ``clock`` when one is given (an ``hsm.Clock``, such as a test clock
    that brings a wait early), so the bot, every part it starts and the actors run their timers on it.
    Each actor is started in order on its class's ``model``, then the bot with ``bot.start``, watched by
    the ``observe`` observers (``bot.observe(...)``), which record and never act. When the block ends, even by raising, the bot is stopped with ``bot.stop``, then the actors in reverse.

    Raises:
        TypeError: If an actor's class declares no ``model``.
        bot.StartError: If the bot does not start; the actors started are stopped first.
    """

    environment = Environment(clock=clock)
    started: list[hsm.Instance] = []
    try:
        for actor in actors:
            declared = bound_model(actor)
            if declared is None:
                raise TypeError(f"actor {type(actor).__name__} declares no model to start it on")
            started.append(await mosfet.started(environment, actor, declared))
        instance = await start(environment, model, observe=observe)
        try:
            yield Run(bot=instance, environment=environment)
        finally:
            await stop(instance)
    finally:
        for actor in reversed(started):
            await hsm.stop(actor, environment)


def snapshot(instance: body.Bot) -> hsm.Snapshot:
    """The running body's own snapshot, with its devices folded in: for assertions only, never coordination."""

    return instance.take_snapshot()


__all__ = ["Choice", "Run", "Scripted", "running", "snapshot"]

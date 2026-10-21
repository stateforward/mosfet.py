"""``bot.start`` and ``bot.stop``: one bot, built from its model and brought to life in an environment.

``bot.start(ctx, model, data)`` follows ``hsm.start(ctx, instance, data)``. It builds the bot's body,
``mosfet.Bot``, around the very parts the bot file constructed and gave to ``bot.define``: its
devices by id, its abilities in their roles, its sensory register and timeouts, and a cognition
built once from ``bot.cognition``'s processor, flags and seeds on one shared memory (a fresh in-memory
``memory.Memory()`` when none was given), whose intuition decides on ``intuition=``'s decision
processor while reasoning, reflection and consolidation think on the processor, with a persona
when the model has traits and, with ``bot.voice``, the inception that gives the bot its voice:
conceived on cognition's processor from the traits, said by the one ``speaking.Speaking`` given to
``bot.output``, and kept in that memory. The body takes the bot's name, so it runs as ``/<name>`` and its devices as ``/<name><DeviceClass>``. It then starts
the body in ``ctx`` with ``data`` as its start data, joins it to the environment, and awaits the
body's answer to its activation:

>>> import asyncio
>>> from mosfet import bot
>>> from mosfet.bot import testing
>>> from mosfet.devices import computer
>>> from mosfet.environment import Environment
>>> model = bot.define(
...     "greeter",
...     bot.cognition(testing.Scripted(), intuition=testing.Decider()),
...     bot.attach(computer.Computer(id="desk")),
... )
>>> async def lifetime() -> tuple[str, str]:
...     greeter = await bot.start(Environment(), model)
...     running = greeter.state()
...     await bot.stop(greeter)
...     return running, greeter.state()
>>> asyncio.run(lifetime())
('/greeter/active/unfocused', '')

``observe=`` is the boundary observers whoever starts the bot opts into, for logs, harness records and
summaries (AGENTS: observable at the owning runtime boundary when that boundary opts in). Each is a
:data:`Observer`, made by :func:`observe`, which is ``hsm.observe`` itself; the body runs on its model
redefined with them, so they watch this start of this bot only, and they never act on it:

>>> import typing
>>> import hsm
>>> import mosfet
>>> seen: list[str] = []
>>> def record(ctx: hsm.Context, instance: hsm.Instance, observed: hsm.Event[typing.Any]) -> None:
...     seen.append(typing.cast(dict[str, hsm.Event[typing.Any]], observed.data)["event"].name)
>>> async def watched() -> None:
...     greeter = await bot.start(Environment(), model, observe=(bot.observe(mosfet.ActivateEvent, record),))
...     await bot.stop(greeter)
>>> asyncio.run(watched())
>>> seen
['bot.activate']

An observer is not part of a bot's definition: given to ``bot.define`` it is ``B0015``.

A body that activates answers ``bot.activated``; one whose parts fail to attach answers the typed
``bot.activating.failed`` and is returned as it is, inactive, for ``bot.stop`` to stop. ``bot.stop``
asks the body to deactivate, awaits its answer (``bot.deactivated``, or ``bot.degraded`` when its
teardown could not finish), takes it out of the environment, and stops it, and the body stops every
part it started: its devices, its abilities, its register, and its cognition with its faculties and
memory, which closes the database it opened. A model with firmware gets a ``bot.Firmware`` of its own
on each start, sending to the bot's devices and abilities by id; the body starts it last and stops it
first.

A model holds the instances its bot file built, so it runs one bot at a time: starting it again
while any of its parts runs is ``B0101``, and once ``bot.stop`` has stopped them it starts again. A part that raises while the bot is built is ``B0102``, and
a ``ctx`` that is not an ``Environment`` is ``B0103``. Each is a :class:`StartError` rendered like
``bot.define``'s errors, pointing at the bot file's own ``bot.start`` call or at the element that
declared the part.

Telemetry (scope ``bot.model``): one span ``bot.model.start`` per start and the counter
``bot.model.start.count``, each with ``bot.outcome`` {ok, failed} and ``bot.failure.kind`` {none,
already_started, constructor_raised, not_an_environment, attach_failed, timeout_error}: at most 7
series. ``attach_failed`` is a ``bot.activating.failed`` answer, or anything else raised while the
body starts and joins, which the span records by its own normalized kind. No name, id, class or
value is an attribute; one INFO log line per started bot names it.
"""

import collections.abc
import datetime
import logging
import typing
import uuid

import hsm
from opentelemetry import metrics

import mosfet
from mosfet import abilities, behavior, lifecycle
from mosfet.abilities import cognition, identification, memory, persona, processing, speaking, vocal
from mosfet.behavior import schedule
from mosfet.device import Device
from mosfet.environment import Environment
from mosfet.telemetry import span

from . import body, diagnostics, elements, firmware

# The package's ``define`` is the function; the module is imported by name for its error base.
from .define import Error

_LOG = logging.getLogger(__name__)
_SCOPE = "bot.model"
_STARTS = metrics.get_meter(_SCOPE).create_counter(
    "bot.model.start.count",
    unit="{start}",
    description="bot.start calls, by outcome and failure kind.",
)
_OUTCOME = "bot.outcome"
_FAILURE_KIND = "bot.failure.kind"

# A backstop past the body's own bounds, never the bound itself: the body answers activation within
# its attachment bound (30 seconds by default) and deactivation within its deactivation bound (five
# minutes), answering a failure or ``bot.degraded`` when one expires.
_ANSWER_BOUND = datetime.timedelta(minutes=10)

type Observer = hsm.ObservationElement[typing.Any]
"""A boundary observer, ``bot.observe(...)``: it watches a running bot for logs and records, and never acts on it."""

observe = hsm.observe
"""``hsm.observe`` itself: ``observe(callback)`` watches everything, ``observe(event, ..., callback)`` those events."""


class StartError(Error):
    """``bot.start`` could not start a bot: B0101, B0102 or B0103.

    ``diagnostics`` holds the one diagnostic, and ``str()`` is hsm's ``file:line:`` of the bot file's
    own code, then the diagnostic rendered and the summary line. ``failure_kind`` is the closed kind
    the start telemetry records.
    """

    def __init__(self, name: str, found: diagnostics.Diagnostic, failure_kind: str) -> None:
        super().__init__(
            hsm.Location(filename=found.primary.filename, lineno=found.primary.lineno),
            diagnostics.report((found,), action="start", name=name),
        )
        self.diagnostics: tuple[diagnostics.Diagnostic, ...] = (found,)
        self.failure_kind: str = failure_kind


def start(
    ctx: Environment, model: elements.Model, data: object = None, *, observe: collections.abc.Sequence[Observer] = ()
) -> collections.abc.Awaitable[body.Bot]:
    """Build the bot ``model`` defines and start it in ``ctx``, as ``hsm.start`` starts a machine.

    ``data`` is the body's start data, and ``observe`` the observers that watch this start of the bot.
    The awaitable resolves to the body once it has answered its activation: active, or inactive after
    ``bot.activating.failed``.

    Raises:
        StartError: B0101 if a part of ``model`` is already running, B0102 if building a part
            raised (chained to the cause), B0103 if ``ctx`` is not an ``Environment``.
        TypeError: If ``model`` is not a ``bot.Model``, or something in ``observe`` is not an observer.
        TimeoutError: If the body never answers its activation.
    """

    # Captured here, in the bot file's own call, before the coroutine runs in another frame.
    return _start(ctx, model, data, tuple(observe), diagnostics.Location.capture())


async def _start(
    ctx: Environment,
    model: elements.Model,
    data: object,
    observers: tuple[Observer, ...],
    location: diagnostics.Location,
) -> body.Bot:
    if not isinstance(typing.cast(object, model), elements.Model):
        raise TypeError(
            f"bot.start takes a bot.Model, not {type(model).__name__}: pass the model `bot.define` returned"
        )
    name = model.qualified_name[1:]
    try:
        with span.operation("bot.model.start", scope=_SCOPE, component="mosfet.Bot", stage="start") as active:
            if not isinstance(typing.cast(object, ctx), Environment):
                raise _outside(name, ctx, location)
            instance = construct(model, location, observe=observers)
            # The body's own model under the bot's name with its observers, as ``mosfet.Bot(name=, observe=)``
            # runs it (DSL-25, DSL-26).
            _ = await mosfet.started(
                ctx, instance, hsm.redefine(body.Bot.model, name, *observers), hsm.Config(data=data)
            )
            try:
                ctx.join(instance)
                answer = await abilities.run_terminal_operation(
                    ctx,
                    child=instance,
                    request=_activate,
                    terminals=(mosfet.ActivatingDoneEvent, mosfet.ActivatingFailedEvent),
                    timeout=_ANSWER_BOUND,
                )
            except BaseException:
                ctx.leave(instance)
                await hsm.stop(instance, ctx)
                raise
            if isinstance(answer.data, mosfet.ActivatingFailedEventData):
                span.record_failure(active, "attach_failed")
                _STARTS.add(1, {_OUTCOME: span.Failed, _FAILURE_KIND: "attach_failed"})
                _LOG.warning("bot activation failed name=%s", name)
                return instance
    except Exception as error:
        if isinstance(error, StartError):
            kind = error.failure_kind
        else:
            kind = "timeout_error" if isinstance(error, TimeoutError) else "attach_failed"
        _STARTS.add(1, {_OUTCOME: span.Failed, _FAILURE_KIND: kind})
        raise
    _STARTS.add(1, {_OUTCOME: span.Ok, _FAILURE_KIND: "none"})
    held = [member.instance for member in model.members.values() if isinstance(member, elements.AttachmentElement)]
    _LOG.info(
        "bot started name=%s devices=%d abilities=%d faculties=%s firmware=%s",
        name,
        sum(isinstance(part, Device) for part in held),
        sum(isinstance(part, abilities.Ability) for part in held),
        ",".join(
            (
                *_cognition(model).faculties(),
                *(
                    ["voice"]
                    if any(isinstance(member, elements.VoiceElement) for member in model.members.values())
                    else []
                ),
            )
        ),
        "yes" if any(isinstance(member, firmware.FirmwareElement) for member in model.members.values()) else "no",
    )
    return instance


def construct(
    model: elements.Model, location: diagnostics.Location, *, observe: collections.abc.Sequence[Observer] = ()
) -> body.Bot:
    """The body ``model`` defines, built around the very parts it holds and not yet started.

    ``location`` is the bot file's own call that asked for it, ``bot.start``'s or ``bot.run``'s, which
    its diagnostics point at, and ``observe`` the observers that call gave. ``bot.run`` builds every bot
    this way before its process starts.

    Raises:
        StartError: B0101 if a part of ``model`` is already running, B0102 if building a part raised
            (chained to the cause).
        TypeError: If something in ``observe`` is not an observer.
    """

    for observer in observe:
        if not isinstance(typing.cast(object, observer), hsm.ObservationElement):
            raise TypeError(f"observe takes hsm observers (bot.observe(...)), not {type(observer).__name__}")
    name = model.qualified_name[1:]
    running = [(part, member) for part, member in _parts(model) if lifecycle.is_started(part)]
    if running:
        raise _again(name, running, location)
    return _body(model, name, tuple(observe), location)


def _activate(reply_to: hsm.Instance) -> hsm.Event[typing.Any]:
    return mosfet.ActivateEvent.with_data_and_id(mosfet.ActivateEventData(reply_to=reply_to), uuid.uuid4().hex)


def _deactivate(reply_to: hsm.Instance) -> hsm.Event[typing.Any]:
    return mosfet.DeactivateEvent.with_data_and_id(mosfet.DeactivateEventData(reply_to=reply_to), uuid.uuid4().hex)


async def stop(instance: body.Bot) -> None:
    """Deactivate the bot ``bot.start`` started, await its answer, take it out of its environment and stop it.

    Stopping the body stops every part it started, so the model it was built from can be started
    again. A bot that is not running is already stopped. A body that answers ``bot.degraded``, or
    never answers, is stopped all the same.
    """

    if not lifecycle.is_started(instance):
        return
    environment = Environment.from_context(instance.context())
    try:
        answer = await abilities.run_terminal_operation(
            environment,
            child=instance,
            request=_deactivate,
            terminals=(mosfet.DeactivatingDoneEvent, mosfet.DegradedEvent),
            timeout=_ANSWER_BOUND,
        )
        if isinstance(answer.data, mosfet.DegradedEventData):
            _LOG.warning("bot stopped degraded stage=%s", answer.data.stage)
    except TimeoutError:
        _LOG.warning("bot stopped without answering its deactivation")
    finally:
        environment.leave(instance)
        await hsm.stop(instance, environment)


def _cognition(model: elements.Model) -> elements.CognitionElement:
    for member in model.members.values():
        if isinstance(member, elements.CognitionElement):
            return member
    raise TypeError(f"bot `{model.qualified_name[1:]}` has no cognition: it was not made by bot.define")


def _parts(model: elements.Model) -> list[tuple[hsm.Instance, elements.Member]]:
    """Every instance the bot file gave the model, once, with the member that holds it, in order."""

    parts: list[tuple[hsm.Instance, elements.Member]] = []
    for member in model.members.values():
        if isinstance(member, elements.AttachmentElement):
            part: object = member.instance
        elif isinstance(member, elements.RegisterElement):
            part = member.store
        elif isinstance(member, elements.CognitionElement):
            part = member.memory
        else:
            continue
        if isinstance(part, hsm.Instance) and not any(known is part for known, _ in parts):
            parts.append((part, member))
    return parts


def _body(
    model: elements.Model, name: str, observers: tuple[Observer, ...], location: diagnostics.Location
) -> body.Bot:
    """The body ``model`` defines, around the very parts it holds."""

    devices: list[Device] = []
    roles: dict[elements.Role, list[abilities.Ability[typing.Any, typing.Any]]] = {
        "attach": [],
        "input": [],
        "output": [],
    }
    traits: elements.TraitsElement | None = None
    voice: elements.VoiceElement | None = None
    timeouts: elements.TimeoutsElement | None = None
    register: memory.StmMemory | None = None
    compiled: firmware.FirmwareElement | None = None
    for member in model.members.values():
        match member:
            case elements.AttachmentElement(instance=Device() as device):
                devices.append(device)
            case elements.AttachmentElement(instance=abilities.Ability() as ability):
                for role in member.roles:
                    roles[role].append(ability)
            case elements.TraitsElement():
                traits = member
            case elements.VoiceElement():
                voice = member
            case elements.TimeoutsElement():
                timeouts = member
            case elements.RegisterElement(store=store):
                register = store
            case firmware.FirmwareElement():
                compiled = member
            case _:
                pass
    declared = _cognition(model)
    # bot.define's validator has checked each value's type; the elements keep them as written.
    processor = typing.cast(processing.Processor, declared.processor)
    given = typing.cast(memory.Memory | None, declared.memory)
    store = memory.Memory() if given is None else given
    nature = None if traits is None else persona.Traits(traits=typing.cast(tuple[str, ...], traits.traits))
    born: vocal.voice.Inception | None = None
    # With a voice, the validator has checked the bot has traits and exactly one Speaking in bot.output.
    if voice is not None and nature is not None:
        (mouth,) = [ability for ability in roles["output"] if isinstance(ability, speaking.Speaking)]
        try:
            born = vocal.voice.Inception(
                traits=nature,
                languages=tuple(typing.cast(collections.abc.Sequence[str], voice.languages)),
                speaking=mouth,
                processor=processor,
                catalog=typing.cast(vocal.voice.Catalog, voice.catalog),
                designer=typing.cast(vocal.voice.Designer | None, voice.designer),
                memory=store,
                allowed=tuple(typing.cast(collections.abc.Sequence[str], voice.allowed)),
            )
        except Exception as error:
            raise _raised(name, voice.location, "voice", error, location) from error
    try:
        mind = _mind(declared, processor, store, register, nature, born, None if timeouts is None else timeouts.step)
    except Exception as error:
        raise _raised(name, declared.location, "cognition", error, location) from error
    # A firmware of its own for every body: it starts fresh with each start of the model, and takes
    # and lets go of the bot's devices within the body's own attachment bound.
    try:
        machine = (
            None
            if compiled is None
            else firmware.Firmware(
                model=compiled.model,
                peers=_peers(devices, roles),
                attachment_timeout=None if timeouts is None else timeouts.attachment,
            )
        )
        return body.Bot(
            devices,
            cognition=mind,
            input=tuple(roles["input"]),
            output=tuple(roles["output"]),
            acquired_abilities=tuple(roles["attach"]),
            processing_timeout=None if timeouts is None else timeouts.processing,
            deactivation_timeout=None if timeouts is None else timeouts.deactivation,
            attachment_timeout=None if timeouts is None else timeouts.attachment,
            stm_memory=register,
            firmware=machine,
            name=name,
            observe=observers,
        )
    except Exception as error:
        raise _raised(name, model.location, "body", error, location) from error


def _peers(
    devices: collections.abc.Sequence[Device],
    roles: collections.abc.Mapping[elements.Role, collections.abc.Sequence[abilities.Ability[typing.Any, typing.Any]]],
) -> dict[str, hsm.Instance]:
    """The bot's devices and abilities by id: what its firmware may send to, never its body or its mind.

    An ability in two roles is one peer.

    Raises:
        ValueError: If two parts share one id, so a send by that id could reach only one of them.
    """

    peers: dict[str, hsm.Instance] = {}
    for part in (*devices, *(ability for given in roles.values() for ability in given)):
        identity = part.identity()
        if peers.setdefault(identity, part) is not part:
            raise ValueError(
                f"Bot parts must have distinct ids; {identity!r} is two parts' id, and the firmware sends by id."
            )
    return peers


def _mind(
    declared: elements.CognitionElement,
    processor: processing.Processor,
    store: memory.Memory,
    register: memory.StmMemory | None,
    nature: persona.Traits | None,
    voice: vocal.voice.Inception | None,
    step: datetime.timedelta | None,
) -> cognition.Cognition:
    """Cognition, its faculties, its persona and the ``voice`` it is given at inception, all on ``store``.

    ``declared`` is ``bot.cognition`` as written, ``processor`` its processor, and ``nature`` the traits
    of ``bot.traits``, which give the bot a persona. Intuition decides on ``declared.intuition``, the
    decision processor; every other faculty thinks on ``processor``. ``step`` is ``bot.timeouts(step=)``,
    the step timeout of cognition and of its reflection; ``None`` keeps cognition's own step timeout for both.
    ``register`` is ``bot.register``, the body's sensory register: cognition reads the recent turns from it.

    Raises:
        TypeError: If ``declared.intuition`` is no decision processor, which ``bot.define`` reports first.
    """

    decider = declared.intuition
    if not isinstance(decider, processing.DecisionProcessor):
        raise TypeError(f"bot.cognition(intuition=) is a processing.DecisionProcessor, not a {type(decider).__name__}.")
    now = typing.cast(schedule.WallClock | None, declared.routines)
    self_knowledge = None if nature is None else persona.Persona(traits=nature, memory=store)
    return cognition.Cognition(
        intuition=cognition.Intuition(processor=decider),
        reasoning=cognition.Reasoning(processor=processor),
        reflection=cognition.Reflection(
            processor=processor, memory=store, persona=self_knowledge, child_operation_timeout=step
        ),
        autonomy=cognition.Autonomy(
            memory=store, seeded_behaviors=tuple(typing.cast(collections.abc.Sequence[behavior.Seed], declared.seeds))
        )
        if declared.autonomy
        else None,
        routines=None if now is None else cognition.Routines(memory=store, now=now),
        memory=store,
        associative=memory.Associative(
            memory=store,
            registries=tuple(typing.cast(collections.abc.Sequence[identification.Registry], declared.registries)),
        )
        if declared.associative
        else None,
        stm_memory=register,
        consolidation=memory.Consolidation(memory=store, processor=processor) if declared.consolidation else None,
        persona=self_knowledge,
        voice=voice,
        child_operation_timeout=step,
    )


def _outside(name: str, ctx: object, location: diagnostics.Location) -> StartError:
    """B0103: ``ctx`` is not an ``Environment``."""

    call = diagnostics.Span.of(location)
    found = diagnostics.argument(call, 0)
    given = type(ctx).__name__
    return StartError(
        name,
        diagnostics.Diagnostic(
            code="B0103",
            level="error",
            message=f"`bot.start` starts a bot in an `Environment`, not a `{given}`",
            primary=(call if found is None else found[0]).model_copy(update={"label": f"this is a `{given}`"}),
            notes=(
                "a bot joins an environment: the world it perceives and acts in, and the scope its parts "
                + "are addressed in",
            ),
            help=diagnostics.Help(
                message="start it in an `Environment` (`from mosfet.environment import Environment`)",
                suggestion=None
                if found is None
                else diagnostics.Suggestion(
                    span=found[0], replacement="Environment()", applicability="maybe-incorrect"
                ),
            ),
        ),
        "not_an_environment",
    )


def _again(
    name: str, running: collections.abc.Sequence[tuple[hsm.Instance, elements.Member]], location: diagnostics.Location
) -> StartError:
    """B0101: a part of the model is already running."""

    def label(member: elements.Member) -> str:
        if isinstance(member, elements.AttachmentElement):
            return f"`{member.name()}` is already running"
        return f"this {'memory' if isinstance(member, elements.CognitionElement) else 'register'} is already running"

    return StartError(
        name,
        diagnostics.Diagnostic(
            code="B0101",
            level="error",
            message=f"bot `{name}` is already started",
            primary=diagnostics.Span.of(location, "started again here"),
            secondary=tuple(diagnostics.Span.of(member.location, label(member)) for _, member in running),
            notes=("a model holds the very parts its bot file built, and they run in one bot at a time",),
            help=diagnostics.Help(
                message="stop the running bot with `bot.stop` first, or define a second model, with new parts, "
                + "for a second bot"
            ),
        ),
        "already_started",
    )


def _raised(
    name: str, declared: hsm.Location, part: str, error: Exception, location: diagnostics.Location
) -> StartError:
    """B0102: building the bot's ``part`` (its cognition or its body), declared at ``declared``, raised ``error``."""

    cause = type(error).__name__
    return StartError(
        name,
        diagnostics.Diagnostic(
            code="B0102",
            level="error",
            message=f"bot `{name}` could not be built: building its {part} raised `{cause}`",
            primary=diagnostics.Span.of(declared, f"`{cause}` raised building what this declares"),
            secondary=(diagnostics.Span.of(location, "started here"),),
            notes=(diagnostics.cause(error),),
            help=diagnostics.Help(message="fix what the cause names in this element's arguments, then start again"),
        ),
        "constructor_raised",
    )


__all__ = ["Observer", "StartError", "construct", "observe", "start", "stop"]

"""``bot.start`` and ``bot.stop``: one bot, built from its model and brought to life in an environment.

``bot.start(ctx, model, data)`` follows ``hsm.start(ctx, instance, data)``. It builds the bot's body,
``mosfet.Bot``, around the very parts the bot file constructed and gave to ``bot.define``: its
devices by id, its abilities in their roles, its sensory register and timeouts, and a cognition
built once from ``bot.cognition``'s processor and flags on one shared memory (a fresh in-memory
``memory.Memory()`` when none was given), with a persona when the model has traits. It then starts
the body in ``ctx`` with ``data`` as its start data, joins it to the environment, and awaits the
body's answer to its activation::

    greeter = await bot.start(Environment(), model)
    ...
    await bot.stop(greeter)

A body that activates answers ``bot.activated``; one whose parts fail to attach answers the typed
``bot.activating.failed`` and is returned as it is, inactive, for ``bot.stop`` to stop. ``bot.stop``
asks the body to deactivate, awaits its answer (``bot.deactivated``, or ``bot.degraded`` when its
teardown could not finish), takes it out of the environment, and stops it.

A model holds the instances its bot file built, so it starts once per process: starting it again
while any of its parts runs is ``B0101``. A part that raises while the bot is built is ``B0102``, and
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
from mosfet import abilities, lifecycle
from mosfet.abilities import cognition, memory, persona, processing
from mosfet.behavior import schedule
from mosfet.device import Device
from mosfet.environment import Environment
from mosfet.telemetry import span

from . import body, diagnostics, elements

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


def start(ctx: Environment, model: elements.Model, data: object = None) -> collections.abc.Awaitable[body.Bot]:
    """Build the bot ``model`` defines and start it in ``ctx``, as ``hsm.start`` starts a machine.

    ``data`` is the body's start data. The awaitable resolves to the body once it has answered its
    activation: active, or inactive after ``bot.activating.failed``.

    Raises:
        StartError: B0101 if a part of ``model`` is already running, B0102 if building a part
            raised (chained to the cause), B0103 if ``ctx`` is not an ``Environment``.
        TypeError: If ``model`` is not a ``bot.Model``.
        TimeoutError: If the body never answers its activation.
    """

    # Captured here, in the bot file's own call, before the coroutine runs in another frame.
    return _start(ctx, model, data, diagnostics.Location.capture())


async def _start(ctx: Environment, model: elements.Model, data: object, location: diagnostics.Location) -> body.Bot:
    if not isinstance(typing.cast(object, model), elements.Model):
        raise TypeError(f"bot.start takes a bot.Model, not {type(model).__name__}: call the function that defines it")
    name = model.qualified_name[1:]
    try:
        with span.operation("bot.model.start", scope=_SCOPE, component="mosfet.Bot", stage="start") as active:
            if not isinstance(typing.cast(object, ctx), Environment):
                raise _outside(name, ctx, location)
            running = [(part, member) for part, member in _parts(model) if lifecycle.is_started(part)]
            if running:
                raise _again(name, running, location)
            instance = _body(model, name, location)
            _ = await mosfet.started(ctx, instance, instance.model, hsm.Config(data=data))
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
        "bot started name=%s devices=%d abilities=%d faculties=%s",
        name,
        sum(isinstance(part, Device) for part in held),
        sum(isinstance(part, abilities.Ability) for part in held),
        ",".join(_cognition(model).faculties()),
    )
    return instance


def _activate(reply_to: hsm.Instance) -> hsm.Event[typing.Any]:
    return mosfet.ActivateEvent.with_data_and_id(mosfet.ActivateEventData(reply_to=reply_to), uuid.uuid4().hex)


def _deactivate(reply_to: hsm.Instance) -> hsm.Event[typing.Any]:
    return mosfet.DeactivateEvent.with_data_and_id(mosfet.DeactivateEventData(reply_to=reply_to), uuid.uuid4().hex)


async def stop(instance: body.Bot) -> None:
    """Deactivate the bot ``bot.start`` started, await its answer, take it out of its environment and stop it.

    A bot that is not running is already stopped. A body that answers ``bot.degraded``, or never
    answers, is stopped all the same.
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


def _body(model: elements.Model, name: str, location: diagnostics.Location) -> body.Bot:
    """The body ``model`` defines, around the very parts it holds."""

    devices: list[Device] = []
    roles: dict[elements.Role, list[abilities.Ability[typing.Any, typing.Any]]] = {
        "attach": [],
        "input": [],
        "output": [],
    }
    traits: elements.TraitsElement | None = None
    timeouts: elements.TimeoutsElement | None = None
    register: memory.StmMemory | None = None
    for member in model.members.values():
        match member:
            case elements.AttachmentElement(instance=Device() as device):
                devices.append(device)
            case elements.AttachmentElement(instance=abilities.Ability() as ability):
                for role in member.roles:
                    roles[role].append(ability)
            case elements.TraitsElement():
                traits = member
            case elements.TimeoutsElement():
                timeouts = member
            case elements.RegisterElement(store=store):
                register = store
            case _:
                pass
    declared = _cognition(model)
    try:
        mind = _mind(declared, traits)
    except Exception as error:
        raise _raised(name, declared.location, "cognition", error, location) from error
    try:
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
        )
    except Exception as error:
        raise _raised(name, model.location, "body", error, location) from error


def _mind(declared: elements.CognitionElement, traits: elements.TraitsElement | None) -> cognition.Cognition:
    """Cognition, its faculties and its persona, all on one memory, from ``bot.cognition`` and ``bot.traits``."""

    # bot.define's validator has checked each value's type; the element keeps them as written.
    processor = typing.cast(processing.Processor, declared.processor)
    given = typing.cast(memory.Memory | None, declared.memory)
    store = memory.Memory() if given is None else given
    clock = typing.cast(schedule.WallClock | None, declared.routines)
    self_knowledge = (
        None
        if traits is None
        else persona.Persona(traits=persona.Traits(traits=typing.cast(tuple[str, ...], traits.traits)), memory=store)
    )
    return cognition.Cognition(
        intuition=cognition.Intuition(processor=processor),
        reasoning=cognition.Reasoning(processor=processor),
        reflection=cognition.Reflection(processor=processor, memory=store, persona=self_knowledge),
        autonomy=cognition.Autonomy(memory=store) if declared.autonomy else None,
        routines=None if clock is None else cognition.Routines(memory=store, clock=clock),
        memory=store,
        associative=memory.Associative(memory=store) if declared.associative else None,
        consolidation=memory.Consolidation(memory=store, processor=processor) if declared.consolidation else None,
        persona=self_knowledge,
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
            notes=("a model holds the very parts its bot file built, and they run in one bot, once per process",),
            help=diagnostics.Help(message="define a second model, with new parts, for a second bot"),
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


__all__ = ["StartError", "start", "stop"]

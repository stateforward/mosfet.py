"""The bot's firmware: the hsm partials a bot file gives ``bot.define``, compiled into one machine.

A bot's mind is its cognition; its firmware is the machinery its owner wrote for it, the way a phone's
call handling is the phone's. The hsm element functions are re-exported here as the very same
objects (``bot.state is hsm.state``), and every one given at the top level of ``bot.define``, mixed
freely with ``bot.*`` elements, is part of the bot's **one** firmware::

    model = bot.define(
        "timer",
        bot.cognition(processor),
        bot.attribute("minutes", int, 0),
        bot.initial(bot.target("idle")),
        bot.state("idle", bot.transition(bot.on(SetEvent), bot.target("../running"))),
        bot.state("running", bot.transition(bot.after(duration), bot.target("../idle"))),
    )

They compile once, after the bot's own elements are walked, into ``mosfet.define(f"{name}_firmware",
...)``: a separate machine that never touches ``mosfet.Bot.model``. With no partials, the bot has no
firmware. A :class:`FirmwareElement` at ``/<bot>/.firmware`` holds the compiled model; its
:meth:`~FirmwareElement.commands` are the tool events (``mosfet.event.EventKind``) it reacts to, which
cognition is offered.

hsm stops at the first mistake in a machine, so the firmware contributes at most one hsm error per
pass (B0014), with hsm's own message, pointing at the partial as the bot file wrote it.

``bot.start`` runs it as a :class:`Firmware` the body owns: started once the bot's devices and
abilities are attached, stopped first, and started again, fresh, with the bot. It is shown to
cognition as one ``<firmware ref="firmware" id state>`` element, the way a device is, and its commands
are offered under the actor key ``firmware`` while a state that reacts to them is active. It acts on
its own initiative: :func:`send` sends a typed event to one of the bot's devices or abilities by id,
and :func:`report` hands an observation to the body, which wakes cognition with it. What cognition does
about a report is its own choice.

Telemetry (scope ``bot.firmware``): ``bot.firmware.send.count`` by ``bot.outcome`` {delivered,
dropped}: two series. No id, name or payload is an attribute.
"""

import ast
import collections.abc
import dataclasses
import logging
import posixpath
import re
import typing
import uuid

import hsm
from opentelemetry import metrics

import mosfet
from mosfet import event, lifecycle, telemetry
from mosfet.abilities import cognition
from mosfet.telemetry import observer
from mosfet.telemetry.hsm import Traced

from . import diagnostics, elements

_LOG = logging.getLogger(__name__)
_SCOPE = "bot.firmware"
_SENDS = metrics.get_meter(_SCOPE).create_counter(
    "bot.firmware.send.count",
    unit="{event}",
    description=(
        "Events the bot's firmware sent to its devices and abilities, by outcome: delivered to a running "
        "peer, or dropped because the peer was not running."
    ),
)

state = hsm.state
initial = hsm.initial
transition = hsm.transition
on = hsm.on
entry = hsm.entry
exit = hsm.exit
activity = hsm.activity
effect = hsm.effect
guard = hsm.guard
after = hsm.after
every = hsm.every
at = hsm.at
when = hsm.when
choice = hsm.choice
final = hsm.final
defer = hsm.defer
source = hsm.source
target = hsm.target
attribute = hsm.attribute
operation = hsm.operation
on_call = hsm.on_call
on_set = hsm.on_set
shallow_history = hsm.shallow_history
deep_history = hsm.deep_history
entry_point = hsm.entry_point
exit_point = hsm.exit_point
submachine_state = hsm.submachine_state

FirmwareKind = hsm.kind.Make(elements.MemberKind)

# hsm's own diagnostic for a firmware mistake, and how the same thing is written in the DSL.
_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^initial state is required"),
        'a firmware starts in the state its `bot.initial(bot.target("<state>"))` names',
    ),
    (
        re.compile(r"^target Vertex .* not found"),
        "a target is relative to the state its transition is written in: `../<state>` is a sibling",
    ),
    (
        re.compile(r"^hsm\.Trigger\(\) must be called within"),
        "`bot.on(...)` goes inside a `bot.transition(...)`",
    ),
    (
        re.compile(r"^time event must be called within"),
        "`bot.after(...)`, `bot.every(...)` and `bot.at(...)` go inside a `bot.transition(...)`",
    ),
    (
        re.compile(r"^attribute must be called within"),
        "`bot.attribute(...)` is given to `bot.define` itself, not to a state",
    ),
    (
        re.compile(r"has no events$"),
        "a transition needs a trigger: `bot.on(...)`, `bot.after(...)` or `bot.when(...)`",
    ),
    (re.compile(r"already defined$"), "name each state once within the state it is written in"),
    (re.compile(r"^concurrent behavior must be an async function"), "`bot.activity(...)` takes an `async def`"),
)


@typing.runtime_checkable
class _Triggered(typing.Protocol):
    """An hsm partial that names the events its transition triggers on: what ``bot.on(...)`` returns."""

    events: list[hsm.Event[typing.Any]]


_LOCATED = re.compile(r"^(.+?):(\d+): (.*)$", re.DOTALL)
_QUOTED = re.compile(r"'(/[^']*)'")
_NAMED_CALLS = frozenset({"state", "final", "submachine_state", "entry_point", "exit_point"})


@dataclasses.dataclass(kw_only=True)
class FirmwareElement(elements.Member, kind=FirmwareKind):
    """The bot's one firmware: its compiled model, and where each event it reacts to is named.

    ``named`` maps each event name the firmware's transitions trigger on to where the bot file first
    wrote that trigger (its ``bot.on(...)``), so a diagnostic about a command points at its use.
    """

    function: typing.ClassVar[str] = "bot.define"
    model: hsm.Model
    named: dict[str, hsm.Location] = dataclasses.field(default_factory=dict[str, hsm.Location])

    def commands(self) -> tuple[hsm.Event[typing.Any], ...]:
        """The tool events the firmware reacts to, in the order it names them: what cognition is offered."""

        return tuple(found for found in self.model.events.values() if found.kind == event.EventKind)


@dataclasses.dataclass(kw_only=True)
class Partial(hsm.Element):
    """An hsm partial given to ``bot.define``: part of the bot's firmware, kept with where it was given.

    ``location`` is the ``bot.define`` (or ``bot.redefine``) call and ``index`` the partial's position
    in it, so a mistake in the firmware is reported at the argument the bot file wrote.
    """

    partial: hsm.Element
    index: int = 0


class Firmware(Traced):
    """The bot's firmware, running: one machine on the model ``bot.define`` compiled, beside the body.

    The body that starts it owns it (``mosfet.Bot(firmware=)``). ``model`` is the compiled firmware
    model, the one it runs and the one cognition's tool menu reads its commands from. Its behaviors
    are ``(ctx, instance, event)`` with ``instance`` this firmware: they keep observable state in its
    ``bot.attribute``\\ s with ``instance.get`` and ``instance.set``, act with :meth:`send`, and report
    with :meth:`report`. ``peers`` are the bot's devices and abilities by id: addresses it sends to,
    never reads.
    """

    model: hsm.Model
    _peers: dict[str, hsm.Instance]

    def __init__(self, *, model: hsm.Model, peers: collections.abc.Mapping[str, hsm.Instance]) -> None:
        super().__init__()
        self.model = model
        self._peers = dict(peers)

    @staticmethod
    def send(
        ctx: hsm.Context, instance: "Firmware", to: str, event: hsm.Event[typing.Any], *, trigger: hsm.Event[typing.Any]
    ) -> None:
        """Send ``event`` to the bot's device or ability ``to`` (its id), on the firmware's own initiative.

        It is stamped from ``trigger``, the event the sending behavior runs for: its id (or the
        event's own, or a fresh one) and its trace, so what the peer does stays in the trigger's
        trace. A peer that is not running drops it, counted and logged.

        Raises:
            LookupError: If ``to`` is not the id of one of the bot's devices or abilities.
        """

        peer = instance._peers.get(to)
        if peer is None:
            known = ", ".join(f"`{name}`" for name in sorted(instance._peers)) or "none"
            raise LookupError(f"the bot has no device or ability `{to}`; its ids are {known}")
        if not lifecycle.is_started(peer):
            _SENDS.add(1, {"bot.outcome": "dropped"})
            _LOG.warning("firmware send dropped: the peer is not running event=%s", event.name)
            return
        sent = dataclasses.replace(
            event,
            id=trigger.id or event.id or uuid.uuid4().hex,
            source=hsm.id(instance),
            target=hsm.id(peer),
            metadata=dict(trigger.metadata),
        )
        _ = telemetry.capture.record_dispatch(
            hsm.dispatch(ctx, peer, sent), component="firmware.send", event=sent, source=sent.source, target=sent.target
        )
        _SENDS.add(1, {"bot.outcome": "delivered"})

    @staticmethod
    def report(
        ctx: hsm.Context, instance: "Firmware", observation: hsm.Event[typing.Any], *, trigger: hsm.Event[typing.Any]
    ) -> None:
        """Report ``observation`` to the body that started this firmware, which wakes cognition with it.

        It goes the way an ability's product goes: a ``cognition.InputEvent`` whose stimulus is
        ``observation``, stamped as the firmware's, with ``trigger``'s id when it has none of its own and
        ``trigger``'s trace. A firmware started by no machine has no body to tell; the report is logged
        and dropped.
        """

        found = instance.context().value(hsm.Keys.Owner)
        if not isinstance(found, hsm.HSM):
            _LOG.warning("firmware report dropped: no body started the firmware event=%s", observation.name)
            return
        # The owner key holds the starting machine's runtime; which instance it runs is not typed.
        owner = typing.cast(hsm.HSM[typing.Any], found)
        stimulus = dataclasses.replace(
            observation,
            id=observation.id or trigger.id or uuid.uuid4().hex,
            source=hsm.id(instance),
            target="",
            metadata=dict(trigger.metadata),
        )
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=stimulus), uuid.uuid4().hex),
                source=hsm.id(instance),
                target=owner.id,
                metadata=dict(trigger.metadata),
            ),
        )


send = Firmware.send
report = Firmware.report


def partial(value: object) -> bool:
    """Whether ``value``, given to ``bot.define``, is an hsm partial: part of the bot's firmware."""

    return isinstance(value, hsm.RedefinableElement) and not isinstance(value, elements.Element)


def _callee(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


@dataclasses.dataclass(kw_only=True)
class _Places:
    """Where the bot file wrote each element of its firmware, and each named state by qualified name."""

    elements: list[tuple[hsm.Element, hsm.Location]] = dataclasses.field(
        default_factory=list[tuple[hsm.Element, hsm.Location]]
    )
    states: dict[str, list[hsm.Location]] = dataclasses.field(default_factory=dict[str, list[hsm.Location]])


def _located(
    element: hsm.Element,
    location: hsm.Location,
    node: ast.expr | None,
    span: diagnostics.Span,
    path: str,
    places: _Places,
) -> None:
    """Record where the bot file wrote ``element`` and what it owns, in ``places``.

    ``node`` is ``element``'s own expression, from the call ``span`` is, and ``path`` the qualified
    name of the state it is written in. A call whose element arguments are its owned elements, one
    to one, gives each its own expression; otherwise what it owns shares its place. The depth is the
    bot file's own nesting.
    """

    places.elements.append((element, location))
    children = list(element.owned_elements)
    arguments: list[ast.expr] = []
    if isinstance(node, ast.Call) and not any(isinstance(item, ast.Starred) for item in node.args):
        # A state's or a point's leading name is a string; every other argument is one owned element.
        leading = node.args[0] if node.args else None
        if _callee(node) in _NAMED_CALLS and isinstance(leading, ast.Constant) and isinstance(leading.value, str):
            path = posixpath.join(path, leading.value)
            places.states.setdefault(path, []).append(location)
            arguments = node.args[1:]
        else:
            arguments = list(node.args)
    paired = len(arguments) == len(children)
    for index, child in enumerate(children):
        own = arguments[index] if paired else None
        _located(
            child,
            location if own is None else diagnostics.Location.at(diagnostics.node_span(span, own)),
            own,
            span,
            path,
            places,
        )


def _hint(message: str) -> str | None:
    return next((hint for pattern, hint in _HINTS if pattern.search(message)), None)


def compile(model: elements.Model, partials: collections.abc.Sequence[Partial]) -> FirmwareElement | None:
    """Compile ``partials`` into the bot's one firmware and register it in ``model``; ``None`` without any.

    Each partial is compiled where the bot file wrote it: hsm's own places are its machinery's, so
    for the compile each element carries its argument's place in the ``bot.define`` call, and its
    own place is put back after. A firmware hsm rejects is one B0014 among ``model``'s findings.
    """

    if not partials:
        return None
    name = f"{model.name() or 'bot'}_firmware"
    places = _Places()
    first: hsm.Location | None = None
    for given in partials:
        call = diagnostics.Span.of(given.location)
        argument = diagnostics.argument(call, given.index)
        where = given.location if argument is None else diagnostics.Location.at(argument[0])
        first = first or where
        _located(given.partial, where, None if argument is None else argument[1], call, f"/{name}", places)
    placed = places.elements
    original = [(element, element.location) for element, _ in placed]
    for element, location in placed:
        element.location = location
    try:
        compiled = mosfet.define(name, *(given.partial for given in partials), hsm.observe(observer))
    except hsm.ErrorValidatingModel as error:
        model.findings.append(_rejected(model, str(error), places, partials))
        return None
    finally:
        for element, location in original:
            element.location = location
    named: dict[str, hsm.Location] = {}
    for element, location in placed:
        if isinstance(element, _Triggered):
            for trigger in typing.cast(list[object], element.events):
                if isinstance(trigger, hsm.Event):
                    _ = named.setdefault(trigger.name, location)
    member = FirmwareElement(
        qualified_name=posixpath.join(model.qualified_name, ".firmware"),
        model=compiled,
        named=named,
        location=first or model.location,
    )
    model.members[member.qualified_name] = member
    return member


def _rejected(
    model: elements.Model, text: str, places: _Places, partials: collections.abc.Sequence[Partial]
) -> diagnostics.Diagnostic:
    """B0014 for hsm's ``file:line: message``, at the partial the bot file wrote on that line.

    hsm's message names qualified names, so the error is placed at the state the first of them is
    written in; else at the partial on hsm's line; else at the firmware's first partial.
    """

    located = _LOCATED.match(text)
    message = text if located is None else located.group(3)
    primary: diagnostics.Span | None = None
    for quoted in _QUOTED.finditer(message):
        name = typing.cast(str, quoted.group(1))
        while name not in places.states and name not in ("", "/"):
            name = posixpath.dirname(name)
        if name in places.states:
            # A state written twice is reported at the second; any other at the state named.
            primary = diagnostics.Span.of(places.states[name][-1], "hsm rejects what is written here")
            break
    if primary is None:
        written = next(
            (
                location
                for _, location in places.elements
                if text.startswith(f"{location.filename}:{location.lineno}: ")
            ),
            None,
        )
        primary = (
            diagnostics.Span.of(written, "hsm rejects this")
            if written is not None
            else diagnostics.Span.of(places.elements[0][1], "the bot's firmware")
            if places.elements
            else diagnostics.head(diagnostics.Span.of(model.location), "its firmware")
        )
    hint = _hint(message)
    suggestion: diagnostics.Suggestion | None = None
    if message.startswith("initial state is required"):
        for given in partials:
            argument = diagnostics.argument(diagnostics.Span.of(given.location), given.index)
            node = None if argument is None else argument[1]
            if argument is None or not isinstance(node, ast.Call) or _callee(node) != "state":
                continue
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                start = argument[0]
                suggestion = diagnostics.Suggestion(
                    span=start.model_copy(update={"end_lineno": start.lineno, "end_col": start.col}),
                    replacement=f'bot.initial(bot.target("{first.value}")), ',
                    applicability="machine-applicable",
                )
                break
    return diagnostics.Diagnostic(
        code="B0014",
        level="error",
        message=f"hsm rejects the bot's firmware: {message}",
        primary=primary,
        notes=("the hsm partials given to `bot.define` are the bot's firmware, one machine hsm defines",),
        help=diagnostics.Help(
            message=hint or 'fix what hsm names; `bot.explain("B0014")` shows a firmware that defines',
            suggestion=suggestion,
        ),
    )


__all__ = [
    "Firmware",
    "FirmwareElement",
    "FirmwareKind",
    "Partial",
    "activity",
    "after",
    "at",
    "attribute",
    "choice",
    "compile",
    "deep_history",
    "defer",
    "effect",
    "entry",
    "entry_point",
    "every",
    "exit",
    "exit_point",
    "final",
    "guard",
    "initial",
    "on",
    "on_call",
    "on_set",
    "operation",
    "partial",
    "report",
    "shallow_history",
    "send",
    "source",
    "state",
    "submachine_state",
    "target",
    "transition",
    "when",
]

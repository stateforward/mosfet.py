"""The elements a bot file is written with: ``bot.attach``, ``bot.cognition``, ``bot.traits``, ...

Each element function returns an inert ``hsm.RedefinableElement`` that remembers where it was written
(a column-aware :class:`~mosfet.bot.diagnostics.Location`) and does nothing else: it constructs, starts
and reads nothing. ``bot.define`` walks them into a :class:`Model`, the way ``hsm.define`` walks
states, and each element's ``redefine`` registers one member there or records what is wrong with it
as a :class:`~mosfet.bot.diagnostics.Diagnostic`. The walk never raises for a mistake in the bot
file, so ``bot.define`` reports every mistake in one pass.

Members are keyed like hsm's: a device or an ability, in any role, at ``/<bot>/<id>`` (one namespace
of ids, which the bot's firmware sends by), and the bot's one traits, cognition, voice, timeouts and
sensory register at ``/<bot>/.traits``, ``.cognition``, ``.voice``, ``.timeouts`` and ``.register``.
A second member at a key is a duplicate, caught exactly as hsm catches a second state of one name.
"""

import ast
import collections.abc
import dataclasses
import difflib
import datetime
import inspect
import posixpath
import typing

import hsm

import mosfet.device
from mosfet import abilities, behavior, environment
from mosfet.abilities import memory, persona, processing, vocal
from mosfet.abilities.communication.conversation.conversation import snake_case
from mosfet.behavior import schedule

from . import diagnostics

TMember = typing.TypeVar("TMember", bound=hsm.Element)

Role = typing.Literal["attach", "input", "output"]
"""How the body takes an attached instance: held or acquired, as a sense organ, or as an effector port."""

ModelKind = hsm.kind.Make(hsm.NamespaceKind)
MemberKind = hsm.kind.Make(hsm.ElementKind)
AttachmentKind = hsm.kind.Make(MemberKind)
TraitsKind = hsm.kind.Make(MemberKind)
CognitionKind = hsm.kind.Make(MemberKind, hsm.NamespaceKind)
TimeoutsKind = hsm.kind.Make(MemberKind)
VoiceKind = hsm.kind.Make(MemberKind)
RegisterKind = hsm.kind.Make(MemberKind)
AllowKind = hsm.kind.Make(MemberKind)

RESERVED_IDS: frozenset[str] = frozenset({"bot", "cognition", "firmware"})
"""Actor keys the body keeps for itself: the body, its cognition, and its firmware."""

_FUNCTIONS: dict[Role, str] = {"attach": "bot.attach", "input": "bot.input", "output": "bot.output"}


@dataclasses.dataclass(kw_only=True)
class Model(hsm.NamespaceElement, kind=ModelKind):
    """One bot, as data: what ``bot.define`` returns and what is started.

    ``members`` holds the bot's parts by qualified name, in the order they were written, and
    ``owned_elements`` the elements it was defined from, so ``bot.redefine`` can add to them.
    ``findings`` collects the diagnostics the walk and the validator found, ``suggested_ids`` the ids
    their ``help:`` rewrites give parts, and ``suggested_elements`` the singular elements
    (``bot.cognition``, ``bot.register``) they write, so no two rewrites of one pass give the same id
    or a second one of an element.
    """

    findings: list[diagnostics.Diagnostic] = dataclasses.field(default_factory=list[diagnostics.Diagnostic])
    suggested_ids: set[str] = dataclasses.field(default_factory=set[str])
    suggested_elements: set[str] = dataclasses.field(default_factory=set[str])

    def __post_init__(self) -> None:
        super().__post_init__()
        self.members[self.qualified_name] = self


@dataclasses.dataclass(kw_only=True)
class Member(hsm.Element, kind=MemberKind):
    """A part of a bot, registered in its :class:`Model` by the element that declared it."""

    function: typing.ClassVar[str] = "bot.define"


@dataclasses.dataclass(kw_only=True)
class AttachmentElement(Member, kind=AttachmentKind):
    """A device the bot holds, or an ability it has: the constructed, unstarted ``instance``."""

    function: typing.ClassVar[str] = "bot.attach"
    instance: mosfet.device.Device | abilities.Ability[typing.Any, typing.Any]
    roles: list[Role] = dataclasses.field(default_factory=list[Role])


@dataclasses.dataclass(kw_only=True)
class TraitsElement(Member, kind=TraitsKind):
    """The owner's fixed seed of the bot's nature, as written (validated by :class:`~mosfet.bot.define.Validator`)."""

    function: typing.ClassVar[str] = "bot.traits"
    traits: tuple[object, ...] = ()


@dataclasses.dataclass(kw_only=True)
class CognitionElement(Member, hsm.NamespaceElement, kind=CognitionKind):
    """The bot's cognition: its processor, its memory, and which faculties it has."""

    function: typing.ClassVar[str] = "bot.cognition"
    processor: object = None
    memory: object = None
    associative: bool = False
    consolidation: bool = False
    autonomy: bool = False
    seeds: object = ()
    routines: object = None

    def faculties(self) -> tuple[str, ...]:
        """The faculties this cognition has, in a fixed order: the three it always has, then each flag's."""

        flags = (
            ("associative", self.associative),
            ("consolidation", self.consolidation),
            ("autonomy", self.autonomy),
            ("routines", self.routines is not None),
        )
        return ("intuition", "reasoning", "reflection", *(name for name, given in flags if given))


@dataclasses.dataclass(kw_only=True)
class VoiceElement(Member, kind=VoiceKind):
    """The voice source the bot is given its voice from at inception, as written (checked by the validator)."""

    function: typing.ClassVar[str] = "bot.voice"
    catalog: object = None
    designer: object = None
    languages: object = ()


@dataclasses.dataclass(kw_only=True)
class TimeoutsElement(Member, kind=TimeoutsKind):
    """The bot's timeout overrides; ``None`` keeps the default of the body or cognition that owns it."""

    function: typing.ClassVar[str] = "bot.timeouts"
    processing: datetime.timedelta | None = None
    deactivation: datetime.timedelta | None = None
    attachment: datetime.timedelta | None = None
    step: datetime.timedelta | None = None


@dataclasses.dataclass(kw_only=True)
class RegisterElement(Member, kind=RegisterKind):
    """The body's sensory register, which records every admitted stimulus."""

    function: typing.ClassVar[str] = "bot.register"
    store: memory.StmMemory


@dataclasses.dataclass(kw_only=True)
class AllowElement(Member, kind=AllowKind):
    """The lints this bot allows, by name, as written (checked by :class:`~mosfet.bot.define.Validator`)."""

    function: typing.ClassVar[str] = "bot.allow"
    lints: tuple[object, ...] = ()


@dataclasses.dataclass(kw_only=True)
class Element(hsm.RedefinableElement[TMember]):
    """A bot element: written at the top level of ``bot.define`` and walked into its :class:`Model`."""

    function: typing.ClassVar[str] = "bot.define"

    @typing.override
    def redefine(
        self, model: hsm.Model | Model, stack: list[hsm.Element], element: TMember | None = None
    ) -> TMember | None:
        """Register this element's member in ``model``, or record what is wrong with it.

        Raises:
            hsm.ErrorValidatingModel: If it is walked by ``hsm.define`` rather than ``bot.define``.
        """

        del element
        function = self._function()
        if not isinstance(model, Model):
            raise hsm.ErrorValidatingModel(self.location, f"{function} must be called within bot.define()")
        enclosing = stack[-1] if stack else None
        if enclosing is not model:
            where = enclosing.function if isinstance(enclosing, Member) else "an element"
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0008",
                    level="error",
                    message=f"`{function}` is written inside `{where}`",
                    primary=diagnostics.Span.of(self.location, f"`{function}` belongs to `bot.define`"),
                    notes=(f"`{where}` takes no `{function}`; every part of a bot is given to `bot.define`",),
                    help=diagnostics.Help(message=f"move it out of `{where}` to `bot.define`'s own arguments"),
                )
            )
            return None
        return self._redefine(model, stack)

    def _redefine(self, model: Model, stack: list[hsm.Element]) -> TMember | None:
        del model, stack
        raise NotImplementedError

    def _function(self) -> str:
        """The element function this element was written with, as a bot file spells it."""

        return self.function


def kind(value: object) -> str:
    """The kind of a class or an instance: its class name in snake case, the id it defaults to."""

    cls = value if isinstance(value, type) else type(value)
    return snake_case(cls.__name__.lstrip("_"))


def _argument(location: hsm.Location) -> tuple[diagnostics.Span, ast.expr | None]:
    """The span of an element call written at ``location`` and its first argument, when it parses."""

    span = diagnostics.Span.of(location)
    found = diagnostics.argument(span, 0)
    return span, None if found is None else found[1]


def free_id(base: str, taken: collections.abc.Collection[str]) -> str:
    """``base`` if it is free, else ``base-2``, ``base-3``, ...: an id no other part of the bot has.

    The number joins with a hyphen, which every id allows, a computer's peer id included.
    """

    candidate, number = base, 2
    while candidate in taken or candidate in RESERVED_IDS:
        candidate, number = f"{base}-{number}", number + 1
    return candidate


def _identity(value: object) -> str | None:
    """``value``'s id when it is a constructed device or ability whose base set one, else ``None``."""

    if isinstance(value, type) or not isinstance(value, mosfet.device.Device | abilities.Ability):
        return None
    part = typing.cast(mosfet.device.Device | abilities.Ability[typing.Any, typing.Any], value)
    try:
        return part.identity()
    except AttributeError:
        return None


def _written_ids(model: Model, *, attached: bool = False) -> set[str]:
    """The ids of the parts the bot file wrote anywhere in the definition, those after this one too.

    ``attached`` keeps only the parts given to an element that takes them, which are keyed by id;
    otherwise a part written where nothing takes it (a misplaced or a stray one) counts as well, since
    its rewrite may give it that id.
    """

    ids: set[str] = set()
    for element in model.owned_elements:
        if isinstance(element, RedefinableAttachment):
            value = element.instance if not attached or _attachable(element.instance, element.role) else None
        elif isinstance(element, RedefinableRegister | Stray) and not attached:
            value = element.store if isinstance(element, RedefinableRegister) else element.value
        else:
            continue
        identity = _identity(value)
        if identity is not None:
            ids.add(identity)
    return ids


def _suggest_id(model: Model, base: str) -> str:
    """A free id for a rewrite to give a part: none the bot file wrote, and none another rewrite gave.

    The ids written are every part's in the definition, those after this one and those written where
    nothing takes them too, so applying every rewrite of the pass at once defines clean.
    """

    suggested = free_id(base, _written_ids(model) | _actor_ids(model) | model.suggested_ids)
    model.suggested_ids.add(suggested)
    return suggested


def _with_id(span: diagnostics.Span, node: ast.Call, new_id: str) -> str:
    """The constructor call ``node``, written under ``span``'s file, as written but with ``id=new_id``."""

    def piece(item: ast.expr | ast.keyword) -> str:
        return diagnostics.text(diagnostics.node_span(span, item))

    others = [piece(keyword) for keyword in node.keywords if keyword.arg != "id"]
    return f"{piece(node.func)}({', '.join([*map(piece, node.args), f'id="{new_id}"', *others])})"


def _rename(location: hsm.Location, instance: object, new_id: str, message: str) -> diagnostics.Help:
    """Help that gives ``instance``, constructed inline at ``location``, the id ``new_id``."""

    span, argument = _argument(location)
    if not isinstance(argument, ast.Call):
        return diagnostics.Help(message=f'{message}: `{type(instance).__name__}(id="{new_id}")`')
    for keyword in argument.keywords:
        if keyword.arg == "id":
            return diagnostics.Help(
                message=message,
                suggestion=diagnostics.Suggestion(
                    span=diagnostics.node_span(span, keyword.value),
                    replacement=f'"{new_id}"',
                    applicability="machine-applicable",
                ),
            )
    return diagnostics.Help(
        message=message,
        suggestion=diagnostics.Suggestion(
            span=diagnostics.node_span(span, argument),
            replacement=_with_id(span, argument, new_id),
            applicability="machine-applicable",
        ),
    )


def _constructible(cls: type) -> bool:
    """Whether ``cls(id=...)`` needs nothing else: every other parameter has a default."""

    try:
        parameters = inspect.signature(cls).parameters.values()
    except (TypeError, ValueError):
        return False
    return all(
        typing.cast(object, parameter.default) is not inspect.Parameter.empty
        or parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters
        if parameter.name != "id"
    )


def misplaced(
    value: object,
    primary: diagnostics.Span,
    written: str,
    where: str,
    *,
    code: str = "B0006",
    model: Model | None = None,
    node: ast.expr | None = None,
) -> diagnostics.Diagnostic | None:
    """``code`` for ``value``, written as ``written`` at ``primary`` where ``where`` takes something else.

    The help names the element that takes it. With the ``model`` being defined, it carries the rewrite
    of ``primary`` into that element, machine-applicable only when applying it with every other rewrite
    of the pass defines clean: a part keeps its id when no other part has it, and is given a free one
    when ``node``, the expression written, is its constructor call; a second ``bot.cognition`` or
    ``bot.register`` is never certain. ``None`` when no element takes it, so the caller says what
    ``where`` takes instead.
    """

    def found(name: str, element: str, suggestion: diagnostics.Suggestion | None) -> diagnostics.Diagnostic:
        return diagnostics.Diagnostic(
            code=code,
            level="error",
            message=f"`{where}` does not take {name}",
            primary=primary.model_copy(update={"label": f"this is {name}"}),
            help=diagnostics.Help(message=f"{name} is given to `{element}`", suggestion=suggestion),
        )

    def rewrite(replacement: str, certain: bool) -> diagnostics.Suggestion:
        return diagnostics.Suggestion(
            span=primary,
            replacement=replacement,
            applicability="machine-applicable" if certain else "maybe-incorrect",
        )

    def single(element: type[Element[typing.Any]], function: str) -> diagnostics.Suggestion | None:
        if model is None or written == "...":
            return None
        taken = function in model.suggested_elements or any(
            isinstance(given, element) for given in model.owned_elements
        )
        model.suggested_elements.add(function)
        return rewrite(f"{function}({written})", not taken)

    def attached() -> diagnostics.Suggestion | None:
        if model is None or written == "...":
            return None
        identity = _identity(value)
        if identity is not None and identity not in (
            _written_ids(model, attached=True) | model.suggested_ids | RESERVED_IDS
        ):
            model.suggested_ids.add(identity)
            return rewrite(f"bot.attach({written})", True)
        if identity is not None and isinstance(node, ast.Call):
            return rewrite(f"bot.attach({_with_id(primary, node, _suggest_id(model, kind(value)))})", True)
        return rewrite(f"bot.attach({written})", False)

    if isinstance(value, mosfet.device.Device):
        return found("a device", "bot.attach", attached())
    if isinstance(value, processing.Processor):
        return found("a processor", "bot.cognition", single(RedefinableCognition, "bot.cognition"))
    if isinstance(value, memory.StmMemory):
        return found("a sensory register", "bot.register", single(RedefinableRegister, "bot.register"))
    if isinstance(value, memory.Memory):
        return found("a memory", "bot.cognition(processor, memory=...)", None)
    if isinstance(value, vocal.voice.Catalog):
        return found("a voice catalog", "bot.voice", single(RedefinableVoice, "bot.voice"))
    if isinstance(value, behavior.Seed):
        return found("a seed", "bot.cognition(processor, autonomy=True, seeds=(...))", None)
    if isinstance(value, abilities.Ability):
        return found("an ability", "bot.attach", attached())
    if isinstance(value, str):
        return found("a string", "bot.traits", None)
    return None


@dataclasses.dataclass(kw_only=True)
class RedefinableAttachment(Element[AttachmentElement]):
    """``bot.attach``, ``bot.input`` or ``bot.output`` of one constructed, unstarted instance."""

    instance: object
    role: Role = "attach"

    @typing.override
    def _function(self) -> str:
        return _FUNCTIONS[self.role]

    @typing.override
    def _redefine(self, model: Model, stack: list[hsm.Element]) -> AttachmentElement | None:
        function = _FUNCTIONS[self.role]
        instance = self.instance
        if isinstance(instance, environment.Object):
            model.findings.append(self._object(model, instance))
            return None
        if not _attachable(instance, self.role):
            span, argument = _argument(self.location)
            written = diagnostics.text(diagnostics.node_span(span, argument)) if argument is not None else "..."
            if isinstance(instance, type):
                model.findings.append(self._class(model, instance, span, argument, written))
            else:
                model.findings.append(self._unattachable(model, instance, written, argument))
            return None
        held = typing.cast(mosfet.device.Device | abilities.Ability[typing.Any, typing.Any], instance)
        try:
            identity = held.identity()
        except AttributeError:
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0016",
                    level="error",
                    message=f"`{type(held).__name__}` has no id",
                    primary=diagnostics.Span.of(self.location, "given here"),
                    notes=(f"`{type(held).__name__}.__init__` never called its base's `__init__`, which sets the id",),
                    help=diagnostics.Help(
                        message=f"call `super().__init__(id=id)` in `{type(held).__name__}.__init__`"
                    ),
                )
            )
            return None
        # One namespace of ids for every role: the firmware sends to devices and abilities alike by id.
        key = posixpath.join(model.qualified_name, identity)
        if identity in RESERVED_IDS:
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0007",
                    level="error",
                    message=f"id `{identity}` is reserved",
                    primary=diagnostics.Span.of(self.location, f"`{identity}` is the body's own"),
                    notes=(
                        "the body keeps `bot`, `cognition` and `firmware` for itself, the way cognition "
                        + "addresses them",
                    ),
                    help=_rename(self.location, held, _suggest_id(model, kind(held)), "give it another id"),
                )
            )
            return None
        existing = model.members.get(key)
        if isinstance(existing, AttachmentElement):
            if existing.instance is held and self.role not in existing.roles:
                existing.roles.append(self.role)
                return existing
            what = "device" if isinstance(held, mosfet.device.Device) else "ability"
            if existing.instance is held:
                message, note = f"`{identity}` is given to `{function}` twice", "a part is given to the bot once"
                help = diagnostics.Help(message=f"remove the second `{function}`")
            else:
                message = f"{what} id `{identity}` is used twice"
                note = f"a bot keys its devices and abilities by `id`; the bot addresses `{identity}` by it"
                help = _rename(
                    self.location, held, _suggest_id(model, kind(held)), f"give the second {what} its own id"
                )
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0003",
                    level="error",
                    message=message,
                    primary=diagnostics.Span.of(self.location, "used again here"),
                    secondary=(diagnostics.Span.of(existing.location, "first used here"),),
                    notes=(note,),
                    help=help,
                )
            )
            return None
        member = AttachmentElement(qualified_name=key, instance=held, roles=[self.role], location=self.location)
        model.members[key] = member
        return member

    def _class(
        self, model: Model, cls: type, span: diagnostics.Span, argument: ast.expr | None, written: str
    ) -> diagnostics.Diagnostic:
        function = _FUNCTIONS[self.role]
        what = "device or ability" if self.role == "attach" else "ability"
        suggestion = None
        if argument is not None:
            suggestion = diagnostics.Suggestion(
                span=diagnostics.node_span(span, argument),
                replacement=f'{written}(id="{_suggest_id(model, kind(cls))}")',
                applicability="machine-applicable" if _constructible(cls) else "maybe-incorrect",
            )
        primary = diagnostics.node_span(span, argument) if argument is not None else span
        return diagnostics.Diagnostic(
            code="B0004",
            level="error",
            message=f"`{function}` takes a constructed {what}, not a class",
            primary=primary.model_copy(update={"label": f"this is the class `{written}`"}),
            notes=("the bot file builds its parts; the bot holds the very instances it is given",),
            help=diagnostics.Help(message="construct it", suggestion=suggestion),
        )

    def _object(self, model: Model, placed: environment.Object) -> diagnostics.Diagnostic:
        """B0019: ``placed`` is an object, which no element takes; a part sits on it instead."""

        function = _FUNCTIONS[self.role]
        _, argument = _argument(self.location)
        written = argument.id if isinstance(argument, ast.Name) else f'environment.Object(id="{placed.id}")'
        span = diagnostics.removal(diagnostics.Span.of(model.location), diagnostics.Span.of(self.location))
        return diagnostics.Diagnostic(
            code="B0019",
            level="error",
            message=f"`{function}` does not take an object",
            primary=diagnostics.Span.of(self.location, f"`{placed.id}` is an object"),
            notes=(
                "an object does nothing: no power, no state and no machine, so the bot has nothing to hold or start",
            ),
            help=diagnostics.Help(
                message=f"an object is not attached; place a part on it with `placement={written}.surface()`, "
                + "and remove this element",
                suggestion=None
                if span is None
                else diagnostics.Suggestion(span=span, replacement="", applicability="machine-applicable"),
            ),
        )

    def _unattachable(
        self, model: Model, instance: object, written: str, argument: ast.expr | None
    ) -> diagnostics.Diagnostic:
        function = _FUNCTIONS[self.role]
        takes = "a device or an ability" if self.role == "attach" else "an ability"
        found = misplaced(instance, diagnostics.Span.of(self.location), written, function, model=model, node=argument)
        if found is not None:
            return found.model_copy(update={"notes": (f"`{function}` takes {takes}",)})
        return diagnostics.Diagnostic(
            code="B0006",
            level="error",
            message=f"`{function}` takes {takes}, not `{type(instance).__name__}`",
            primary=diagnostics.Span.of(self.location, f"this is a `{type(instance).__name__}`"),
            help=diagnostics.Help(message=f"pass {takes}, for example `{function}(phone.Phone())`"),
        )


def _attachable(instance: object, role: Role) -> bool:
    if isinstance(instance, type):
        return False
    if role == "attach":
        return isinstance(instance, mosfet.device.Device | abilities.Ability)
    return isinstance(instance, abilities.Ability) and not isinstance(instance, memory.StmMemory)


def _actor_ids(model: Model) -> set[str]:
    return {
        member.name()
        for member in model.members.values()
        if isinstance(member, AttachmentElement) and member.owner() == model.qualified_name
    }


@dataclasses.dataclass(kw_only=True)
class _Single(Element[TMember]):
    """An element a bot has at most one of, registered at ``/<bot>/<key>``."""

    key: typing.ClassVar[str] = ""

    def _register(self, model: Model, member: TMember) -> TMember | None:
        existing = model.members.get(member.qualified_name)
        if existing is not None:
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0009",
                    level="error",
                    message=f"`{self.function}` is given twice",
                    primary=diagnostics.Span.of(self.location, "given again here"),
                    secondary=(diagnostics.Span.of(existing.location, "first given here"),),
                    notes=(f"a bot has one `{self.function}`",),
                    help=diagnostics.Help(message=f"merge them into one `{self.function}(...)`"),
                )
            )
            return None
        model.members[member.qualified_name] = member
        return member


@dataclasses.dataclass(kw_only=True)
class RedefinableTraits(_Single[TraitsElement]):
    """``bot.traits``: the owner's traits, as written."""

    function: typing.ClassVar[str] = "bot.traits"
    traits: tuple[object, ...] = ()

    @typing.override
    def _redefine(self, model: Model, stack: list[hsm.Element]) -> TraitsElement | None:
        key = posixpath.join(model.qualified_name, ".traits")
        return self._register(model, TraitsElement(qualified_name=key, traits=self.traits, location=self.location))


@dataclasses.dataclass(kw_only=True)
class RedefinableCognition(_Single[CognitionElement]):
    """``bot.cognition``: the processor, the memory, the faculty flags, and its faculty markers."""

    function: typing.ClassVar[str] = "bot.cognition"
    processor: object = None
    memory: object = None
    associative: bool = False
    consolidation: bool = False
    autonomy: bool = False
    seeds: object = ()
    routines: object = None

    @typing.override
    def _redefine(self, model: Model, stack: list[hsm.Element]) -> CognitionElement | None:
        member = CognitionElement(
            qualified_name=posixpath.join(model.qualified_name, ".cognition"),
            processor=self.processor,
            memory=self.memory,
            associative=self.associative,
            consolidation=self.consolidation,
            autonomy=self.autonomy,
            seeds=self.seeds,
            routines=self.routines,
            location=self.location,
        )
        registered = self._register(model, member)
        span = diagnostics.Span.of(self.location) if self.owned_elements else None
        node = diagnostics.call(span) if span is not None else None
        arguments = node.args[1:] if node is not None and not any(isinstance(a, ast.Starred) for a in node.args) else []
        for index, faculty in enumerate(self.owned_elements):
            if isinstance(faculty, Element):
                _ = typing.cast(Element[hsm.Element], faculty).redefine(model, [*stack, member])
                continue
            primary = (
                diagnostics.node_span(span, arguments[index])
                if span is not None and index < len(arguments)
                else diagnostics.Span.of(self.location)
            )
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0017",
                    level="error",
                    message=f"`bot.cognition` takes a processor, then faculty elements; this is a `{type(faculty).__name__}`",
                    primary=primary.model_copy(update={"label": "not a faculty"}),
                    notes=(
                        "its faculties are keyword flags: `associative=`, `consolidation=`, `autonomy=`, `routines=`",
                    ),
                    help=diagnostics.Help(message="remove it, or give it to the element that takes it"),
                )
            )
        return registered


@dataclasses.dataclass(kw_only=True)
class RedefinableVoice(_Single[VoiceElement]):
    """``bot.voice``: the voice source the bot is given its voice from, as written."""

    function: typing.ClassVar[str] = "bot.voice"
    catalog: object = None
    designer: object = None
    languages: object = ()

    @typing.override
    def _redefine(self, model: Model, stack: list[hsm.Element]) -> VoiceElement | None:
        member = VoiceElement(
            qualified_name=posixpath.join(model.qualified_name, ".voice"),
            catalog=self.catalog,
            designer=self.designer,
            languages=self.languages,
            location=self.location,
        )
        return self._register(model, member)


@dataclasses.dataclass(kw_only=True)
class RedefinableTimeouts(_Single[TimeoutsElement]):
    """``bot.timeouts``: the bot's timeout overrides."""

    function: typing.ClassVar[str] = "bot.timeouts"
    processing: datetime.timedelta | None = None
    deactivation: datetime.timedelta | None = None
    attachment: datetime.timedelta | None = None
    step: datetime.timedelta | None = None

    @typing.override
    def _redefine(self, model: Model, stack: list[hsm.Element]) -> TimeoutsElement | None:
        member = TimeoutsElement(
            qualified_name=posixpath.join(model.qualified_name, ".timeouts"),
            processing=self.processing,
            deactivation=self.deactivation,
            attachment=self.attachment,
            step=self.step,
            location=self.location,
        )
        return self._register(model, member)


@dataclasses.dataclass(kw_only=True)
class RedefinableRegister(_Single[RegisterElement]):
    """``bot.register``: the body's sensory register."""

    function: typing.ClassVar[str] = "bot.register"
    store: object = None

    @typing.override
    def _redefine(self, model: Model, stack: list[hsm.Element]) -> RegisterElement | None:
        store = self.store
        given = type(store).__name__
        if not isinstance(store, memory.StmMemory):
            span, argument = _argument(self.location)
            written = diagnostics.text(diagnostics.node_span(span, argument)) if argument is not None else "..."
            cls = store if isinstance(store, type) else None
            if cls is not None and issubclass(cls, memory.StmMemory):
                primary = diagnostics.node_span(span, argument) if argument is not None else span
                model.findings.append(
                    diagnostics.Diagnostic(
                        code="B0004",
                        level="error",
                        message="`bot.register` takes a constructed sensory register, not a class",
                        primary=primary.model_copy(update={"label": f"this is the class `{written}`"}),
                        help=diagnostics.Help(
                            message="construct it",
                            suggestion=None
                            if argument is None
                            else diagnostics.Suggestion(
                                span=primary, replacement=f"{written}()", applicability="machine-applicable"
                            ),
                        ),
                    )
                )
                return None
            found = misplaced(
                store, diagnostics.Span.of(self.location), written, "bot.register", model=model, node=argument
            )
            model.findings.append(
                found.model_copy(update={"notes": ("`bot.register` takes a `memory.StmMemory`",)})
                if found is not None
                else diagnostics.Diagnostic(
                    code="B0006",
                    level="error",
                    message=f"`bot.register` takes a `memory.StmMemory`, not `{given}`",
                    primary=diagnostics.Span.of(self.location, f"this is a `{given}`"),
                    help=diagnostics.Help(message="pass a sensory register: `bot.register(memory.StmMemory())`"),
                )
            )
            return None
        member = RegisterElement(
            qualified_name=posixpath.join(model.qualified_name, ".register"), store=store, location=self.location
        )
        return self._register(model, member)


@dataclasses.dataclass(kw_only=True)
class Stray(Element[Member]):
    """An argument to ``bot.define`` that is not a bot element, kept so the walk can say what it is.

    ``location`` is the ``bot.define`` (or ``bot.redefine``) call and ``index`` the argument's position in it.
    """

    value: object = None
    index: int = 0

    @typing.override
    def _redefine(self, model: Model, stack: list[hsm.Element]) -> Member | None:
        value = self.value
        call = diagnostics.Span.of(self.location)
        found = diagnostics.argument(call, self.index)
        primary = call if found is None else found[0]
        written = "..." if found is None else diagnostics.text(primary)
        if isinstance(value, hsm.ObservationElement):
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0015",
                    level="error",
                    message="an observer is not part of a bot's definition",
                    primary=primary.model_copy(update={"label": "this observes a running bot"}),
                    notes=("an observer watches a bot while it runs, and is given where the bot is started",),
                    help=diagnostics.Help(
                        message="remove it from `bot.define`, and give it to `bot.start(..., observe=(...))`"
                    ),
                )
            )
            return None
        items = typing.cast(collections.abc.Sequence[object], value) if type(value) in (tuple, list) else ()
        if items and all(isinstance(item, Element) for item in items):
            suggestion = (
                None
                if found is None
                else diagnostics.Suggestion(span=primary, replacement=f"*{written}", applicability="machine-applicable")
            )
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0017",
                    level="error",
                    message=f"`bot.define` takes elements one by one, not a `{type(value).__name__}` of them",
                    primary=primary.model_copy(update={"label": f"a `{type(value).__name__}` of bot elements"}),
                    help=diagnostics.Help(message="unpack it", suggestion=suggestion),
                )
            )
            return None
        named = next(
            (name for function, name in (*_ELEMENT_FUNCTIONS, *_FIRMWARE_FUNCTIONS) if value is function), None
        )
        if named is not None:
            model.findings.append(
                diagnostics.Diagnostic(
                    code="B0017",
                    level="error",
                    message=f"`bot.{named}` is given, not called",
                    primary=primary.model_copy(update={"label": "this is the function itself"}),
                    help=diagnostics.Help(message=f"call it: `bot.{named}(...)`"),
                )
            )
            return None
        misplacement = misplaced(
            value, primary, written, "bot.define", code="B0017", model=model, node=None if found is None else found[1]
        )
        if misplacement is not None:
            model.findings.append(misplacement.model_copy(update={"notes": ("`bot.define` takes bot elements",)}))
            return None
        nearest = difflib.get_close_matches(kind(value), [name for _, name in _ELEMENT_FUNCTIONS], n=1, cutoff=0.5)
        model.findings.append(
            diagnostics.Diagnostic(
                code="B0017",
                level="error",
                message=f"`bot.define` takes bot elements, not a `{type(value).__name__}`",
                primary=primary.model_copy(update={"label": "not a bot element"}),
                notes=(_DEFINED_FROM,),
                help=diagnostics.Help(
                    message=f"did you mean `bot.{nearest[0]}(...)`?"
                    if nearest
                    else "give it to the element that takes it, or remove it"
                ),
            )
        )
        return None


def attach(instance: mosfet.device.Device | abilities.Ability[typing.Any, typing.Any]) -> RedefinableAttachment:
    """Hold a device, keyed by its ``id`` (``bot.attach(computer.Computer(id="desk"))``), or acquire an ability.

    ``instance`` is constructed by the bot file and unstarted; the bot holds that very instance, so it
    may also be passed to another part's constructor.
    """

    return RedefinableAttachment(instance=instance, role="attach", location=diagnostics.Location.capture())


def input(ability: abilities.Ability[typing.Any, typing.Any]) -> RedefinableAttachment:
    """A sense organ: an ability the body fans every environment stimulus out to, in parallel."""

    return RedefinableAttachment(instance=ability, role="input", location=diagnostics.Location.capture())


def output(ability: abilities.Ability[typing.Any, typing.Any]) -> RedefinableAttachment:
    """An effector port: an ability the bot acts through, off the stimulus fan-out."""

    return RedefinableAttachment(instance=ability, role="output", location=diagnostics.Location.capture())


def register(store: memory.StmMemory) -> RedefinableRegister:
    """The body's sensory register (``bot.register(memory.StmMemory())``), recording every admitted stimulus."""

    return RedefinableRegister(store=store, location=diagnostics.Location.capture())


def traits(*traits: persona.Trait) -> RedefinableTraits:
    """The owner's fixed seed of the bot's nature (``bot.traits("curious", "patient")``); with traits, it has a persona.

    One to 24 traits, each a word or a short phrase of at most 200 characters. They say who the bot is,
    never what it must do: how it acts is the bot's own to develop.
    """

    return RedefinableTraits(traits=tuple(traits), location=diagnostics.Location.capture())


def cognition(
    processor: processing.Processor,
    /,
    *faculties: Element[hsm.Element],
    memory: memory.Memory | None = None,
    associative: bool = False,
    consolidation: bool = False,
    autonomy: bool = False,
    seeds: collections.abc.Sequence[behavior.Seed] = (),
    routines: schedule.WallClock | None = None,
) -> RedefinableCognition:
    """The bot's cognition, thinking with ``processor`` (``bot.cognition(processor, associative=True)``).

    Args:
        processor: What selects the bot's events on each turn.
        faculties: Faculty elements that take their own sub-elements; none exists yet.
        memory: The memory every faculty shares; a fresh in-memory one when omitted.
        associative: Whether it recalls associations before it decides.
        consolidation: Whether it consolidates what it remembers.
        autonomy: Whether it can wake itself, and run the behaviors it has before it deliberates.
        seeds: Native behaviors its autonomy runs beside the ones it learns, such as
            ``communication.speech_heard_seed()``, which takes heard speech into a conversation; they need
            ``autonomy=True``.
        routines: The wall clock its routines run on; no routines when omitted.
    """

    return RedefinableCognition(
        processor=processor,
        memory=memory,
        associative=associative,
        consolidation=consolidation,
        autonomy=autonomy,
        seeds=seeds,
        routines=routines,
        owned_elements=list(faculties),
        location=diagnostics.Location.capture(),
    )


def timeouts(
    *,
    processing: datetime.timedelta | None = None,
    deactivation: datetime.timedelta | None = None,
    attachment: datetime.timedelta | None = None,
    step: datetime.timedelta | None = None,
) -> RedefinableTimeouts:
    """The bot's timeout overrides (``bot.timeouts(processing=datetime.timedelta(seconds=30))``); each positive.

    ``processing``, ``deactivation`` and ``attachment`` are the body's own bounds. ``step`` bounds each step
    of a turn in cognition (intuition, reasoning, ...; two minutes by default) and each step of its
    reflection; a step that outlasts it fails only its turn, which the bot then perceives.
    """

    return RedefinableTimeouts(
        processing=processing,
        deactivation=deactivation,
        attachment=attachment,
        step=step,
        location=diagnostics.Location.capture(),
    )


def voice(
    catalog: vocal.voice.Catalog,
    /,
    *,
    designer: vocal.voice.Designer | None = None,
    languages: collections.abc.Sequence[vocal.speech.Locale] = ("en",),
) -> RedefinableVoice:
    """The source the bot is given its voice from when it is born (``bot.voice(catalog, languages=("en",))``).

    The first time the bot starts, its inception conceives a voice from the bot's traits, in ``languages``:
    one ``catalog`` holds, or a new one ``designer`` makes (of the catalog's own source). Its one
    ``speaking.Speaking`` given to ``bot.output`` speaks in it from then on, on every start after, and
    cognition is told the voice it was given. The bot needs ``bot.traits`` to conceive it from.

    Args:
        catalog: The voices the source holds.
        designer: What makes a new voice from a description, at the same source; none when omitted.
        languages: The languages the voice speaks, as language tags.
    """

    return RedefinableVoice(
        catalog=catalog, designer=designer, languages=languages, location=diagnostics.Location.capture()
    )


@dataclasses.dataclass(kw_only=True)
class RedefinableAllow(_Single[AllowElement]):
    """``bot.allow``: the lints this bot allows."""

    function: typing.ClassVar[str] = "bot.allow"
    lints: tuple[object, ...] = ()

    @typing.override
    def _redefine(self, model: Model, stack: list[hsm.Element]) -> AllowElement | None:
        key = posixpath.join(model.qualified_name, ".allow")
        return self._register(model, AllowElement(qualified_name=key, lints=self.lints, location=self.location))


def allow(*lints: str) -> RedefinableAllow:
    """Allow these lints for this bot (``bot.allow("no_devices")``), when what they warn about is deliberate.

    A lint is a warning, never an error: allowing one says the bot file means it. Each name is a lint's
    (``bot.explain("<lint>")`` says what it warns about).
    """

    return RedefinableAllow(lints=tuple(lints), location=diagnostics.Location.capture())


_ELEMENT_FUNCTIONS: tuple[tuple[object, str], ...] = (
    (attach, "attach"),
    (input, "input"),
    (output, "output"),
    (register, "register"),
    (traits, "traits"),
    (cognition, "cognition"),
    (voice, "voice"),
    (timeouts, "timeouts"),
    (allow, "allow"),
)
"""Every element function, with the name it is written by after ``bot.``."""

_FIRMWARE_FUNCTIONS: tuple[tuple[object, str], ...] = (
    (hsm.state, "state"),
    (hsm.initial, "initial"),
    (hsm.transition, "transition"),
    (hsm.attribute, "attribute"),
    (hsm.operation, "operation"),
    (hsm.final, "final"),
    (hsm.choice, "choice"),
    (hsm.submachine_state, "submachine_state"),
    (hsm.entry, "entry"),
    (hsm.exit, "exit"),
    (hsm.activity, "activity"),
    (hsm.defer, "defer"),
    (hsm.on, "on"),
    (hsm.after, "after"),
    (hsm.every, "every"),
    (hsm.at, "at"),
    (hsm.when, "when"),
    (hsm.on_call, "on_call"),
    (hsm.on_set, "on_set"),
    (hsm.target, "target"),
    (hsm.source, "source"),
    (hsm.guard, "guard"),
    (hsm.effect, "effect"),
    (hsm.entry_point, "entry_point"),
    (hsm.exit_point, "exit_point"),
    (hsm.shallow_history, "shallow_history"),
    (hsm.deep_history, "deep_history"),
)
"""Every firmware element function (``bot.state is hsm.state``), with its name after ``bot.``: the first twelve
are given at the top level of ``bot.define`` (and inside a state), the rest inside them."""

_TOP_LEVEL_FIRMWARE = 12


def _listed(names: collections.abc.Sequence[str]) -> str:
    written = [f"`bot.{name}`" for name in names]
    return f"{', '.join(written[:-1])} and {written[-1]}"


_DEFINED_FROM: typing.Final = (
    f"a bot is defined from {_listed([name for _, name in _ELEMENT_FUNCTIONS])}; its firmware from "
    + f"{_listed([name for _, name in _FIRMWARE_FUNCTIONS[:_TOP_LEVEL_FIRMWARE]])}, with "
    + f"{_listed([name for _, name in _FIRMWARE_FUNCTIONS[_TOP_LEVEL_FIRMWARE:]])} inside them"
)
"""What ``bot.define`` takes, for a diagnostic about something it does not: every element, firmware included."""


__all__ = [
    "RESERVED_IDS",
    "AllowElement",
    "AttachmentElement",
    "CognitionElement",
    "Element",
    "Member",
    "Model",
    "RedefinableAllow",
    "RedefinableAttachment",
    "RedefinableCognition",
    "RedefinableRegister",
    "RedefinableTimeouts",
    "RedefinableTraits",
    "RedefinableVoice",
    "RegisterElement",
    "Role",
    "Stray",
    "TimeoutsElement",
    "TraitsElement",
    "VoiceElement",
    "allow",
    "attach",
    "cognition",
    "free_id",
    "input",
    "kind",
    "misplaced",
    "output",
    "register",
    "timeouts",
    "traits",
    "voice",
]

"""People: the bot's capacity to know who people are, under the owner's household.

Perception tells who a print is (the voice and face registries mint and match ids); memory knows who a
person is (each participant ref belongs to exactly one person, and the bot itself is ``self``); the owner
declares who lives in the house. People is how the bot itself says who is who. It offers four
model-callable tools, and choosing any of them is the bot's: hearing a voice nobody has named is a fact
memory shows the bot on its turn (``known`` false), never a reason this ability acts.

* ``LinkEvent`` (``LinkData``): this ref is that person's, or a new person's (named, or not yet).
* ``UnlinkEvent`` (``UnlinkData``): this ref is nobody's I know.
* ``NameEvent`` (``NameData``): this person is called that.
* ``ForgetEvent`` (``ForgetData``): forget this person, and their voice and face prints with them.

The owner's facts are the owner's: the household (``memory.Member``: id, name, adult or child, birth
year) is declared to memory when People is attached, and memory refuses every statement that would
change an owner fact or the bot itself (``declared``). No tool carries a role or a birth year. People
never decides what the bot says to anyone, and treats no one differently: it only refuses a capacity
memory or a registry refuses.

Memory and the registries are peers reached by message only (``memory.PersonEvent`` and
``identification.LinkEvent`` / ``identification.ForgetEvent``, each through a one-shot reply actor
bounded by ``timeout``, a modeled ``hsm.after``). Memory is the authority on who is who, and a
registry's link mark follows it: a ``voice:`` or ``face:`` print is kept through retention exactly while
its ref belongs to a person who is known, named or declared (D-W8-18). So every statement is memory's
first, and the marks then follow what memory holds:

* a link into a known person marks the ref; a link into a person nobody named or declared marks
  nothing, so a stranger's print is still pruned. When the registry refuses the mark because the id
  is the bot's own (``conflict``), holds no print (``not_found``) or is no id, memory's link is taken
  back (the ref is unlinked, and a person the link made is forgotten). A mark the registry cannot set
  yet is answered failed with memory's link standing; linking again sets it.
* a name marks every ref the person has, since naming them makes them known;
* an unlink clears the mark;
* a forget erases each print from its registry (memory answers with every ref the person had):
  erasing is part of forgetting;
* the household's declaration marks each ``voice:`` and ``face:`` ref the owner declares. Memory
  takes every owner ref from whoever holds it except the bot itself, whose own ref it refuses alone;
  People logs such a refusal once, by the ref's kind and never its value, counts it, and serves. A
  mark a registry refuses for good is logged and counted the same way; one it cannot set yet fails
  the declaration, which is made again. An owner ref no member lists any more is one memory drops
  to a fresh unknown person, so its mark is cleared and the print is a stranger's again; a mark that
  cannot be cleared is logged and counted the same way and fails nothing, since memory has dropped
  the ref and the mark left behind only keeps a print longer. A declaration that fails after
  memory dropped a ref hands the marks it could not clear yet to the next declaration (memory
  drops each ref once), so every attempt clears them again until one is declared; only then is a
  mark still left behind logged and counted, once.

Topology (``People`` behavior inside the ability lifecycle)::

    declaring     the household is declared, its owner prints  -> serving ; failed -> unavailable
                  marked
    unavailable   every selection answered failed{unavailable}
      waiting     the failed declaration's marks still to clear wait (hsm.defer); after(retry) -> due
                  (the retry starts at 100 ms and doubles up to 30 s; a declaration resets it)
      due         those marks -> declaring, which clears them again
    serving
      idle        link, unlink, name and forget are offered here, and only here
      working     one statement at a time; the rest wait (hsm.defer)

Every selection is answered once, as the stimulus of a ``cognition.InputEvent`` People hands its owner
(the body, which wakes cognition with it): ``ChangedEvent`` (``ChangedData``: what was done and the
person as memory now holds them) or ``FailedEvent`` (``FailureData`` with a closed ``FailureKind``),
carrying the selection's id. A detached People answers a selection at its source with ``FailedEvent``.

Telemetry (scope ``bot.abilities.people``; every attribute is a closed set from code):

* ``bot.people.count`` — ``bot.people.op`` {declare, link, unlink, name, forget, none} × ``bot.outcome``
  {done, failed} × ``bot.failure.kind`` (``FailureKind`` or ``none``): at most 6 × (1 + 7) = 48 series
  (``none`` is a selection whose payload was not a People statement).
* Spans ``bot.people.<op>`` (component ``people``) record a failure with the same kind.

A person id, a name, a ref, a reason or a print is never an attribute, a log field or a metric value.
"""

import collections.abc
import dataclasses
import datetime
import enum
import logging
import typing
import uuid

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from mosfet import event
from mosfet.abilities import cognition
from mosfet.telemetry import span

from .. import ability, identification, memory

_LOG = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: typing.Final = datetime.timedelta(seconds=10)
# A memory its own owner is still bringing up is asked again soon; one that keeps refusing, less often.
_FIRST_RETRY: typing.Final = datetime.timedelta(milliseconds=100)
_LAST_RETRY: typing.Final = datetime.timedelta(seconds=30)
_SCOPE: typing.Final = "bot.abilities.people"

_COUNT = metrics.get_meter(_SCOPE).create_counter(
    "bot.people.count",
    unit="{statement}",
    description="People statements answered, by statement, outcome and failure kind.",
)

Op: typing.TypeAlias = typing.Literal["link", "unlink", "name", "forget"]
"""A statement the bot makes about people."""

_ID = pydantic.TypeAdapter[str](identification.Id)


class FailureKind(enum.StrEnum):
    """Why a People statement failed; values match ``telemetry/FAILURE_KINDS.md``."""

    CONFLICT = "conflict"
    """The ref is already someone else's (unlink it from them first), or it is the bot's own voice or face."""
    DECLARED = "declared"
    """The statement would change an owner fact or the bot itself: a declared person, an owner's ref, ``self``."""
    NOT_FOUND = "not_found"
    """The statement names a person, ref or print nobody holds."""
    INVALID_INPUT = "invalid_input"
    """The statement is malformed."""
    STORE_FAILED = "store_failed"
    """Memory or a registry failed to keep the change; nothing was changed by that step."""
    TIMEOUT = "timeout"
    """Memory or a registry did not answer within People's timeout."""
    UNAVAILABLE = "unavailable"
    """People is not serving: not attached, or the household is not declared yet."""


_KINDS: typing.Final[dict[str, FailureKind]] = {kind.value: kind for kind in FailureKind}
# A registry refuses these marks for good: the id is the bot's own, holds no print, or is no id.
_PERMANENT: typing.Final = frozenset({FailureKind.CONFLICT, FailureKind.NOT_FOUND, FailureKind.INVALID_INPUT})
_PRINT_KINDS: typing.Final[frozenset[str]] = frozenset(typing.get_args(identification.Kind))


class LinkData(pydantic.BaseModel):
    """Say whose a voice, face or other ref is."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Say that a ref you perceive belongs to a person: one of the voices, faces or other identities "
                "your turn shows you, under people. Give person for someone memory already knows, or leave it "
                "out to make a new person for the ref, with a name when you know what they are called. A person "
                "may have many voices and faces, but each ref is exactly one person's: a ref that already "
                "belongs to someone else is refused until it is unlinked from them, and your own voice is "
                "always yours. Owner-declared facts (who is an adult or a child, birth years) are not yours to "
                "set. This only records who the ref belongs to; it says nothing to anyone."
            ),
        },
    )

    # No example, deliberately: the ref is one the turn showed, and an example ref is nobody's.
    ref: str = pydantic.Field(
        min_length=1,
        max_length=200,
        description=(
            "The ref exactly as the turn shows it, such as a voice or face ref from people; never one made up."
        ),
    )
    person: memory.PersonId | None = pydantic.Field(
        default=None,
        description="The id of the person it belongs to, as memory showed them; leave it out for a new person.",
    )
    # No example either: a name is a complete answer, and a name nobody gave names nobody.
    name: memory.Name | None = pydantic.Field(
        default=None,
        description=(
            "The new person's name, exactly as it was given to you; only with no person, and only a name you "
            "were actually told."
        ),
    )
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Optional short reason; it is remembered with what you did.",
        examples=["They said it was their voice."],
    )

    @pydantic.model_validator(mode="after")
    def _validate_name(self) -> typing.Self:
        if self.person is not None and self.name is not None:
            raise ValueError("name only names a new person (person is left out)")
        return self


class UnlinkData(pydantic.BaseModel):
    """Say a ref is no longer the person's it was linked to."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Say that a ref does not belong to the person it is linked to: it becomes nobody's you know "
                "again, and what you remember from before stays with whoever you thought it was then. Your own "
                "voice and refs the owner declared cannot be unlinked."
            ),
        },
    )

    ref: str = pydantic.Field(
        min_length=1, max_length=200, description="The ref exactly as the turn or memory shows it."
    )


class NameData(pydantic.BaseModel):
    """Say what a person is called."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Say what a person memory knows is called. It replaces any name you gave them before. People "
                "the owner declared, and you yourself, keep the names they have."
            ),
        },
    )

    person: memory.PersonId = pydantic.Field(description="The id of the person, as memory showed them.")
    name: memory.Name = pydantic.Field(
        description="Their name, exactly as it was given to you: the name itself and nothing around it."
    )


class ForgetData(pydantic.BaseModel):
    """Forget a person."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Forget a person: who they are, which voices and faces were theirs, and the voice and face "
                "prints themselves are erased, so you will not recognize them again. What you remember "
                "happening stays, without them in it. People the owner declared, and you yourself, cannot be "
                "forgotten."
            ),
        },
    )

    person: memory.PersonId = pydantic.Field(description="The id of the person to forget, as memory showed them.")
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Optional short reason; it is remembered with what you did.",
        examples=["They asked me to forget them."],
    )


class ChangedData(pydantic.BaseModel):
    """What a People statement did."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Observation: a statement you made about people is done. person is who it was about, as memory "
                "now holds them (for forget, as they were)."
            ),
            "examples": [
                {
                    "op": "link",
                    "person": {"id": "3f2a9c0d4b5e6f708192a3b4c5d6e7f8", "declared": False, "refs": ["voice:3f2a9c"]},
                }
            ],
        },
    )

    op: Op = pydantic.Field(description="Which statement: link, unlink, name or forget.", examples=["link"])
    person: memory.Person = pydantic.Field(
        description=(
            "link: whose the ref now is; unlink: the ref's new, unknown person; name: the person named; "
            "forget: the person as they were."
        )
    )


class FailureData(ability.FailureData):
    """A People statement failed; nothing that step would have changed was changed."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Observation: a statement you made about people was not done. kind says why: conflict (the ref "
                "is someone else's, or your own), declared (an owner fact or you yourself), not_found, "
                "invalid_input, store_failed, timeout or unavailable."
            ),
            "examples": [{"message": "That ref belongs to another person.", "kind": "conflict", "op": "link"}],
        },
    )

    kind: FailureKind = pydantic.Field(
        description="Closed failure kind; the same value is reported on spans and metrics.", examples=["conflict"]
    )
    op: Op | None = pydantic.Field(
        default=None,
        description="Which statement failed; None when the selection was not a People statement at all.",
        examples=["link"],
    )


LinkEvent = hsm.Event[LinkData](name="bot.ability.people.link", kind=event.EventKind, schema=LinkData)
UnlinkEvent = hsm.Event[UnlinkData](name="bot.ability.people.unlink", kind=event.EventKind, schema=UnlinkData)
NameEvent = hsm.Event[NameData](name="bot.ability.people.name", kind=event.EventKind, schema=NameData)
ForgetEvent = hsm.Event[ForgetData](name="bot.ability.people.forget", kind=event.EventKind, schema=ForgetData)
ChangedEvent = hsm.Event[ChangedData](name="bot.ability.people.changed", schema=ChangedData)
"""A statement is done; handed to the owner as a cognition input's stimulus."""
FailedEvent = hsm.Event[FailureData](name="bot.ability.people.failed", schema=FailureData)
"""A statement failed; handed to the owner as a cognition input's stimulus (or, detached, to its source)."""

_REQUESTS: typing.Final[tuple[hsm.Event[typing.Any], ...]] = (LinkEvent, UnlinkEvent, NameEvent, ForgetEvent)
_OPS: typing.Final[dict[type[pydantic.BaseModel], Op]] = {
    LinkData: "link",
    UnlinkData: "unlink",
    NameData: "name",
    ForgetData: "forget",
}
_Outcome: typing.TypeAlias = ChangedData | FailureData


class _DeclaredData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")


class _DoneData(pydantic.BaseModel):
    """The one outcome of the statement being served, carried back by its own activity."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    outcome: _Outcome


_DeclaredEvent = hsm.Event[_DeclaredData](
    name="bot.ability.people.declared", kind=hsm.CompletionEventKind, schema=_DeclaredData
)
_DeclareFailedEvent = hsm.Event[FailureData](
    name="bot.ability.people.declare.failed", kind=hsm.ErrorEventKind, schema=FailureData
)
_DoneEvent = hsm.Event[_DoneData](name="bot.ability.people.done", kind=hsm.CompletionEventKind, schema=_DoneData)


class _RetryData(pydantic.BaseModel):
    """A failed declaration's refs memory has dropped whose marks are still to clear, for the next one.

    At most the refs memory dropped for this household: memory drops each once, so a retry adds none.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    clear: tuple[str, ...] = ()


_RetryEvent = hsm.Event[_RetryData](name="bot.ability.people.declare.retry", schema=_RetryData)


def _op(data: object) -> Op | None:
    return _OPS.get(type(data)) if isinstance(data, pydantic.BaseModel) else None


def _declaration(household: collections.abc.Sequence[memory.Member]) -> memory.PersonDeclareData:
    """The owner's household as one declaration (``People.__init__``'s ``memory`` shadows the module there)."""

    return memory.PersonDeclareData(members=tuple(household))


def _count(op: str, outcome: _Outcome | None) -> None:
    """One ``bot.people.count`` point: ``outcome`` is the statement's (None for a declaration that was made)."""

    failure = outcome.kind if isinstance(outcome, FailureData) else None
    _COUNT.add(
        1,
        {
            "bot.people.op": op,
            "bot.outcome": "done" if failure is None else "failed",
            "bot.failure.kind": "none" if failure is None else failure.value,
        },
    )


class People(ability.Ability[LinkData, cognition.InputData]):
    """The bot's capacity to know people: link, unlink, name and forget, under the owner's household.

    ``memory`` is the bot's one memory (the cognition's), a peer reached by message. ``registries`` are the
    voice and face registries the perceiving abilities own, at most one of each kind: a ``voice:`` or
    ``face:`` ref is marked and erased in its kind's registry, and a ref of a kind with none is memory's only.
    ``household`` is the owner's declaration, made to memory each time People is attached (an empty one
    declares nobody, and undeclares whoever an earlier one declared). ``timeout`` bounds each step: one
    request to memory or to one registry.

    Raises:
        ValueError: if two registries serve one kind, ``timeout`` is not positive, or ``household`` is not a
            valid declaration (each member and each ref once, at most ``memory.MAX_MEMBERS``).
    """

    input_event: typing.ClassVar[hsm.Event[LinkData]] = LinkEvent
    output_event: typing.ClassVar[hsm.Event[cognition.InputData]] = cognition.InputEvent
    failed_event: typing.ClassVar[hsm.Event[FailureData]] = FailedEvent
    detached_requests: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = _REQUESTS

    _memory: memory.Memory
    _registries: dict[str, identification.Registry]
    _household: memory.PersonDeclareData
    _timeout: datetime.timedelta
    # Durable machine data: how long the next declaration waits after a failed one, and the id of the
    # statement being served (cleared when it is answered), so a stale completion correlates to nothing.
    _retry: datetime.timedelta
    _serving: str | None

    @classmethod
    @typing.override
    def detached_failure(cls, event: hsm.Event[typing.Any]) -> pydantic.BaseModel:
        return FailureData(kind=FailureKind.UNAVAILABLE, op=_op(event.data), message="People is not attached.")

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        memory: memory.Memory,
        registries: collections.abc.Sequence[identification.Registry] = (),
        household: collections.abc.Sequence[memory.Member] = (),
        timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
    ) -> None:
        if timeout <= datetime.timedelta(0):
            raise ValueError("People timeout must be positive.")
        by_kind: dict[str, identification.Registry] = {}
        for registry in registries:
            kind = registry.kind()
            if kind in by_kind:
                raise ValueError(f"People takes one {kind} registry; two were given.")
            by_kind[kind] = registry
        declaration = _declaration(household)
        super().__init__(id=id, clock=clock)
        self._memory = memory
        self._registries = by_kind
        self._household = declaration
        self._timeout = timeout
        self._retry = _FIRST_RETRY
        self._serving = None

    # --- asking peers ------------------------------------------------------------------------------

    @staticmethod
    async def _ask(
        ctx: hsm.Context,
        instance: "People",
        child: hsm.Instance,
        request: hsm.Event[typing.Any],
        terminals: tuple[hsm.Event[typing.Any], hsm.Event[typing.Any]],
    ) -> object:
        """Send ``request`` to a peer by message; return its reply's data, or the typed failure."""

        try:
            terminal = await ability.run_terminal_operation(
                ctx,
                child=child,
                request=request.with_data_and_id(request.data, uuid.uuid4().hex),
                terminals=terminals,
                timeout=instance._timeout,
            )
        except TimeoutError:
            return FailureData(kind=FailureKind.TIMEOUT, message=f"{type(child).__name__} did not answer in time.")
        except RuntimeError as error:
            return FailureData(
                kind=FailureKind.UNAVAILABLE, message=f"{type(child).__name__} cannot be reached: {error}"
            )
        reply = terminal.data
        if isinstance(reply, memory.FailureData | identification.FailureData):
            return FailureData(kind=_KINDS.get(reply.kind.value, FailureKind.STORE_FAILED), message=reply.message)
        if isinstance(reply, ability.FailureData):
            return FailureData(kind=FailureKind.UNAVAILABLE, message=reply.message)
        return reply

    @staticmethod
    async def _state(ctx: hsm.Context, instance: "People", data: memory.PersonData) -> object:
        """One person statement to memory: its ``PersonDoneData``, or the typed failure."""

        return await People._ask(
            ctx,
            instance,
            instance._memory,
            memory.PersonEvent.with_data(data),
            (memory.PersonDoneEvent, memory.FailedEvent),
        )

    @staticmethod
    def _print(instance: "People", ref: str) -> tuple[identification.Registry, str] | FailureData | None:
        """The registry and id a biometric ``ref`` names; None for a ref no registry here holds."""

        kind, _, id = ref.partition(":")
        registry = instance._registries.get(kind)
        if registry is None:
            return None
        try:
            return registry, _ID.validate_python(id)
        except pydantic.ValidationError:
            return FailureData(kind=FailureKind.INVALID_INPUT, message=f"That is not a valid {kind} ref.")

    @staticmethod
    async def _mark(
        ctx: hsm.Context, instance: "People", held: tuple[identification.Registry, str], linked: bool
    ) -> FailureData | None:
        registry, id = held
        reply = await People._ask(
            ctx,
            instance,
            registry,
            identification.LinkEvent.with_data(identification.LinkData(id=id, linked=linked)),
            (identification.LinkedEvent, identification.FailedEvent),
        )
        return reply if isinstance(reply, FailureData) else None

    @staticmethod
    async def _keep(
        ctx: hsm.Context, instance: "People", refs: collections.abc.Iterable[str]
    ) -> tuple[list[tuple[str, FailureData]], FailureData | None]:
        """Mark each print among ``refs`` as a known person's.

        Returns the refs a registry refused for good, each with its failure, and the first failure of a
        mark the registry could not set yet; a ref of a kind no registry here holds is memory's only.
        """

        refused: list[tuple[str, FailureData]] = []
        pending: FailureData | None = None
        for ref in refs:
            held = People._print(instance, ref)
            if held is None:
                continue
            failure = held if isinstance(held, FailureData) else await People._mark(ctx, instance, held, True)
            if failure is None:
                continue
            if failure.kind in _PERMANENT:
                refused.append((ref, failure))
            elif pending is None:
                pending = failure
        return refused, pending

    @staticmethod
    async def _clear(
        ctx: hsm.Context, instance: "People", refs: collections.abc.Iterable[str]
    ) -> list[tuple[str, FailureData]]:
        """Clear the mark of each print among ``refs``, now a person's nobody knows.

        Returns the refs whose mark a registry could not clear, each with its failure; a print a
        registry no longer holds has no mark left to clear.
        """

        uncleared: list[tuple[str, FailureData]] = []
        for ref in refs:
            held = People._print(instance, ref)
            if not isinstance(held, tuple):
                continue
            failure = await People._mark(ctx, instance, held, False)
            if failure is not None and failure.kind is not FailureKind.NOT_FOUND:
                uncleared.append((ref, failure))
        return uncleared

    @staticmethod
    def _done(op: Op, reply: object) -> _Outcome:
        if isinstance(reply, FailureData):
            return reply.model_copy(update={"op": op})
        if isinstance(reply, memory.PersonDoneData) and reply.person is not None:
            return ChangedData(op=op, person=reply.person)
        return FailureData(kind=FailureKind.STORE_FAILED, op=op, message="Memory answered without the person.")

    # --- statements --------------------------------------------------------------------------------

    @staticmethod
    async def _link(ctx: hsm.Context, instance: "People", data: LinkData) -> _Outcome:
        held = People._print(instance, data.ref)
        if isinstance(held, FailureData):
            return held.model_copy(update={"op": "link"})
        reply = await People._state(
            ctx, instance, memory.PersonLinkData(ref=data.ref, person=data.person, name=data.name)
        )
        person = reply.person if isinstance(reply, memory.PersonDoneData) else None
        if held is None or person is None or (person.name is None and not person.declared):
            # Memory refused and changed nothing, or the ref is a person's nobody named or declared,
            # or no registry holds it: no mark changes.
            return People._done("link", reply)
        refused = await People._mark(ctx, instance, held, True)
        if refused is None:
            return People._done("link", reply)
        if refused.kind not in _PERMANENT:
            return FailureData(
                kind=refused.kind,
                op="link",
                message="The ref is linked, but its print could not be kept yet; link it again to keep it.",
            )
        # The id is the bot's own or holds no print, so nobody may hold the ref: memory's link is taken
        # back, and a person this link made is forgotten with it.
        undone = await People._state(ctx, instance, memory.PersonUnlinkData(ref=data.ref))
        if not isinstance(undone, FailureData) and data.person is None:
            undone = await People._state(ctx, instance, memory.PersonForgetData(person=person.id))
        if isinstance(undone, FailureData):
            _LOG.warning("people: a link the registry refused could not be taken back from memory (%s)", undone.kind)
        return refused.model_copy(update={"op": "link"})

    @staticmethod
    async def _unlink(ctx: hsm.Context, instance: "People", data: UnlinkData) -> _Outcome:
        held = People._print(instance, data.ref)
        if isinstance(held, FailureData):
            return held.model_copy(update={"op": "unlink"})
        reply = await People._state(ctx, instance, memory.PersonUnlinkData(ref=data.ref))
        if held is not None and not isinstance(reply, FailureData):
            kept = await People._mark(ctx, instance, held, False)
            if kept is not None:
                # Memory is the authority and has unlinked; a mark left behind only keeps a print longer.
                _LOG.warning("people: a link mark could not be cleared after an unlink (%s)", kept.kind)
        return People._done("unlink", reply)

    @staticmethod
    async def _name(ctx: hsm.Context, instance: "People", data: NameData) -> _Outcome:
        reply = await People._state(ctx, instance, memory.PersonNameData(person=data.person, name=data.name))
        if not isinstance(reply, memory.PersonDoneData):
            return People._done("name", reply)
        refused, pending = await People._keep(ctx, instance, reply.refs)
        # A print a registry no longer holds has nothing left to keep.
        unkept = pending or next((failure for _, failure in refused if failure.kind is not FailureKind.NOT_FOUND), None)
        if unkept is not None:
            return FailureData(
                kind=unkept.kind,
                op="name",
                message="The person is named, but not every one of their prints could be kept; name them again.",
            )
        return People._done("name", reply)

    @staticmethod
    async def _forget(ctx: hsm.Context, instance: "People", data: ForgetData) -> _Outcome:
        reply = await People._state(ctx, instance, memory.PersonForgetData(person=data.person))
        if not isinstance(reply, memory.PersonDoneData):
            return People._done("forget", reply)
        unerased: FailureData | None = None
        for ref in reply.refs:
            held = People._print(instance, ref)
            if not isinstance(held, tuple):
                continue
            registry, id = held
            erased = await People._ask(
                ctx,
                instance,
                registry,
                identification.ForgetEvent.with_data(identification.ForgetData(id=id)),
                (identification.ForgottenEvent, identification.FailedEvent),
            )
            if isinstance(erased, FailureData) and unerased is None:
                unerased = erased
        if unerased is not None:
            return FailureData(
                kind=unerased.kind,
                op="forget",
                message="The person is forgotten, but not every one of their prints could be erased.",
            )
        return People._done("forget", reply)

    # --- declaring ---------------------------------------------------------------------------------

    @staticmethod
    async def _declare(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> None:
        """Declare the owner's household to memory, mark its owner prints, and report how that went here.

        A ref refused for good (memory keeps the bot's own ref its own; a registry holds no such print)
        is logged once and counted, and the household is declared without it. A failure that may pass
        (memory or a registry cannot answer yet) fails the declaration, which is made again. Each ref
        memory dropped has its mark cleared, on this attempt or, when ``event`` is a failed one's
        ``_RetryEvent``, again on this one; a mark still not cleared when the household is declared is
        logged once and counted, and one a failed attempt leaves is handed on to the next.
        """

        # Memory answers a dropped ref once, so the marks a failed attempt could not clear come with its retry.
        clear = event.data.clear if isinstance(event.data, _RetryData) else ()
        refused: list[tuple[str, FailureData]] = []
        uncleared: list[tuple[str, FailureData]] = []
        with span.operation("bot.people.declare", scope=_SCOPE, component="people", stage="declare") as active:
            try:
                reply = await People._state(ctx, instance, instance._household)
                if isinstance(reply, memory.PersonDoneData):
                    refused = [
                        (ref, FailureData(kind=FailureKind.CONFLICT, message="The ref is the bot's own."))
                        for ref in reply.refused
                    ]
                    owner = [
                        ref for member in instance._household.members for ref in member.refs if ref not in reply.refused
                    ]
                    marked, pending = await People._keep(ctx, instance, owner)
                    refused.extend(marked)
                    clear = tuple(dict.fromkeys((*reply.dropped, *clear)))
                    uncleared = await People._clear(ctx, instance, clear)
                    clear = tuple(ref for ref, _ in uncleared)
                    if pending is not None:
                        reply = pending
            except Exception:
                # The boundary of the declaration's one outcome: an unexpected error is retried, never left hanging.
                _LOG.exception("people: declaring the household failed unexpectedly")
                reply = FailureData(kind=FailureKind.STORE_FAILED, message="People could not declare the household.")
            if isinstance(reply, FailureData):
                span.record_failure(active, reply.kind.value)
        if isinstance(reply, FailureData):
            _count("declare", reply)
            _ = hsm.dispatch(ctx, instance, _DeclareFailedEvent.with_data(reply))
            # Queued after the failure, so it waits in unavailable for the retry it starts.
            _ = hsm.dispatch(ctx, instance, _RetryEvent.with_data(_RetryData(clear=clear)))
            return
        for ref, failure in refused:
            kind = ref.partition(":")[0]
            _LOG.warning(
                "people: an owner-declared %s ref was refused (%s); the rest of the household is declared",
                kind if kind in _PRINT_KINDS else "other",
                failure.kind,
            )
            _count("declare", failure)
        for ref, failure in uncleared:
            kind = ref.partition(":")[0]
            _LOG.warning(
                "people: the mark of a %s ref the owner no longer declares could not be cleared (%s)",
                kind if kind in _PRINT_KINDS else "other",
                failure.kind,
            )
            _count("declare", failure)
        _count("declare", None)
        _ = hsm.dispatch(ctx, instance, _DeclaredEvent.with_data(_DeclaredData()))

    @staticmethod
    def _declared(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._retry = _FIRST_RETRY

    @staticmethod
    def _retry_delay(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._retry

    @staticmethod
    def _back_off(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._retry = min(instance._retry * 2, _LAST_RETRY)

    # --- answering the bot -------------------------------------------------------------------------

    @staticmethod
    def _observe(ctx: hsm.Context, instance: "People", request: hsm.Event[typing.Any], outcome: _Outcome) -> None:
        """Hand the owner ``outcome`` as the stimulus of a cognition input; the stimulus carries the request's id."""

        observation = (
            ChangedEvent.with_data(outcome) if isinstance(outcome, ChangedData) else FailedEvent.with_data(outcome)
        )
        stimulus = dataclasses.replace(
            observation,
            id=request.id or uuid.uuid4().hex,
            source=hsm.id(instance),
            target=None,
            metadata=dict(request.metadata),
        )
        handoff = dataclasses.replace(
            cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=stimulus), uuid.uuid4().hex),
            source=hsm.id(instance),
            target=None,
            metadata=dict(stimulus.metadata),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(handoff))
        _count(outcome.op or "none", outcome)

    @staticmethod
    def _refuse_unavailable(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> None:
        People._observe(
            ctx,
            instance,
            event,
            FailureData(
                kind=FailureKind.UNAVAILABLE,
                op=_op(event.data),
                message="People is not serving yet: the household is not declared.",
            ),
        )

    @staticmethod
    def _refuse_invalid(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> None:
        People._observe(
            ctx,
            instance,
            event,
            FailureData(
                kind=FailureKind.INVALID_INPUT,
                message=f"The selection carried {type(event.data).__name__}, not a People statement.",
            ),
        )

    # --- serving -----------------------------------------------------------------------------------

    @staticmethod
    def _is_statement(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, LinkData | UnlinkData | NameData | ForgetData)

    @staticmethod
    def _begin(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._serving = event.id or None

    @staticmethod
    def _end(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._serving = None

    @staticmethod
    async def _work(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> None:
        """Serve the one statement in flight and report its outcome to this machine under its id."""

        data = event.data
        assert isinstance(data, LinkData | UnlinkData | NameData | ForgetData)
        op = _OPS[type(data)]
        with span.operation(f"bot.people.{op}", scope=_SCOPE, component="people", stage=op) as active:
            outcome: _Outcome
            try:
                if isinstance(data, LinkData):
                    outcome = await People._link(ctx, instance, data)
                elif isinstance(data, UnlinkData):
                    outcome = await People._unlink(ctx, instance, data)
                elif isinstance(data, NameData):
                    outcome = await People._name(ctx, instance, data)
                else:
                    outcome = await People._forget(ctx, instance, data)
            except pydantic.ValidationError:
                outcome = FailureData(kind=FailureKind.INVALID_INPUT, op=op, message="The statement is not valid.")
            except Exception:
                # The boundary of the statement's one answer: an unexpected error is answered, never left hanging.
                _LOG.exception("people: the %s statement failed unexpectedly", op)
                outcome = FailureData(
                    kind=FailureKind.STORE_FAILED, op=op, message="People could not finish the statement."
                )
            if isinstance(outcome, FailureData):
                span.record_failure(active, outcome.kind.value)
        done = _DoneEvent.with_data_and_id(_DoneData(outcome=outcome), event.id)
        _ = hsm.dispatch(ctx, instance, dataclasses.replace(done, metadata=dict(event.metadata)))

    @staticmethod
    def _is_current(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return instance._serving is not None and event.id == instance._serving and isinstance(event.data, _DoneData)

    @staticmethod
    def _answer(ctx: hsm.Context, instance: "People", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _DoneData)
        People._observe(ctx, instance, event, data.outcome)

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "People",
        hsm.initial(hsm.target("/People/declaring")),
        hsm.state(
            "declaring",
            hsm.defer(*_REQUESTS),
            hsm.activity(_declare),
            hsm.transition(hsm.on(_DeclaredEvent), hsm.effect(_declared), hsm.target("/People/serving")),
            hsm.transition(hsm.on(_DeclareFailedEvent), hsm.target("/People/unavailable")),
        ),
        hsm.state(
            "unavailable",
            hsm.initial(hsm.target("/People/unavailable/waiting")),
            # Not offered here, yet answered at once, selected by its typed payload: the bot learns now.
            hsm.transition(hsm.on(hsm.AnyEvent), hsm.guard(_is_statement), hsm.effect(_refuse_unavailable)),
            hsm.state(
                "waiting",
                hsm.defer(_RetryEvent),
                hsm.transition(hsm.after(_retry_delay), hsm.effect(_back_off), hsm.target("/People/unavailable/due")),
            ),
            # The failed declaration's retry carries the marks it left to clear into the next declaration.
            hsm.state("due", hsm.transition(hsm.on(_RetryEvent), hsm.target("/People/declaring"))),
        ),
        hsm.state(
            "serving",
            hsm.initial(hsm.target("/People/serving/idle")),
            hsm.state(
                "idle",
                # Offered to the bot from here, and only here: the tools come from live topology.
                hsm.transition(hsm.on(*_REQUESTS), hsm.guard(_is_statement), hsm.target("/People/serving/working")),
                hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_refuse_invalid)),
            ),
            hsm.state(
                "working",
                hsm.entry(_begin),
                hsm.exit(_end),
                hsm.defer(*_REQUESTS),
                hsm.activity(_work),
                hsm.transition(
                    hsm.on(_DoneEvent),
                    hsm.guard(_is_current),
                    hsm.effect(_answer),
                    hsm.target("/People/serving/idle"),
                ),
            ),
        ),
    )


__all__ = [
    "ChangedData",
    "ChangedEvent",
    "FailedEvent",
    "FailureData",
    "FailureKind",
    "ForgetData",
    "ForgetEvent",
    "LinkData",
    "LinkEvent",
    "NameData",
    "NameEvent",
    "Op",
    "People",
    "UnlinkData",
    "UnlinkEvent",
]

"""``Consolidation``: what the bot does with its memory while it rests.

The owner (Cognition) sends ``RestEvent`` when the bot comes to rest and ``WakeEvent`` when anything
wakes it. While resting, Consolidation works through Memory by message only:

1. **backfill**: Memory indexes nodes that have no vector in its embedder's space, batch after
   batch, until none is left for this rest (mechanical; nodes that can never be indexed are left
   out by Memory).
2. **lease**: Memory leases the next unit, its oldest unreviewed episodes with their nearest nodes
   and the people they involve, named by unit aliases.
3. **decide**: the injected processor (a model) is shown the unit and offered exactly the
   model-callable events Memory declares (``bot.ability.memory.consolidate``). What to link,
   merge or forget, or whether to leave the unit as it is, is the model's decision; a forget is a
   hard delete. Selections are dispatched to Memory with this Consolidation as their authority,
   so Memory's reply comes back here, correlated by the decision id.
4. Memory applies a decision in one transaction; the unit is then reviewed and the next one is
   leased, until nothing unreviewed is left.

Waking, or detaching, at any point stops the rest: a model call still running is cancelled with
its typed cancel, and a unit not yet committed is given back unreviewed for a later rest. A
decision Memory is already applying still commits whole (Memory is atomic and does not observe
wake); one still waiting on its merged statements' embed is voided by that release, and nothing
of it is applied.

Memory takes one decision per lease. A model may select the decision more than once; every copy
carries the decision id, and Memory refuses each copy after the first ``stale_lease``. Such a
refusal can arrive before the first copy's outcome (while the first waits on its merge embed), so
``stale_lease`` is never read as the unit's outcome: it is counted as a stale reply, and the first
copy's own reply (or the commit timeout) settles the unit.

Bookkeeping is the only thing Consolidation chooses: which unit comes next (Memory's oldest
unreviewed) and whether a unit is reviewed. A unit whose decision cannot be made or applied
(a model or provider failure, a commit that timed out) is given back unreviewed and the rest
ends; when the same unit fails again the next time it is leased, it is released reviewed so
leasing moves past it rather than stalling every rest on it.

The owner is told what became of every unit with ``UnitOutcomeEvent`` (``UnitOutcomeData``).

Telemetry (scope ``bot.abilities.memory``; every attribute value comes from code):

* ``bot.memory.consolidation.unit.count`` — ``bot.memory.unit.outcome`` (5 values): 5 series.
* ``bot.memory.consolidation.change.count`` — ``bot.memory.change`` ∈ {link, merge, forget}: 3 series.
* ``bot.memory.rest.count`` — ``bot.memory.rest.transition`` ∈ {rest, wake}: 2 series.
* ``bot.memory.rest.duration`` — histogram, no attributes.
* ``bot.memory.consolidation.reply.dropped.count`` — replies for a unit already settled, no attributes.
* ``bot.ability.request.dropped.count`` — rest and wake notices that reached a detaching,
  degraded, or detached Consolidation.

The span ``bot.memory.consolidation.unit`` (component ``memory.consolidation``) covers one
decision and carries only the unit's node count.
"""

import collections.abc
import dataclasses
import datetime
import hashlib
import time
import typing
import uuid

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from mosfet import event
from mosfet.protocols import attachment
from mosfet.telemetry import span

from .. import ability, processing
from . import memory, payload

INSTRUCTIONS: typing.Final = (
    "You are resting, and this is one unit of your own memory: some episodes you have not reviewed yet, the "
    "nodes nearest them, and the people they involve. Every node is named by a ref such as 'n1'; the edges "
    "are the links that already exist between them.\n"
    "You can call bot.ability.memory.consolidate once with your decision about this unit, naming it by its "
    "unit id and its nodes only by their refs. A decision can link two nodes with a relation in your own "
    "words, merge several nodes into one statement in your own words (its sources can be forgotten with it), "
    "and forget episode or neighbor nodes. Forgetting deletes a node for good; people cannot be forgotten. "
    "Every part of a decision may be empty, and you may also call nothing and leave the unit as it is."
)
"""What the consolidating model is told: the unit and its capacities (link, merge, forget, or leave)."""

_DEFAULT_TIMEOUT: typing.Final = datetime.timedelta(minutes=2)

_METER = metrics.get_meter("bot.abilities.memory")
_UNITS = _METER.create_counter(
    "bot.memory.consolidation.unit.count",
    unit="{unit}",
    description="Consolidation units settled, by what became of them.",
)
_CHANGES = _METER.create_counter(
    "bot.memory.consolidation.change.count",
    unit="{change}",
    description="Links, merges, and forgets committed by consolidation decisions.",
)
_RESTS = _METER.create_counter(
    "bot.memory.rest.count",
    unit="{transition}",
    description="Rest and wake transitions of consolidation.",
)
_REST_DURATION = _METER.create_histogram(
    "bot.memory.rest.duration",
    unit="s",
    description="How long one rest lasted, from rest to wake.",
)
_STALE = _METER.create_counter(
    "bot.memory.consolidation.reply.dropped.count",
    unit="{reply}",
    description="Memory or processing replies that arrived after their unit was settled.",
)

RestEvent = hsm.Event[payload.RestData](name="bot.ability.memory.consolidation.rest", schema=payload.RestData)
"""The bot came to rest (from the owner); consolidation begins."""
WakeEvent = hsm.Event[payload.WakeData](name="bot.ability.memory.consolidation.wake", schema=payload.WakeData)
"""The bot woke (from the owner); consolidation stops."""
UnitOutcomeEvent = hsm.Event[payload.UnitOutcomeData](
    name="bot.ability.memory.consolidation.output", schema=payload.UnitOutcomeData
)
"""What became of one unit, sent to the owner."""

_Decided = typing.Literal["dispatched", "nothing", "failed"]


class _StepData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


class _LeasedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    unit: payload.UnitData | None = None
    failed: bool = False


class _DecidedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    outcome: _Decided


_BackfilledEvent = hsm.Event[_StepData](
    name="bot.ability.memory.consolidation.backfilled", kind=hsm.CompletionEventKind, schema=_StepData
)
_LeasedEvent = hsm.Event[_LeasedData](
    name="bot.ability.memory.consolidation.leased", kind=hsm.CompletionEventKind, schema=_LeasedData
)
_DecidedEvent = hsm.Event[_DecidedData](
    name="bot.ability.memory.consolidation.decided", kind=hsm.CompletionEventKind, schema=_DecidedData
)
_ReleasedEvent = hsm.Event[_StepData](
    name="bot.ability.memory.consolidation.released", kind=hsm.CompletionEventKind, schema=_StepData
)


@dataclasses.dataclass(frozen=True)
class _Decision:
    """The unit being decided (durable machine data from lease to settle)."""

    id: str
    unit: str
    # Identifies the unit's own episodes across leases (lease ids are new every time).
    fingerprint: str
    # How the unit is being released once settled: None while it is still undecided.
    reviewed: bool | None = None


def _fingerprint(unit: payload.UnitData) -> str:
    episodes = sorted((node.created_at.isoformat(), node.text) for node in unit.nodes if node.role == "episode")
    digest = hashlib.sha256()
    for created_at, words in episodes:
        digest.update(created_at.encode("utf-8"))
        digest.update(b"\0")
        digest.update(words.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class Consolidation(ability.Ability[payload.RestData, payload.UnitOutcomeData]):
    """Consolidate memory while the bot rests: backfill vectors, then decide unit after unit.

    ``memory`` is a peer ``Memory`` reached only by message. ``processor`` makes the decisions;
    it is wrapped in a child ``processing.Processing`` stamped with ``instructions``
    (default ``INSTRUCTIONS``). ``lease`` sizes a unit; ``backfill`` sizes a backfill batch
    (its ``pass_id`` is replaced with each rest's own). ``timeout`` bounds every request to
    Memory, one model decision, and the wait for a decision's commit.

    Raises:
        ValueError: if ``timeout`` is not positive, or ``instructions`` is blank.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = payload.RestData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = payload.UnitOutcomeData
    input_event: typing.ClassVar[hsm.Event[payload.RestData]] = RestEvent
    output_event: typing.ClassVar[hsm.Event[payload.UnitOutcomeData]] = UnitOutcomeEvent
    failed_event: typing.ClassVar[hsm.Event[typing.Any]] = ability.FailedEvent
    detached_notifications: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = (RestEvent, WakeEvent)
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True

    _attachment_group: attachment.Group | None
    _memory: memory.Memory
    _processing: processing.Processing
    _lease: payload.LeaseData
    _backfill: payload.BackfillData
    _timeout: datetime.timedelta
    # Durable machine data: this rest's pass id (backfill pass and step correlation), when it
    # began (telemetry only), the unit being decided, and the last unit whose decision failed.
    _pass_id: str
    _rested_at: float | None
    _decision: _Decision | None
    _failed_unit: str | None

    def __init__(
        self,
        *,
        memory: memory.Memory,
        processor: processing.Processor,
        instructions: str | None = None,
        lease: payload.LeaseData = payload.LeaseData(),
        backfill: payload.BackfillData = payload.BackfillData(limit=32, pass_id="rest"),
        timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
    ) -> None:
        if timeout <= datetime.timedelta(0):
            raise ValueError("Consolidation timeout must be positive.")
        super().__init__()
        self._memory = memory
        self._processing = processing.Processing(
            processor=processor, instructions=INSTRUCTIONS if instructions is None else instructions
        )
        self._attachment_group = attachment.Group(self._processing)
        self._lease = lease
        self._backfill = backfill
        self._timeout = timeout
        self._pass_id = ""
        self._rested_at = None
        self._decision = None
        self._failed_unit = None

    # --- outcomes ------------------------------------------------------------------------------

    @staticmethod
    def _emit(ctx: hsm.Context, instance: "Consolidation", unit: str | None, outcome: payload.UnitOutcome) -> None:
        """Tell the owner what became of a unit."""

        _UNITS.add(1, {"bot.memory.unit.outcome": outcome})
        terminal = dataclasses.replace(
            UnitOutcomeEvent.with_data(payload.UnitOutcomeData(unit=unit, outcome=outcome)),
            source=hsm.id(instance),
            target="",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _settle(ctx: hsm.Context, instance: "Consolidation", *, reviewed: bool, outcome: payload.UnitOutcome) -> None:
        """Settle the unit being decided: tell the owner, and record how it is to be released."""

        decision = instance._decision
        assert decision is not None
        instance._decision = dataclasses.replace(decision, reviewed=reviewed)
        Consolidation._emit(ctx, instance, decision.unit, outcome)

    @staticmethod
    def _leave_reviewed(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        """The model left the unit as it is: a decision; the unit is reviewed and not offered again."""

        Consolidation._forgive(ctx, instance, event)
        Consolidation._settle(ctx, instance, reviewed=True, outcome="reviewed")

    @staticmethod
    def _fail_reviewed(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        """The decision was refused, or the unit failed twice running: leave it reviewed, move on."""

        Consolidation._forgive(ctx, instance, event)
        Consolidation._settle(ctx, instance, reviewed=True, outcome="failed")

    @staticmethod
    def _fail_returned(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        """First failure of this unit: remember it and give the unit back for a later rest."""

        del event
        decision = instance._decision
        assert decision is not None
        instance._failed_unit = decision.fingerprint
        Consolidation._settle(ctx, instance, reviewed=False, outcome="failed")

    @staticmethod
    def _emit_exhausted(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _LeasedData)
        Consolidation._emit(ctx, instance, None, "failed" if data.failed else "exhausted")

    @staticmethod
    def _commit(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, payload.ConsolidatedData)
        for change, count in (("link", data.linked), ("merge", data.merged), ("forget", data.forgotten)):
            if count:
                _CHANGES.add(count, {"bot.memory.change": change})
        Consolidation._forgive(ctx, instance, event)
        Consolidation._emit(ctx, instance, data.unit, "committed")
        instance._decision = None

    @staticmethod
    def _forgive(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        """The unit that failed before is settled now: forget its failure."""

        del ctx, event
        decision = instance._decision
        if decision is not None and instance._failed_unit == decision.fingerprint:
            instance._failed_unit = None

    @staticmethod
    def _count_stale(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _STALE.add(1)

    @staticmethod
    def _drop_detaching(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        ability.Ability._count_dropped(instance, reason="detaching", replied=False)

    @staticmethod
    def _drop_degraded(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        ability.Ability._count_dropped(instance, reason="degraded", replied=False)

    # --- rest ----------------------------------------------------------------------------------

    @staticmethod
    def _begin_rest(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._pass_id = uuid.uuid4().hex
        instance._rested_at = time.monotonic()  # telemetry only
        _RESTS.add(1, {"bot.memory.rest.transition": "rest"})

    @staticmethod
    def _end_rest(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        """Leaving rest (wake, detach, or stop): a unit not yet decided is given back unreviewed.

        A unit already settled is released the way it was settled; if its release was already on
        its way, this second one finds no lease and is counted as a stale reply.
        """

        del event
        decision = instance._decision
        if decision is not None:
            peer = instance._memory
            _ = hsm.dispatch(
                ctx,
                peer,
                dataclasses.replace(
                    memory.ReleaseEvent.with_data_and_id(
                        payload.ReleaseData(unit=decision.unit, reviewed=bool(decision.reviewed)), uuid.uuid4().hex
                    ),
                    source=hsm.id(instance),
                    target=hsm.id(peer),
                ),
            )
            if decision.reviewed is None:
                Consolidation._emit(ctx, instance, decision.unit, "interrupted")
            instance._decision = None
        _RESTS.add(1, {"bot.memory.rest.transition": "wake"})
        if instance._rested_at is not None:
            _REST_DURATION.record(time.monotonic() - instance._rested_at)
            instance._rested_at = None

    # --- correlation ---------------------------------------------------------------------------

    @staticmethod
    def _is_this_pass(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return bool(instance._pass_id) and event.id == instance._pass_id

    @staticmethod
    def _has_unit(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        return (
            Consolidation._is_this_pass(ctx, instance, event)
            and isinstance(data, _LeasedData)
            and data.unit is not None
        )

    @staticmethod
    def _is_this_decision(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return instance._decision is not None and event.id == instance._decision.id

    @staticmethod
    def _decided(outcome: _Decided) -> collections.abc.Callable[..., bool]:
        def decided(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> bool:
            data = event.data
            return (
                Consolidation._is_this_decision(ctx, instance, event)
                and isinstance(data, _DecidedData)
                and data.outcome == outcome
            )

        decided.__name__ = decided.__qualname__ = f"decided_{outcome}"
        return decided

    @staticmethod
    def _is_committed(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> bool:
        return Consolidation._is_this_decision(ctx, instance, event) and isinstance(
            event.data, payload.ConsolidatedData
        )

    @staticmethod
    def _is_duplicate(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> bool:
        """Memory refused a copy of this decision because the unit is already decided by another copy."""

        data = event.data
        return (
            Consolidation._is_this_decision(ctx, instance, event)
            and isinstance(data, payload.FailureData)
            and data.kind is payload.FailureKind.STALE_LEASE
        )

    @staticmethod
    def _is_rejected(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> bool:
        """Memory refused the decision itself (it names nodes outside the unit, or forgets a person): not retried."""

        data = event.data
        return (
            Consolidation._is_this_decision(ctx, instance, event)
            and isinstance(data, payload.FailureData)
            and data.kind is payload.FailureKind.INVALID_INPUT
        )

    @staticmethod
    def _is_unapplied(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> bool:
        """Memory could not apply the decision (timeout, unavailable, store failure); nothing changed."""

        data = event.data
        return (
            Consolidation._is_this_decision(ctx, instance, event)
            and isinstance(data, payload.FailureData)
            and data.kind not in (payload.FailureKind.INVALID_INPUT, payload.FailureKind.STALE_LEASE)
        )

    @staticmethod
    def _is_repeated_failure(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        decision = instance._decision
        return decision is not None and instance._failed_unit == decision.fingerprint

    @staticmethod
    def _take_unit(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _LeasedData) and data.unit is not None
        instance._decision = _Decision(id=uuid.uuid4().hex, unit=data.unit.unit, fingerprint=_fingerprint(data.unit))

    @staticmethod
    def _settle_unit(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._decision = None

    @staticmethod
    def _timeout_delay(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._timeout

    # --- work ----------------------------------------------------------------------------------

    @staticmethod
    async def _ask(
        ctx: hsm.Context, instance: "Consolidation", request: hsm.Event[typing.Any], answer: hsm.Event[typing.Any]
    ) -> object | None:
        """Send one request to Memory by message; its answer's data, or None when it failed or timed out."""

        try:
            terminal = await ability.run_terminal_operation(
                ctx,
                child=instance._memory,
                request=request.with_data_and_id(request.data, uuid.uuid4().hex),
                terminals=(answer, memory.FailedEvent),
                timeout=instance._timeout,
            )
        except (TimeoutError, RuntimeError):
            return None
        data = terminal.data
        return None if isinstance(data, payload.FailureData) else data

    @staticmethod
    def _step(
        ctx: hsm.Context, instance: "Consolidation", step: hsm.Event[typing.Any], data: pydantic.BaseModel
    ) -> None:
        _ = instance.dispatch(ctx, step.with_data_and_id(data, instance._pass_id))

    @staticmethod
    async def _backfill_activity(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        """Index batch after batch until Memory reports nothing left for this pass, or a batch fails."""

        del event
        batch = instance._backfill.model_copy(update={"pass_id": instance._pass_id})
        while True:
            done = await Consolidation._ask(
                ctx, instance, memory.BackfillEvent.with_data(batch), memory.BackfilledEvent
            )
            if not isinstance(done, payload.BackfilledData) or not done.remaining:
                break
        Consolidation._step(ctx, instance, _BackfilledEvent, _StepData())

    @staticmethod
    async def _lease_activity(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        del event
        leased = await Consolidation._ask(
            ctx, instance, memory.LeaseEvent.with_data(instance._lease), memory.LeasedEvent
        )
        Consolidation._step(
            ctx,
            instance,
            _LeasedEvent,
            _LeasedData(unit=leased.unit) if isinstance(leased, payload.LeasedData) else _LeasedData(failed=True),
        )

    @staticmethod
    def _offered(peer: memory.Memory) -> tuple[hsm.Event[typing.Any], ...]:
        """The model-callable events Memory declares in its model: what a decision can be.

        Read from the declared model, never a live snapshot: Memory may be between requests when
        the frame is built, and whether a decision applies is Memory's own topology to answer.
        """

        model = type(peer).model
        assert model is not None
        return tuple(declared for declared in model.events.values() if declared.kind == event.EventKind)

    @staticmethod
    async def _decide_activity(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        """Show the model the unit; Processing dispatches its selection to Memory under this authority."""

        data = event.data
        decision = instance._decision
        assert isinstance(data, _LeasedData) and data.unit is not None and decision is not None
        unit = data.unit
        peer = instance._memory
        child = instance._processing
        offered = Consolidation._offered(peer)
        frame = processing.InputData(
            input=unit,
            schemas=offered,
            actors={"memory": peer},
            actor_events={declared.name: ("memory",) for declared in offered},
            authority=instance,
        )
        outcome: _Decided = "failed"
        with span.operation(
            "bot.memory.consolidation.unit",
            scope="bot.abilities.memory",
            component="memory.consolidation",
            stage="decide",
        ) as active:
            active.set_attribute("bot.memory.unit.nodes", len(unit.nodes))
            try:
                terminal = await ability.run_terminal_operation(
                    ctx,
                    child=child,
                    request=child.input_event.with_data_and_id(frame, decision.id),
                    terminals=(child.output_event, child.failed_event),
                    timeout=instance._timeout,
                )
            except TimeoutError:
                # The model is still working: stop it, so a late selection never lands.
                processing.dispatch_child_cancel(
                    ctx,
                    instance,
                    child,
                    event,
                    request_id=decision.id,
                    parent_operation_id=decision.id,
                    token=uuid.uuid4().hex,
                )
                span.record_failure(active, "timeout")
            except RuntimeError:
                span.record_failure(active, "unavailable")
            else:
                completed = terminal.data
                if isinstance(completed, processing.CompletionData):
                    output = completed.output
                    outcome = "dispatched" if output.handled and output.events else "nothing"
                else:
                    # The model or its provider could not produce a decision: a dependency not serving.
                    span.record_failure(active, "unavailable")
        _ = instance.dispatch(ctx, _DecidedEvent.with_data_and_id(_DecidedData(outcome=outcome), decision.id))

    @staticmethod
    def _abandon_decision(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        """Leaving ``deciding`` before the model answered (wake, detach, stop): cancel the model call."""

        decision = instance._decision
        if decision is None or isinstance(event.data, _DecidedData):
            return
        processing.dispatch_child_cancel(
            ctx,
            instance,
            instance._processing,
            event,
            request_id=decision.id,
            parent_operation_id=decision.id,
            token=uuid.uuid4().hex,
        )

    @staticmethod
    async def _release_activity(ctx: hsm.Context, instance: "Consolidation", event: hsm.Event[typing.Any]) -> None:
        """Give the settled unit back to Memory, reviewed or not as it was settled."""

        del event
        decision = instance._decision
        assert decision is not None and decision.reviewed is not None
        _ = await Consolidation._ask(
            ctx,
            instance,
            memory.ReleaseEvent.with_data(payload.ReleaseData(unit=decision.unit, reviewed=decision.reviewed)),
            memory.ReleasedEvent,
        )
        Consolidation._step(ctx, instance, _ReleasedEvent, _StepData())

    _STALE_REPLIES: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = (
        memory.ConsolidatedEvent,
        memory.FailedEvent,
        memory.ReleasedEvent,
        processing.CancelledEvent,
    )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Consolidation",
        hsm.initial(hsm.target("/Consolidation/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent, RestEvent, WakeEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Consolidation/awake"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                # In initializing the only composite operation is the attach, so its "detached"
                # terminal is the attach failure: the owner is told, and Consolidation fails closed.
                hsm.guard(ability.Ability._is_composite_detached_terminal),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Consolidation/degraded"),
            ),
        ),
        hsm.state(
            "awake",
            hsm.transition(hsm.on(RestEvent), hsm.target("/Consolidation/resting")),
            hsm.transition(hsm.on(*_STALE_REPLIES), hsm.effect(_count_stale)),
        ),
        hsm.state(
            "resting",
            hsm.entry(_begin_rest),
            hsm.exit(_end_rest),
            hsm.initial(hsm.target("/Consolidation/resting/backfilling")),
            hsm.transition(hsm.on(WakeEvent), hsm.target("/Consolidation/awake")),
            # Replies for a unit already settled (a second copy of one decision, a release sent on
            # wake) match nothing below; they are counted, never silently lost.
            hsm.transition(hsm.on(*_STALE_REPLIES), hsm.effect(_count_stale)),
            hsm.state(
                "backfilling",
                hsm.activity(_backfill_activity),
                hsm.transition(
                    hsm.on(_BackfilledEvent), hsm.guard(_is_this_pass), hsm.target("/Consolidation/resting/leasing")
                ),
            ),
            hsm.state(
                "leasing",
                hsm.activity(_lease_activity),
                hsm.transition(
                    hsm.on(_LeasedEvent),
                    hsm.guard(_has_unit),
                    hsm.effect(_take_unit),
                    hsm.target("/Consolidation/resting/deciding"),
                ),
                hsm.transition(
                    hsm.on(_LeasedEvent),
                    hsm.guard(_is_this_pass),
                    hsm.effect(_emit_exhausted),
                    hsm.target("/Consolidation/resting/settled"),
                ),
            ),
            hsm.state(
                "deciding",
                # Memory may answer the decision before Processing's own terminal gets here.
                hsm.defer(memory.ConsolidatedEvent, memory.FailedEvent),
                hsm.activity(_decide_activity),
                hsm.exit(_abandon_decision),
                hsm.transition(
                    hsm.on(_DecidedEvent),
                    hsm.guard(_decided("dispatched")),
                    hsm.target("/Consolidation/resting/committing"),
                ),
                hsm.transition(
                    hsm.on(_DecidedEvent),
                    # Leaving the unit as it is is a decision: it is reviewed and not offered again.
                    hsm.guard(_decided("nothing")),
                    hsm.effect(_leave_reviewed),
                    hsm.target("/Consolidation/resting/reviewing"),
                ),
                hsm.transition(
                    hsm.on(_DecidedEvent),
                    hsm.guard(_decided("failed")),
                    hsm.target("/Consolidation/resting/failing"),
                ),
            ),
            hsm.state(
                "committing",
                # A copy of this decision refused stale: the first copy's own reply settles the unit.
                hsm.transition(hsm.on(memory.FailedEvent), hsm.guard(_is_duplicate), hsm.effect(_count_stale)),
                hsm.transition(
                    hsm.on(memory.ConsolidatedEvent),
                    hsm.guard(_is_committed),
                    hsm.effect(_commit),
                    hsm.target("/Consolidation/resting/leasing"),
                ),
                hsm.transition(
                    hsm.on(memory.FailedEvent),
                    hsm.guard(_is_rejected),
                    hsm.effect(_fail_reviewed),
                    hsm.target("/Consolidation/resting/reviewing"),
                ),
                hsm.transition(
                    hsm.on(memory.FailedEvent),
                    hsm.guard(_is_unapplied),
                    hsm.target("/Consolidation/resting/failing"),
                ),
                hsm.transition(hsm.after(_timeout_delay), hsm.target("/Consolidation/resting/failing")),
            ),
            hsm.choice(
                "failing",
                # The same unit failed on its previous lease too: leave it reviewed and move on.
                hsm.transition(
                    hsm.guard(_is_repeated_failure),
                    hsm.effect(_fail_reviewed),
                    hsm.target("/Consolidation/resting/reviewing"),
                ),
                # First failure: give it back for a later rest, and end this one.
                hsm.transition(hsm.effect(_fail_returned), hsm.target("/Consolidation/resting/returning")),
            ),
            hsm.state(
                "reviewing",
                hsm.activity(_release_activity),
                hsm.transition(
                    hsm.on(_ReleasedEvent),
                    hsm.guard(_is_this_pass),
                    hsm.effect(_settle_unit),
                    hsm.target("/Consolidation/resting/leasing"),
                ),
            ),
            hsm.state(
                "returning",
                hsm.activity(_release_activity),
                hsm.transition(
                    hsm.on(_ReleasedEvent),
                    hsm.guard(_is_this_pass),
                    hsm.effect(_settle_unit),
                    hsm.target("/Consolidation/resting/settled"),
                ),
            ),
            # Nothing left for this rest; waits for wake.
            hsm.state("settled"),
        ),
        hsm.state(
            "detaching",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(hsm.on(RestEvent, WakeEvent), hsm.effect(_drop_detaching)),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Consolidation/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Consolidation/awake"),
            ),
        ),
        hsm.state(
            "degraded",
            hsm.transition(hsm.on(RestEvent, WakeEvent), hsm.effect(_drop_degraded)),
        ),
    )


__all__ = [
    "INSTRUCTIONS",
    "Consolidation",
    "RestEvent",
    "UnitOutcomeEvent",
    "WakeEvent",
]

"""The chart intuition and reasoning share: two tiers of the same kind of thinking at different depths.

``Deliberation`` is their one base: initializing → idle → applying → idle, with a bounded selection repair
through ``routing_rejection`` and what is still being written drafted to its targets as it streams. Each tier
declares only its own private events (:class:`Events`), its effort ceiling, whether it is offered the
deliberative handoff, and the span it reports under; :meth:`Deliberation.chart` builds its machine from those.
"""

from .. import ability
from ..ability import Effort
from .. import processing

import dataclasses
import typing
import uuid

import hsm
import mosfet

import pydantic

from mosfet import telemetry
from mosfet.telemetry import span

from . import input as cognition_input
from . import types


class InputData(pydantic.BaseModel):
    """Typed Cognition request for one deliberative stage: the turn and the host's frame for it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    processing_input: processing.InputData


class _Decided(typing.Protocol):
    """What a deliberative processor's answer holds: the selected events."""

    result: types.OutputData


class InitializingCompleteData(pydantic.BaseModel):
    """Typed empty signal that a deliberative stage finished initialization."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Empty initialization-complete signal for a deliberative stage; carries no payload.",
            "examples": [{}],
        },
    )


class Capability(pydantic.BaseModel):
    """JSON-safe authority minted for exactly one live deliberative operation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str
    actor_id: str
    token: str
    reply_to: str | None = None


class ReasonedData(pydantic.BaseModel):
    """Private completion: the processor decided this turn, and what it selected is dispatched.

    ``turn`` carries what this stage dispatched before handing the turn on, when it did (``acted``);
    ``output`` is None when it handed the turn on.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    capability: Capability
    output: types.OutputData | None


class StageFailedData(pydantic.BaseModel):
    """Private failure: the stage could not decide this turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    failure: ability.FailureData
    turn: types.TurnData
    capability: Capability


class SelectionRejectedData(pydantic.BaseModel):
    """Private retry context: the model's selection for this turn was rejected before delivery.

    ``message`` is the redacted diagnostic the model repairs from; the raw exception never crosses the HSM
    boundary, because Pydantic may include the rejected value in it.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    request: InputData = pydantic.Field(description="The request being repaired.")
    capability: Capability
    message: str = pydantic.Field(min_length=1, description="Redacted rejection diagnostic for model repair.")
    normalized: str = pydantic.Field(
        min_length=1, description="Stable rejection key used to detect the same rejection twice running."
    )
    previous_normalized: str | None = pydantic.Field(
        default=None, min_length=1, description="The key of the rejection just before this one, when any."
    )
    attempt: int = pydantic.Field(
        ge=0,
        le=types.MAX_SELECTION_REPAIRS,
        description="Zero-based bounded invocation attempt that produced this rejection.",
    )


@dataclasses.dataclass(frozen=True)
class Events:
    """One tier's own private events, named in its namespace (``bot.ability.<tier>.*``)."""

    initializing_complete: hsm.Event[InitializingCompleteData]
    reasoned: hsm.Event[ReasonedData]
    stage_failed: hsm.Event[StageFailedData]
    selection_rejected: hsm.Event[SelectionRejectedData]

    @classmethod
    def named(cls, tier: str) -> "Events":
        """The private events of the tier named ``tier``."""

        return cls(
            initializing_complete=hsm.Event[InitializingCompleteData](
                name=f"bot.ability.{tier}.initializing.complete",
                kind=hsm.CompletionEventKind,
                schema=InitializingCompleteData,
            ),
            reasoned=hsm.Event[ReasonedData](
                name=f"bot.ability.{tier}.reasoned",
                kind=hsm.CompletionEventKind,
                schema=ReasonedData,
            ),
            stage_failed=hsm.Event[StageFailedData](
                name=f"bot.ability.{tier}.stage.failed",
                kind=hsm.ErrorEventKind,
                schema=StageFailedData,
            ),
            selection_rejected=hsm.Event[SelectionRejectedData](
                name=f"bot.ability.{tier}.selection.rejected",
                kind=hsm.ErrorEventKind,
                schema=SelectionRejectedData,
            ),
        )


@dataclasses.dataclass(frozen=True)
class Spans:
    """The span one tier reports its steps under: a constant name, component, two stages and its held count."""

    name: str
    component: str
    initialize: str
    reason: str
    held: str


def _selections_from_output(output: types.OutputData) -> processing.Events:
    return tuple(
        processing.SelectedEvent(
            event=item.event,
            target=item.target,
            data=item.data,
            reason=item.reason,
        )
        for item in output
    )


class Deliberation(processing.Processing):
    """A deliberative tier: inject a ``processing.Processor`` and apply it to the host's frame for the turn.

    Chart phases: initializing → idle → applying → idle. A rejected selection (one the processor refuses, or
    one dispatch refuses) passes through ``routing_rejection``: the model is asked again with the diagnostic,
    at most ``types.MAX_SELECTION_REPAIRS`` times, and the same rejection twice running fails the turn.
    Turn payload rides private completion events (HSM-COMPLETION-001). What memory surfaced for the turn
    arrives on the host frame as ``recalled`` and is passed on to the processor unchanged. The processor is not
    an HSM child; the stage calls ``process`` in-activity.

    A tier offered the deliberative handoff (``bot.ability.reasoning.input``) is told, in ``<held>``, what is
    above its ceiling, and selecting the handoff hands the turn on: its other selections are dispatched first
    and ride the turn's ``acted`` to the next stage, and its terminal leaves the turn unhandled.
    """

    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = types.CompletionData
    failed_event: typing.ClassVar[hsm.Event[types.FailureData]] = hsm.Event[types.FailureData](
        name=ability.FailedEvent.name,
        kind=hsm.ErrorEventKind,
        schema=types.FailureData,
    )
    # Declared by each tier.
    _events: typing.ClassVar[Events]
    _spans: typing.ClassVar[Spans]
    _request: typing.ClassVar[type[InputData]]
    _processor_input: typing.ClassVar[type[pydantic.BaseModel]]
    _decision: typing.ClassVar[type[pydantic.BaseModel]]
    _offers_handoff: typing.ClassVar[bool]
    _processor: processing.Processor
    _instructions: str
    _max_effort: Effort | None

    def __init__(
        self,
        *,
        id: ability.AbilityId | None,
        clock: hsm.Clock | None,
        processor: processing.Processor,
        instructions: str | None,
        max_effort: Effort | None,
    ) -> None:
        resolved = type(self).instructions if instructions is None else instructions
        if instructions is not None and not instructions.strip():
            raise ValueError("instructions must not be blank when provided.")
        # Empty default is intentional: system channel is only the per-turn world XML when present.
        # Composition host: leaf Processor is transport; any static policy is ability instructions.
        super(processing.Processing, self).__init__(id=id, clock=clock)
        self._instructions = resolved.strip() if resolved else ""
        self._processor = processor
        self._max_effort = max_effort

    @staticmethod
    async def _initialize_activity(
        ctx: hsm.Context,
        instance: "Deliberation",
        event: hsm.Event[typing.Any],
    ) -> None:
        spans = type(instance)._spans
        with span.operation(
            spans.name,
            scope="bot.abilities.cognition",
            component=spans.component,
            stage=spans.initialize,
            context=telemetry.event_context(event),
        ):
            del event
            _ = hsm.dispatch(
                ctx, instance, type(instance)._events.initializing_complete.with_data(InitializingCompleteData())
            )

    @staticmethod
    def _has_request(ctx: hsm.Context, instance: "Deliberation", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, type(instance)._request)

    @staticmethod
    def _matches_operation(instance: "Deliberation", event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        capability = (
            data.capability if isinstance(data, ReasonedData | StageFailedData | SelectionRejectedData) else None
        )
        if not isinstance(capability, Capability):
            return False
        return (
            event.id == capability.operation_id
            and event.source == capability.actor_id
            and event.target == hsm.id(instance)
            and processing.matches_operation(instance, capability.operation_id, capability.actor_id)
        )

    @staticmethod
    def _has_reasoned(ctx: hsm.Context, instance: "Deliberation", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, ReasonedData) and Deliberation._matches_operation(instance, event)

    @staticmethod
    def _has_stage_failure(ctx: hsm.Context, instance: "Deliberation", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, StageFailedData) and Deliberation._matches_operation(instance, event)

    @staticmethod
    def _has_selection_rejection(ctx: hsm.Context, instance: "Deliberation", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, SelectionRejectedData) and Deliberation._matches_operation(instance, event)

    @staticmethod
    def _is_same_selection_rejection(ctx: hsm.Context, instance: "Deliberation", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, SelectionRejectedData) and data.previous_normalized == data.normalized

    @staticmethod
    def _can_repair_selection(ctx: hsm.Context, instance: "Deliberation", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, SelectionRejectedData) and data.attempt < types.MAX_SELECTION_REPAIRS

    @staticmethod
    async def _start_operation(
        instance: "Deliberation",
        operation_id: str,
        *,
        reply_to: str | None,
    ) -> Capability:
        operation = processing.active_operation(instance, operation_id)
        if operation is None:
            operation = await processing.start_operation(instance, operation_id)
        return Capability(
            operation_id=operation_id,
            actor_id=hsm.id(operation),
            token=uuid.uuid4().hex,
            reply_to=reply_to,
        )

    @staticmethod
    def _cancel(ctx: hsm.Context, instance: "Deliberation", event: hsm.Event[typing.Any]) -> None:
        processing.Processing._emit_cancelled(ctx, instance, event)

    @staticmethod
    def _dispatch_terminal(
        ctx: hsm.Context,
        instance: "Deliberation",
        terminal: hsm.Event[typing.Any],
        *,
        operation_id: str | None,
    ) -> None:
        wrapped = (
            ability.TerminalErrorEvent.with_data(terminal)
            if isinstance(terminal.data, types.FailureData)
            else ability.TerminalOutputEvent.with_data(terminal)
        )
        _ = hsm.dispatch(ctx, instance, wrapped)
        if operation_id is not None:
            processing.finish_operation(ctx, instance, operation_id)

    @staticmethod
    def _fail(
        ctx: hsm.Context,
        instance: "Deliberation",
        event: hsm.Event[typing.Any],
        failure: types.FailureData,
        reply_to: str | None,
    ) -> None:
        Deliberation._dispatch_terminal(
            ctx,
            instance,
            dataclasses.replace(
                instance.failed_event.with_data(failure),
                id=event.id or None,
                metadata=dict(event.metadata),
                source=hsm.id(instance),
                target=reply_to,
            ),
            operation_id=event.id or None,
        )

    @staticmethod
    def _fail_from_stage(ctx: hsm.Context, instance: "Deliberation", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, StageFailedData)
        Deliberation._fail(
            ctx,
            instance,
            event,
            types.FailureData(message=data.failure.message, turn=data.turn),
            data.capability.reply_to,
        )

    @staticmethod
    def _fail_selection_rejection(ctx: hsm.Context, instance: "Deliberation", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, SelectionRejectedData)
        Deliberation._fail(
            ctx,
            instance,
            event,
            types.FailureData(message=data.message, turn=data.request.turn),
            data.capability.reply_to,
        )

    @staticmethod
    def _complete_from_reasoned(ctx: hsm.Context, instance: "Deliberation", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ReasonedData)
        Deliberation._dispatch_terminal(
            ctx,
            instance,
            dataclasses.replace(
                instance.output_event.with_data(types.CompletionData(turn=data.turn, output=data.output)),
                id=event.id or None,
                metadata=dict(event.metadata),
                source=hsm.id(instance),
                target=data.capability.reply_to,
            ),
            operation_id=event.id or None,
        )

    @staticmethod
    def _decided(instance: "Deliberation", value: object) -> types.OutputData:
        """The events the processor's answer selects, whichever shape it answered in."""

        if isinstance(value, type(instance)._decision):
            return typing.cast(_Decided, typing.cast(object, value)).result
        if isinstance(value, processing.Result):
            result = typing.cast(processing.Result[object], value)
            return Deliberation._decided(instance, result.output) if result.is_handled else ()
        selections = processing.coerce_event_selections(value)
        if selections is None:
            raise TypeError(f"{type(instance).__name__} produced output that does not match its output schema.")
        return types.OUTPUT_SCHEMA_CONTRACT.validate_python(
            tuple(
                {
                    "event": item.event,
                    "target": item.target,
                    "data": item.data,
                    "reason": item.reason,
                }
                for item in selections
            )
        )

    @staticmethod
    async def _reason_activity(ctx: hsm.Context, instance: "Deliberation", event: hsm.Event[typing.Any]) -> None:
        tier = type(instance)
        with span.operation(
            tier._spans.name,
            scope="bot.abilities.cognition",
            component=tier._spans.component,
            stage=tier._spans.reason,
            context=telemetry.event_context(event),
        ) as active:
            # A repair re-enters applying with the rejection, which carries the request and the live capability.
            retry = event.data if isinstance(event.data, SelectionRejectedData) else None
            attempt = retry.attempt + 1 if retry is not None else 0
            previous_normalized = retry.normalized if retry is not None else None
            if retry is not None:
                capability = retry.capability
                operation_id = capability.operation_id
            else:
                operation_id = event.id if event.id else uuid.uuid4().hex
                capability = await Deliberation._start_operation(
                    instance,
                    operation_id,
                    reply_to=event.source if event.target == hsm.id(instance) and event.source else None,
                )
            active.set_attribute("bot.selection.attempt", attempt)
            metadata = dict(event.metadata)

            def dispatch_stage(event_type: hsm.Event[typing.Any], data: object) -> None:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        event_type.with_data(data),
                        id=capability.operation_id,
                        source=capability.actor_id,
                        target=hsm.id(instance),
                        metadata=dict(metadata),
                    ),
                )

            def reject(error: processing.SelectionRejectionError) -> None:
                active.set_attribute("bot.selection.rejected", True)
                dispatch_stage(
                    tier._events.selection_rejected,
                    SelectionRejectedData(
                        request=request,
                        capability=capability,
                        message=types.selection_rejection_message(error.normalized),
                        normalized=types.selection_rejection_key(error.normalized),
                        previous_normalized=previous_normalized,
                        attempt=attempt,
                    ),
                )

            def fail(error: Exception) -> None:
                dispatch_stage(
                    tier._events.stage_failed,
                    StageFailedData(
                        failure=ability.FailureData(message=str(error)), turn=request.turn, capability=capability
                    ),
                )

            if retry is not None:
                request = retry.request
            else:
                data = event.data
                if not isinstance(data, tier._request):
                    raise TypeError(f"{tier.__name__} input must be {tier._request.__name__}, got {type(data)!r}.")
                request = data
            try:
                # The host's frame narrowed to this stage's effort ceiling. Narrowing reads live actor
                # snapshots, so a frame that cannot be read fails the turn.
                narrowed = cognition_input.within_effort(request.processing_input, instance._max_effort)
                if tier._offers_handoff:
                    # What only a later tier holds is named, reached by the handoff this tier is offered.
                    held = cognition_input.held(request.processing_input, narrowed)
                    active.set_attribute(tier._spans.held, len(held))
                    host_input = cognition_input.with_held(
                        narrowed, held, reached_by=types.DELIBERATIVE_HANDOFF_EVENT_NAME
                    )
                else:
                    # The handoff's recipient never offers the handoff to itself.
                    host_input = cognition_input.without_handoff(narrowed)
                # Keep host tools/actors, the host's live instructions (device-state block), and what
                # memory surfaced for this turn; nest the stage's payload as the process input.
                child_input = processing.Processing._input_for_processor(
                    instance,
                    processing.InputData(
                        input=tier._processor_input(host_input=host_input),
                        schemas=host_input.schemas,
                        actors=host_input.actors,
                        instructions=host_input.instructions,
                        recalled=host_input.recalled,
                    ),
                )
                if retry is not None:
                    child_input = types.with_selection_feedback(child_input, retry.message)
                # Selections are dispatched as returned, so what the stage is still writing may be drafted to
                # its targets as it streams; the drafts carry this operation's id, as the selection will.
                drafts = (
                    processing.Processing._draft_sink(
                        ctx, instance, host_input, operation_id=operation_id, metadata=dict(metadata)
                    )
                    if host_input.actors
                    else None
                )
                raw = await instance._processor.process(child_input, drafts=drafts)
                result = Deliberation._decided(instance, raw)
            except processing.SelectionRejectionError as error:
                # The processor refused the selection (it did not decode, or fails its schema): the model's to repair.
                reject(error)
                return
            except Exception as error:
                fail(error)
                return
            # A tier offered the handoff hands the turn on by selecting it, after its other selections run. A tier
            # never offered it dispatches its result as it is, so dispatch rejects the handoff like any event the
            # turn does not offer, and the model repairs it.
            handed_on = tier._offers_handoff and any(
                types.is_deliberative_handoff_event_name(item.event) for item in result
            )
            selected = (
                tuple(item for item in result if not types.is_deliberative_handoff_event_name(item.event))
                if tier._offers_handoff
                else result
            )
            if tier._offers_handoff:
                active.set_attribute("bot.cognition.escalated", handed_on)
            # Only what reaches an actor was done; with no actor on the frame, nothing was.
            dispatched = selected if host_input.actors else ()
            if dispatched:
                try:
                    await types.dispatch_selected_events(
                        ctx,
                        host_input,
                        _selections_from_output(dispatched),
                        operation_id=operation_id,
                        source=instance,
                        focus_candidates=request.turn.input.focus_candidates,
                        metadata=dict(metadata),
                        dispatch_trust=processing.DispatchTrust.MODEL,
                    )
                except processing.SelectionRejectionError as error:
                    # Every selection is validated before any is delivered, so a rejection delivered nothing.
                    reject(error)
                    return
                except Exception as error:
                    fail(error)
                    return
            dispatch_stage(
                tier._events.reasoned,
                ReasonedData(
                    turn=types.with_acts(request.turn, dispatched) if handed_on else request.turn,
                    capability=capability,
                    output=None if handed_on else result,
                ),
            )

    @staticmethod
    def chart(name: str, input_event: hsm.Event[typing.Any], events: Events) -> hsm.Model:
        """The machine of the deliberative tier named ``name``, taking ``input_event`` and sending ``events``."""

        root = f"/{name}"
        return mosfet.define(
            name,
            hsm.initial(hsm.target(f"{root}/initializing")),
            hsm.state(
                "initializing",
                hsm.defer(input_event),
                hsm.activity(Deliberation._initialize_activity),
                hsm.transition(
                    hsm.on(events.initializing_complete),
                    hsm.target(f"{root}/idle"),
                ),
            ),
            hsm.state(
                "idle",
                hsm.transition(
                    hsm.on(processing.CancelEvent),
                    hsm.guard(processing.Processing._is_cancel_request),
                    hsm.effect(Deliberation._cancel),
                ),
                hsm.transition(
                    hsm.on(input_event),
                    hsm.guard(Deliberation._has_request),
                    hsm.target(f"{root}/applying"),
                ),
            ),
            hsm.state(
                "applying",
                hsm.defer(input_event),
                hsm.activity(Deliberation._reason_activity),
                hsm.transition(
                    hsm.on(processing.CancelEvent),
                    hsm.guard(processing.Processing._is_cancel_request),
                    hsm.effect(Deliberation._cancel),
                    hsm.target(f"{root}/idle"),
                ),
                hsm.transition(
                    hsm.on(events.reasoned),
                    hsm.guard(Deliberation._has_reasoned),
                    hsm.effect(Deliberation._complete_from_reasoned),
                    hsm.target(f"{root}/idle"),
                ),
                hsm.transition(
                    hsm.on(events.stage_failed),
                    hsm.guard(Deliberation._has_stage_failure),
                    hsm.effect(Deliberation._fail_from_stage),
                    hsm.target(f"{root}/idle"),
                ),
                hsm.transition(
                    hsm.on(events.selection_rejected),
                    hsm.guard(Deliberation._has_selection_rejection),
                    hsm.target(f"{root}/routing_rejection"),
                ),
            ),
            hsm.choice(
                "routing_rejection",
                hsm.transition(
                    hsm.guard(Deliberation._is_same_selection_rejection),
                    hsm.effect(Deliberation._fail_selection_rejection),
                    hsm.target(f"{root}/idle"),
                ),
                hsm.transition(
                    hsm.guard(Deliberation._can_repair_selection),
                    hsm.target(f"{root}/applying"),
                ),
                hsm.transition(
                    hsm.effect(Deliberation._fail_selection_rejection),
                    hsm.target(f"{root}/idle"),
                ),
            ),
        )


__all__ = [
    "Deliberation",
    "Events",
    "InputData",
    "Spans",
]

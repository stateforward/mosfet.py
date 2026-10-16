from .. import ability
from .. import memory
from .. import persona
from .. import processing

import dataclasses
import datetime
import logging
import typing
import uuid

import hsm
import mosfet
from mosfet.define import define

from mosfet.protocols import attachment
import pydantic

from mosfet import media, telemetry
from mosfet.event import event_json_value
from mosfet.telemetry import span

from mosfet import behavior

from .input import InputData, is_input
from . import autonomy
from . import input

# ``event`` names every callback's event parameter here, so the projection module goes by its role.
from . import event as projection
from . import intuition
from . import reasoning
from . import reflection
from . import routines
from . import types
from .types import OUTPUT_SCHEMA_CONTRACT, EventData, OutputData, is_output

# Private event-chain key: carries this turn's cognition input with child requests.
# Not a metric/span attribute; not stored on the cognition instance (HSM-COMPLETION-001).
_AUTONOMY_ID_SUFFIX = ":autonomy"
_ASSOCIATE_ID_SUFFIX = ":associate"
_INTUITION_ID_SUFFIX = ":intuition"
_REASONING_ID_SUFFIX = ":reasoning"
_REFLECTION_ID_SUFFIX = ":reflection"
_REMEMBER_ID_SUFFIX = ":remember"
_CHILD_OPERATION_TIMEOUT = datetime.timedelta(seconds=30)
# How long cognition stays idle before it rests (and consolidation may run): a capacity of the
# composition, like the processing timeout, not a choice the bot makes.
_DEFAULT_REST_AFTER: typing.Final = datetime.timedelta(minutes=10)
_CANCEL_TEARDOWN_TIMEOUT = datetime.timedelta(seconds=5)
# What memory keeps of a stimulus's images: at most one turn's worth, of every accepted media type.
_MEMORY_IMAGES: typing.Final = media.ImageCapacity(
    max_images=media.MAX_TURN_IMAGES,
    max_total_bytes=media.MAX_TURN_IMAGES * media.MAX_IMAGE_BYTES,
    media_types=frozenset(typing.get_args(media.ImageMediaType)),
)
# Bounds of what one remembered act keeps (memory.Act's own limits).
_MAX_ACTS: typing.Final = 32
_MAX_ACT_REASON: typing.Final = 4000

_LOG = logging.getLogger(__name__)


class CancelData(pydantic.BaseModel):
    """Request cancellation of one active cognition turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(
        min_length=1,
        description="Bot turn identifier whose active cognition work must be cancelled.",
        examples=["turn-123"],
    )
    token: str = pydantic.Field(
        min_length=1,
        description="Opaque cancellation capability created by the Bot operation actor.",
        examples=["8d72b83f17654f1788c012ab132b4afd"],
    )


class CancelledData(pydantic.BaseModel):
    """Confirmation that one cognition turn no longer owns active child work."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(
        min_length=1,
        description="Bot turn identifier whose cognition work reached the idle boundary.",
        examples=["turn-123"],
    )
    token: str = pydantic.Field(
        min_length=1,
        description="Exact cancellation capability that reached the idle boundary.",
        examples=["8d72b83f17654f1788c012ab132b4afd"],
    )


class _UseFailedEventData(pydantic.BaseModel):
    """Private failure payload carrying failure details and operation identity."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    failure: ability.FailureData = pydantic.Field(
        description="FailureData signal produced while the cognition decision was active.",
    )
    operation_id: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Private apply-operation identifier from the input event.",
    )


InputEvent = hsm.Event[InputData](
    name="bot.ability.cognition.input",
    schema=InputData,
)
OutputEvent = hsm.Event[OutputData](
    name="bot.ability.cognition.output",
    schema=OUTPUT_SCHEMA_CONTRACT,
)
CancelEvent = hsm.Event[CancelData](
    name="bot.ability.cognition.cancel",
    schema=CancelData,
)
CancelledEvent = hsm.Event[CancelledData](
    name="bot.ability.cognition.cancelled",
    kind=hsm.CompletionEventKind,
    schema=CancelledData,
)
_CancelPendingEvent = hsm.Event[CancelData](
    name="bot.ability.cognition.cancel.pending",
    schema=CancelData,
)
_ApplyFailedEvent = hsm.Event[_UseFailedEventData](
    name="bot.ability.cognition.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=_UseFailedEventData,
)
_ChildHopTimedOutEvent = hsm.Event[object](
    name="bot.ability.cognition.child.hop.timed_out",
    kind=hsm.ErrorEventKind,
    schema=object,
)


class _AssociatedData(pydantic.BaseModel):
    """Private completion: what memory surfaced for the turn now rides it into intuition."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    turn: types.TurnData


_AssociatedEvent = hsm.Event[_AssociatedData](
    name="bot.ability.cognition.associated",
    kind=hsm.CompletionEventKind,
    schema=_AssociatedData,
)


class _TurnOpenedData(pydantic.BaseModel):
    """Private: the input of the turn Cognition just took up, sent to itself under the turn's id.

    ``processing`` defers it for the whole turn, so it reaches whichever state the turn ends in:
    ``cancelling`` remembers the turn as cancelled and ``rebooting`` as failed, each with what was
    said and who took part. A turn that completes or fails is remembered from its own terminal,
    which carries the turn, so ``ready`` lets this event fall through.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    input: InputData


_TurnOpenedEvent = hsm.Event[_TurnOpenedData](
    name="bot.ability.cognition.turn.opened",
    kind=hsm.CompletionEventKind,
    schema=_TurnOpenedData,
)


def _failure(operation_id: str | None, message: str) -> _UseFailedEventData:
    return _UseFailedEventData(
        failure=ability.FailureData(message=message),
        operation_id=operation_id,
    )


def _coerce_output(output: object) -> OutputData:
    if isinstance(output, processing.Result):
        result = typing.cast(processing.Result[object], output)
        if not result.is_handled:
            return ()
        return _coerce_output(result.output)
    if output is None:
        return ()
    if is_output(output):
        return output
    if isinstance(output, EventData):
        return (output,)
    selections = processing.coerce_event_selections(output)
    if selections is not None:
        validated = OUTPUT_SCHEMA_CONTRACT.validate_python(
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
        return validated
    if isinstance(output, list | tuple):
        values = typing.cast(list[object] | tuple[object, ...], output)
        validated = OUTPUT_SCHEMA_CONTRACT.validate_python(tuple(values))
        return validated
    if isinstance(output, dict) and "event" in output:
        validated = OUTPUT_SCHEMA_CONTRACT.validate_python((output,))
        return validated
    raise TypeError("Cognition processing produced output that does not match its output schema.")


def _act(selection: EventData) -> memory.Act:
    """One selection as memory keeps it: event, target, JSON data, and the reason (bounded)."""

    data = event_json_value(selection.data) if selection.data is not None else {}
    return memory.Act(
        event=selection.event,
        target=selection.target,
        data=typing.cast(dict[str, pydantic.JsonValue], data)
        if isinstance(data, dict)
        else {"value": typing.cast(pydantic.JsonValue, data)},
        reason=None if selection.reason is None else selection.reason[:_MAX_ACT_REASON],
    )


def _public_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Metadata safe to forward on host apply / terminals (no turn payload)."""

    return dict(metadata)


def _parent_operation_id(event: hsm.Event[typing.Any]) -> str | None:
    child_id = event.id if event.id else None
    if child_id is None:
        return None
    for suffix in (_AUTONOMY_ID_SUFFIX, _ASSOCIATE_ID_SUFFIX, _INTUITION_ID_SUFFIX, _REASONING_ID_SUFFIX):
        if child_id.endswith(suffix):
            parent = child_id[: -len(suffix)]
            return parent or None
    return child_id


class Cognition(ability.Ability[InputData, OutputData]):
    """Cognitive ability: autonomy → associating → intuition → reasoning, then reflection.

    Chart states are lifecycle phases only. The active turn rides the child request
    and terminal event chain, not instance fields (HSM-COMPLETION-001).

    With ``memory`` composed, Cognition attaches it and remembers every turn it settles —
    completed, failed, or cancelled — as one ``memory.RememberEvent`` it does not wait on. With
    ``associative`` composed too, a turn autonomy leaves unhandled first asks it what memory
    associates with the stimulus; the associations ride the turn to intuition and reasoning (as
    their frames' ``recalled`` context) and to reflection. An association that fails or times out
    leaves the turn without associations; it never fails the turn.

    Every frame Cognition builds also holds ``associative`` (under ``memory``), so a stage whose
    effort ceiling admits it (Reasoning, not Intuition) is offered ``bot.ability.memory.recall``.
    What a recall brings to mind (``memory.RecollectedEvent``), or its typed failure, comes back to
    Cognition and goes to the body as a ``cognition.InputEvent`` observing it: a new turn the bot
    decides on, the way a routine tick is.

    With ``consolidation`` composed too, Cognition rests once it has been idle for ``rest_after``:
    it sends ``consolidation`` a ``memory.RestEvent``, and the first cognition input wakes it with
    ``memory.WakeEvent``.

    With ``routines`` composed, Reflection settling a turn sends Routines a reconcile (its
    inventory writes may have created, changed, or broken a persistent routine), and every
    routine tick Routines forwards goes to the body as a ``cognition.InputEvent`` observing
    ``bot.behavior.tick``: a new turn the bot decides on, never a hardcoded reaction.

    With ``persona`` composed, Cognition attaches it and keeps what it last told: the traits the
    bot's owner gave it and the description it wrote of itself (``persona.OutputEvent``, on waking
    and after every revision written down). Every frame Cognition builds shows it after the world
    block as the self block (``SelfData``), so the bot knows who it is on every turn. Cognition only
    keeps what it was told; it never revises the persona.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    _attachment_group: attachment.Group | None
    input_event: typing.ClassVar[hsm.Event[InputData]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[OutputData]] = OutputEvent
    cancel_event: typing.ClassVar[hsm.Event[typing.Any] | None] = CancelEvent
    cancelled_event: typing.ClassVar[hsm.Event[typing.Any] | None] = CancelledEvent
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    _autonomy: autonomy.Autonomy | None
    _intuition: intuition.Intuition
    _reasoning: reasoning.Reasoning
    _reflection: reflection.Reflection
    _routines: routines.Routines | None
    _memory: memory.Memory | None
    _associative: memory.Associative | None
    _consolidation: memory.Consolidation | None
    _persona: persona.Persona | None
    # What persona last told Cognition, by event: durable self-knowledge, not turn scratch.
    _persona_told: persona.OutputData | None
    _rest_after: datetime.timedelta
    # Injected operation bounds (None = live module default, so tests may still tune
    # the module constants around construction). Never hardcode per-caller durations.
    _child_operation_timeout: datetime.timedelta | None
    _cancel_teardown_timeout: datetime.timedelta | None

    @staticmethod
    def _matches_child_terminal(
        event: hsm.Event[typing.Any],
        *,
        owner: "Cognition",
        child: ability.Ability[typing.Any, typing.Any],
        suffix: str,
        data_type: type[types.CompletionData] | type[types.FailureData],
    ) -> bool:
        data = event.data
        if not isinstance(data, data_type):
            return False
        operation = processing.active_operation(owner, data.turn.operation_id)
        # The enclosing stage topology owns phase identity. Envelope id + generation + source
        # correlate the turn (HSM-DELIVERY-001) — not event.name after topology selected it.
        return (
            event.source == hsm.id(child)
            and event.target == hsm.id(owner)
            and operation is not None
            and data.turn.generation == hsm.id(operation)
            and event.id == f"{data.turn.operation_id}{suffix}"
        )

    @staticmethod
    def _autonomy_operation_id(event: hsm.Event[typing.Any]) -> str:
        operation_id = event.id if event.id else uuid.uuid4().hex
        return f"{operation_id}{_AUTONOMY_ID_SUFFIX}"

    @staticmethod
    def _intuition_operation_id(event: hsm.Event[typing.Any]) -> str:
        operation_id = event.id if event.id else uuid.uuid4().hex
        return f"{operation_id}{_INTUITION_ID_SUFFIX}"

    @staticmethod
    def _reasoning_operation_id(operation_id: str | None) -> str:
        base = operation_id if operation_id else uuid.uuid4().hex
        return f"{base}{_REASONING_ID_SUFFIX}"

    @staticmethod
    def _reflection_operation_id(operation_id: str | None) -> str:
        base = operation_id if operation_id else uuid.uuid4().hex
        return f"{base}{_REFLECTION_ID_SUFFIX}"

    @staticmethod
    def _build_processing_input(
        instance: "Cognition",
        cognition_input: InputData,
        *,
        recalled: memory.AssociationsData | None = None,
    ) -> processing.InputData:
        associative = instance._associative
        told = instance._persona_told
        frame = input.build_processing_input(
            cognition_input,
            # Deliberate recall: offered to a stage whose effort ceiling admits the associative.
            extra_actors=None if associative is None else {"memory": associative},
            authority=instance,
            known=None if told is None else input.SelfData(persona=told),
        )
        return frame if recalled is None else frame.model_copy(update={"recalled": recalled})

    @staticmethod
    def _turn(
        instance: "Cognition",
        cognition_input: InputData,
        operation_id: str,
        *,
        associations: memory.AssociationsData | None = None,
    ) -> types.TurnData:
        operation = processing.active_operation(instance, operation_id)
        assert operation is not None
        return types.TurnData(
            input=cognition_input,
            operation_id=operation_id,
            generation=hsm.id(operation),
            associations=associations,
        )

    @staticmethod
    def _remember(
        ctx: hsm.Context,
        instance: "Cognition",
        *,
        operation_id: str,
        outcome: memory.Outcome,
        metadata: dict[str, object],
        cognition_input: InputData,
        output: OutputData = (),
    ) -> None:
        """Send memory the turn just settled, stamped here, where the turn is owned; never waited on.

        Every outcome carries the turn's input: a completed or failed turn from its terminal, a
        cancelled or rebooted one from its ``_TurnOpenedEvent``. Memory's reply is not awaited:
        Memory counts its own outcome, and no transition here consumes it.
        """

        store = instance._memory
        if store is None:
            return
        try:
            episode = memory.EpisodeData(
                turn_id=operation_id,
                stimulus=input.stimulus_name(cognition_input.stimulus),
                parts=projection.memory_content(
                    cognition_input.stimulus, capacity=_MEMORY_IMAGES, budget_bytes=memory.MAX_EPISODE_TEXT_BYTES
                ),
                participants=cognition_input.participants,
                focus=cognition_input.focus,
                focus_candidates=cognition_input.focus_candidates[:_MAX_ACTS],
                acts=tuple(_act(selection) for selection in output[:_MAX_ACTS]),
                outcome=outcome,
            )
        except pydantic.ValidationError as error:
            _LOG.warning("cognition did not remember a turn: invalid episode (%d errors)", error.error_count())
            return
        _ = hsm.dispatch(
            ctx,
            store,
            dataclasses.replace(
                memory.RememberEvent.with_data_and_id(episode, f"{operation_id}{_REMEMBER_ID_SUFFIX}"),
                source=hsm.id(instance),
                target=hsm.id(store),
                metadata=_public_metadata(metadata),
            ),
        )

    @staticmethod
    def _is_child_hop_timeout(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return event.source == hsm.id(instance) and event.target == hsm.id(instance)

    @staticmethod
    async def _await_child_hop(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
        *,
        child: ability.Ability[typing.Any, typing.Any],
        request: hsm.Event[typing.Any],
        hop_id: str,
    ) -> None:
        """Await one child hop through the settled one-shot. Timeout is the hop bound."""

        child_timeout = instance._child_operation_timeout
        try:
            terminal = await ability.run_terminal_operation(
                instance.context(),
                child=child,
                request=request,
                terminals=(child.output_event, child.failed_event),
                timeout=child_timeout if child_timeout is not None else _CHILD_OPERATION_TIMEOUT,
            )
        except TimeoutError:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ChildHopTimedOutEvent.with_data_and_id(None, hop_id),
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=_public_metadata(dict(event.metadata)),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                terminal,
                source=hsm.id(child),
                target=hsm.id(instance),
            ),
        )

    @staticmethod
    async def _start_autonomy(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.cognition.stage",
            scope="bot.abilities.cognition",
            component="cognition",
            stage="autonomy",
            context=telemetry.event_context(event),
        ):
            data = event.data
            assert is_input(data)
            autonomy_ability = instance._autonomy
            assert autonomy_ability is not None
            operation_id = event.id or uuid.uuid4().hex
            if processing.active_operation(instance, operation_id) is None:
                _ = await processing.start_operation(instance, operation_id)
            turn = Cognition._turn(instance, data, operation_id)
            hop_id = f"{operation_id}{_AUTONOMY_ID_SUFFIX}"
            input_event = dataclasses.replace(
                autonomy_ability.input_event.with_data_and_id(
                    turn,
                    hop_id,
                ),
                metadata=dict(event.metadata),
            )
            # Route the autonomy child through the settled one-shot terminal operation so its
            # correlated terminal is accepted regardless of the parent transient state
            # (HSM-CONTEXT-001 / HSM-DELIVERY-001). The envelope stamps source/target on the
            # child request so the child's reply_to echoes its terminal back to the operation
            # actor, which settles it here.
            await Cognition._await_child_hop(
                ctx,
                instance,
                event,
                child=autonomy_ability,
                request=input_event,
                hop_id=hop_id,
            )

    @staticmethod
    async def _hop_intuition(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
        *,
        cognition_input: InputData,
        operation_id: str | None,
        associations: memory.AssociationsData | None,
    ) -> None:
        """Build intuition's frame (with what memory surfaced as ``recalled``) and hop to it."""

        public_metadata = _public_metadata(dict(event.metadata))
        base = operation_id or uuid.uuid4().hex
        try:
            input = Cognition._build_processing_input(
                instance,
                cognition_input,
                recalled=associations,
            )
        except Exception as error:
            Cognition._remember(
                ctx,
                instance,
                operation_id=base,
                outcome=memory.Outcome.FAILED,
                metadata=public_metadata,
                cognition_input=cognition_input,
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(_failure(operation_id, str(error))),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        if processing.active_operation(instance, base) is None:
            _ = await processing.start_operation(instance, base)
        turn = Cognition._turn(instance, cognition_input, base, associations=associations)
        hop_id = f"{base}{_INTUITION_ID_SUFFIX}"
        input_event = dataclasses.replace(
            instance._intuition.input_event.with_data_and_id(
                intuition.InputData(turn=turn, processing_input=input),
                hop_id,
            ),
            metadata=public_metadata,
        )
        # Route the intuition child through the settled one-shot terminal operation so the
        # child's reply_to echoes its correlated terminal back to the operation actor.
        await Cognition._await_child_hop(
            ctx,
            instance,
            event,
            child=instance._intuition,
            request=input_event,
            hop_id=hop_id,
        )

    @staticmethod
    def _matches_autonomy_output(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._autonomy
        if child is None:
            return False
        return Cognition._matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_AUTONOMY_ID_SUFFIX,
            data_type=types.CompletionData,
        )

    @staticmethod
    def _matches_autonomy_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._autonomy
        if child is None:
            return False
        return Cognition._matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_AUTONOMY_ID_SUFFIX,
            data_type=types.FailureData,
        )

    @staticmethod
    def _autonomy_is_handled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Cognition._matches_autonomy_output(ctx, instance, event)
            and isinstance(event.data, types.CompletionData)
            and event.data.output is not None
        )

    @staticmethod
    def _autonomy_is_unhandled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Cognition._matches_autonomy_output(ctx, instance, event)
            and isinstance(event.data, types.CompletionData)
            and event.data.output is None
        )

    @staticmethod
    def _complete_processing(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, types.CompletionData)
        cognition_input = data.turn.input
        operation_id = data.turn.operation_id
        public_metadata = _public_metadata(dict(event.metadata))
        try:
            output = _coerce_output(data.output)
        except Exception as error:
            Cognition._remember(
                ctx,
                instance,
                operation_id=operation_id,
                outcome=memory.Outcome.FAILED,
                metadata=public_metadata,
                cognition_input=cognition_input,
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(_failure(operation_id, str(error))),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        Cognition._remember(
            ctx,
            instance,
            operation_id=operation_id,
            outcome=memory.Outcome.COMPLETED,
            metadata=public_metadata,
            cognition_input=cognition_input,
            output=output,
        )
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=operation_id,
            metadata=public_metadata,
            source=hsm.id(instance),
        )
        # Deliver the body terminal before reflection starts. Reflection is a peer HSM;
        # dispatching it first occupies the loop while this machine's TerminalOutputEvent
        # is still queued, so Bot stays in processing until reflection yields.
        ability.Ability._forward_terminal_event(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))
        reflection_ability = instance._reflection
        turn = reflection.InputData(
            cognition_input=cognition_input, cognition_output=output, associations=data.turn.associations
        )
        reflection_event = dataclasses.replace(
            reflection_ability.input_event.with_data_and_id(
                turn,
                Cognition._reflection_operation_id(operation_id),
            ),
            source=hsm.id(instance),
            target=hsm.id(reflection_ability),
            metadata=public_metadata,
        )
        _ = hsm.dispatch(ctx, reflection_ability, reflection_event)

    @staticmethod
    async def _start_intuition_activity(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.cognition.stage",
            scope="bot.abilities.cognition",
            component="cognition",
            stage="intuition",
            context=telemetry.event_context(event),
        ) as active:
            data = event.data
            if Cognition._matches_associated(ctx, instance, event):
                assert isinstance(data, _AssociatedData)
                active.set_attribute("bot.cognition.entered_from", "associating")
                await Cognition._hop_intuition(
                    ctx,
                    instance,
                    event,
                    cognition_input=data.turn.input,
                    operation_id=data.turn.operation_id,
                    associations=data.turn.associations,
                )
                return
            # Prefer full unhandled-autonomy match (envelope id, generation, live op, suffix).
            # Suffix is still set when this activity starts from the autonomy→intuition transition.
            if Cognition._autonomy_is_unhandled(ctx, instance, event):
                assert isinstance(data, types.CompletionData)
                active.set_attribute("bot.cognition.entered_from", "autonomy")
                await Cognition._hop_intuition(
                    ctx,
                    instance,
                    event,
                    cognition_input=data.turn.input,
                    operation_id=data.turn.operation_id,
                    associations=None,
                )
                return
            active.set_attribute("bot.cognition.entered_from", "input")
            assert is_input(data)
            await Cognition._hop_intuition(
                ctx, instance, event, cognition_input=data, operation_id=event.id or None, associations=None
            )

    @staticmethod
    async def _start_associating(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Ask Associative, by message, what memory associates with the stimulus; never fails the turn."""

        with span.operation(
            "bot.cognition.stage",
            scope="bot.abilities.cognition",
            component="cognition",
            stage="associating",
            context=telemetry.event_context(event),
        ) as active:
            data = event.data
            if isinstance(data, types.CompletionData):
                cognition_input = data.turn.input
                operation_id = data.turn.operation_id
            else:
                assert is_input(data)
                cognition_input = data
                operation_id = event.id or uuid.uuid4().hex
            associative = instance._associative
            assert associative is not None
            if processing.active_operation(instance, operation_id) is None:
                _ = await processing.start_operation(instance, operation_id)
            associations: memory.AssociationsData | None = None
            child_timeout = instance._child_operation_timeout
            try:
                terminal = await ability.run_terminal_operation(
                    ctx,
                    child=associative,
                    request=dataclasses.replace(
                        memory.AssociateEvent.with_data_and_id(
                            memory.CueData(
                                cue=projection.memory_content(
                                    cognition_input.stimulus,
                                    capacity=_MEMORY_IMAGES,
                                    budget_bytes=memory.MAX_EPISODE_TEXT_BYTES,
                                ),
                                participants=cognition_input.participants,
                            ),
                            f"{operation_id}{_ASSOCIATE_ID_SUFFIX}",
                        ),
                        metadata=_public_metadata(dict(event.metadata)),
                    ),
                    terminals=(memory.AssociatedEvent, ability.FailedEvent),
                    timeout=child_timeout if child_timeout is not None else _CHILD_OPERATION_TIMEOUT,
                )
            except (TimeoutError, RuntimeError):
                terminal = None
            if terminal is not None and isinstance(terminal.data, memory.AssociationsData):
                associations = terminal.data
            active.set_attribute("bot.cognition.associated", associations is not None)
            turn = Cognition._turn(instance, cognition_input, operation_id, associations=associations)
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _AssociatedEvent.with_data(_AssociatedData(turn=turn)),
                    id=operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=_public_metadata(dict(event.metadata)),
                ),
            )

    @staticmethod
    def _matches_associated(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _AssociatedData):
            return False
        operation = processing.active_operation(instance, data.turn.operation_id)
        return (
            event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
            and event.id == data.turn.operation_id
            and operation is not None
            and data.turn.generation == hsm.id(operation)
        )

    @staticmethod
    def _has_associative(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._associative is not None

    @staticmethod
    def _acknowledge_associating_cancel(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Associative has no cancel event and settles by its own timeout: acknowledge at once."""

        data = event.data
        assert isinstance(data, CancelData)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _CancelPendingEvent.with_data(data),
                id=data.operation_id,
                source=event.source,
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _matches_intuition_output(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._intuition
        return Cognition._matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_INTUITION_ID_SUFFIX,
            data_type=types.CompletionData,
        )

    @staticmethod
    def _matches_intuition_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._intuition
        return Cognition._matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_INTUITION_ID_SUFFIX,
            data_type=types.FailureData,
        )

    @staticmethod
    def _intuition_is_handled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Cognition._matches_intuition_output(ctx, instance, event)
            and isinstance(event.data, types.CompletionData)
            and event.data.output is not None
        )

    @staticmethod
    def _intuition_is_unhandled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        # Host cascade only: an unhandled intuition result advances Cognition to reasoning.
        return (
            Cognition._matches_intuition_output(ctx, instance, event)
            and isinstance(event.data, types.CompletionData)
            and event.data.output is None
        )

    @staticmethod
    async def _start_reasoning(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.cognition.stage",
            scope="bot.abilities.cognition",
            component="cognition",
            stage="reasoning",
            context=telemetry.event_context(event),
        ):
            completion = event.data
            cognition_input = completion.turn.input if isinstance(completion, types.CompletionData) else None
            operation_id = completion.turn.operation_id if isinstance(completion, types.CompletionData) else None
            public_metadata = _public_metadata(dict(event.metadata))
            if cognition_input is None:
                # No turn reached this stage, so there is nothing to remember.
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _ApplyFailedEvent.with_data(
                            _failure(operation_id, "Cognition reasoning start is missing turn correlation.")
                        ),
                        id=operation_id,
                        metadata=public_metadata,
                    ),
                )
                return
            associations = completion.turn.associations if isinstance(completion, types.CompletionData) else None
            base = operation_id if operation_id else uuid.uuid4().hex
            try:
                input = Cognition._build_processing_input(
                    instance,
                    cognition_input,
                    recalled=associations,
                )
            except Exception as error:
                Cognition._remember(
                    ctx,
                    instance,
                    operation_id=base,
                    outcome=memory.Outcome.FAILED,
                    metadata=public_metadata,
                    cognition_input=cognition_input,
                )
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _ApplyFailedEvent.with_data(_failure(operation_id, str(error))),
                        id=operation_id,
                        metadata=public_metadata,
                    ),
                )
                return
            turn = Cognition._turn(instance, cognition_input, base, associations=associations)
            hop_id = f"{base}{_REASONING_ID_SUFFIX}"
            input_event = dataclasses.replace(
                instance._reasoning.input_event.with_data_and_id(
                    reasoning.InputData(turn=turn, processing_input=input),
                    hop_id,
                ),
                metadata=public_metadata,
            )
            # Route the reasoning child through the settled one-shot terminal operation so the
            # child's reply_to echoes its correlated terminal back to the operation actor.
            await Cognition._await_child_hop(
                ctx,
                instance,
                event,
                child=instance._reasoning,
                request=input_event,
                hop_id=hop_id,
            )

    @staticmethod
    def _matches_reasoning_output(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._reasoning
        return Cognition._matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_REASONING_ID_SUFFIX,
            data_type=types.CompletionData,
        )

    @staticmethod
    def _matches_reasoning_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._reasoning
        return Cognition._matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_REASONING_ID_SUFFIX,
            data_type=types.FailureData,
        )

    @staticmethod
    def _fail_child(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        operation_id = _parent_operation_id(event)
        failure = (
            event.data
            if isinstance(event.data, ability.FailureData)
            else ability.FailureData(message="Cognition ability failed.")
        )
        if operation_id is not None and isinstance(event.data, types.FailureData):
            Cognition._remember(
                ctx,
                instance,
                operation_id=operation_id,
                outcome=memory.Outcome.FAILED,
                metadata=dict(event.metadata),
                cognition_input=event.data.turn.input,
            )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ApplyFailedEvent.with_data(_UseFailedEventData(failure=failure, operation_id=operation_id)),
                id=operation_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=_public_metadata(dict(event.metadata)),
            ),
        )

    @staticmethod
    def _open_turn(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        """Send this turn's input to Cognition itself; ``processing`` holds it until the turn ends."""

        data = event.data
        assert is_input(data)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _TurnOpenedEvent.with_data_and_id(_TurnOpenedData(input=data), event.id),
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=_public_metadata(dict(event.metadata)),
            ),
        )

    @staticmethod
    def _is_opened_turn(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> bool:
        """The turn Cognition opened for itself; a turn without an id cannot be kept by id."""

        del ctx
        return (
            isinstance(event.data, _TurnOpenedData)
            and bool(event.id)
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _remember_opened_turn(
        ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any], outcome: memory.Outcome
    ) -> None:
        data = event.data
        assert isinstance(data, _TurnOpenedData)
        Cognition._remember(
            ctx,
            instance,
            operation_id=event.id,
            outcome=outcome,
            metadata=dict(event.metadata),
            cognition_input=data.input,
        )

    @staticmethod
    def _remember_cancelled(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        """An accepted cancel ended the turn, whether its teardown then settles or reboots."""

        Cognition._remember_opened_turn(ctx, instance, event, memory.Outcome.CANCELLED)

    @staticmethod
    def _remember_abandoned(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        """A reboot (a child hop that never answered, or reflection's) ended the turn: it failed."""

        Cognition._remember_opened_turn(ctx, instance, event, memory.Outcome.FAILED)

    @staticmethod
    def _request_reboot(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        # Child-wait timeout / teardown path only (wired from hsm.after transitions).
        processing.request_reboot(ctx, instance, event, reason="cognition_child_teardown_failed")

    @staticmethod
    def _request_detach_rollback_reboot(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        # Detach-terminal rollback failure path only (wired on composite attachment terminal).
        processing.request_reboot(ctx, instance, event, reason="cognition_detach_rollback_failed")

    @staticmethod
    def _dispatch_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _UseFailedEventData)
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data.failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _has_autonomy(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._autonomy is not None

    @staticmethod
    def _has_input(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        data = event.data
        if not is_input(data):
            return False
        return True

    @staticmethod
    def _finish_turn(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        if isinstance(data, types.CompletionData):
            operation_id = data.turn.operation_id
        elif isinstance(data, types.FailureData):
            operation_id = data.turn.operation_id
        elif isinstance(data, CancelData):
            operation_id = data.operation_id
        elif isinstance(data, processing.CancelledData):
            operation_id = data.parent_operation_id
        elif isinstance(data, _UseFailedEventData):
            operation_id = event.id
        else:
            operation_id = _parent_operation_id(event)
        if operation_id is not None:
            processing.finish_operation(ctx, instance, operation_id)

    @staticmethod
    def _has_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, _UseFailedEventData)

    @staticmethod
    def _is_cancel_request(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not (
            isinstance(data, CancelData)
            and event.id == data.operation_id
            and event.target == hsm.id(instance)
            and bool(instance._attachments)
            and event.source == hsm.id(instance._attachments[0])
        ):
            return False
        return processing.active_operation(instance, data.operation_id) is not None

    @staticmethod
    async def _dispatch_cancel_to_child(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
        child: ability.Ability[typing.Any, typing.Any],
        suffix: str,
        *,
        pending_after_child: bool = False,
    ) -> None:
        data = event.data
        assert isinstance(data, CancelData)
        child_operation_id = f"{data.operation_id}{suffix}"
        cancellation_id = processing.cancellation_operation_id(
            data.operation_id,
            child_operation_id,
            data.token,
            hsm.id(child),
        )
        if processing.active_operation(instance, cancellation_id) is None:
            _ = await processing.start_operation(instance, cancellation_id)
        await hsm.dispatch(
            ctx,
            child,
            dataclasses.replace(
                processing.CancelEvent.with_data(
                    processing.CancelData(
                        operation_id=child_operation_id,
                        token=data.token,
                        parent_operation_id=data.operation_id,
                    )
                ),
                id=child_operation_id,
                source=hsm.id(instance),
                target=hsm.id(child),
                metadata=dict(event.metadata),
            ),
        )
        if pending_after_child:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _CancelPendingEvent.with_data(data),
                    id=data.operation_id,
                    source=event.source,
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    async def _cancel_autonomy(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        child = instance._autonomy
        assert child is not None
        await Cognition._dispatch_cancel_to_child(
            ctx,
            instance,
            event,
            child,
            _AUTONOMY_ID_SUFFIX,
            pending_after_child=True,
        )

    @staticmethod
    async def _cancel_intuition(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        await Cognition._dispatch_cancel_to_child(ctx, instance, event, instance._intuition, _INTUITION_ID_SUFFIX)

    @staticmethod
    async def _cancel_reasoning(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        await Cognition._dispatch_cancel_to_child(ctx, instance, event, instance._reasoning, _REASONING_ID_SUFFIX)

    @staticmethod
    def _matches_child_cancelled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if (
            not isinstance(data, processing.CancelledData)
            or event.id != data.operation_id
            or event.target != hsm.id(instance)
        ):
            return False
        parent_operation_id = _parent_operation_id(event)
        if parent_operation_id is None or data.parent_operation_id != parent_operation_id:
            return False
        cancellation_id = processing.cancellation_operation_id(
            parent_operation_id,
            data.operation_id,
            data.token,
            event.source,
        )
        return processing.active_operation(instance, cancellation_id) is not None and event.source in {
            hsm.id(child)
            for child in (instance._autonomy, instance._intuition, instance._reasoning)
            if child is not None
        }

    @staticmethod
    def _matches_cancelled_child_terminal(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Cognition._matches_autonomy_output(ctx, instance, event) or Cognition._matches_autonomy_failure(
            ctx, instance, event
        )

    @staticmethod
    def _emit_cancelled(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        if isinstance(event.data, CancelData):
            operation_id = event.data.operation_id
            token = event.data.token
            suffix = _AUTONOMY_ID_SUFFIX
            child = instance._autonomy
            if child is not None:
                child_operation_id = f"{operation_id}{suffix}"
                cancellation_id = processing.cancellation_operation_id(
                    operation_id,
                    child_operation_id,
                    token,
                    hsm.id(child),
                )
                processing.finish_operation(ctx, instance, cancellation_id)
        else:
            data = event.data
            assert isinstance(data, processing.CancelledData)
            operation_id = _parent_operation_id(event)
            assert operation_id is not None
            token = data.token
            cancellation_id = processing.cancellation_operation_id(
                operation_id,
                data.operation_id,
                token,
                event.source,
            )
            processing.finish_operation(ctx, instance, cancellation_id)
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                CancelledEvent.with_data(CancelledData(operation_id=operation_id, token=token)),
                id=operation_id,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _cancel_timeout_delay(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, event
        cancel_timeout = instance._cancel_teardown_timeout
        return cancel_timeout if cancel_timeout is not None else _CANCEL_TEARDOWN_TIMEOUT

    @staticmethod
    def _request_cancel_reboot(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        processing.request_reboot(ctx, instance, event, reason="cognition_cancel_teardown_failed")

    @staticmethod
    def _finish_operations(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        del event
        processing.finish_operations(ctx, instance)

    @staticmethod
    def _is_reflection_reboot(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, mosfet.RebootEventData)
            and event.source == hsm.id(instance._reflection)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _forward_reflection_reboot(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        if not instance._attachments:
            return
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                event,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=_public_metadata(dict(event.metadata)),
            ),
        )

    @staticmethod
    def _is_reflection_settled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            instance._routines is not None
            and event.source == hsm.id(instance._reflection)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _is_attach_complete_with_routines(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return instance._routines is not None and ability.Ability._is_composite_attach_complete(ctx, instance, event)

    @staticmethod
    def _reconcile_routines(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Inventory is reachable or was just written: let Routines match it.

        Sent once every child (memory among them) is attached, which rehydrates the routines a
        restart left behind, and after Reflection settles a turn (Reflection writes inventory).
        """

        routines_ability = instance._routines
        assert routines_ability is not None
        _ = hsm.dispatch(
            ctx,
            routines_ability,
            dataclasses.replace(
                routines.ReconcileEvent.with_data(routines.ReconcileData()),
                id=event.id or None,
                source=hsm.id(instance),
                target=hsm.id(routines_ability),
                metadata=_public_metadata(dict(event.metadata)),
            ),
        )

    @staticmethod
    def _is_routine_tick(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            instance._routines is not None
            and isinstance(event.data, behavior.TickData)
            and event.source == hsm.id(instance._routines)
            and bool(instance._attachments)
        )

    @staticmethod
    def _is_recall_outcome(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """What the bot's own recall brought to mind, or why it could not: from Associative, by data type."""

        del ctx
        return (
            instance._associative is not None
            and event.source == hsm.id(instance._associative)
            and isinstance(event.data, memory.RecollectionData | ability.FailureData)
            and bool(instance._attachments)
        )

    @staticmethod
    def _observe(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Hand a routine tick or a recall outcome to the body as an observation; the body grants it a turn when free."""

        owner = instance._attachments[0]
        stimulus = dataclasses.replace(event, target="", metadata=_public_metadata(dict(event.metadata)))
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                InputEvent.with_data_and_id(InputData(stimulus=stimulus), uuid.uuid4().hex),
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=_public_metadata(dict(event.metadata)),
            ),
        )

    @staticmethod
    def _is_persona_told(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """What the bot knows of itself, told by its own persona child (by source and payload type)."""

        del ctx
        return (
            instance._persona is not None
            and event.source == hsm.id(instance._persona)
            and isinstance(event.data, persona.OutputData)
        )

    @staticmethod
    def _know_persona(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Keep what persona told: every frame built from now on shows it as the self block."""

        del ctx
        told = event.data
        if not isinstance(told, persona.OutputData):
            # Unreachable: paired with hsm.guard(_is_persona_told), which narrows this payload.
            return
        instance._persona_told = told

    @staticmethod
    def _has_consolidation(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._consolidation is not None

    @staticmethod
    def _rest_delay(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, event
        return instance._rest_after

    @staticmethod
    def _notify_consolidation(
        ctx: hsm.Context,
        instance: "Cognition",
        notice: hsm.Event[typing.Any],
    ) -> None:
        consolidation = instance._consolidation
        assert consolidation is not None, "only a cognition with consolidation rests"
        _ = hsm.dispatch(
            ctx,
            consolidation,
            dataclasses.replace(notice, source=hsm.id(instance), target=hsm.id(consolidation)),
        )

    @staticmethod
    def _rest(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        """Resting: memory may be consolidated until anything wakes the bot."""

        del event
        Cognition._notify_consolidation(ctx, instance, memory.RestEvent.with_data(memory.RestData()))

    @staticmethod
    def _wake(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        """Leaving rest for any reason (input, reboot, detach): consolidation stops."""

        del event
        Cognition._notify_consolidation(ctx, instance, memory.WakeEvent.with_data(memory.WakeData()))

    @staticmethod
    def _accept_ignore(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Acknowledge an explicit ignore selection; cognition only (no body/device work)."""

        del ctx, instance, event

    submodel: typing.ClassVar[hsm.Model | None] = define(
        "Cognition",
        hsm.initial(hsm.target("/Cognition/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(_is_attach_complete_with_routines),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal, _reconcile_routines),
                hsm.target("/Cognition/ready/settling"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Cognition/ready/settling"),
            ),
        ),
        # Not processing a turn. A cognition that cannot consolidate is simply idle. One that can
        # is awake, and rests once it has been awake for rest_after; every input leaves rest through
        # these shared transitions, which wakes it.
        hsm.state(
            "ready",
            hsm.initial(hsm.target("/Cognition/ready/idle")),
            hsm.choice(
                "settling",
                hsm.transition(hsm.guard(_has_consolidation), hsm.target("/Cognition/ready/awake")),
                hsm.transition(hsm.target("/Cognition/ready/idle")),
            ),
            hsm.state("idle"),
            hsm.state(
                "awake",
                hsm.transition(hsm.after(_rest_delay), hsm.target("/Cognition/ready/resting")),
            ),
            hsm.state("resting", hsm.entry(_rest), hsm.exit(_wake)),
            # Call surface for deliberative tools (snapshot-offered when host is an actor).
            hsm.transition(
                hsm.on(types.IgnoreEvent),
                hsm.effect(_accept_ignore),
            ),
            hsm.transition(
                hsm.on(reflection.OutputEvent, ability.FailedEvent),
                hsm.guard(_is_reflection_settled),
                hsm.effect(_reconcile_routines),
            ),
            hsm.transition(
                hsm.on(behavior.TickEvent),
                hsm.guard(_is_routine_tick),
                hsm.effect(_observe),
            ),
            hsm.transition(
                hsm.on(memory.RecollectedEvent, ability.FailedEvent),
                hsm.guard(_is_recall_outcome),
                hsm.effect(_observe),
            ),
            hsm.transition(
                hsm.on(mosfet.RebootEvent),
                hsm.guard(_is_reflection_reboot),
                hsm.effect(_forward_reflection_reboot),
                hsm.target("/Cognition/rebooting"),
            ),
            hsm.transition(
                hsm.on(CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
            ),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_input),
                hsm.guard(_has_autonomy),
                hsm.effect(_open_turn),
                hsm.target("/Cognition/processing/autonomy"),
            ),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_input),
                hsm.guard(_has_associative),
                hsm.effect(_open_turn),
                hsm.target("/Cognition/processing/associating"),
            ),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_input),
                hsm.effect(_open_turn),
                hsm.target("/Cognition/processing/intuition"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
            ),
        ),
        hsm.state(
            "processing",
            hsm.initial(hsm.target("/Cognition/processing/intuition")),
            # The turn's own input waits here until the state the turn ends in takes it up.
            hsm.defer(input_event, _TurnOpenedEvent),
            # Enabled while a child stage runs so ignore stays snapshot-offered mid-turn.
            hsm.transition(
                hsm.on(types.IgnoreEvent),
                hsm.effect(_accept_ignore),
            ),
            # Reflection of the previous turn, routine ticks and recall outcomes keep arriving
            # mid-turn; the body defers their handoff until this turn ends.
            hsm.transition(
                hsm.on(reflection.OutputEvent, ability.FailedEvent),
                hsm.guard(_is_reflection_settled),
                hsm.effect(_reconcile_routines),
            ),
            hsm.transition(
                hsm.on(behavior.TickEvent),
                hsm.guard(_is_routine_tick),
                hsm.effect(_observe),
            ),
            hsm.transition(
                hsm.on(memory.RecollectedEvent, ability.FailedEvent),
                hsm.guard(_is_recall_outcome),
                hsm.effect(_observe),
            ),
            hsm.transition(
                hsm.on(mosfet.RebootEvent),
                hsm.guard(_is_reflection_reboot),
                hsm.effect(_forward_reflection_reboot),
                hsm.target("/Cognition/rebooting"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
                hsm.effect(_finish_turn),
                hsm.target("/Cognition/ready/settling"),
            ),
            hsm.state(
                "autonomy",
                hsm.activity(_start_autonomy),
                hsm.transition(
                    hsm.on(CancelEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.target("/Cognition/processing/cancelling/autonomy"),
                ),
                hsm.transition(
                    hsm.on(autonomy.OutputEvent),
                    hsm.guard(_autonomy_is_unhandled),
                    hsm.guard(_has_associative),
                    hsm.target("/Cognition/processing/associating"),
                ),
                hsm.transition(
                    hsm.on(autonomy.OutputEvent),
                    hsm.guard(_autonomy_is_unhandled),
                    hsm.target("/Cognition/processing/intuition"),
                ),
                hsm.transition(
                    hsm.on(autonomy.OutputEvent),
                    hsm.guard(_autonomy_is_handled),
                    hsm.effect(_complete_processing),
                    hsm.effect(_finish_turn),
                    hsm.target("/Cognition/ready/settling"),
                ),
                hsm.transition(
                    hsm.on(ability.FailedEvent),
                    hsm.guard(_matches_autonomy_failure),
                    hsm.effect(_fail_child),
                    hsm.effect(_finish_turn),
                    hsm.target("/Cognition/ready/settling"),
                ),
                hsm.transition(
                    hsm.on(_ChildHopTimedOutEvent),
                    hsm.guard(_is_child_hop_timeout),
                    hsm.effect(_request_reboot),
                    hsm.target("/Cognition/rebooting"),
                ),
            ),
            hsm.state(
                "associating",
                hsm.activity(_start_associating),
                hsm.transition(
                    hsm.on(CancelEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.target("/Cognition/processing/cancelling/associating"),
                ),
                # No reboot path: an association that fails or times out leaves the turn without
                # associations, and the turn goes on.
                hsm.transition(
                    hsm.on(_AssociatedEvent),
                    hsm.guard(_matches_associated),
                    hsm.target("/Cognition/processing/intuition"),
                ),
            ),
            hsm.state(
                "intuition",
                hsm.activity(_start_intuition_activity),
                hsm.transition(
                    hsm.on(CancelEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.target("/Cognition/processing/cancelling/intuition"),
                ),
                hsm.transition(
                    hsm.on(intuition.OutputEvent),
                    hsm.guard(_intuition_is_unhandled),
                    hsm.target("/Cognition/processing/reasoning"),
                ),
                hsm.transition(
                    hsm.on(intuition.OutputEvent),
                    hsm.guard(_intuition_is_handled),
                    hsm.effect(_complete_processing),
                    hsm.effect(_finish_turn),
                    hsm.target("/Cognition/ready/settling"),
                ),
                hsm.transition(
                    hsm.on(ability.FailedEvent),
                    hsm.guard(_matches_intuition_failure),
                    hsm.effect(_fail_child),
                    hsm.effect(_finish_turn),
                    hsm.target("/Cognition/ready/settling"),
                ),
                hsm.transition(
                    hsm.on(_ChildHopTimedOutEvent),
                    hsm.guard(_is_child_hop_timeout),
                    hsm.effect(_request_reboot),
                    hsm.target("/Cognition/rebooting"),
                ),
            ),
            hsm.state(
                "reasoning",
                hsm.activity(_start_reasoning),
                hsm.transition(
                    hsm.on(CancelEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.target("/Cognition/processing/cancelling/reasoning"),
                ),
                hsm.transition(
                    hsm.on(reasoning.OutputEvent),
                    hsm.guard(_matches_reasoning_output),
                    hsm.effect(_complete_processing),
                    hsm.effect(_finish_turn),
                    hsm.target("/Cognition/ready/settling"),
                ),
                hsm.transition(
                    hsm.on(ability.FailedEvent),
                    hsm.guard(_matches_reasoning_failure),
                    hsm.effect(_fail_child),
                    hsm.effect(_finish_turn),
                    hsm.target("/Cognition/ready/settling"),
                ),
                hsm.transition(
                    hsm.on(_ChildHopTimedOutEvent),
                    hsm.guard(_is_child_hop_timeout),
                    hsm.effect(_request_reboot),
                    hsm.target("/Cognition/rebooting"),
                ),
            ),
            hsm.state(
                "cancelling",
                # Cancellation remains pending until the child has terminated.
                hsm.defer(_CancelPendingEvent),
                hsm.initial(hsm.target("intuition")),
                hsm.transition(hsm.on(_TurnOpenedEvent), hsm.guard(_is_opened_turn), hsm.effect(_remember_cancelled)),
                hsm.transition(
                    hsm.on(processing.CancelledEvent),
                    hsm.guard(_matches_child_cancelled),
                    hsm.effect(_emit_cancelled),
                    hsm.effect(_finish_turn),
                    hsm.target("/Cognition/ready/settling"),
                ),
                # A directed autonomy cancel settles through its pending acknowledgement: autonomy
                # is an Ability (not Processing) whose cancellation is best-effort — an empty
                # inventory leaves nothing to cancel, and a completed match may have already
                # settled its terminal inside the one-shot operation. Deliver the typed cancel
                # (above) and acknowledge when it lands, rather than blocking on a confirmation
                # the child may never emit.
                hsm.transition(
                    hsm.on(_CancelPendingEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.effect(_emit_cancelled),
                    hsm.effect(_finish_turn),
                    hsm.target("/Cognition/ready/settling"),
                ),
                hsm.transition(
                    hsm.on(autonomy.OutputEvent),
                    hsm.guard(_matches_cancelled_child_terminal),
                    hsm.target("/Cognition/processing/cancelling/acknowledging"),
                ),
                hsm.transition(
                    hsm.on(ability.FailedEvent),
                    hsm.guard(_matches_autonomy_failure),
                    hsm.target("/Cognition/processing/cancelling/acknowledging"),
                ),
                hsm.transition(
                    hsm.after(_cancel_timeout_delay),
                    hsm.effect(_request_cancel_reboot),
                    hsm.target("/Cognition/rebooting"),
                ),
                hsm.state("autonomy", hsm.activity(_cancel_autonomy)),
                # Associative has no cancel event: it settles by its own timeout, so this stage
                # acknowledges through _CancelPendingEvent exactly like autonomy.
                hsm.state("associating", hsm.entry(_acknowledge_associating_cancel)),
                hsm.state("intuition", hsm.activity(_cancel_intuition)),
                hsm.state("reasoning", hsm.activity(_cancel_reasoning)),
                hsm.state(
                    "acknowledging",
                    hsm.transition(
                        hsm.on(_CancelPendingEvent),
                        hsm.guard(_is_cancel_request),
                        hsm.effect(_emit_cancelled),
                        hsm.effect(_finish_turn),
                        hsm.target("/Cognition/ready/settling"),
                    ),
                ),
            ),
        ),
        hsm.state(
            "rebooting",
            hsm.entry(_finish_operations),
            hsm.defer(input_event),
            hsm.transition(hsm.on(_TurnOpenedEvent), hsm.guard(_is_opened_turn), hsm.effect(_remember_abandoned)),
        ),
        # Persona tells what the bot knows of itself whenever it knows something new: on waking,
        # usually while Cognition is still attaching its children, and after each revision, in
        # whatever phase Cognition is then. Keeping it is internal and never moves a phase.
        hsm.transition(
            hsm.on(persona.OutputEvent),
            hsm.guard(_is_persona_told),
            hsm.effect(_know_persona),
        ),
        hsm.state(
            "detaching",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(
                    ability.Ability._deliver_composite_attachment_terminal,
                    _request_detach_rollback_reboot,
                ),
                hsm.target("/Cognition/rebooting"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Cognition/ready/settling"),
            ),
        ),
    )

    def __init__(
        self,
        *,
        intuition: intuition.Intuition,
        reasoning: reasoning.Reasoning,
        reflection: reflection.Reflection,
        autonomy: autonomy.Autonomy | None = None,
        routines: routines.Routines | None = None,
        memory: memory.Memory | None = None,
        associative: memory.Associative | None = None,
        consolidation: memory.Consolidation | None = None,
        persona: persona.Persona | None = None,
        rest_after: datetime.timedelta = _DEFAULT_REST_AFTER,
        child_operation_timeout: datetime.timedelta | None = None,
        cancel_teardown_timeout: datetime.timedelta | None = None,
    ) -> None:
        super().__init__()
        if child_operation_timeout is not None and child_operation_timeout <= datetime.timedelta():
            raise ValueError("child_operation_timeout must be positive.")
        if cancel_teardown_timeout is not None and cancel_teardown_timeout <= datetime.timedelta():
            raise ValueError("cancel_teardown_timeout must be positive.")
        if associative is not None and memory is None:
            raise ValueError("associative requires memory: Cognition attaches the memory it associates from.")
        if consolidation is not None and memory is None:
            raise ValueError("consolidation requires memory: Cognition attaches the memory it consolidates.")
        if rest_after <= datetime.timedelta():
            raise ValueError("rest_after must be positive.")
        self._autonomy = autonomy
        self._intuition = intuition
        self._reasoning = reasoning
        self._reflection = reflection
        self._routines = routines
        self._memory = memory
        self._associative = associative
        self._consolidation = consolidation
        self._persona = persona
        self._persona_told = None
        self._rest_after = rest_after
        self._child_operation_timeout = child_operation_timeout
        self._cancel_teardown_timeout = cancel_teardown_timeout
        children: list[hsm.Instance] = []
        if autonomy is not None:
            children.append(autonomy)
        children.extend((intuition, reasoning))
        children.append(reflection)
        if memory is not None:
            children.append(memory)
        if associative is not None:
            children.append(associative)
        if consolidation is not None:
            children.append(consolidation)
        if persona is not None:
            children.append(persona)
        if routines is not None:
            children.append(routines)
        self._attachment_group = attachment.Group(*children)


__all__ = [
    "InputEvent",
    "OutputEvent",
    "Cognition",
    "InputData",
    "OutputData",
]

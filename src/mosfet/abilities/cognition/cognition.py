from .. import ability
from .. import memory
from .. import persona
from .. import processing
from .. import vocal
from ..language import text

import asyncio
import dataclasses
import datetime
import logging
import typing
import uuid

import hsm
import mosfet
from mosfet.define import define
from opentelemetry import metrics

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
_RECENT_ID_SUFFIX = ":recent"
_INTUITION_ID_SUFFIX = ":intuition"
_REASONING_ID_SUFFIX = ":reasoning"
_MAX_SHOWN_PARENTS = 8
"""Most causal parents followed up a stimulus for the ids of the turns it already shows."""
_REFLECTION_ID_SUFFIX = ":reflection"
_REMEMBER_ID_SUFFIX = ":remember"
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
# What an episode keeps of the turn's own output, rendered after what was perceived: text only (an image the bot
# showed is kept as its placeholder), within a quarter of the episode's text, so the stimulus keeps the rest.
_OUTPUT_IMAGES: typing.Final = media.ImageCapacity()
_MAX_OUTPUT_BYTES: typing.Final = memory.MAX_EPISODE_TEXT_BYTES // 4
# What an episode keeps of a failed turn's typed failure, rendered after its output, within its own bound.
_MAX_FAILURE_BYTES: typing.Final = memory.MAX_EPISODE_TEXT_BYTES // 16
# What deliberation is told behaviors already did this turn (D-W12-9): text only, after the world and self
# blocks, within its own bound so the instructions keep room for them.
_MAX_ACTED_BYTES: typing.Final = memory.MAX_EPISODE_TEXT_BYTES // 16

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


class StepTimedOutData(ability.FailureData):
    """A step of the bot's cognition took longer than its step timeout, so the turn it was taking failed.

    The bot perceives this as a turn of its own: which step it was in, how long that step was
    allowed, and which turn failed for which stimulus. Only the step was stopped; nothing was
    rebooted and every ability kept its state.
    """

    # The base names its own description in json_schema_extra; this payload's docstring describes it.
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, json_schema_extra=None)

    stage: typing.Literal["autonomy", "intuition", "reasoning"] = pydantic.Field(
        description=(
            "The step of cognition that outlasted its timeout: autonomy (a learned behavior running), "
            "intuition (the fast first decision) or reasoning (the deliberate one)."
        ),
        examples=["reasoning"],
    )
    timeout_seconds: float = pydantic.Field(
        gt=0,
        description="How long the step was allowed, in seconds, before the turn failed.",
        examples=[120.0],
    )
    turn_id: str = pydantic.Field(
        min_length=1,
        description="Identifier of the turn that failed.",
        examples=["turn-123"],
    )
    stimulus: str | None = pydantic.Field(
        default=None,
        description="Name of the stimulus the failed turn was taken up for, when it has one.",
        examples=["bot.input"],
    )


StepTimedOutEvent = hsm.Event[StepTimedOutData](
    name="bot.ability.cognition.step.timed_out",
    schema=StepTimedOutData,
)


class StepFailedData(ability.FailureData):
    """A step of the bot's cognition failed, so the turn it was taking failed.

    The bot perceives this as a turn of its own: which step failed and why, and which turn failed for
    which stimulus. Only that turn failed; nothing was rebooted and every ability kept its state. A
    step that fails while the bot perceives a failed step is not handed back again. An intuition that
    fails is not handed back either: its turn goes on to reasoning, so the stage is autonomy or
    reasoning in practice.
    """

    # The base names its own description in json_schema_extra; this payload's docstring describes it.
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, json_schema_extra=None)

    stage: typing.Literal["autonomy", "intuition", "reasoning"] = pydantic.Field(
        description=(
            "The step of cognition that failed: autonomy (a learned behavior running), intuition (the fast "
            "first decision) or reasoning (the deliberate one)."
        ),
        examples=["reasoning"],
    )
    turn_id: str = pydantic.Field(
        min_length=1,
        description="Identifier of the turn that failed.",
        examples=["turn-123"],
    )
    stimulus: str | None = pydantic.Field(
        default=None,
        description="Name of the stimulus the failed turn was taken up for, when it has one.",
        examples=["bot.input"],
    )


StepFailedEvent = hsm.Event[StepFailedData](
    name="bot.ability.cognition.step.failed",
    schema=StepFailedData,
)

_HANDED_BACK = metrics.get_meter("mosfet.abilities.cognition").create_counter(
    "bot.cognition.step.failure.count",
    unit="{failure}",
    description=(
        "Steps of a turn that failed or outlasted the step timeout, by stage (autonomy, intuition, reasoning), "
        "kind (failed, timed_out) and outcome: handed_back (the bot perceives it as its next turn), bounded "
        "(the turn that failed was itself perceiving a failed step, so it is not handed back again) or cascaded "
        "(an intuition that failed, or whose frame could not be built, so reasoning took the same turn)."
    ),
)


class _HopTimedOutData(pydantic.BaseModel):
    """Private: the step that outlasted its timeout, with the turn it was taking, sent to Cognition itself."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    turn: types.TurnData
    failure: StepTimedOutData


_ChildHopTimedOutEvent = hsm.Event[_HopTimedOutData](
    name="bot.ability.cognition.child.hop.timed_out",
    kind=hsm.ErrorEventKind,
    schema=_HopTimedOutData,
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
# Private completion: intuition passed the turn without deciding it (``output`` is None), so reasoning takes it.
_IntuitionPassedEvent = hsm.Event[types.CompletionData](
    name="bot.ability.cognition.intuition.passed",
    kind=hsm.CompletionEventKind,
    schema=types.CompletionData,
)


class _TurnOpenedData(pydantic.BaseModel):
    """Private: the input of the turn Cognition just took up, sent to itself under the turn's id.

    ``processing`` defers it for the whole turn, so it reaches whichever state the turn ends in:
    ``cancelling`` remembers the turn as cancelled and ``rebooting`` as failed, each with what was
    said and who took part. A turn that completes or fails is remembered from its own terminal,
    which carries the turn, so ``ready`` lets this event fall through; so is a turn whose step
    outlasted its timeout, so ``stopping`` only spends it.
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

    Each step of a turn (autonomy, associating, intuition, reasoning) is bounded by the step
    timeout, ``child_operation_timeout`` (``bot.timeouts(step=)``; two minutes by default, as one
    step of a local model can take half a minute and more). A step that outlasts it fails only its
    turn: the body gets the turn's ``StepTimedOutData`` failure and then a ``cognition.InputEvent``
    observing ``StepTimedOutEvent``, a new turn the bot decides on, while the step is stopped the
    way a cancel stops it. Nothing reboots and no ability loses its state, unless the step cannot be
    stopped within the cancel teardown bound. An association that outlasts it leaves the turn
    without associations. A step that fails (autonomy or reasoning, or a frame that cannot be built
    for reasoning) is handed back the same way, as a ``cognition.InputEvent`` observing
    ``StepFailedEvent``. Both are bounded: a step that fails or outlasts its timeout on a turn
    taken up for a failed step is not handed back again (``bot.cognition.step.failure.count``
    counts each, by stage, kind and outcome). An intuition that fails, or whose frame cannot be
    built, hands the same turn to reasoning instead (outcome ``cascaded``): nothing is handed back
    and reasoning settles the turn. A turn taken up for a handed-back intuition failure (an
    intuition step that outlasted its timeout) passes intuition, so reasoning, a stage that can
    perceive that failure, decides it.

    With ``memory`` composed, Cognition attaches it and remembers every turn it settles —
    completed, failed, or cancelled — as one ``memory.RememberEvent`` it does not wait on: what it
    perceived, then its own output (``OutputEvent``), so recall covers what it did as well. With
    ``associative`` composed too, a turn autonomy leaves unhandled first asks it what memory
    associates with the stimulus; the associations ride the turn to intuition and reasoning (as
    their frames' ``recalled`` context) and to reflection. An association that fails or times out
    leaves the turn without associations; it never fails the turn.

    With ``stm_memory`` (the body's short-term register, injected and reached only by message), a
    turn autonomy leaves unhandled also asks it, in parallel with associating, for the recent turns:
    what was heard and said just before, oldest first, without the turn's own stimulus. They ride the
    turn to intuition and reasoning as ``recalled`` (``types.RecalledData``), ahead of what memory
    associates. A read that fails or times out leaves the turn without them; it never fails the turn.

    Every frame Cognition builds also holds ``associative`` (under ``memory``), so a stage whose
    effort ceiling admits it (Reasoning, not Intuition) is offered ``bot.ability.memory.recall``,
    ``bot.ability.memory.know`` and ``bot.ability.memory.forget``. What a recall brings to mind
    (``memory.RecollectedEvent``), what memory now knows of a person (``memory.KnownEvent``), or a
    typed failure, comes back to Cognition and goes to the body as a ``cognition.InputEvent``
    observing it: a new turn the bot decides on, the way a routine tick is.

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
    block as the self block (``SelfData``), so the bot knows who it is on every turn, and hands it to
    reflection with the turn. Cognition only keeps what it was told; reflection, composed with the
    same persona, is the only place a revision may be selected.

    With ``voice`` composed, Cognition attaches the bot's inception (``vocal.voice.Inception``) and
    keeps the voice it was told the bot speaks in (``vocal.voice.OutputEvent``), with what that
    voice can do with an expression; the self block shows it from then on. Inception is the bot's
    making, never a turn: what it fails at (a voice estranged or lost, a source or model that cannot
    answer) goes to the body as a ``cognition.InputEvent`` observing it, never silently dropped.
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
    # The body's short-term register, injected: a peer reached only by message, never attached here.
    _stm_memory: memory.StmMemory | None
    _consolidation: memory.Consolidation | None
    _persona: persona.Persona | None
    # What persona last told Cognition, by event: durable self-knowledge, not turn scratch.
    _persona_told: persona.OutputData | None
    # Quoted: Inception reaches Speaking, whose imports reach this module (vocal.voice exports it lazily).
    _voice: "vocal.voice.Inception | None"
    # The voice inception told Cognition the bot speaks in, by event: durable self-knowledge.
    _voice_told: vocal.voice.VoiceData | None
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
        associations: memory.AssociationsData | None = None,
        recent: tuple[memory.stm.Turn, ...] = (),
        acted: types.ActedData | None = None,
    ) -> processing.InputData:
        """Deliberation's frame for the turn: offered events, the self block (``instructions``), and the
        world block then what behaviors already did on its percepts (``acted``), as the turn's ``situation``.

        What the bot has in mind beside the stimulus rides as ``recalled``: the ``recent`` turns
        short-term memory holds, then what long-term memory ``associations`` surfaced; none when neither."""

        associative = instance._associative
        told = instance._persona_told
        voiced = instance._voice_told
        frame = input.build_processing_input(
            cognition_input,
            # Deliberate recall: offered to a stage whose effort ceiling admits the associative.
            extra_actors=None if associative is None else {"memory": associative},
            authority=instance,
            known=None if told is None and voiced is None else input.SelfData(persona=told, voice=voiced),
        )
        if acted is not None:
            fact = "".join(
                part.text
                for part in projection.memory_content(acted, capacity=_OUTPUT_IMAGES, budget_bytes=_MAX_ACTED_BYTES)
                if isinstance(part, text.TextPart)
            )
            frame = frame.model_copy(
                update={"situation": "\n".join(block for block in (frame.situation, fact) if block)}
            )
        if not recent and associations is None:
            return frame
        return frame.model_copy(update={"recalled": types.RecalledData(recent=recent, associations=associations)})

    @staticmethod
    def _turn(
        instance: "Cognition",
        cognition_input: InputData,
        operation_id: str,
        *,
        associations: memory.AssociationsData | None = None,
        recent: tuple[memory.stm.Turn, ...] = (),
        acted: types.ActedData | None = None,
    ) -> types.TurnData:
        operation = processing.active_operation(instance, operation_id)
        assert operation is not None
        return types.TurnData(
            input=cognition_input,
            operation_id=operation_id,
            generation=hsm.id(operation),
            associations=associations,
            recent=recent,
            acted=acted,
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
        failure: ability.FailureData | None = None,
    ) -> None:
        """Send memory the turn just settled, stamped here, where the turn is owned; never waited on.

        Every outcome carries the turn's input: a completed or failed turn from its terminal, a
        cancelled or rebooted one from its ``_TurnOpenedEvent``. The episode's parts are what the bot
        perceived, then, when the turn selected anything, its own output (``OutputEvent``: what it
        showed, said and asked of its tools), then, when the turn failed with a typed failure, that
        failure (``failed_event``), so what it did and what became of it are recalled with what it
        perceived; a tool's result is perceived on a later turn and kept as that turn's. Memory's
        reply is not awaited: Memory counts its own outcome, and no transition here consumes it.
        """

        store = instance._memory
        if store is None:
            return
        acts = output[:_MAX_ACTS]
        did = (
            projection.memory_content(
                OutputEvent.with_data(acts), capacity=_OUTPUT_IMAGES, budget_bytes=_MAX_OUTPUT_BYTES
            )
            if acts
            else ()
        )
        if failure is not None:
            did = (
                *did,
                *projection.memory_content(
                    instance.failed_event.with_data(failure), capacity=_OUTPUT_IMAGES, budget_bytes=_MAX_FAILURE_BYTES
                ),
            )
        # One newline joins the output to the last text of what was perceived.
        spent = sum(len(part.text.encode()) + 1 for part in did if isinstance(part, text.TextPart))
        perceived = projection.memory_content(
            cognition_input.stimulus, capacity=_MEMORY_IMAGES, budget_bytes=memory.MAX_EPISODE_TEXT_BYTES - spent
        )
        last = perceived[-1] if perceived else None
        parts = (
            (*perceived[:-1], text.TextPart(text="\n".join(part.text for part in (last, *did) if part.kind == "text")))
            if isinstance(last, text.TextPart) and did
            else (*perceived, *did)
        )
        try:
            episode = memory.EpisodeData(
                turn_id=operation_id,
                stimulus=input.stimulus_name(cognition_input.stimulus),
                parts=parts,
                participants=cognition_input.participants,
                focus=cognition_input.focus,
                focus_candidates=cognition_input.focus_candidates[:_MAX_ACTS],
                acts=tuple(_act(selection) for selection in acts),
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
        data = event.data
        if not isinstance(data, _HopTimedOutData):
            return False
        operation = processing.active_operation(instance, data.turn.operation_id)
        # The stage topology selected the step; source, target and generation correlate the turn.
        return (
            event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
            and operation is not None
            and data.turn.generation == hsm.id(operation)
        )

    @staticmethod
    async def _await_child_hop(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
        *,
        child: ability.Ability[typing.Any, typing.Any],
        request: hsm.Event[typing.Any],
        hop_id: str,
        turn: types.TurnData,
        stage: typing.Literal["autonomy", "intuition", "reasoning"],
    ) -> None:
        """Await one child hop through the settled one-shot. The step timeout is the hop bound."""

        child_timeout = instance._child_operation_timeout
        timeout = child_timeout if child_timeout is not None else types.STEP_TIMEOUT
        try:
            terminal = await ability.run_terminal_operation(
                instance.context(),
                child=child,
                request=request,
                terminals=(child.output_event, child.failed_event),
                timeout=timeout,
            )
        except TimeoutError:
            seconds = timeout.total_seconds()
            failure = StepTimedOutData(
                message=f"Cognition's {stage} step took longer than {seconds:g} seconds, so the turn failed.",
                stage=stage,
                timeout_seconds=seconds,
                turn_id=turn.operation_id,
                stimulus=input.stimulus_name(turn.input.stimulus),
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ChildHopTimedOutEvent.with_data_and_id(_HopTimedOutData(turn=turn, failure=failure), hop_id),
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
                turn=turn,
                stage="autonomy",
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
        recent: tuple[memory.stm.Turn, ...],
        acted: types.ActedData | None,
    ) -> None:
        """Build intuition's frame (with the recent turns and what memory surfaced as ``recalled``, and what behaviors
        already did as ``acted``) and hop to it.

        A turn that perceives a handed-back intuition failure (its stimulus, or any percept of its
        ``bot.perceived`` batch) passes intuition unhandled, so reasoning, a stage that can perceive
        that failure, takes it (D-R-30). So does a turn whose intuition frame cannot be built: the
        same turn cascades to reasoning (D-R-36).
        """

        public_metadata = _public_metadata(dict(event.metadata))
        base = operation_id or uuid.uuid4().hex
        perceived = cognition_input.stimulus
        stimuli = (
            tuple(percept.stimulus for percept in perceived.data.percepts)
            if isinstance(perceived, hsm.Event) and isinstance(perceived.data, mosfet.PerceivedData)
            else (perceived,)
        )
        frame: processing.InputData | None = None
        if not any(
            isinstance(stimulus, hsm.Event)
            and isinstance(stimulus.data, StepFailedData | StepTimedOutData)
            and stimulus.data.stage == "intuition"
            for stimulus in stimuli
        ):
            try:
                frame = Cognition._build_processing_input(
                    instance,
                    cognition_input,
                    associations=associations,
                    recent=recent,
                    acted=acted,
                )
            except Exception:
                # The documented boundary: an intuition frame that cannot be built cascades to reasoning, whose
                # own frame build reports what failed if it fails too (D-R-36).
                _HANDED_BACK.add(1, {"stage": "intuition", "kind": "failed", "outcome": "cascaded"})
        if processing.active_operation(instance, base) is None:
            _ = await processing.start_operation(instance, base)
        turn = Cognition._turn(instance, cognition_input, base, associations=associations, recent=recent, acted=acted)
        if frame is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _IntuitionPassedEvent.with_data(types.CompletionData(turn=turn)),
                    id=base,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=public_metadata,
                ),
            )
            return
        hop_id = f"{base}{_INTUITION_ID_SUFFIX}"
        input_event = dataclasses.replace(
            instance._intuition.input_event.with_data_and_id(
                intuition.InputData(turn=turn, processing_input=frame),
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
            turn=turn,
            stage="intuition",
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
        # What behaviors already did on this turn's percepts is the turn's too, ahead of deliberation's (D-W12-9).
        acted = () if data.turn.acted is None else data.turn.acted.acts
        try:
            output = (*acted, *_coerce_output(data.output))
        except Exception as error:
            failed = _failure(operation_id, str(error))
            Cognition._remember(
                ctx,
                instance,
                operation_id=operation_id,
                outcome=memory.Outcome.FAILED,
                metadata=public_metadata,
                cognition_input=cognition_input,
                output=acted,
                failure=failed.failure,
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(failed),
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
            cognition_input=cognition_input,
            cognition_output=output,
            associations=data.turn.associations,
            persona=instance._persona_told,
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
                    recent=data.turn.recent,
                    acted=data.turn.acted,
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
                    recent=(),
                    acted=data.turn.acted,
                )
                return
            active.set_attribute("bot.cognition.entered_from", "input")
            assert is_input(data)
            await Cognition._hop_intuition(
                ctx,
                instance,
                event,
                cognition_input=data,
                operation_id=event.id or None,
                associations=None,
                recent=(),
                acted=None,
            )

    @staticmethod
    async def _start_associating(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Ask, by message and at once, what memory associates with the stimulus and what the recent turns were.

        Associative (long-term memory) and the short-term register answer in parallel, each within the
        step timeout. Either one absent, failing or late leaves the turn without what it would have
        given; neither fails the turn. The recent turns leave out every turn whose id is this turn's
        stimulus (or one of the percepts of its ``bot.perceived`` batch), which the turn already shows.
        """

        with span.operation(
            "bot.cognition.stage",
            scope="bot.abilities.cognition",
            component="cognition",
            stage="associating",
            context=telemetry.event_context(event),
        ) as active:
            data = event.data
            acted: types.ActedData | None = None
            if isinstance(data, types.CompletionData):
                cognition_input = data.turn.input
                operation_id = data.turn.operation_id
                acted = data.turn.acted
            else:
                assert is_input(data)
                cognition_input = data
                operation_id = event.id or uuid.uuid4().hex
            if processing.active_operation(instance, operation_id) is None:
                _ = await processing.start_operation(instance, operation_id)
            child_timeout = instance._child_operation_timeout
            timeout = child_timeout if child_timeout is not None else types.STEP_TIMEOUT
            metadata = _public_metadata(dict(event.metadata))

            async def associate() -> memory.AssociationsData | None:
                associative = instance._associative
                if associative is None:
                    return None
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
                            metadata=metadata,
                        ),
                        terminals=(memory.AssociatedEvent, ability.FailedEvent),
                        timeout=timeout,
                    )
                except (TimeoutError, RuntimeError):
                    return None
                return terminal.data if isinstance(terminal.data, memory.AssociationsData) else None

            async def recall_recent() -> tuple[memory.stm.Turn, ...]:
                register = instance._stm_memory
                if register is None:
                    return ()
                try:
                    terminal = await ability.run_terminal_operation(
                        ctx,
                        child=register,
                        request=dataclasses.replace(
                            memory.stm.RecentTurnsEvent.with_data_and_id(
                                memory.stm.RecentTurnsData(), f"{operation_id}{_RECENT_ID_SUFFIX}"
                            ),
                            metadata=metadata,
                        ),
                        terminals=(memory.stm.RecentTurnsOutputEvent, register.failed_event),
                        timeout=timeout,
                    )
                except (TimeoutError, RuntimeError):
                    return ()
                if not isinstance(terminal.data, memory.stm.RecentTurnsOutputData):
                    return ()
                stimulus = cognition_input.stimulus
                stimuli = (
                    tuple(percept.stimulus for percept in stimulus.data.percepts)
                    if isinstance(stimulus, hsm.Event) and isinstance(stimulus.data, mosfet.PerceivedData)
                    else ()
                )
                # The ids this turn already shows: its stimulus's, or each percept's of its batch, and
                # the causal parents each carries (``parent``, stamped by its producer). Speech admitted
                # through Conversation carries the heard speech two parents up, and Listening wrote
                # its heard turn under that speech's id.
                shown: set[str] = set()
                for given in (stimulus, *stimuli):
                    observed = given if isinstance(given, hsm.Event) else given.observation
                    if observed is not None and observed.id:
                        shown.add(observed.id)
                    data: object = None if observed is None else observed.data
                    for _ in range(_MAX_SHOWN_PARENTS):
                        parent: object = getattr(data, "parent", None)
                        if not isinstance(parent, mosfet.StimulusData):
                            break
                        cause = typing.cast(mosfet.StimulusData[object], parent)
                        if cause.id:
                            shown.add(cause.id)
                        data = cause.data
                return tuple(turn for turn in terminal.data.turns if turn.id not in shown)

            associations, recent = await asyncio.gather(associate(), recall_recent())
            active.set_attribute("bot.cognition.associated", associations is not None)
            turn = Cognition._turn(
                instance, cognition_input, operation_id, associations=associations, recent=recent, acted=acted
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _AssociatedEvent.with_data(_AssociatedData(turn=turn)),
                    id=operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=metadata,
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
    def _is_intuition_passed(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, types.CompletionData):
            return False
        operation = processing.active_operation(instance, data.turn.operation_id)
        # The intuition state selected the pass; envelope and generation correlate the turn.
        return (
            event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
            and event.id == data.turn.operation_id
            and operation is not None
            and data.turn.generation == hsm.id(operation)
        )

    @staticmethod
    def _can_recall(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """Whether a turn has anything to recall beside its stimulus: long-term associations or recent turns."""

        del ctx, event
        return instance._associative is not None or instance._stm_memory is not None

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
            # Intuition's completion (unhandled, or passed), or its failure, which cascades here (D-R-36).
            completion = event.data
            reached = completion.turn if isinstance(completion, types.CompletionData | types.FailureData) else None
            operation_id = None if reached is None else reached.operation_id
            public_metadata = _public_metadata(dict(event.metadata))
            if reached is None:
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
            cognition_input = reached.input
            associations = reached.associations
            recent = reached.recent
            acted = reached.acted
            base = operation_id if operation_id else uuid.uuid4().hex
            try:
                input = Cognition._build_processing_input(
                    instance,
                    cognition_input,
                    associations=associations,
                    recent=recent,
                    acted=acted,
                )
            except Exception as error:
                failed = _failure(operation_id, str(error))
                Cognition._remember(
                    ctx,
                    instance,
                    operation_id=base,
                    outcome=memory.Outcome.FAILED,
                    metadata=public_metadata,
                    cognition_input=cognition_input,
                    output=() if acted is None else acted.acts,
                    failure=failed.failure,
                )
                Cognition._hand_back_step_failure(
                    ctx,
                    instance,
                    stage="reasoning",
                    message=str(error),
                    turn_id=base,
                    cognition_input=cognition_input,
                    metadata=public_metadata,
                )
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _ApplyFailedEvent.with_data(failed),
                        id=operation_id,
                        metadata=public_metadata,
                    ),
                )
                return
            turn = Cognition._turn(
                instance, cognition_input, base, associations=associations, recent=recent, acted=acted
            )
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
                turn=turn,
                stage="reasoning",
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
                output=() if event.data.turn.acted is None else event.data.turn.acted.acts,
                # What failed, without the turn it correlates: the episode already holds that turn.
                failure=ability.FailureData(message=failure.message),
            )
            # The failure's modeled source says which step it was; intuition's failure cascades instead.
            stage: typing.Literal["autonomy", "reasoning"] = (
                "autonomy"
                if instance._autonomy is not None and event.source == hsm.id(instance._autonomy)
                else "reasoning"
            )
            Cognition._hand_back_step_failure(
                ctx,
                instance,
                stage=stage,
                message=failure.message,
                turn_id=operation_id,
                cognition_input=event.data.turn.input,
                metadata=_public_metadata(dict(event.metadata)),
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
    def _cascade_intuition_failure(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        """Count an intuition that failed: its turn goes on to reasoning, with nothing handed back (D-R-36)."""

        del ctx, instance, event
        _HANDED_BACK.add(1, {"stage": "intuition", "kind": "failed", "outcome": "cascaded"})

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
        """A reboot (a step that could not be stopped, or reflection's) ended the turn: it failed."""

        Cognition._remember_opened_turn(ctx, instance, event, memory.Outcome.FAILED)

    @staticmethod
    def _spend_opened_turn(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        """A turn whose step outlasted its timeout was remembered from that timeout: nothing is left to keep."""

        del ctx, instance, event

    @staticmethod
    def _fail_timed_out_step(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        """Fail the turn whose step outlasted its timeout, and hand the body that failure to perceive.

        The turn is remembered as failed from the timeout, which carries it. The body gets the
        turn's typed failure first, then a ``cognition.InputEvent`` observing it with the turn's
        participants: a new turn the bot decides on, the way a recall outcome is.
        """

        data = event.data
        assert isinstance(data, _HopTimedOutData)
        public_metadata = _public_metadata(dict(event.metadata))
        Cognition._remember(
            ctx,
            instance,
            operation_id=data.turn.operation_id,
            outcome=memory.Outcome.FAILED,
            metadata=public_metadata,
            cognition_input=data.turn.input,
            output=() if data.turn.acted is None else data.turn.acted.acts,
            failure=data.failure,
        )
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data.failure),
            id=data.turn.operation_id,
            metadata=public_metadata,
            source=hsm.id(instance),
        )
        ability.Ability._forward_terminal_event(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))
        Cognition._hand_back_failure(
            ctx,
            instance,
            data.failure,
            cognition_input=data.turn.input,
            metadata=public_metadata,
        )

    @staticmethod
    def _hand_back_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        failure: StepFailedData | StepTimedOutData,
        *,
        cognition_input: InputData,
        metadata: dict[str, object],
    ) -> None:
        """Hand the body ``failure``, a failed step's typed failure, to perceive as a new turn.

        The body gets a ``cognition.InputEvent`` observing it with the failed turn's participants: a
        turn the bot decides on, the way a recall outcome is. Bounded: when the failed turn was itself
        taken up only for failed steps (its stimulus carries one, or every percept of its
        ``bot.perceived`` batch does), nothing is handed back, so a step that keeps failing cannot
        keep waking the bot. A batch that also perceived anything else is handed back once.
        """

        timed_out = isinstance(failure, StepTimedOutData)
        kind = "timed_out" if timed_out else "failed"
        perceived = cognition_input.stimulus
        stimuli = (
            tuple(percept.stimulus for percept in perceived.data.percepts)
            if isinstance(perceived, hsm.Event) and isinstance(perceived.data, mosfet.PerceivedData)
            else (perceived,)
        )
        if all(
            isinstance(stimulus, hsm.Event) and isinstance(stimulus.data, StepFailedData | StepTimedOutData)
            for stimulus in stimuli
        ):
            _HANDED_BACK.add(1, {"stage": failure.stage, "kind": kind, "outcome": "bounded"})
            return
        if not instance._attachments:
            return
        owner = instance._attachments[0]
        observed = StepTimedOutEvent.with_data(failure) if timed_out else StepFailedEvent.with_data(failure)
        stimulus = dataclasses.replace(observed, source=hsm.id(instance), metadata=metadata)
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                InputEvent.with_data_and_id(
                    InputData(stimulus=stimulus, participants=cognition_input.participants), uuid.uuid4().hex
                ),
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=metadata,
            ),
        )
        _HANDED_BACK.add(1, {"stage": failure.stage, "kind": kind, "outcome": "handed_back"})

    @staticmethod
    def _hand_back_step_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        *,
        stage: typing.Literal["autonomy", "intuition", "reasoning"],
        message: str,
        turn_id: str,
        cognition_input: InputData,
        metadata: dict[str, object],
    ) -> None:
        """Hand the body the failure of a step that failed (not one that outlasted its timeout)."""

        failure = StepFailedData(
            message=f"Cognition's {stage} step failed, so the turn failed: {message}",
            stage=stage,
            turn_id=turn_id,
            stimulus=input.stimulus_name(cognition_input.stimulus),
        )
        Cognition._hand_back_failure(
            ctx,
            instance,
            failure,
            cognition_input=cognition_input,
            metadata=metadata,
        )

    @staticmethod
    async def _stop_step(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
        child: ability.Ability[typing.Any, typing.Any],
        suffix: str,
        *,
        pending_after_child: bool = False,
    ) -> None:
        """Stop the step that outlasted its timeout the way a cancel stops it.

        The cancellation capability is the turn's own generation, so only this turn's stop settles.
        """

        data = event.data
        assert isinstance(data, _HopTimedOutData)
        operation_id = data.turn.operation_id
        stop = dataclasses.replace(
            CancelEvent.with_data(CancelData(operation_id=operation_id, token=data.turn.generation)),
            id=operation_id,
            source=hsm.id(instance),
            target=hsm.id(instance),
            metadata=dict(event.metadata),
        )
        await Cognition._dispatch_cancel_to_child(
            ctx, instance, stop, child, suffix, pending_after_child=pending_after_child
        )

    @staticmethod
    async def _stop_autonomy(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        child = instance._autonomy
        assert child is not None
        # Autonomy's cancellation is best-effort (see cancelling): the stop settles once it lands.
        await Cognition._stop_step(ctx, instance, event, child, _AUTONOMY_ID_SUFFIX, pending_after_child=True)

    @staticmethod
    async def _stop_intuition(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        await Cognition._stop_step(ctx, instance, event, instance._intuition, _INTUITION_ID_SUFFIX)

    @staticmethod
    async def _stop_reasoning(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        await Cognition._stop_step(ctx, instance, event, instance._reasoning, _REASONING_ID_SUFFIX)

    @staticmethod
    def _is_step_stopped(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> bool:
        """Autonomy's stop landed: Cognition's own acknowledgement, for a turn still live."""

        del ctx
        data = event.data
        return (
            isinstance(data, CancelData)
            and event.id == data.operation_id
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
            and processing.active_operation(instance, data.operation_id) is not None
        )

    @staticmethod
    def _finish_stop(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        """The stopped step no longer holds work: retire the capability its stop was sent under."""

        _ = Cognition._finish_cancellation(ctx, instance, event)

    @staticmethod
    def _request_reboot(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        # A step that could not be stopped within the teardown bound (wired from hsm.after).
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
    def _finish_cancellation(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> tuple[str, str]:
        """Retire the capability a child's cancel was sent under; the cancelled turn's id and token."""

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
        return operation_id, token

    @staticmethod
    def _emit_cancelled(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        operation_id, token = Cognition._finish_cancellation(ctx, instance, event)
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
    def _is_memory_outcome(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """What the bot's own recall brought to mind, what memory now knows of a person, or why it could not:
        from Associative, by data type."""

        del ctx
        return (
            instance._associative is not None
            and event.source == hsm.id(instance._associative)
            and isinstance(event.data, memory.RecollectionData | memory.KnownData | ability.FailureData)
            and bool(instance._attachments)
        )

    @staticmethod
    def _observe(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Hand a routine tick, a recall outcome or an inception failure to the body as an observation; the body
        grants it a turn when free."""

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
    def _is_voice_told(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """The voice the bot speaks in, told by its own inception child (by source and payload type)."""

        del ctx
        return (
            instance._voice is not None
            and event.source == hsm.id(instance._voice)
            and isinstance(event.data, vocal.voice.VoiceData)
        )

    @staticmethod
    def _know_voice(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Keep the voice inception told: every frame built from now on shows it in the self block."""

        del ctx
        told = event.data
        if not isinstance(told, vocal.voice.VoiceData):
            # Unreachable: paired with hsm.guard(_is_voice_told), which narrows this payload.
            return
        instance._voice_told = told

    @staticmethod
    def _is_inception_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """What the bot's inception could not do, from its own inception child, with a body to observe it."""

        del ctx
        return (
            instance._voice is not None
            and event.source == hsm.id(instance._voice)
            and isinstance(event.data, ability.FailureData)
            and bool(instance._attachments)
        )

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

    @staticmethod
    def _accept_handoff(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Acknowledge a handoff to reasoning; intuition leaves its turn unhandled, which is the handoff itself."""

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
            # The handoff to deliberate reasoning: intuition keeps it on its menu only when reasoning
            # holds capacities above intuition's ceiling, and reasoning never offers it to itself.
            hsm.transition(
                hsm.on(reasoning.InputEvent),
                hsm.effect(_accept_handoff),
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
                hsm.on(memory.RecollectedEvent, memory.KnownEvent, ability.FailedEvent),
                hsm.guard(_is_memory_outcome),
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
                hsm.guard(_can_recall),
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
            # Enabled while a child stage runs so ignore and the handoff stay snapshot-offered mid-turn.
            hsm.transition(
                hsm.on(types.IgnoreEvent),
                hsm.effect(_accept_ignore),
            ),
            hsm.transition(
                hsm.on(reasoning.InputEvent),
                hsm.effect(_accept_handoff),
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
                hsm.on(memory.RecollectedEvent, memory.KnownEvent, ability.FailedEvent),
                hsm.guard(_is_memory_outcome),
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
                    hsm.guard(_can_recall),
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
                    hsm.effect(_fail_timed_out_step),
                    hsm.target("/Cognition/processing/stopping/autonomy"),
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
                    hsm.on(_IntuitionPassedEvent),
                    hsm.guard(_is_intuition_passed),
                    hsm.target("/Cognition/processing/reasoning"),
                ),
                hsm.transition(
                    hsm.on(intuition.OutputEvent),
                    hsm.guard(_intuition_is_handled),
                    hsm.effect(_complete_processing),
                    hsm.effect(_finish_turn),
                    hsm.target("/Cognition/ready/settling"),
                ),
                # A failing intuition hands the same turn to reasoning (D-R-36); a timed-out one is stopped below.
                hsm.transition(
                    hsm.on(ability.FailedEvent),
                    hsm.guard(_matches_intuition_failure),
                    hsm.effect(_cascade_intuition_failure),
                    hsm.target("/Cognition/processing/reasoning"),
                ),
                hsm.transition(
                    hsm.on(_ChildHopTimedOutEvent),
                    hsm.guard(_is_child_hop_timeout),
                    hsm.effect(_fail_timed_out_step),
                    hsm.target("/Cognition/processing/stopping/intuition"),
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
                    hsm.effect(_fail_timed_out_step),
                    hsm.target("/Cognition/processing/stopping/reasoning"),
                ),
            ),
            # A step outlasted its timeout: its turn has failed, and the step is stopped the way a
            # cancel stops it. Only a step that cannot be stopped within the teardown bound reboots.
            hsm.state(
                "stopping",
                hsm.initial(hsm.target("intuition")),
                # The turn was remembered as failed with its timeout; its opened input is spent here
                # (unguarded, so it is taken whenever it arrives), never kept for rebooting.
                hsm.transition(hsm.on(_TurnOpenedEvent), hsm.effect(_spend_opened_turn)),
                hsm.transition(
                    hsm.on(processing.CancelledEvent),
                    hsm.guard(_matches_child_cancelled),
                    hsm.effect(_finish_stop, _finish_turn),
                    hsm.target("/Cognition/ready/settling"),
                ),
                hsm.transition(
                    hsm.on(_CancelPendingEvent),
                    hsm.guard(_is_step_stopped),
                    hsm.effect(_finish_stop, _finish_turn),
                    hsm.target("/Cognition/ready/settling"),
                ),
                hsm.transition(
                    hsm.after(_cancel_timeout_delay),
                    hsm.effect(_request_reboot),
                    hsm.target("/Cognition/rebooting"),
                ),
                hsm.state("autonomy", hsm.activity(_stop_autonomy)),
                hsm.state("intuition", hsm.activity(_stop_intuition)),
                hsm.state("reasoning", hsm.activity(_stop_reasoning)),
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
        # Inception gives the bot its voice once, usually while Cognition is still attaching its
        # children, and what it fails at can happen in any phase; neither moves a phase.
        hsm.transition(
            hsm.on(vocal.voice.OutputEvent),
            hsm.guard(_is_voice_told),
            hsm.effect(_know_voice),
        ),
        hsm.transition(
            hsm.on(ability.FailedEvent),
            hsm.guard(_is_inception_failure),
            hsm.effect(_observe),
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
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        intuition: intuition.Intuition,
        reasoning: reasoning.Reasoning,
        reflection: reflection.Reflection,
        autonomy: autonomy.Autonomy | None = None,
        routines: routines.Routines | None = None,
        memory: memory.Memory | None = None,
        associative: memory.Associative | None = None,
        stm_memory: memory.StmMemory | None = None,
        consolidation: memory.Consolidation | None = None,
        persona: persona.Persona | None = None,
        voice: "vocal.voice.Inception | None" = None,
        rest_after: datetime.timedelta = _DEFAULT_REST_AFTER,
        child_operation_timeout: datetime.timedelta | None = None,
        cancel_teardown_timeout: datetime.timedelta | None = None,
    ) -> None:
        super().__init__(id=id, clock=clock)
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
        self._stm_memory = stm_memory
        self._consolidation = consolidation
        self._persona = persona
        self._persona_told = None
        self._voice = voice
        self._voice_told = None
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
        if voice is not None:
            children.append(voice)
        if routines is not None:
            children.append(routines)
        self._attachment_group = attachment.Group(*children)


__all__ = [
    "InputEvent",
    "OutputEvent",
    "Cognition",
    "InputData",
    "OutputData",
    "StepFailedData",
    "StepFailedEvent",
    "StepTimedOutData",
    "StepTimedOutEvent",
]

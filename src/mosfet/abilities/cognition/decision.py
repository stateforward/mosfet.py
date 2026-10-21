from .. import ability
from ..ability import Effort
from .. import processing

import dataclasses
import math
import typing
import uuid

import hsm
import mosfet
import pydantic

from mosfet import telemetry
from mosfet.telemetry import span

from . import input as cognition_input
from . import types

# Absolute floor used only during tuner warmup (few samples). After warmup the
# threshold is mean - k_sigma * std, floored by a soft minimum. Scale is 0–100.
_DEFAULT_MAX_EFFORT = Effort.S
_DEFAULT_CONFIDENCE_FLOOR = 35
_DEFAULT_TUNER_ALPHA = 0.08
_DEFAULT_TUNER_K_SIGMA = 1.0
_DEFAULT_TUNER_WARMUP = 8
# Variance floor on 0–100 scale (~std 10 when expressed as variance 100).
_DEFAULT_TUNER_VAR_FLOOR = 100.0
_DEFAULT_TUNER_MEAN = 70.0
# No static system prose: the model-facing system channel is only the self block (stamped on
# ``InputData.instructions``), followed by the ``<held>`` capacities above this layer's ceiling when
# there are any. The world block (``Environment.model_snapshot`` XML) is the turn's ``situation``.
# Tool schemas and the stimulus carry the rest of the contract.
DEFAULT_INSTRUCTIONS = ""
# How the turn reaches what a later tier holds: the decision layer is never offered the handoff, so its pass.
_PASSING_ON = "passing this turn on to a later tier of thinking"


class EventPatch(pydantic.BaseModel):
    """Decision-layer model-facing fields patched onto every offered event for this turn.

    Processing applies this when ``InputData.patch is EventPatch``; other faculties may supply
    a different BaseModel patch or leave patch None.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    confidence: int = pydantic.Field(
        ge=0,
        le=100,
        description=(
            "Required integer self-assessment of how sure you are that selecting THIS event is "
            "correct right now, on a whole-number scale from 0 to 100. "
            "0 = no idea / wrong event / missing capability; "
            "100 = certain this event should run with the given data. "
            "Use whole integers only (for example 86), never fractions or decimals (not 0.86). "
            "Calibration guide: "
            "80–100 when the stimulus clearly matches the event and required data is known; "
            "40–70 when an action is plausible but the situation is incomplete or ambiguous; "
            "0–35 when you are guessing, the host likely cannot fulfill the request with this "
            "event, or slower deliberation should also run. "
            "The host compares this score to a self-tuned baseline: low confidence can keep "
            "environment actions while passing the turn on to a later tier of thinking."
        ),
        examples=[100, 86, 55, 20, 0],
    )


_AppliedEvent = hsm.Event[types.CompletionData](
    name="bot.ability.decision.applied",
    kind=hsm.CompletionEventKind,
    schema=types.CompletionData,
)
_ApplyFailedEvent = hsm.Event[types.FailureData](
    name="bot.ability.decision.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=types.FailureData,
)


class InputData(pydantic.BaseModel):
    """Typed Cognition request for one intuitive processing stage."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    processing_input: processing.InputData


class _SelectionRejectionData(pydantic.BaseModel):
    """Typed retry context for one rejected model selection.

    ``message`` is a normalized, redacted diagnostic. The raw exception is deliberately not
    carried across the HSM boundary because Pydantic may include the rejected value in it.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    request: InputData = pydantic.Field(description="Original decision request being retried.")
    message: str = pydantic.Field(
        min_length=1,
        description="Redacted dispatch rejection diagnostic suitable for model repair feedback.",
    )
    normalized: str = pydantic.Field(
        min_length=1,
        description="Stable rejection key used to detect the same failure twice consecutively.",
    )
    previous_normalized: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Normalized key from the immediately preceding rejected selection, when any.",
    )
    attempt: int = pydantic.Field(
        ge=0,
        le=types.MAX_SELECTION_REPAIRS,
        description="Zero-based bounded invocation attempt that produced this rejection.",
    )


_SelectionRejectedEvent = hsm.Event[_SelectionRejectionData](
    name="bot.ability.decision.selection.rejected",
    kind=hsm.ErrorEventKind,
    schema=_SelectionRejectionData,
)


class _InvocationCompletedData(pydantic.BaseModel):
    """Typed processor result awaiting confidence and dispatch classification."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    request: InputData
    input: processing.InputData
    product: types.OutputData | None
    confidence: int | None
    attempt: int = pydantic.Field(ge=0, le=types.MAX_SELECTION_REPAIRS)
    previous_normalized: str | None = None
    event_confidences: tuple[int | None, ...] | None = None
    hand_to: processing.HandOn | None = None


class _DispatchPlanData(pydantic.BaseModel):
    """Typed classified result requiring environment dispatch before publication."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    request: InputData
    input: processing.InputData
    selections: types.OutputData
    terminal: types.OutputData | None
    attempt: int = pydantic.Field(ge=0, le=types.MAX_SELECTION_REPAIRS)
    previous_normalized: str | None = None


class _ClassificationData(pydantic.BaseModel):
    """Typed immutable input for classification after the RTC tuner update."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    invocation: _InvocationCompletedData
    escalate: bool


class _PublishData(pydantic.BaseModel):
    """Typed classified or dispatched result ready for terminal publication."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    completion: types.CompletionData


_InvocationCompletedEvent = hsm.Event[_InvocationCompletedData](
    name="bot.ability.decision.invocation.completed",
    kind=hsm.CompletionEventKind,
    schema=_InvocationCompletedData,
)
_ClassificationReadyEvent = hsm.Event[_ClassificationData](
    name="bot.ability.decision.classification.ready",
    kind=hsm.CompletionEventKind,
    schema=_ClassificationData,
)
_DispatchPlannedEvent = hsm.Event[_DispatchPlanData](
    name="bot.ability.decision.dispatch.planned",
    kind=hsm.CompletionEventKind,
    schema=_DispatchPlanData,
)
_PublishReadyEvent = hsm.Event[_PublishData](
    name="bot.ability.decision.publish.ready",
    kind=hsm.CompletionEventKind,
    schema=_PublishData,
)


class OutputData(pydantic.BaseModel):
    """Reflex selection produced by a decision processor.

    Confidence is not on this envelope. Tool-calling processors patch ``confidence`` onto each
    offered event payload; the host unpatches into ``SelectedEvent.confidence`` and aggregates
    for escalate policy. This type is only for explicit result / unhandled / reason shapes.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Reflex selection produced by the decision layer. result is a list of selected modeled "
                "events from the offered schemas (multi-select allowed). Confidence is reported on "
                "each selected event's patched payload (not here). "
                "Omit result / leave unhandled when no offered event should run."
            ),
            "examples": [
                {
                    "result": [],
                    "reason": "The active runtime already owns the input.",
                },
                # No worked example of an utterance here, and none anywhere a model reads: an
                # example the bot can say is a crib rather than documentation, and an unsure bot
                # says it. Selections shown here are structural for that reason.
                {
                    "result": [
                        {
                            "target": "bot",
                            "event": "bot.focus_device",
                            "data": {"device": "device-a"},
                            "reason": "That device is where this turn is happening.",
                        }
                    ],
                    "reason": "Recognized the stimulus without needing to deliberate.",
                },
                {
                    "reason": "The decision processor did not select an output.",
                },
            ],
        },
    )

    result: types.OutputData | None = pydantic.Field(
        default=None,
        description=(
            "Selected modeled events for this turn (multi-select). Empty tuple selects no events "
            "but is still handled when per-event confidence is high enough. Omit/None leaves the "
            "input unhandled for the host cascade."
        ),
        examples=[[]],
    )
    hand_to: processing.HandOn | None = pydantic.Field(
        default=None,
        description=(
            "The later tier a pass (``result`` None) hands the turn to (R-23): intuition, to answer it in words "
            "now, or reasoning, to think it through first. None hands it to the first tier the turn offers."
        ),
        examples=["reasoning"],
    )
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional concise rationale for the intuitive selection.",
        examples=["The active runtime already owns the input."],
    )


@dataclasses.dataclass
class ConfidenceTuner:
    """Self-tuning confidence baseline for passing a turn on to a later tier.

    Scores are integers **0–100**. Tracks a running mean/variance of reported confidences
    (EMA-style, cortext-like local normalization). Escalate when confidence falls below an
    adaptive floor:

        threshold = max(soft_floor, mean - k_sigma * std)

    During warmup, only the absolute ``floor`` is used so a single model cannot be
    judged against another model's scale.
    """

    mean: float = _DEFAULT_TUNER_MEAN
    var: float = _DEFAULT_TUNER_VAR_FLOOR
    n: int = 0
    alpha: float = _DEFAULT_TUNER_ALPHA
    k_sigma: float = _DEFAULT_TUNER_K_SIGMA
    floor: int = _DEFAULT_CONFIDENCE_FLOOR
    warmup: int = _DEFAULT_TUNER_WARMUP
    var_floor: float = _DEFAULT_TUNER_VAR_FLOOR

    def observe(self, confidence: int) -> None:
        """Update running stats with a newly reported 0–100 confidence."""

        conf = float(max(0, min(100, confidence)))
        if self.n == 0:
            self.mean = conf
            self.var = self.var_floor
            self.n = 1
            return
        delta = conf - self.mean
        self.mean = self.mean + self.alpha * delta
        self.var = max(
            self.var_floor,
            (1.0 - self.alpha) * self.var + self.alpha * delta * delta,
        )
        self.n += 1

    def threshold(self) -> float:
        """Current escalate threshold on the 0–100 scale (confidence below this passes the turn on)."""

        if self.n < self.warmup:
            return float(self.floor)
        std = math.sqrt(max(self.var, self.var_floor))
        adaptive = self.mean - self.k_sigma * std
        # Soft floor avoids collapsing to zero when variance spikes; no global fixed cut.
        soft_floor = 0.5 * float(self.floor)
        return max(soft_floor, adaptive)

    def should_escalate(self, confidence: int | None) -> bool:
        """True when reported confidence is too low for this ability's baseline."""

        if confidence is None:
            # No score → cannot self-tune; do not force escalate (explicit unhandled still cascades).
            return False
        return float(confidence) < self.threshold()


def _events_as_output(value: object) -> tuple[types.OutputData | None, int | None, tuple[int | None, ...] | None]:
    """Coerce event selections; return (product, min aggregate, per-event confidences).

    The per-event numbers ride the invoke frame for the host's own policy (a composition
    confidence gate reads them at classify); the model-facing EventData envelope stays
    clean — host-computed values never enter a model-facing contract.
    """

    selections = processing.coerce_event_selections(value, patch=EventPatch)
    if selections is None:
        return None, None, None
    product = types.OUTPUT_SCHEMA_CONTRACT.validate_python(
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
    per_event = tuple(item.confidence for item in selections)
    return product, processing.selection_confidence(selections), per_event


def _product_from_processor_output(
    output: object,
) -> tuple[types.OutputData | None, int | None, tuple[int | None, ...] | None, processing.HandOn | None]:
    """Return (event product, min aggregate confidence, per-event confidences, the tier a pass chose)."""

    if isinstance(output, processing.Result):
        result = typing.cast(processing.Result[object], output)
        if not result.is_handled:
            return None, None, None, None
        return _product_from_processor_output(result.output)
    if output is None:
        return None, None, None, None
    if isinstance(output, OutputData):
        # Envelope has no confidence; only explicit unhandled (result is None), with its chosen tier, vs product.
        if output.result is None:
            return None, None, None, output.hand_to
        product, aggregate, per_event = _events_as_output(output.result)
        return product, aggregate, per_event, None
    if types.is_output(output):
        return output, None, None, None
    events, aggregate, per_event = _events_as_output(output)
    if events is not None:
        return events, aggregate, per_event, None
    raise TypeError("Decision processor produced output that does not match a cognitive output schema.")


def _is_deliberative_input_event(
    event_name: str,
    schemas: dict[str, hsm.Event[typing.Any]],
) -> bool:
    """True for deliberate handoff tools — host cascade / multi-select handoff to System 2.

    Detected without importing reasoning: shared event-name constant and/or the schema
    marker ``__deliberative_handoff__`` (see :func:`processing.is_deliberative_handoff_schema`).
    Name match covers cascade selections that are not in the current tool menu.
    """

    if types.is_deliberative_handoff_event_name(event_name):
        return True
    schema_event = schemas.get(event_name)
    if schema_event is None:
        return False
    return processing.is_deliberative_handoff_schema(getattr(schema_event, "schema", None))


def _environment_actions(
    output: types.OutputData,
    *,
    current_input: processing.InputData,
) -> types.OutputData:
    """Action events safe to fire before passing an uncertain turn on to a later tier.

    Cognition ignore is internal (not an environment action) and is never pre-fired on escalate.
    """

    schemas = {event.name: event for event in current_input.schemas}
    return tuple(
        item
        for item in output
        if not _is_deliberative_input_event(item.event, schemas) and not types.is_ignore_event(item.event)
    )


def _selections_from_output(
    output: types.OutputData,
) -> processing.Events:
    """Build ordinary actor selections owned by this decision result."""

    selections: list[processing.SelectedEvent] = []
    for item in output:
        raw: object = item.data if item.data is not None else {}
        selections.append(
            processing.SelectedEvent(
                event=item.event,
                target=item.target,
                data=raw,
                reason=item.reason,
            )
        )
    return tuple(selections)


class Decision(processing.Processing):
    """Reflex's fallback layer: the decision model, on what no behavior handled; multi-dispatch when handled.

    It decides on a ``processing.DecisionProcessor``, a decision model, never a chat model;
    any other processor is a ``TypeError``. It is never offered the deliberative handoff: its pass
    (an unhandled result) is how it hands the turn on. Processor-reported ``confidence`` updates a
    per-instance baseline. When confidence is low relative to that baseline, environment actions
    still dispatch, but the terminal is unhandled so Cognition passes the turn on to the next stage.
    High confidence non-empty terminals keep the selection list (no hand-on). Empty selections and
    explicit unhandled (``result is None``) hand the turn on as well; a deliberate pass on the
    stimulus itself uses ``bot.ability.cognition.ignore``.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = types.CompletionData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.decision.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[types.CompletionData]] = hsm.Event[types.CompletionData](
        name="bot.ability.decision.output",
        schema=types.CompletionData,
    )
    failed_event: typing.ClassVar[hsm.Event[types.FailureData]] = hsm.Event[types.FailureData](
        name=ability.FailedEvent.name,
        kind=hsm.ErrorEventKind,
        schema=types.FailureData,
    )
    instructions: typing.ClassVar[str] = DEFAULT_INSTRUCTIONS
    _processor: processing.DecisionProcessor
    _instructions: str
    _confidence_tuner: ConfidenceTuner
    _confidence_gate: int | None
    _max_effort: Effort | None

    @staticmethod
    def _has_decision_input(ctx: hsm.Context, instance: "Decision", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, InputData)

    @staticmethod
    def _has_decision_applied(ctx: hsm.Context, instance: "Decision", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, types.CompletionData)

    @staticmethod
    def _has_apply_failure(ctx: hsm.Context, instance: "Decision", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, types.FailureData)

    @staticmethod
    def _has_invocation_completed(
        ctx: hsm.Context,
        instance: "Decision",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, _InvocationCompletedData)

    @staticmethod
    def _has_dispatch_plan(
        ctx: hsm.Context,
        instance: "Decision",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, _DispatchPlanData)

    @staticmethod
    def _has_classification_ready(
        ctx: hsm.Context,
        instance: "Decision",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, _ClassificationData)

    @staticmethod
    def _has_publish_ready(ctx: hsm.Context, instance: "Decision", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _PublishData)

    @staticmethod
    def _is_same_selection_rejection(
        ctx: hsm.Context,
        instance: "Decision",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _SelectionRejectionData) and data.previous_normalized == data.normalized

    @staticmethod
    def _can_retry_selection_rejection(
        ctx: hsm.Context,
        instance: "Decision",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _SelectionRejectionData) and data.attempt < types.MAX_SELECTION_REPAIRS

    @staticmethod
    def _dispatch_terminal_output(
        ctx: hsm.Context,
        instance: "Decision",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        output: types.CompletionData,
        target: str | None,
    ) -> None:
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=operation_id,
            metadata=dict(metadata),
            source=hsm.id(instance),
            target=target,
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))
        if operation_id is not None:
            processing.finish_operation(ctx, instance, operation_id)

    @staticmethod
    def _dispatch_terminal_failure(
        ctx: hsm.Context,
        instance: "Decision",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        failure: types.FailureData,
        target: str | None,
    ) -> None:
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=operation_id,
            metadata=dict(metadata),
            source=hsm.id(instance),
            target=target,
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))
        if operation_id is not None:
            processing.finish_operation(ctx, instance, operation_id)

    @staticmethod
    def _decision_input_for_processor(
        instance: "Decision",
        narrowed: processing.InputData,
        held: tuple[hsm.Event[typing.Any], ...],
    ) -> processing.InputData:
        """Stamp instructions, what a later tier holds, and the EventPatch on the narrowed frame.

        ``held`` (see :func:`cognition_input.held`) renders after the self block as one
        ``<held>`` element (:func:`cognition_input.with_held`). The decision layer is never offered the
        handoff: its pass is how the turn reaches them.
        """

        stamped = cognition_input.with_held(
            processing.Processing._input_for_processor(instance, narrowed), held, reached_by=_PASSING_ON
        )
        return stamped if stamped.patch is not None else stamped.model_copy(update={"patch": EventPatch})

    @staticmethod
    async def _invoke_activity(ctx: hsm.Context, instance: "Decision", event: hsm.Event[typing.Any]) -> None:
        with span.operation(
            "bot.reflex.invoke",
            scope="bot.abilities.cognition",
            component="cognition.decision",
            stage="decision_invoke",
            context=telemetry.event_context(event),
        ) as active:
            retry = event.data if isinstance(event.data, _SelectionRejectionData) else None
            data = retry.request if retry is not None else event.data
            assert isinstance(data, InputData)
            reply_to = (
                event.source
                if retry is not None or (event.target == hsm.id(instance) and bool(event.source))
                else hsm.id(instance)
            )
            attempt = retry.attempt + 1 if retry is not None else 0
            previous_normalized = retry.normalized if retry is not None else None
            operation_id = event.id if event.id else uuid.uuid4().hex
            if processing.active_operation(instance, operation_id) is None:
                _ = await processing.start_operation(instance, operation_id)
            metadata = dict(event.metadata)
            try:
                # Narrowing reads live actor snapshots, so a frame that cannot be read fails the turn.
                narrowed = cognition_input.within_effort(data.processing_input, instance._max_effort)
                held = cognition_input.held(data.processing_input, narrowed)
                # The decision layer is never offered the handoff: its pass is the hand-on.
                narrowed = cognition_input.without_handoff(narrowed)
                active.set_attribute("bot.reflex.held.count", len(held))
                input = Decision._decision_input_for_processor(instance, narrowed, held)
                if retry is not None:
                    input = types.with_selection_feedback(input, retry.message)
                raw = await instance._processor.process(input)
                product, confidence, event_confidences, hand_to = _product_from_processor_output(raw)
            except processing.SelectionRejectionError as error:
                # The processor refused a selection (an event this turn does not offer): the model's to repair.
                active.set_attribute("bot.selection.rejected", True)
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _SelectionRejectedEvent.with_data(
                            _SelectionRejectionData(
                                request=data,
                                message=types.selection_rejection_message(error.normalized),
                                normalized=types.selection_rejection_key(error.normalized),
                                previous_normalized=previous_normalized,
                                attempt=attempt,
                            )
                        ),
                        id=operation_id,
                        source=reply_to,
                        target=hsm.id(instance),
                        metadata=metadata,
                    ),
                )
                return
            except Exception as error:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _ApplyFailedEvent.with_data(types.FailureData(message=str(error), turn=data.turn)),
                        id=operation_id,
                        source=reply_to,
                        target=hsm.id(instance),
                        metadata=metadata,
                    ),
                )
                return
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _InvocationCompletedEvent.with_data(
                        _InvocationCompletedData(
                            request=data,
                            input=input,
                            product=product,
                            confidence=confidence,
                            attempt=attempt,
                            previous_normalized=previous_normalized,
                            event_confidences=event_confidences,
                            hand_to=hand_to,
                        )
                    ),
                    id=operation_id,
                    source=reply_to,
                    target=hsm.id(instance),
                    metadata=metadata,
                ),
            )

    @staticmethod
    def _tune_confidence(ctx: hsm.Context, instance: "Decision", event: hsm.Event[typing.Any]) -> None:
        """Update the owned tuner during RTC and emit immutable classification input."""

        data = event.data
        assert isinstance(data, _InvocationCompletedData)
        if data.confidence is not None:
            instance._confidence_tuner.observe(data.confidence)
        classification = _ClassificationData(
            invocation=data,
            escalate=instance._confidence_tuner.should_escalate(data.confidence),
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ClassificationReadyEvent.with_data(classification),
                id=event.id,
                source=event.source,
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _classify_activity(ctx: hsm.Context, instance: "Decision", event: hsm.Event[typing.Any]) -> None:
        classification = event.data
        assert isinstance(classification, _ClassificationData)
        data = classification.invocation
        with span.operation(
            "bot.reflex.classify",
            scope="bot.abilities.cognition",
            component="cognition.decision",
            stage="decision_classify",
            context=telemetry.event_context(event),
        ) as active:
            escalate = classification.escalate
            product = data.product
            if product is None or len(product) == 0:
                terminal: types.OutputData | None = None
                selections: types.OutputData = ()
                escalation = "empty_product"
            else:
                gate = instance._confidence_gate
                if gate is not None and data.event_confidences is not None:
                    product_before_gate = len(product)
                    per_event = data.event_confidences
                    kept = tuple(
                        (item, conf)
                        for item, conf in zip(product, per_event, strict=True)
                        if conf is None or conf >= gate
                    )
                    product = tuple(item for item, _ in kept)
                    active.set_attribute("bot.before_gate.count", product_before_gate)
                    active.set_attribute("bot.after_gate.count", len(product))
                if len(product) == 0:
                    # The gate dropped everything: the turn is an explicit unhandled cascade,
                    # never a handled-empty selection (the pass-criterion lesson).
                    selections = ()
                    terminal = None
                    escalation = "gate_dropped_all"
                elif escalate or any(
                    _is_deliberative_input_event(item.event, {schema.name: schema for schema in data.input.schemas})
                    for item in product
                ):
                    selections = _environment_actions(product, current_input=data.input)
                    terminal = None
                    escalation = "low_confidence" if escalate else "deliberative_input"
                else:
                    selections = product
                    terminal = product
                    escalation = "none"
            active.set_attribute("bot.selection.count", len(selections))
            active.set_attribute("bot.cognition.escalated", terminal is None)
            active.set_attribute("bot.cognition.escalation.reason", escalation)
            # Only the decision model's own pass chooses the tier (R-23); every other hand-on keeps the first one
            # the turn offers.
            turn = (
                data.request.turn.model_copy(update={"hand_to": data.hand_to})
                if terminal is None and data.hand_to is not None
                else data.request.turn
            )
            if terminal is None:
                offered = data.input.hand_ons or ("reasoning",)
                active.set_attribute("bot.reflex.handed_to", data.hand_to or offered[0])
            if selections and data.input.actors:
                next_event: hsm.Event[typing.Any] = _DispatchPlannedEvent.with_data(
                    _DispatchPlanData(
                        request=data.request,
                        input=data.input,
                        selections=selections,
                        terminal=terminal,
                        attempt=data.attempt,
                        previous_normalized=data.previous_normalized,
                    )
                )
            else:
                next_event = _PublishReadyEvent.with_data(
                    _PublishData(completion=types.CompletionData(turn=turn, output=terminal))
                )
            await hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    next_event,
                    id=event.id,
                    source=event.source,
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    async def _dispatch_selections_activity(
        ctx: hsm.Context,
        instance: "Decision",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _DispatchPlanData)
        with span.operation(
            "bot.reflex.dispatch",
            scope="bot.abilities.cognition",
            component="cognition.decision",
            stage="decision_dispatch",
            context=telemetry.event_context(event),
        ) as active:
            try:
                await types.dispatch_selected_events(
                    ctx,
                    data.input,
                    _selections_from_output(data.selections),
                    operation_id=event.id or hsm.id(instance),
                    source=instance,
                    focus_candidates=data.request.turn.input.focus_candidates,
                    metadata=dict(event.metadata),
                    dispatch_trust=processing.DispatchTrust.MODEL,
                )
            except processing.SelectionRejectionError as error:
                normalized = types.selection_rejection_key(error.normalized)
                active.set_attribute("bot.selection.rejected", True)
                active.set_attribute(
                    "bot.selection.rejection_repeat",
                    data.previous_normalized == normalized,
                )
                active.set_attribute("bot.selection.retry", data.attempt < types.MAX_SELECTION_REPAIRS)
                next_event = _SelectionRejectedEvent.with_data(
                    _SelectionRejectionData(
                        request=data.request,
                        message=types.selection_rejection_message(error.normalized),
                        normalized=normalized,
                        previous_normalized=data.previous_normalized,
                        attempt=data.attempt,
                    )
                )
            except Exception as error:
                next_event = _ApplyFailedEvent.with_data(types.FailureData(message=str(error), turn=data.request.turn))
            else:
                # Actions dispatched before passing the turn on ride its ``acted`` to the next stage (D-R-52).
                turn = (
                    data.request.turn
                    if data.terminal is not None
                    else types.with_acts(data.request.turn, data.selections)
                )
                next_event = _PublishReadyEvent.with_data(
                    _PublishData(completion=types.CompletionData(turn=turn, output=data.terminal))
                )
            await hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    next_event,
                    id=event.id,
                    source=event.source,
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    async def _publish_activity(ctx: hsm.Context, instance: "Decision", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _PublishData)
        await hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _AppliedEvent.with_data(data.completion),
                id=event.id,
                source=event.source,
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _complete_apply(ctx: hsm.Context, instance: "Decision", event: hsm.Event[typing.Any]) -> None:
        output = event.data
        assert isinstance(output, types.CompletionData)
        Decision._dispatch_terminal_output(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            output=output,
            target=event.source if event.source != hsm.id(instance) else None,
        )

    @staticmethod
    def _fail_apply(ctx: hsm.Context, instance: "Decision", event: hsm.Event[typing.Any]) -> None:
        failure = event.data
        assert isinstance(failure, types.FailureData)
        Decision._dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            failure=failure,
            target=event.source if event.source != hsm.id(instance) else None,
        )

    @staticmethod
    def _fail_selection_rejection(ctx: hsm.Context, instance: "Decision", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _SelectionRejectionData)
        Decision._dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            failure=types.FailureData(message=data.message, turn=data.request.turn),
            target=event.source if event.source != hsm.id(instance) else None,
        )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Decision",
        hsm.initial(hsm.target("/Decision/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(processing.Processing._emit_cancelled),
            ),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_decision_input),
                hsm.target("/Decision/workflow"),
            ),
        ),
        hsm.state(
            "workflow",
            hsm.initial(hsm.target("/Decision/workflow/invoking")),
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(processing.Processing._emit_cancelled),
                hsm.target("/Decision/idle"),
            ),
            hsm.state(
                "invoking",
                hsm.activity(_invoke_activity),
                hsm.transition(
                    hsm.on(_InvocationCompletedEvent),
                    hsm.guard(_has_invocation_completed),
                    hsm.effect(_tune_confidence),
                    hsm.target("/Decision/workflow/tuning"),
                ),
                hsm.transition(
                    hsm.on(_SelectionRejectedEvent),
                    hsm.target("/Decision/workflow/routing_rejection"),
                ),
                hsm.transition(
                    hsm.on(_ApplyFailedEvent),
                    hsm.guard(_has_apply_failure),
                    hsm.effect(_fail_apply),
                    hsm.target("/Decision/idle"),
                ),
            ),
            hsm.state(
                "tuning",
                hsm.transition(
                    hsm.on(_ClassificationReadyEvent),
                    hsm.guard(_has_classification_ready),
                    hsm.target("/Decision/workflow/classifying"),
                ),
            ),
            hsm.state(
                "classifying",
                hsm.activity(_classify_activity),
                hsm.transition(
                    hsm.on(_DispatchPlannedEvent),
                    hsm.guard(_has_dispatch_plan),
                    hsm.target("/Decision/workflow/dispatching"),
                ),
                hsm.transition(
                    hsm.on(_PublishReadyEvent),
                    hsm.guard(_has_publish_ready),
                    hsm.target("/Decision/workflow/publishing"),
                ),
            ),
            hsm.state(
                "dispatching",
                hsm.activity(_dispatch_selections_activity),
                hsm.transition(
                    hsm.on(_PublishReadyEvent),
                    hsm.guard(_has_publish_ready),
                    hsm.target("/Decision/workflow/publishing"),
                ),
                hsm.transition(
                    hsm.on(_SelectionRejectedEvent),
                    hsm.target("/Decision/workflow/routing_rejection"),
                ),
                hsm.transition(
                    hsm.on(_ApplyFailedEvent),
                    hsm.guard(_has_apply_failure),
                    hsm.effect(_fail_apply),
                    hsm.target("/Decision/idle"),
                ),
            ),
            hsm.choice(
                "routing_rejection",
                hsm.transition(
                    hsm.guard(_is_same_selection_rejection),
                    hsm.effect(_fail_selection_rejection),
                    hsm.target("/Decision/idle"),
                ),
                hsm.transition(
                    hsm.guard(_can_retry_selection_rejection),
                    hsm.target("/Decision/workflow/invoking"),
                ),
                hsm.transition(
                    hsm.effect(_fail_selection_rejection),
                    hsm.target("/Decision/idle"),
                ),
            ),
            hsm.state(
                "publishing",
                hsm.activity(_publish_activity),
                hsm.transition(
                    hsm.on(_AppliedEvent),
                    hsm.guard(_has_decision_applied),
                    hsm.effect(_complete_apply),
                    hsm.target("/Decision/idle"),
                ),
            ),
        ),
    )

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        processor: processing.DecisionProcessor,
        instructions: str | None = None,
        confidence_floor: int = _DEFAULT_CONFIDENCE_FLOOR,
        confidence_tuner: ConfidenceTuner | None = None,
        confidence_gate: int | None = None,
        max_effort: Effort | None = _DEFAULT_MAX_EFFORT,
    ) -> None:
        # Precondition (R-6): decision decides on a decision model, never a chat model. The checker
        # sees the annotation; this holds for a caller it cannot see.
        if processing.DecisionProcessor not in type(processor).__mro__:
            raise TypeError(f"Decision decides on a processing.DecisionProcessor, not a {type(processor).__name__}.")
        if not 0 <= confidence_floor <= 100:
            raise ValueError("confidence_floor must be an integer between 0 and 100.")
        if confidence_gate is not None and not 0 <= confidence_gate <= 100:
            raise ValueError("confidence_gate must be None or an integer between 0 and 100.")
        resolved = type(self).instructions if instructions is None else instructions
        if instructions is not None and not instructions.strip():
            raise ValueError("instructions must not be blank when provided.")
        # Empty default is intentional: system channel is only the per-turn world XML when present.
        super(processing.Processing, self).__init__(id=id, clock=clock)
        self._instructions = resolved.strip() if resolved else ""
        self._processor = processor
        self._confidence_tuner = confidence_tuner or ConfidenceTuner(floor=confidence_floor)
        self._confidence_gate = confidence_gate
        self._max_effort = max_effort


InputEvent = Decision.input_event
OutputEvent = Decision.output_event

__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "ConfidenceTuner",
    "EventPatch",
    "InputEvent",
    "InputData",
    "OutputEvent",
    "Decision",
    "OutputData",
]

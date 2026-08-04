from .. import ability
from .. import processing

import dataclasses
import math
import typing
import uuid

import hsm
import pydantic

from bot.telemetry import observer

from . import types

# Absolute floor used only during tuner warmup (few samples). After warmup the
# threshold is mean - k_sigma * std, floored by a soft minimum. Scale is 0–100.
_DEFAULT_CONFIDENCE_FLOOR = 35
_DEFAULT_TUNER_ALPHA = 0.08
_DEFAULT_TUNER_K_SIGMA = 1.0
_DEFAULT_TUNER_WARMUP = 8
# Variance floor on 0–100 scale (~std 10 when expressed as variance 100).
_DEFAULT_TUNER_VAR_FLOOR = 100.0
_DEFAULT_TUNER_MEAN = 70.0
# No static system prose: the model-facing system channel is only the per-turn world block
# (``Environment.model_snapshot`` XML stamped on ``InputData.instructions``). Tool schemas and
# the stimulus carry the rest of the contract.
DEFAULT_INSTRUCTIONS = ""


class EventPatch(pydantic.BaseModel):
    """Intuition-only model-facing fields patched onto every offered event for this turn.

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
            "environment actions while escalating the turn to deliberate reasoning."
        ),
        examples=[100, 86, 55, 20, 0],
    )


_AppliedEvent = hsm.Event[object](
    name="bot.ability.intuition.applied",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)
_ApplyFailedEvent = hsm.Event[types.FailureData](
    name="bot.ability.intuition.apply.failed",
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


class OutputData(pydantic.BaseModel):
    """Fast cognitive selection produced by an intuition processor.

    Confidence is not on this envelope. Tool-calling processors patch ``confidence`` onto each
    offered event payload; the host unpatches into ``SelectedEvent.confidence`` and aggregates
    for escalate policy. This type is only for explicit result / unhandled / reason shapes.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Fast cognitive selection produced by intuition. result is a list of selected modeled "
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
                    "reason": "The intuition processor did not select an output.",
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
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional concise rationale for the intuitive selection.",
        examples=["The active runtime already owns the input."],
    )


IntuitionProcessor: typing.TypeAlias = processing.Processor


@dataclasses.dataclass
class ConfidenceTuner:
    """Self-tuning confidence baseline for escalate-to-reasoning.

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
        """Current escalate threshold on the 0–100 scale (confidence below this → System 2)."""

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


def _events_as_output(value: object) -> tuple[types.OutputData | None, int | None]:
    """Coerce event selections and lift patched per-event confidence (min aggregate)."""

    selections = processing.coerce_event_selections(value, patch=EventPatch)
    if selections is None:
        return None, None
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
    return product, processing.selection_confidence(selections)


def _product_from_processor_output(output: object) -> tuple[types.OutputData | None, int | None]:
    """Return (event product, optional confidence) without applying escalate policy."""

    if isinstance(output, processing.Result):
        result = typing.cast(processing.Result[object], output)
        if not result.is_handled:
            return None, None
        return _product_from_processor_output(result.output)
    if output is None:
        return None, None
    if isinstance(output, OutputData):
        # Envelope has no confidence; only explicit unhandled (result is None) vs product.
        if output.result is None:
            return None, None
        return output.result, processing.selection_confidence(
            processing.coerce_event_selections(output.result, patch=EventPatch) or ()
        )
    if types.is_output(output):
        return output, None
    events, confidence = _events_as_output(output)
    if events is not None:
        return events, confidence
    raise TypeError("Intuition processor produced output that does not match a cognitive output schema.")


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
    """Action events safe to fire before handing an uncertain turn to deliberate reasoning.

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
    """Build ordinary actor selections owned by this intuition result."""

    selections: list[processing.SelectedEvent] = []
    for item in output:
        raw = item.data if item.data is not None else {}
        selections.append(
            processing.SelectedEvent(
                event=item.event,
                target=item.target,
                data=raw,
                reason=item.reason,
            )
        )
    return tuple(selections)


class Intuition(processing.Processing):
    """Fast cognitive ability: multi-dispatch when handled; self-tuning confidence gates System 2.

    Processor-reported ``confidence`` updates a per-instance baseline. When confidence is
    low relative to that baseline, environment actions still dispatch, but the terminal is
    unhandled so Cognition cascades to reasoning. High confidence non-empty terminals keep
    the selection list (no cascade). Empty selections and explicit unhandled
    (``result is None``) cascade to reasoning by default; deliberate pass uses
    ``bot.ability.cognition.ignore``.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = types.CompletionData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.intuition.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[types.CompletionData]] = hsm.Event[types.CompletionData](
        name="bot.ability.intuition.output",
        schema=types.CompletionData,
    )
    failed_event: typing.ClassVar[hsm.Event[types.FailureData]] = hsm.Event[types.FailureData](
        name=ability.FailedEvent.name,
        kind=hsm.ErrorEventKind,
        schema=types.FailureData,
    )
    instructions: typing.ClassVar[str] = DEFAULT_INSTRUCTIONS
    _processor: processing.Processor
    _instructions: str
    _confidence_tuner: ConfidenceTuner

    @staticmethod
    def _has_intuition_input(ctx: hsm.Context, instance: "Intuition", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, InputData)

    @staticmethod
    def _has_intuition_applied(ctx: hsm.Context, instance: "Intuition", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        # Topology is already on _AppliedEvent; narrow by typed completion payload.
        return isinstance(event.data, types.CompletionData)

    @staticmethod
    def _has_apply_failure(ctx: hsm.Context, instance: "Intuition", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, types.FailureData)

    @staticmethod
    def _dispatch_terminal_output(
        ctx: hsm.Context,
        instance: "Intuition",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        output: types.CompletionData,
    ) -> None:
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=operation_id,
            metadata=dict(metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))
        if operation_id is not None:
            processing.finish_operation(ctx, instance, operation_id)

    @staticmethod
    def _dispatch_terminal_failure(
        ctx: hsm.Context,
        instance: "Intuition",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        failure: types.FailureData,
    ) -> None:
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=operation_id,
            metadata=dict(metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))
        if operation_id is not None:
            processing.finish_operation(ctx, instance, operation_id)

    @staticmethod
    def _intuition_input_for_processor(instance: "Intuition", input: processing.InputData) -> processing.InputData:
        """Stamp instructions and intuition EventPatch (confidence) onto the processing input."""

        stamped = processing.Processing._input_for_processor(instance, input)
        updates: dict[str, object] = {}
        if stamped.patch is None:
            updates["patch"] = EventPatch
        if updates:
            return stamped.model_copy(update=updates)
        return stamped

    @staticmethod
    async def _apply_intuition_activity(ctx: hsm.Context, instance: "Intuition", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, InputData)
        input = Intuition._intuition_input_for_processor(instance, data.processing_input)
        operation_id = event.id if event.id else uuid.uuid4().hex
        if processing.active_operation(instance, operation_id) is None:
            await processing.start_operation(instance, operation_id)
        metadata = dict(event.metadata)
        try:
            raw = await instance._processor.process(input)
            product, confidence = _product_from_processor_output(raw)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(types.FailureData(message=str(error), turn=data.turn)),
                    id=operation_id,
                    metadata=metadata,
                ),
            )
            return

        if confidence is not None:
            instance._confidence_tuner.observe(confidence)
        escalate = instance._confidence_tuner.should_escalate(confidence)
        # Tuner state drives escalate choice only; do not put it in event.metadata.

        # Unhandled cascade (System 2): explicit None, empty dispatch, or low confidence.
        # Empty events: [] is not a deliberate pass — use cognition.ignore for that.
        # On escalate with selections: fire environment actions first, then terminal None.
        if product is None or len(product) == 0:
            terminal: types.OutputData | None = None
            to_dispatch: types.OutputData = ()
        elif escalate or any(
            _is_deliberative_input_event(item.event, {schema.name: schema for schema in input.schemas})
            for item in product
        ):
            to_dispatch = _environment_actions(product, current_input=input)
            terminal = None
        else:
            to_dispatch = product
            terminal = product

        if to_dispatch and input.actors:
            try:
                await types.dispatch_selected_events(
                    ctx,
                    input,
                    _selections_from_output(to_dispatch),
                    operation_id=operation_id,
                    source=instance,
                    focus_candidates=data.turn.input.focus_candidates,
                    focused_device=data.turn.input.focus,
                    metadata=metadata,
                )
            except Exception as error:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _ApplyFailedEvent.with_data(types.FailureData(message=str(error), turn=data.turn)),
                        id=operation_id,
                        metadata=metadata,
                    ),
                )
                return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _AppliedEvent.with_data(types.CompletionData(turn=data.turn, output=terminal)),
                id=operation_id,
                metadata=metadata,
            ),
        )

    @staticmethod
    def _complete_apply(ctx: hsm.Context, instance: "Intuition", event: hsm.Event[typing.Any]) -> None:
        output = event.data
        assert isinstance(output, types.CompletionData)
        Intuition._dispatch_terminal_output(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            output=output,
        )

    @staticmethod
    def _fail_apply(ctx: hsm.Context, instance: "Intuition", event: hsm.Event[typing.Any]) -> None:
        failure = event.data
        assert isinstance(failure, types.FailureData)
        Intuition._dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            failure=failure,
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Intuition",
        hsm.initial(hsm.target("/Intuition/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(processing.Processing._emit_cancelled),
            ),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_intuition_input),
                hsm.target("/Intuition/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(_apply_intuition_activity),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(processing.Processing._emit_cancelled),
                hsm.target("/Intuition/idle"),
            ),
            hsm.transition(
                hsm.on(_AppliedEvent),
                hsm.guard(_has_intuition_applied),
                hsm.effect(_complete_apply),
                hsm.target("/Intuition/idle"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_apply_failure),
                hsm.effect(_fail_apply),
                hsm.target("/Intuition/idle"),
            ),
        ),
        hsm.observe(observer),
    )

    def __init__(
        self,
        *,
        processor: IntuitionProcessor,
        instructions: str | None = None,
        confidence_floor: int = _DEFAULT_CONFIDENCE_FLOOR,
        confidence_tuner: ConfidenceTuner | None = None,
    ) -> None:
        if not 0 <= confidence_floor <= 100:
            raise ValueError("confidence_floor must be an integer between 0 and 100.")
        resolved = type(self).instructions if instructions is None else instructions
        if instructions is not None and not instructions.strip():
            raise ValueError("instructions must not be blank when provided.")
        # Empty default is intentional: system channel is only the per-turn world XML when present.
        ability.Ability.__init__(self)
        self._instructions = resolved.strip() if resolved else ""
        self._processor = processor
        self._confidence_tuner = confidence_tuner or ConfidenceTuner(floor=confidence_floor)


InputEvent = Intuition.input_event
OutputEvent = Intuition.output_event

__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "ConfidenceTuner",
    "EventPatch",
    "InputEvent",
    "InputData",
    "OutputEvent",
    "Intuition",
    "OutputData",
    "IntuitionProcessor",
]

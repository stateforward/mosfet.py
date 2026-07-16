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
DEFAULT_INSTRUCTIONS = (
    "Select zero or more modeled events from the offered schemas. Multiple events may run in one "
    "turn (for example speaking together with another offered ability). Every offered event schema "
    "includes integer confidence 0–100: always set it on each selected event (whole number only, "
    "never a fraction). Use high confidence (80–100) when the match is clear, mid (40–70) when "
    "plausible but incomplete, and low (0–35) when guessing or the host likely cannot fulfill the "
    "request—low confidence may still run world actions while escalating to deliberation. When "
    "deliberation is needed and the user is waiting on speech, prefer multi-select: speaking.input "
    "with a short bridge line together with reasoning.input. Leave the turn unhandled only when "
    "no offered event should run."
)


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
            "world actions while escalating the turn to deliberate reasoning."
        ),
        examples=[100, 86, 55, 20, 0],
    )


_AppliedEvent = hsm.Event[object](
    name="bot.ability.intuition.applied",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)
_ApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.intuition.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


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
                {
                    "result": [
                        {
                            "event": "bot.ability.speaking.input",
                            "data": {"text": "One moment."},
                            "reason": "Acknowledge while deliberating.",
                        }
                    ],
                    "reason": "Can speak now but not sure enough to stop.",
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
    """True for reasoning invoke tools — host cascade / multi-select handoff to System 2."""

    from . import reasoning as reasoning_ability

    if event_name == reasoning_ability.InputEvent.name:
        return True
    schema_event = schemas.get(event_name)
    if schema_event is None:
        return False
    schema = getattr(schema_event, "schema", None)
    return schema is processing.InputData or schema is reasoning_ability.CallData


def _world_actions(
    output: types.OutputData,
    *,
    current_input: processing.InputData,
) -> types.OutputData:
    """Action events safe to fire before handing an uncertain turn to deliberate reasoning."""

    schemas = {event.name: event for event in current_input.schemas}
    return tuple(item for item in output if not _is_deliberative_input_event(item.event, schemas))


def _selections_from_output(
    output: types.OutputData,
    *,
    current_input: processing.InputData,
) -> processing.Events:
    """Build dispatch selections; reasoning.input uses CallData (host frame via metadata)."""

    from . import reasoning as reasoning_ability

    schemas = {event.name: event for event in current_input.schemas}
    selections: list[processing.SelectedEvent] = []
    for item in output:
        raw: object = item.data if item.data is not None else {}
        schema_event = schemas.get(item.event)
        if item.event == reasoning_ability.InputEvent.name:
            # Model-facing CallData is empty; runtime frame is stamped in dispatch metadata.
            raw = {}
        elif (raw is None or raw == {}) and schema_event is not None:
            schema = getattr(schema_event, "schema", None)
            if schema is processing.InputData:
                raw = current_input
        selections.append(
            processing.SelectedEvent(
                event=item.event,
                target=item.target,
                data=raw if raw is not None else {},
                reason=item.reason,
            )
        )
    return tuple(selections)


class Intuition(processing.Processing):
    """Fast cognitive ability: multi-dispatch when handled; self-tuning confidence gates System 2.

    Processor-reported ``confidence`` updates a per-instance baseline. When confidence is
    low relative to that baseline, world actions still dispatch, but the terminal is
    unhandled so Cognition cascades to reasoning. High confidence terminals with the
    selection list (no cascade). Explicit unhandled (``result is None``) always cascades.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = processing.InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = object
    input_event: typing.ClassVar[hsm.Event[processing.InputData]] = ability.ability_input_event(
        "bot.ability.intuition.input",
        processing.InputData,
    )
    output_event: typing.ClassVar[hsm.Event[types.OutputData | None]] = hsm.Event[types.OutputData | None](
        name="bot.ability.intuition.output",
        schema=types.OPTIONAL_OUTPUT_SCHEMA_CONTRACT,
    )
    instructions: typing.ClassVar[str] = DEFAULT_INSTRUCTIONS
    _processor: processing.Processor
    _instructions: str
    _confidence_tuner: ConfidenceTuner

    @staticmethod
    def _has_intuition_input(ctx: hsm.Context, instance: "Intuition", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, processing.InputData)

    @staticmethod
    def _has_applied(ctx: hsm.Context, instance: "Intuition", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return event.name == _AppliedEvent.name

    @staticmethod
    def _has_apply_failure(ctx: hsm.Context, instance: "Intuition", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, ability.FailureData)

    @staticmethod
    def _dispatch_terminal_output(
        ctx: hsm.Context,
        instance: "Intuition",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        output: types.OutputData | None,
    ) -> None:
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=operation_id,
            metadata=dict(metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _dispatch_terminal_failure(
        ctx: hsm.Context,
        instance: "Intuition",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        failure: ability.FailureData,
    ) -> None:
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=operation_id,
            metadata=dict(metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _input_for_processor(instance: "Intuition", input: processing.InputData) -> processing.InputData:
        """Stamp instructions and intuition EventPatch (confidence) onto the processing input."""

        stamped = processing.Processing._input_for_processor(instance, input)
        updates: dict[str, object] = {}
        if stamped.patch is None:
            updates["patch"] = EventPatch
        if updates:
            return stamped.model_copy(update=updates)
        return stamped

    @staticmethod
    async def _apply_activity(ctx: hsm.Context, instance: "Intuition", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, processing.InputData)
        input = Intuition._input_for_processor(instance, typing.cast(processing.InputData, data))
        operation_id = event.id if event.id else uuid.uuid4().hex
        metadata = dict(event.metadata)
        try:
            raw = await instance._processor.process(input)
            product, confidence = _product_from_processor_output(raw)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(ability.FailureData(message=str(error))),
                    id=operation_id,
                    metadata=metadata,
                ),
            )
            return

        if confidence is not None:
            instance._confidence_tuner.observe(confidence)
        escalate = instance._confidence_tuner.should_escalate(confidence)
        # Always stamp tuner state for hosts/logs (not only on escalate).
        metadata = {
            **metadata,
            "bot.intuition.confidence": confidence,
            "bot.intuition.confidence_threshold": instance._confidence_tuner.threshold(),
            "bot.intuition.escalate": escalate,
        }

        # Explicit unhandled, or low confidence → Cognition cascade (System 2).
        # On escalate with selections: fire world actions first, then terminal None.
        if product is None:
            terminal: types.OutputData | None = None
            to_dispatch: types.OutputData = ()
        elif escalate:
            to_dispatch = _world_actions(product, current_input=input)
            terminal = None
        else:
            to_dispatch = product
            terminal = product

        if to_dispatch and input.actors:
            try:
                from . import reasoning as reasoning_ability

                # Host frame for any multi-selected reasoning.input (CallData invoke).
                dispatch_metadata = {
                    **metadata,
                    reasoning_ability.HOST_INPUT_METADATA_KEY: input,
                }
                await types.dispatch_selected_events(
                    ctx,
                    input,
                    _selections_from_output(to_dispatch, current_input=input),
                    operation_id=operation_id,
                    source=instance,
                    metadata=dispatch_metadata,
                )
            except Exception as error:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _ApplyFailedEvent.with_data(ability.FailureData(message=str(error))),
                        id=operation_id,
                        metadata=metadata,
                    ),
                )
                return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _AppliedEvent.with_data(terminal),
                id=operation_id,
                metadata=metadata,
            ),
        )

    @staticmethod
    def _complete_apply(ctx: hsm.Context, instance: "Intuition", event: hsm.Event[typing.Any]) -> None:
        output = typing.cast(types.OutputData | None, event.data)
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
        assert isinstance(failure, ability.FailureData)
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
            hsm.activity(_apply_activity),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(processing.Processing._emit_cancelled),
                hsm.target("/Intuition/idle"),
            ),
            hsm.transition(
                hsm.on(_AppliedEvent),
                hsm.guard(_has_applied),
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
        if not resolved.strip():
            raise ValueError("instructions must not be blank.")
        ability.Ability.__init__(self)
        self._instructions = resolved.strip()
        self._processor = processor
        self._confidence_tuner = confidence_tuner or ConfidenceTuner(floor=confidence_floor)


InputEvent = Intuition.input_event
OutputEvent = Intuition.output_event

__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "ConfidenceTuner",
    "EventPatch",
    "InputEvent",
    "OutputEvent",
    "Intuition",
    "OutputData",
    "IntuitionProcessor",
]

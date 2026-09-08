from .. import processing
from .. import ability
from . import input as cognition_input

import collections.abc
import typing

import hsm
import bot
from bot import event
import pydantic

# Model-facing deliberate (System-2) handoff event name. Reasoning owns the event; intuition
# detects cascade handoff via this constant and/or ``processing.is_deliberative_handoff_schema``
# so stages never import each other for the check.
DELIBERATIVE_HANDOFF_EVENT_NAME: typing.Final[str] = "bot.ability.reasoning.input"


def is_deliberative_handoff_event_name(event_name: str) -> bool:
    """True when ``event_name`` is the shared deliberate handoff event."""

    return event_name == DELIBERATIVE_HANDOFF_EVENT_NAME


_Reference = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description="Stable reference used by a cognition host to resolve a target or modeled event.",
        examples=["runtime"],
    ),
]

# Runtime selections may retain a typed payload produced by a trusted native ability. Model and
# learned-behavior boundaries still use the canonical JSON projection; ``EventData`` serializes
# that projection only when a JSON boundary is crossed.
EventPayload: typing.TypeAlias = object


class EventData(pydantic.BaseModel):
    """One selected HSM event to dispatch."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "description": (
                "One selected modeled HSM event. Processing dispatches it to the resolved target "
                "fire-and-forget; hosts do not re-apply OutputData. Put selection rationale on "
                "reason here — not on the event payload."
            ),
            "examples": [
                {
                    "target": "bot",
                    "event": "bot.focus_device",
                    "data": {"device": "device-a"},
                    "reason": "Attention should move to that device for this turn.",
                }
            ],
        },
    )

    target: _Reference | None = pydantic.Field(
        default=None,
        description=(
            "Stable reference for the host-local receiver of the modeled event. Hosts may omit this when the "
            "event resolves relative to focus or a single actor."
        ),
        examples=["bot", "device-a"],
    )
    event: _Reference = pydantic.Field(
        description="Canonical modeled HSM event name selected by cognition.",
        examples=["bot.focus_device"],
    )
    data: EventPayload | None = pydantic.Field(
        default=None,
        description=(
            "Optional typed domain payload for the selected event. Trusted native abilities may retain a typed "
            "Pydantic payload through internal dispatch; model and learned-behavior JSON projections omit raw "
            "media. Do not put selection rationale here — use reason on this envelope."
        ),
        examples=[{"device": "device-a"}],
    )
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional reason this event was selected.",
        examples=["Attention should move to that device for this turn."],
    )

    @pydantic.field_serializer("data", when_used="json")
    def _serialize_data(self, value: object | None) -> object | None:
        """Project typed domain data at JSON boundaries without mutating runtime payloads."""

        return None if value is None else event.event_json_value(value)


# Always a tuple of selected events. Empty tuple = no dispatch.
OutputData: typing.TypeAlias = tuple[EventData, ...]


class IgnoreData(pydantic.BaseModel):
    """Explicit decision that this cognition turn should run no environment or body actions.

    Prefer selecting this event over an empty ``events`` array so models have a named
    branch under required tool-calling. Host treats ignore-only as handled (no cascade
    to deliberation solely because nothing else was selected) and does not dispatch it
    to devices or the bot body. Selection rationale belongs on the selection envelope
    (``EventData.reason`` / dispatch item ``reason``), not on this payload.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Deliberately ignore this stimulus: no device command, no focus change, no "
                "speech. Required when no other offered event should run. Do not use a device "
                "command, focus, or clear_focus as a stand-in for ignore. Put why on the "
                "selection envelope reason field, not on this empty payload."
            ),
            "examples": [{}],
        },
    )


IgnoreEvent = hsm.Event[IgnoreData](
    name="bot.ability.cognition.ignore",
    kind=event.EventKind,
    schema=IgnoreData,
)


def is_ignore_event(event_name: str) -> bool:
    """True when ``event_name`` is the cognition ignore selection."""

    return event_name == IgnoreEvent.name


def without_ignore_selections(selections: processing.Events) -> processing.Events:
    """Drop cognition ignore selections (they are cognition only, not dispatch targets)."""

    return tuple(item for item in selections if not is_ignore_event(item.event))


class TurnData(pydantic.BaseModel):
    """Typed correlation and body input for one active cognition turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    input: cognition_input.InputData = pydantic.Field(
        description=(
            "Live body input for this turn: the stimulus plus the abilities, named actors, and focus "
            "candidates the stage may select from. Built by the Cognition host from live body context."
        ),
        examples=[{"focus": "device-a", "focus_candidates": ["device-a"]}],
    )
    operation_id: str = pydantic.Field(
        min_length=1,
        description=(
            "Correlation id of the cognition turn. Stage terminals carry it back so the Cognition host "
            "settles exactly this turn."
        ),
        examples=["turn-1"],
    )
    generation: str = pydantic.Field(
        min_length=1,
        description="Live operation-actor identifier that proves this turn is still current.",
        examples=["operation:turn-1"],
    )


class CompletionData(pydantic.BaseModel):
    """Typed terminal from one cognition stage back to the Cognition host."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: TurnData = pydantic.Field(
        description=(
            "The cognition turn this terminal settles. Carries the correlation id and the liveness "
            "proof the host checks before accepting the output."
        ),
    )
    output: OutputData | processing.Events | None = pydantic.Field(
        default=None,
        description=(
            "Selected modeled events for Processing to dispatch fire-and-forget, or None when the stage "
            "selected nothing. Hosts do not re-apply OutputData; selection rationale belongs on each "
            "selection envelope, not here."
        ),
        examples=[[{"target": "bot", "event": "bot.focus_device", "data": {"device": "device-a"}}]],
    )


class FailureData(ability.FailureData):
    """Typed cognition-stage failure correlated to its originating turn."""

    turn: TurnData = pydantic.Field(
        description=(
            "The cognition turn that failed. Carries the correlation id so the host attributes the "
            "failure to exactly this turn."
        ),
    )


OUTPUT_SCHEMA_CONTRACT = pydantic.TypeAdapter(OutputData)
OPTIONAL_OUTPUT_SCHEMA_CONTRACT = typing.cast(
    "pydantic.TypeAdapter[OutputData | None]", pydantic.TypeAdapter(OutputData | None)
)
OUTPUT_SCHEMA = typing.cast(dict[str, object], OUTPUT_SCHEMA_CONTRACT.json_schema())
OPTIONAL_OUTPUT_SCHEMA = typing.cast(
    dict[str, object],
    OPTIONAL_OUTPUT_SCHEMA_CONTRACT.json_schema(),
)


def is_output(value: object) -> typing.TypeGuard[OutputData]:
    """Return whether a value is a tuple of event selections."""

    if not isinstance(value, tuple):
        return False
    return all(isinstance(item, EventData) for item in typing.cast(tuple[object, ...], value))


def attention_selection_error(
    selection: processing.SelectedEvent,
    *,
    focus_candidates: tuple[str, ...],
    configured_device_names: frozenset[str],
    enforce_candidates: bool,
    focused_device: str | None = None,
) -> str | None:
    """Return a fail-closed error for illegal attention selections, else None.

    Attention policy is owned by cognition: this validator decides which
    focus/clear selections are legal for a turn. The body applies the verdict
    mechanically at its turn gates (and here, pre-dispatch) — it never invents
    attention policy of its own.
    """

    if selection.event == bot.FocusDeviceEvent.name:
        if selection.target is not None and selection.target != "bot":
            return "Processing selected focus_device outside available device candidates."
        data = bot.FocusDeviceEventData.model_validate(selection.data or {})
        if enforce_candidates and data.device not in focus_candidates:
            return "Processing selected focus_device outside available device candidates."
        if configured_device_names and data.device not in configured_device_names:
            return "Processing selected focus_device outside available device candidates."
        return None
    if selection.event == bot.ClearFocusEvent.name:
        if selection.target not in (None, "bot"):
            return "Processing selected clear_focus for a non-bot target."
        if not focused_device:
            return "Processing selected clear_focus with no focused device."
        return None
    return None


def _device_actor_names(actors: collections.abc.Mapping[str, hsm.Instance]) -> frozenset[str]:
    """Names of device actors in this turn's actor map, without importing devices.

    Cognition must not depend on the device package (import cycle); device-ness is
    structural — devices expose a firmware model, abilities do not. The ``bot`` and
    ``cognition`` host keys are never focus targets.
    """

    return frozenset(
        name for name, actor in actors.items() if name not in ("bot", "cognition") and hasattr(actor, "firmware_model")
    )


def _is_body_turn_gate(actor: object) -> bool:
    """Whether this turn's ``bot`` actor applies the body attention turn-gate.

    The real body exposes the cognition-owned attention verdict as the public
    ``attention_selection_error`` entry point and gates focus/clear on the live turn;
    unit-test doubles (plain ``hsm.Instance`` recipients) model acceptance themselves
    and gate nothing. Detect the gate structurally so this module never imports the
    body package (import cycle): pre-dispatch fails closed exactly when the body
    would, and stays quiet otherwise.
    """

    return callable(getattr(actor, "attention_selection_error", None))


def attention_bias_for_turn(
    stimulus: object,
    *,
    envelope_source: str,
    device_names: collections.abc.Container[str],
    device_source_refs: collections.abc.Mapping[str, str],
    focused_device: str | None,
) -> tuple[str | None, tuple[str, ...]]:
    """Cognition-owned attention bias for one body turn: (focus, focus_candidates).

    The body stamps identity context (which configured references exist, which live
    actor ids map to them, what is currently focused); this function decides how that
    context combines into the turn's bias. It reads device identity only — never what
    happened, never stimulus insistence — so nothing here arbitrates per stimulus.
    """

    focus = focused_device if focused_device is not None and focused_device in device_names else None
    if isinstance(stimulus, bot.InputEventData):
        target_reference: str | None = stimulus.target_device
        if target_reference is None and envelope_source:
            target_reference = device_source_refs.get(envelope_source)
        stimulus_references = (
            (target_reference,) if target_reference is not None and target_reference in device_names else ()
        )
    else:
        observed_source = stimulus.source if isinstance(stimulus, hsm.Event) else ""
        observed_reference: str | None = device_source_refs.get(observed_source) if observed_source else None
        stimulus_references = (
            (observed_reference,) if observed_reference is not None and observed_reference in device_names else ()
        )
    ordered: list[str] = []
    if focus is not None:
        ordered.append(focus)
    for reference in stimulus_references:
        if reference not in ordered:
            ordered.append(reference)
    return focus, tuple(ordered)


async def dispatch_selected_events(
    ctx: hsm.Context,
    input: processing.InputData,
    selections: processing.Events,
    *,
    operation_id: str,
    source: hsm.Instance,
    focus_candidates: tuple[str, ...],
    focused_device: str | None = None,
    configured_device_names: frozenset[str] | None = None,
    metadata: collections.abc.Mapping[str, object] | None = None,
    dispatch_trust: processing.DispatchTrust = processing.DispatchTrust.MODEL,
) -> None:
    """Validate body-action constraints, then dispatch selected modeled events.

    Cognition ignore is offered from the host model-offerable snapshot and may appear in
    selections as a handled terminal product, but it is not a body/device dispatch target
    (cognition only). Strip it before actor delivery.

    Attention policy (focus/clear legality) is owned by cognition
    (``attention_selection_error``); the body enforces the same verdict at its turn
    gates. ``focused_device`` is the body-stamped current focus for this turn (not a
    private field read). ``configured_device_names`` is injectable for hosts that know
    their device inventory; when omitted it is derived structurally from the actor map
    so this module never imports body or device packages.
    """

    event_metadata = dict(metadata or {})
    configured = configured_device_names if configured_device_names is not None else _device_actor_names(input.actors)
    allowed = set(focus_candidates)
    candidates = tuple(item for item in configured if item in allowed)
    # Enforce candidates when devices are configured or a real body gates this turn
    # (it will turn-gate focus the same way, so pre-dispatch must fail closed too).
    # Plain doubles accept what they model and gate nothing — no enforcement there.
    enforce_candidates = bool(configured) or _is_body_turn_gate(input.actors.get("bot"))
    to_dispatch = without_ignore_selections(selections)
    if not to_dispatch:
        return

    current_focus = focused_device if focused_device else None
    for selection in to_dispatch:
        try:
            error = attention_selection_error(
                selection,
                focus_candidates=candidates,
                configured_device_names=configured,
                enforce_candidates=enforce_candidates,
                focused_device=current_focus,
            )
        except pydantic.ValidationError as validation_error:
            raise processing.SelectionRejectionError.from_validation(
                event_name=selection.event,
                prefix=f"Processing selected invalid focus selection data for event: {selection.event}",
                error=validation_error,
            ) from validation_error
        if error is not None:
            raise processing.SelectionRejectionError(error)

    await processing.dispatch_selected_events(
        ctx,
        input,
        to_dispatch,
        operation_id=operation_id,
        source=source,
        metadata=event_metadata,
        dispatch_trust=dispatch_trust,
    )


__all__ = [
    "DELIBERATIVE_HANDOFF_EVENT_NAME",
    "OUTPUT_SCHEMA",
    "OUTPUT_SCHEMA_CONTRACT",
    "CompletionData",
    "EventData",
    "EventPayload",
    "FailureData",
    "IgnoreData",
    "IgnoreEvent",
    "OutputData",
    "TurnData",
    "OPTIONAL_OUTPUT_SCHEMA",
    "OPTIONAL_OUTPUT_SCHEMA_CONTRACT",
    "attention_bias_for_turn",
    "attention_selection_error",
    "dispatch_selected_events",
    "is_deliberative_handoff_event_name",
    "is_ignore_event",
    "is_output",
    "without_ignore_selections",
]

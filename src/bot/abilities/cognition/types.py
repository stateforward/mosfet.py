from .. import processing
from .. import ability
from . import input as cognition_input

import collections.abc
import dataclasses
import typing

import hsm
from bot import event_schema
import pydantic

_Reference = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        description="Stable reference used by a cognition host to resolve a target or modeled event.",
        examples=["runtime"],
    ),
]

EventPayload = dict[str, object]


class EventData(pydantic.BaseModel):
    """One selected HSM event to dispatch."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "description": (
                "One selected modeled HSM event. Processing dispatches it to the resolved target "
                "fire-and-forget; hosts do not re-apply OutputData."
            ),
            "examples": [
                {
                    "target": "phone",
                    "event": "phone.answer_call",
                    "data": {"call_id": "livekit:caller"},
                    "reason": "Incoming call should be answered.",
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
        examples=["phone"],
    )
    event: _Reference = pydantic.Field(
        description="Canonical modeled HSM event name selected by cognition.",
        examples=["phone.answer_call"],
    )
    data: EventPayload | None = pydantic.Field(
        default=None,
        description=(
            "Optional JSON-serializable event data for the selected event. Do not include raw audio, message text, "
            "credentials, provider-specific objects, or high-cardinality diagnostics."
        ),
        examples=[{"call_id": "incoming-call"}],
    )
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional reason this event was selected.",
        examples=["The incoming call should be answered."],
    )


# Always a tuple of selected events. Empty tuple = no dispatch.
OutputData: typing.TypeAlias = tuple[EventData, ...]


class IgnoreData(pydantic.BaseModel):
    """Explicit judgment that this cognition turn should run no world or body actions.

    Prefer selecting this event over an empty ``events`` array so models have a named
    branch under required tool-calling. Host treats ignore-only as handled (no cascade
    to deliberation solely because nothing else was selected) and does not dispatch it
    to devices or the bot body.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Deliberately ignore this stimulus: no device command, no focus change, no "
                "speech. Required when no other offered event should run. Do not use answer, "
                "decline, focus, or clear_focus as a stand-in for ignore. "
                "Do not select ignore when the stimulus is an actionable phone.ringing "
                "(kind=phone.ringing with call_id) and answer/decline are offered — answer "
                "or decline instead."
            ),
            "examples": [
                {"reason": "Ambient sound is not an actionable phone ring."},
                {"reason": "Stimulus is incomplete; no safe action."},
            ],
        },
    )

    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional short reason this turn is intentionally ignored.",
        examples=["Ambient noise only.", "Not a phone.ringing stimulus."],
    )


IgnoreEvent = hsm.Event[IgnoreData](
    name="bot.ability.cognition.ignore",
    kind=event_schema.EventKind,
    schema=IgnoreData,
)


def is_ignore_event(event_name: str) -> bool:
    """True when ``event_name`` is the cognition ignore selection."""

    return event_name == IgnoreEvent.name


def without_ignore_selections(selections: processing.Events) -> processing.Events:
    """Drop cognition ignore selections (they are judgment only, not dispatch targets)."""

    return tuple(item for item in selections if not is_ignore_event(item.event))


class TurnData(pydantic.BaseModel):
    """Typed correlation and body input for one active cognition turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    input: cognition_input.InputData
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(
        min_length=1,
        description="Live operation-actor identifier that proves this turn is still current.",
    )


class CompletionData(pydantic.BaseModel):
    """Typed terminal from one cognition stage back to the Cognition host."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: TurnData
    output: OutputData | None = None


class FailureData(ability.FailureData):
    """Typed cognition-stage failure correlated to its originating turn."""

    turn: TurnData


OUTPUT_SCHEMA_CONTRACT = typing.cast(
    pydantic.TypeAdapter[OutputData],
    pydantic.TypeAdapter(OutputData),
)
OPTIONAL_OUTPUT_SCHEMA_CONTRACT = typing.cast(
    pydantic.TypeAdapter[OutputData | None],
    pydantic.TypeAdapter(OutputData | None),
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
    return all(isinstance(item, EventData) for item in value)


# Answer/decline must target the live ringing call. Phone elevates ring as world.sound with
# PhoneSoundData.call_id and event.id = call_id; models often still copy schema examples.
_PHONE_CALL_ID_COMMANDS: frozenset[str] = frozenset(
    {
        "phone.answer_call",
        "phone.decline_call",
    }
)
_RING_SOUND_KINDS: frozenset[str] = frozenset({"phone.ringing", "phone.call"})


def _call_id_from_ring_stimulus(stimulus: object) -> str | None:
    """Return live call_id from a ring/call world.sound stimulus, if present.

    Prefers :class:`~bot.devices.phone.events.PhoneSoundData.call_id` so host bind matches
    the model-facing field; falls back to ``event.id`` when only correlation was stamped.
    """

    from bot.devices.phone.events import PhoneSoundData
    from bot.world import SoundData, SoundEvent

    if not isinstance(stimulus, hsm.Event):
        return None
    if stimulus.name != SoundEvent.name:
        return None
    data = stimulus.data
    if not isinstance(data, SoundData):
        return None
    if data.kind is not None and data.kind not in _RING_SOUND_KINDS:
        return None
    if isinstance(data, PhoneSoundData) and data.call_id.strip():
        return data.call_id
    call_id = stimulus.id
    if not call_id or not str(call_id).strip():
        return None
    return str(call_id)


def bind_phone_call_id_from_stimulus(
    input: processing.InputData,
    selections: processing.Events,
) -> processing.Events:
    """Overwrite model call_id on answer/decline with the ring stimulus call_id.

    Prefers PhoneSoundData.call_id, then event.id, so intuition/reasoning cannot dispatch a
    stale schema-example call_id against a live ringing call.
    """

    call_id = _call_id_from_ring_stimulus(input.input)
    if call_id is None:
        return selections
    bound: list[processing.SelectedEvent] = []
    for selection in selections:
        if selection.event not in _PHONE_CALL_ID_COMMANDS:
            bound.append(selection)
            continue
        data = dict(selection.data or {})
        if data.get("call_id") == call_id:
            bound.append(selection)
            continue
        data["call_id"] = call_id
        bound.append(dataclasses.replace(selection, data=data))
    return tuple(bound)


async def dispatch_selected_events(
    ctx: hsm.Context,
    input: processing.InputData,
    selections: processing.Events,
    *,
    operation_id: str,
    source: hsm.Instance,
    focus_candidates: tuple[str, ...],
    focused_device: str | None = None,
    metadata: collections.abc.Mapping[str, object] | None = None,
) -> None:
    """Validate body-action constraints, then dispatch selected modeled events.

    Cognition ignore is offered from the host model-offerable snapshot and may appear in
    selections as a handled terminal product, but it is not a body/device dispatch target
    (judgment only). Strip it before actor delivery.

    Body attention policy (focus/clear legality) is owned by ``bot.Bot``; judgment only
    fails closed on that policy before dispatch so illegal selections are not silent drops.
    ``focused_device`` is the body-stamped current focus for this turn (not a private field read).
    """

    import bot
    from bot import device

    event_metadata = dict(metadata or {})
    configured_device_names = frozenset(
        name for name, actor in input.actors.items() if name != "bot" and isinstance(actor, device.Device)
    )
    allowed = set(focus_candidates)
    candidates = tuple(item for item in configured_device_names if item in allowed)
    bot_actor = input.actors.get("bot")
    # Enforce candidates when devices are configured or the bot actor is a real Bot.
    enforce_candidates = bool(configured_device_names) or isinstance(bot_actor, bot.Bot)
    selections = bind_phone_call_id_from_stimulus(input, selections)
    to_dispatch = without_ignore_selections(selections)
    if not to_dispatch:
        return

    current_focus = focused_device if focused_device else None
    for selection in to_dispatch:
        error = bot.Bot.attention_selection_error(
            selection,
            focus_candidates=candidates,
            configured_device_names=configured_device_names,
            enforce_candidates=enforce_candidates,
            focused_device=current_focus,
        )
        if error is not None:
            raise RuntimeError(error)

    await processing.dispatch_selected_events(
        ctx,
        input,
        to_dispatch,
        operation_id=operation_id,
        source=source,
        metadata=event_metadata,
    )


__all__ = [
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
    "bind_phone_call_id_from_stimulus",
    "dispatch_selected_events",
    "is_ignore_event",
    "is_output",
    "without_ignore_selections",
]

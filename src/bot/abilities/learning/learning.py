"""Learning: decode a lesson, ground runtime input in memory, then author a behavior.

Unlike Reflection (one live turn), Learning starts from external material. It does **not** import
peer ability packages to learn event contracts. The only source of live-input knowledge is memory:
recalled remembered turns supply stimulus_name, focus, and prior selections. If memory cannot name
a stimulus, Learning fails closed.

Topology (children attached; never call child private methods)::

    idle → decoding → selecting → revising → idle

Decoder is a pure injected dependency (not an HSM child). Select Processing is an attached child.
Revision is composition for Starlark authoring only — not a source of input knowledge. Behavior
Starlark is authored only by the injected processor through Revision — never hard-coded here.
"""

from __future__ import annotations

from .. import ability
from .. import decoding
from .. import memory
from .. import processing
# Revision's InputData still requires cognition-shaped turn fields. These imports exist only for
# that authoring handoff — Learning never uses them to discover peer ability event contracts.
from ..cognition import episodes
from ..cognition import input as cognition_input
from ..cognition import types as cognition_types
from ..cognition.reflection import revision

import asyncio
import dataclasses
import datetime
import typing
import uuid

import hsm
from bot import event_schema
import pydantic
from sqlalchemy import select

from bot.behavior import ChangeData, CreateData
from bot.protocols import attachment
from bot.telemetry import observer

_GENERATE_EVENT_NAME = "bot.ability.learning.generate"
# Durable memory query tag for stored turns (same tag writers use when inserting episodes).
_REMEMBERED_TURN_QUERY = "cognitive_episode"

GENERATE_INSTRUCTIONS = (
    "You are the Learning generate step. Input is decoded lesson material (decoded.text, optional "
    "decoded.kind) plus prior_turns recalled from memory (real past turns).\n"
    "Honesty: runtime_input must be grounded in prior_turns. Use a turn's stimulus_name, "
    "focus, focus_candidates, and a selection from turn.output as expected_event/target/data/reason. "
    "Prefer the most recent turn that fits the lesson. You may only add payload fields that fit "
    "that known stimulus (memory stores stimulus_name and output, not full raw media). Do not invent "
    "a stimulus_name, device API, or expected event that is not supported by prior_turns.\n"
    "If prior_turns is empty or has no usable stimulus_name, Learning will fail closed — do not "
    "pretend to know the live input. Learning does not consult other abilities for event contracts; "
    "memory is the only ground truth.\n"
    "Inventory: inventory_event is bot.behavior.create or bot.behavior.change. Triggers derive from "
    "runtime_input.stimulus_name (plus optional extra_triggers). Set reason to the decoded lesson "
    "text. Omit source — Revision authors Starlark later. Exactly one "
    "bot.ability.learning.generate selection."
)

def _no_stimulus_message(prior_turns: tuple["RememberedTurn", ...]) -> str:
    """Fail-closed refusal that names what recall actually returned.

    Recall succeeded here — the operator must be able to tell "memory is empty" from "memory has
    turns but none record a stimulus_name", and neither from a recall error (raised separately).
    """

    if not prior_turns:
        return (
            "Learning cannot determine the input stimulus: memory recall returned no remembered "
            "turns. Refuse to invent runtime_input; store real turns first."
        )
    return (
        f"Learning cannot determine the input stimulus: {len(prior_turns)} remembered turn(s) "
        "recalled but none record a stimulus_name. Refuse to invent runtime_input."
    )

# Recall bound for generate grounding: enough turns to cover recent stimulus variety while
# keeping the model-facing select input small and the query cost fixed (PY-MEM-001).
_RECALL_TURN_LIMIT = 50

# Live stimulus payload fields are identifiers, enum-like kinds, and short scalars; lesson prose
# is sentences. This bound is the schema's discriminator between the two — it is deliberately
# generous for real payload values and deliberately below any usable instruction sentence.
_MAX_PAYLOAD_TEXT_LENGTH = 40

_SELECT_ID_SUFFIX = ":learning:select"
_CHILD_OPERATION_TIMEOUT = datetime.timedelta(seconds=30)
_CANCEL_TEARDOWN_TIMEOUT = datetime.timedelta(seconds=5)


class InputData(pydantic.BaseModel):
    """External material to learn from (decoded before behavior generation)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        extra="forbid",
        json_schema_extra={
            "description": (
                "Learning input payload. A decoder turns content into DecodedData; generate synthesizes "
                "a runtime input and inventory intent for Revision."
            ),
            "examples": [
                {"content": "When the phone rings, answer it.", "media_type": "text/plain"},
            ],
        },
    )

    content: str | bytes = pydantic.Field(
        description="Raw learning material (text string or binary bytes).",
        examples=["When the phone rings, answer it."],
    )
    media_type: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional media type of content (e.g. text/plain).",
        examples=["text/plain"],
    )


_INPUT_EVENT = ability.ability_input_event(
    "bot.ability.learning.input",
    InputData,
    description=(
        "Teach the bot a standing rule from something it was just told, so the behavior runs "
        "automatically next time instead of being reasoned about again. Select this when a turn is "
        "instruction about what to do in future situations ('when the phone rings, answer it') rather "
        "than a request to act now. content is the instruction as heard; the bot grounds it against "
        "what it already remembers and authors the behavior itself."
    ),
    examples=[{"content": "When the phone rings, answer it.", "media_type": "text/plain"}],
)
# Model-callable so judgment can select learning from a live turn, the same way Speaking is
# selected. Programmatic callers still dispatch this event directly.
InputEvent = dataclasses.replace(_INPUT_EVENT, kind=event_schema.EventKind)


class DecodedData(pydantic.BaseModel):
    """Normalized product of the learning decoder."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Decoded learning material used to synthesize a runtime input and inventory intent.",
            "examples": [{"text": "Answer when the phone rings.", "kind": "instruction"}],
        },
    )

    text: str = pydantic.Field(
        min_length=1,
        description="Decoded text used as generate-step evidence.",
        examples=["Answer when the phone rings."],
    )
    kind: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional decoder-assigned material kind.",
        examples=["instruction", "transcript", "correction"],
    )


class RuntimeInputData(pydantic.BaseModel):
    """Synthesized live-turn input the authored behavior must accept at runtime.

    Not lesson text — the stimulus Autonomy would hand the behavior later.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Runtime input for Learning authoring and dry-run. stimulus_name is the external "
                "event that proposes the behavior; payload is example live event data."
            ),
            "examples": [
                {
                    "stimulus_name": "environment.sound",
                    "payload": {"kind": "phone.ringing", "call_id": "incoming-call"},
                    "focus": "phone",
                    "focus_candidates": ["phone"],
                    "expected_event": "bot.focus_device",
                    "expected_data": {"device": "phone"},
                }
            ],
        },
    )

    stimulus_name: str = pydantic.Field(
        min_length=1,
        description="External event name that should propose the behavior (e.g. environment.sound).",
        examples=["environment.sound"],
    )
    payload: dict[str, object] = pydantic.Field(
        min_length=1,
        description=(
            "Example live event data the behavior will receive as Starlark event['data']. "
            "Must look like a real stimulus payload — identifiers, kinds, and short scalars — "
            f"not the lesson text. Every string value must be under {_MAX_PAYLOAD_TEXT_LENGTH} "
            "characters; put the lesson in reason instead."
        ),
        examples=[{"kind": "phone.ringing", "call_id": "incoming-call"}],
    )
    focus: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional focused device/instance name for the synthetic turn.",
        examples=["phone"],
    )
    focus_candidates: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Optional focus candidates for the synthetic turn.",
        examples=[["phone"]],
    )
    expected_event: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional expected selection event the lesson implies (from memory when known).",
        examples=["bot.focus_device", "phone.answer_call"],
    )
    expected_target: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional target for the expected selection.",
        examples=["phone", "bot"],
    )
    expected_data: dict[str, object] | None = pydantic.Field(
        default=None,
        description="Optional data for the expected selection.",
        examples=[{"device": "phone"}],
    )
    expected_reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional reason for the expected selection.",
    )

    @pydantic.field_validator("payload")
    @classmethod
    def _payload_is_a_stimulus_not_a_lesson(cls, payload: dict[str, object]) -> dict[str, object]:
        """Reject lesson prose smuggled in as live event data.

        The behavior authored from this input is dry-run against ``payload`` as its runtime event
        data; lesson text there produces a behavior that only ever matches the lesson.
        """

        for key, value in payload.items():
            if isinstance(value, str) and len(value) > _MAX_PAYLOAD_TEXT_LENGTH:
                size = f"{len(value)} characters"
                raise ValueError(
                    f"runtime_input.payload[{key!r}] reads as lesson text ({size}); put the lesson in reason."
                )
        return payload


class RememberedSelection(pydantic.BaseModel):
    """One prior selection recorded on a remembered turn in memory."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="ignore",
    )

    event: str = pydantic.Field(min_length=1, description="Selected event name from a past turn.")
    target: str | None = pydantic.Field(default=None, min_length=1)
    data: dict[str, object] | None = None
    reason: str | None = pydantic.Field(default=None, min_length=1)


class RememberedTurn(pydantic.BaseModel):
    """Past turn recalled from memory — Learning's only source of live-input knowledge.

    Parsed from memory content JSON (query_tags=cognitive_episode). Learning does not import peer
    ability packages to discover stimulus or selection contracts.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="ignore",
        json_schema_extra={
            "description": (
                "Remembered turn from memory used to ground runtime_input.stimulus_name and expected "
                "selections. Not a live ability snapshot."
            ),
        },
    )

    focus: str | None = pydantic.Field(default=None, examples=["phone"])
    focus_candidates: tuple[str, ...] = pydantic.Field(default=(), examples=[["phone"]])
    stimulus_name: str | None = pydantic.Field(default=None, min_length=1, examples=["environment.sound"])
    output: tuple[RememberedSelection, ...] = pydantic.Field(
        default=(),
        description="Prior selections from that turn (event/target/data/reason).",
    )


class GenerateData(pydantic.BaseModel):
    """Generate-step product: inventory intent plus synthesized runtime input."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Select bot.ability.learning.generate once. runtime_input is the live turn to author "
                "against; inventory_event + name install create/change. Omit source."
            ),
            "examples": [
                {
                    "event": _GENERATE_EVENT_NAME,
                    "inventory_event": "bot.behavior.create",
                    "name": "AnswerIncomingRing",
                    "reason": "When the phone rings, answer it.",
                    "runtime_input": {
                        "stimulus_name": "environment.sound",
                        "payload": {"kind": "phone.ringing"},
                        "focus_candidates": ["phone"],
                        "expected_event": "bot.focus_device",
                        "expected_data": {"device": "phone"},
                    },
                }
            ],
        },
    )

    event: typing.Literal["bot.ability.learning.generate"] = pydantic.Field(
        default="bot.ability.learning.generate",
        description="Learning generate event name (discriminator).",
    )
    inventory_event: typing.Literal["bot.behavior.create", "bot.behavior.change"] = pydantic.Field(
        description="Inventory operation to install after Revision authors Starlark.",
        examples=["bot.behavior.create"],
    )
    name: str = pydantic.Field(
        min_length=1,
        description="PascalCase behavior / HSM model name.",
        examples=["AnswerIncomingRing"],
    )
    runtime_input: RuntimeInputData = pydantic.Field(
        description="Synthesized runtime input the behavior must accept (not lesson text).",
    )
    description: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional human-readable behavior description.",
    )
    reason: str = pydantic.Field(
        min_length=1,
        description="Authoring reason; must carry the decoded lesson text for Revision evidence.",
        examples=["When the phone rings, answer it."],
    )
    lesson_kind: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional echo of decoded.kind for Learning terminal fidelity.",
        examples=["instruction"],
    )
    extra_triggers: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Optional triggers in addition to runtime_input.stimulus_name.",
    )


GenerateEvent = hsm.Event[GenerateData](
    name=_GENERATE_EVENT_NAME,
    kind=event_schema.EventKind,
    schema=GenerateData,
)


class SelectInput(pydantic.BaseModel):
    """Context for the generate step after decoding and memory recall."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Ground runtime_input in prior_turns from memory, then select "
                "bot.ability.learning.generate. Source is authored later by Revision."
            ),
        },
    )

    turn: InputData = pydantic.Field(description="Original learning input for this operation.")
    decoded: DecodedData = pydantic.Field(description="Decoded learning material.")
    prior_turns: tuple[RememberedTurn, ...] = pydantic.Field(
        default=(),
        description="Remembered turns recalled from memory before generate (sole input ground truth).",
    )
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)


class OutputData(pydantic.BaseModel):
    """Learning completed: decoded lesson, runtime input, and applied inventory payload."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )

    decoded: DecodedData = pydantic.Field(description="Decoder product that drove generation.")
    runtime_input: RuntimeInputData = pydantic.Field(
        description="Runtime input used for authoring and dry-run (memory-grounded).",
    )
    behavior: CreateData | ChangeData = pydantic.Field(
        description="Validated behavior inventory payload persisted by Revision.",
    )


class _DecodedEventData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    turn: InputData
    decoded: DecodedData
    prior_turns: tuple[RememberedTurn, ...] = ()
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)


class _SelectedEventData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
    )

    turn: InputData
    decoded: DecodedData
    generate: GenerateData
    prior_turns: tuple[RememberedTurn, ...] = ()
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)


class _StageFailedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    failure: ability.FailureData
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)


_DecodedEvent = hsm.Event[_DecodedEventData](
    name="bot.ability.learning.decoded",
    kind=hsm.CompletionEventKind,
    schema=_DecodedEventData,
)
_SelectedEvent = hsm.Event[_SelectedEventData](
    name="bot.ability.learning.selected",
    kind=hsm.CompletionEventKind,
    schema=_SelectedEventData,
)
_StageFailedEvent = hsm.Event[_StageFailedData](
    name="bot.ability.learning.stage.failed",
    kind=hsm.ErrorEventKind,
    schema=_StageFailedData,
)


def _coerce_generate(value: object) -> GenerateData:
    selections = processing.coerce_event_selections(value)
    if selections is None:
        raise TypeError("Learning generate step must return an events array.")
    if not selections:
        raise TypeError("Learning generate step returned an empty selection.")
    if len(selections) != 1:
        raise TypeError("Learning generate step must return exactly one generate event.")
    chosen = selections[0]
    if chosen.event != GenerateEvent.name:
        raise TypeError(f"Learning generate unsupported event: {chosen.event}.")
    raw = chosen.data
    # RuntimeInputData validates payload shape (non-empty, live-stimulus-like) as its own contract.
    return raw if isinstance(raw, GenerateData) else GenerateData.model_validate(raw if raw is not None else {})


def _inventory_triggers(generate: GenerateData) -> tuple[str, ...]:
    names = [generate.runtime_input.stimulus_name, *generate.extra_triggers]
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def _intent_from_generate(generate: GenerateData) -> CreateData | ChangeData:
    triggers = _inventory_triggers(generate)
    if generate.inventory_event == "bot.behavior.create":
        return CreateData(
            name=generate.name,
            triggers=triggers,
            description=generate.description,
            reason=generate.reason,
        )
    return ChangeData(
        name=generate.name,
        triggers=triggers,
        description=generate.description,
        reason=generate.reason,
    )


def _synthetic_stimulus(runtime_input: RuntimeInputData, *, operation_id: str) -> hsm.Event[object]:
    """Build an ObservedBotEvent-shaped stimulus for Revision dry-run / authoring."""

    # Shaped like the environment stimulus it stands in for (``environment.sound`` and friends declare no
    # kind), not like a tool: this is dry-run authoring input, never offered for selection.
    template = hsm.Event[object](
        name=runtime_input.stimulus_name,
        schema=object,
    )
    return dataclasses.replace(
        template.with_data(dict(runtime_input.payload)),
        id=operation_id,
        source="learning",
        target=None,
    )


def _expected_output(runtime_input: RuntimeInputData) -> cognition_types.OutputData:
    """Revision bridge: optional expected product from memory-grounded runtime_input."""

    if runtime_input.expected_event is None:
        return ()
    return (
        cognition_types.EventData(
            event=runtime_input.expected_event,
            target=runtime_input.expected_target,
            data=runtime_input.expected_data,
            reason=runtime_input.expected_reason,
        ),
    )


def _revision_cognition_input(runtime_input: RuntimeInputData, *, operation_id: str) -> cognition_input.InputData:
    """Revision bridge: synthetic dry-run stimulus from memory-grounded runtime_input only.

    abilities/actors stay empty — Learning does not snapshot peer abilities for authoring input.
    """

    return cognition_input.InputData(
        stimulus=_synthetic_stimulus(runtime_input, operation_id=operation_id),
        abilities=(),
        actors={},
        focus=runtime_input.focus,
        focus_candidates=runtime_input.focus_candidates,
    )


def _revision_prior_episodes(prior_turns: tuple[RememberedTurn, ...]) -> tuple[episodes.CognitiveEpisode, ...]:
    """Revision bridge: map Learning remembered turns into Revision's episode field type."""

    mapped: list[episodes.CognitiveEpisode] = []
    for turn in prior_turns:
        output = tuple(
            cognition_types.EventData(
                event=selection.event,
                target=selection.target,
                data=selection.data,
                reason=selection.reason,
            )
            for selection in turn.output
        )
        mapped.append(
            episodes.CognitiveEpisode(
                focus=turn.focus,
                focus_candidates=turn.focus_candidates,
                stimulus_name=turn.stimulus_name,
                output=output,
            )
        )
    return tuple(mapped)


def _runtime_input_from_revision_stimulus(cognition: cognition_input.InputData) -> RuntimeInputData | None:
    stimulus = cognition.stimulus
    if not isinstance(stimulus, hsm.Event):
        return None
    raw = stimulus.data
    payload: dict[str, object]
    if isinstance(raw, dict):
        payload = {str(key): value for key, value in typing.cast(dict[object, object], raw).items()}
    elif raw is None:
        payload = {}
    else:
        payload = {"value": raw}
    if not payload:
        return None
    return RuntimeInputData(
        stimulus_name=stimulus.name,
        payload=payload,
        focus=cognition.focus,
        focus_candidates=cognition.focus_candidates,
    )


def _decoded_from_revision_input(revision_input: revision.InputData) -> DecodedData | None:
    instruction = revision_input.instruction
    if instruction is not None:
        return DecodedData(text=instruction.text, kind=instruction.kind)
    reason = revision_input.intent.reason
    if isinstance(reason, str) and reason:
        return DecodedData(text=reason, kind=None)
    return None


def _remembered_turns_from_contents(contents: tuple[str, ...]) -> tuple[RememberedTurn, ...]:
    """Parse memory content strings into Learning-local remembered turns.

    A row stored under the remembered-turn query tag that will not parse is a data-integrity
    failure, not an absent memory. Raise so the caller reports it as a recall failure instead of
    letting it read as "nothing is stored".
    """

    turns: list[RememberedTurn] = []
    for index, content in enumerate(contents):
        try:
            turns.append(RememberedTurn.model_validate_json(content))
        except Exception as error:
            raise ValueError(
                f"remembered turn {index} under query tag {_REMEMBERED_TURN_QUERY} is not a valid turn: {error}"
            ) from error
    return tuple(turns)


def _recall_prior_turns(store: memory.Memory) -> tuple[RememberedTurn, ...]:
    """Load remembered turns from memory for generate grounding.

    Store and parse errors propagate: "recall failed" and "nothing is remembered" are different
    outcomes, and only the caller can turn the former into a typed failure that names the cause.
    """

    table = memory.memory_table
    clause = (
        select(table)
        .where(table.c.query_tags == _REMEMBERED_TURN_QUERY)
        .order_by(table.c.created_at)
        .limit(_RECALL_TURN_LIMIT)
    )
    recalled = store.execute(memory.InputData(statements=memory.compile_statements(clause)))
    return _remembered_turns_from_contents(recalled.contents(statement_index=0))


class _GroundedTurn(pydantic.BaseModel):
    """What a remembered turn can honestly ground.

    Memory stores turn identity and prior selections, never the raw stimulus payload — so this
    deliberately carries no payload. The example payload stays the model's, bounded by
    :class:`RuntimeInputData` validation.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    stimulus_name: str = pydantic.Field(min_length=1)
    focus: str | None = None
    focus_candidates: tuple[str, ...] = ()
    expected_event: str | None = None
    expected_target: str | None = None
    expected_data: dict[str, object] | None = None
    expected_reason: str | None = None


def _grounded_from_turn(turn: RememberedTurn) -> _GroundedTurn | None:
    """Project a remembered turn onto the fields memory can vouch for."""

    if not turn.stimulus_name:
        return None
    selection = turn.output[0] if turn.output else None
    return _GroundedTurn(
        stimulus_name=turn.stimulus_name,
        focus=turn.focus,
        focus_candidates=turn.focus_candidates,
        expected_event=selection.event if selection is not None else None,
        expected_target=selection.target if selection is not None else None,
        expected_data=dict(selection.data) if selection is not None and selection.data is not None else None,
        expected_reason=selection.reason if selection is not None else None,
    )


def _groundable_turns(prior_turns: tuple[RememberedTurn, ...]) -> tuple[RememberedTurn, ...]:
    """Turns that can honestly name a live stimulus."""

    return tuple(turn for turn in prior_turns if turn.stimulus_name)


def _require_memory_runtime_input(
    generate: GenerateData,
    prior_turns: tuple[RememberedTurn, ...],
) -> GenerateData:
    """Bind runtime_input to remembered turns; refuse when stimulus is unknown."""

    groundable = _groundable_turns(prior_turns)
    if not groundable:
        raise TypeError(_no_stimulus_message(prior_turns))
    face = generate.runtime_input
    same_stimulus = tuple(turn for turn in groundable if turn.stimulus_name == face.stimulus_name)
    with_output = tuple(turn for turn in groundable if turn.output)
    candidates = same_stimulus or with_output or groundable
    grounded = _grounded_from_turn(candidates[-1])
    if grounded is None:
        raise TypeError(_no_stimulus_message(prior_turns))
    # Memory is authoritative for stimulus identity and expected selection; the example payload
    # stays the model's, already bounded to live-stimulus shape by RuntimeInputData validation.
    preferred = RuntimeInputData(
        stimulus_name=grounded.stimulus_name,
        payload=face.payload,
        focus=grounded.focus if grounded.focus is not None else face.focus,
        focus_candidates=grounded.focus_candidates or face.focus_candidates,
        expected_event=grounded.expected_event or face.expected_event,
        expected_target=grounded.expected_target or face.expected_target,
        expected_data=grounded.expected_data if grounded.expected_data is not None else face.expected_data,
        expected_reason=grounded.expected_reason or face.expected_reason,
    )
    return generate.model_copy(update={"runtime_input": preferred})


class Learning(ability.Ability[InputData, OutputData]):
    """Decode lesson material, ground runtime input in memory, revise into inventory."""

    instructions: typing.ClassVar[str] = GENERATE_INSTRUCTIONS
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[OutputData]] = ability.ability_output_event(
        "bot.ability.learning.output",
        OutputData,
        description="Decoded lesson, memory-grounded runtime input, and inventory payload Revision applied.",
    )
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True

    _decoder: decoding.Decoder[InputData, DecodedData]
    _select_processing: processing.Processing
    _revision: revision.Revision
    _memory: memory.Memory

    @staticmethod
    def _child_id(instance: "Learning", suffix: str) -> str:
        return f"{hsm.id(instance)}{suffix}"

    @staticmethod
    def _has_learning_input(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, InputData)

    @staticmethod
    def _operation(event: hsm.Event[typing.Any]) -> tuple[str, str] | None:
        data = event.data
        if isinstance(data, processing.CompletionData):
            nested = data.input.input
            if isinstance(nested, SelectInput):
                return nested.operation_id, nested.generation
        if isinstance(data, revision.OutputData | revision.FailureData):
            return data.input.parent_operation_id, data.input.parent_generation
        if isinstance(data, _DecodedEventData | _SelectedEventData | _StageFailedData):
            return data.operation_id, data.generation
        return None

    @staticmethod
    def _matches_operation(instance: "Learning", event: hsm.Event[typing.Any]) -> bool:
        return processing.matches_private_terminal(instance, event, Learning._operation(event))

    @staticmethod
    def _private_event(
        instance: "Learning",
        source_event: hsm.Event[typing.Any],
        event_type: hsm.Event[typing.Any],
        data: object,
        *,
        operation_id: str,
        generation: str,
    ) -> hsm.Event[typing.Any]:
        """Build a self-addressed private event correlated to the caller's live operation."""

        if event_type.name == _StageFailedEvent.name and isinstance(data, ability.FailureData):
            data = _StageFailedData(failure=data, operation_id=operation_id, generation=generation)
        return dataclasses.replace(
            event_type.with_data(data),
            id=operation_id,
            source=hsm.id(instance),
            target=hsm.id(instance),
            metadata=dict(source_event.metadata),
        )

    @staticmethod
    def _fail_from_stage(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        if not isinstance(data, _StageFailedData):
            # Unreachable: paired with hsm.guard(_has_stage_failure), which narrows this payload.
            return
        processing.dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            failure=data.failure,
        )

    @staticmethod
    def _has_stage_failure(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _StageFailedData) and Learning._matches_operation(instance, event)

    @staticmethod
    def _has_decoded(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _DecodedEventData) and Learning._matches_operation(instance, event)

    @staticmethod
    def _is_cancel_request(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        if (
            not isinstance(event.data, processing.CancelData)
            or event.id != event.data.operation_id
            or event.target != hsm.id(instance)
        ):
            return False
        return (
            bool(instance._attachments)
            and event.source == hsm.id(instance._attachments[0])
            and processing.active_operation(instance, event.data.operation_id) is not None
        )

    @staticmethod
    def _emit_cancelled(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        if not isinstance(data, processing.CancelData):
            # Unreachable: paired with hsm.guard(_is_cancel_request), which narrows this payload.
            return
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                processing.CancelledEvent.with_data(
                    processing.CancelledData(
                        operation_id=data.operation_id,
                        token=data.token,
                        parent_operation_id=data.parent_operation_id,
                    )
                ),
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=dict(event.metadata),
            ),
        )
        processing.finish_operation(ctx, instance, data.operation_id)

    @staticmethod
    def _child_timeout_delay(
        ctx: hsm.Context,
        instance: "Learning",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, instance, event
        return _CHILD_OPERATION_TIMEOUT

    @staticmethod
    def _cancel_timeout_delay(
        ctx: hsm.Context,
        instance: "Learning",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, instance, event
        return _CANCEL_TEARDOWN_TIMEOUT

    @staticmethod
    async def _decode_activity(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        turn = event.data
        if not isinstance(turn, InputData):
            # Unreachable: paired with hsm.guard(_has_learning_input), which narrows this payload.
            return
        operation_id = event.id if event.id else uuid.uuid4().hex
        active = processing.active_operation(instance, operation_id)
        if active is None:
            active = await processing.start_operation(instance, operation_id)
        generation = hsm.id(active)
        try:
            decoded = await instance._decoder.decode(turn)
        except asyncio.CancelledError:
            # Cancel / decode-timeout transitions own the terminal; leave without double-dispatch.
            raise
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _StageFailedEvent.with_data(
                        _StageFailedData(
                            failure=ability.FailureData(message=f"Learning decode failed: {error}"),
                            operation_id=operation_id,
                            generation=generation,
                        )
                    ),
                    id=operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )
            return
        # Memory first: only source of live-input knowledge (no peer ability inventory).
        # A recall failure is not "nothing is remembered" — surface it as its own typed failure.
        try:
            prior_turns = _recall_prior_turns(instance._memory)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _StageFailedEvent.with_data(
                        _StageFailedData(
                            failure=ability.FailureData(message=f"Learning memory recall failed: {error}"),
                            operation_id=operation_id,
                            generation=generation,
                        )
                    ),
                    id=operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _DecodedEvent.with_data(
                    _DecodedEventData(
                        turn=turn,
                        decoded=decoded,
                        prior_turns=prior_turns,
                        operation_id=operation_id,
                        generation=generation,
                    )
                ),
                id=operation_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _fail_decode_timeout(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        operation_id = processing.active_operation_id(instance)
        processing.dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=operation_id,
            metadata=dict(event.metadata),
            failure=ability.FailureData(message="Learning decode timed out."),
        )

    @staticmethod
    async def _dispatch_select(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        if not isinstance(data, _DecodedEventData):
            # Unreachable: paired with hsm.guard(_has_decoded), which narrows this payload.
            return
        select_input = processing.InputData(
            input=SelectInput(
                turn=data.turn,
                decoded=data.decoded,
                prior_turns=data.prior_turns,
                operation_id=data.operation_id,
                generation=data.generation,
            ),
            schemas=(GenerateEvent,),
            actors={},
        )
        input_event = dataclasses.replace(
            instance._select_processing.input_event.with_data_and_id(
                select_input,
                Learning._child_id(instance, _SELECT_ID_SUFFIX),
            ),
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=hsm.id(instance._select_processing),
        )
        await hsm.dispatch(ctx, instance._select_processing, input_event)

    @staticmethod
    def _matches_select_output(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        child = instance._select_processing
        completion = event.data
        select_input = completion.input.input if isinstance(completion, processing.CompletionData) else None
        if not isinstance(select_input, SelectInput):
            return False
        return processing.matches_child_terminal(
            instance,
            child,
            event,
            name=child.output_event.name,
            request_id=Learning._child_id(instance, _SELECT_ID_SUFFIX),
            operation_id=select_input.operation_id,
            generation=select_input.generation,
        )

    @staticmethod
    def _matches_select_failure(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        child = instance._select_processing
        failure = event.data
        select_input = failure.input.input if isinstance(failure, processing.FailureData) else None
        if not isinstance(select_input, SelectInput):
            return False
        return processing.matches_child_terminal(
            instance,
            child,
            event,
            name=child.failed_event.name,
            request_id=Learning._child_id(instance, _SELECT_ID_SUFFIX),
            operation_id=select_input.operation_id,
            generation=select_input.generation,
        )

    @staticmethod
    def _on_select_output(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        # Both narrowings are unreachable: paired with hsm.guard(_matches_select_output).
        completion = event.data
        if not isinstance(completion, processing.CompletionData):
            return
        select_input = completion.input.input
        if not isinstance(select_input, SelectInput):
            return
        operation_id = select_input.operation_id
        try:
            generate = _require_memory_runtime_input(
                _coerce_generate(completion.output),
                select_input.prior_turns,
            )
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                Learning._private_event(
                    instance,
                    event,
                    _StageFailedEvent,
                    ability.FailureData(message=str(error)),
                    operation_id=operation_id,
                    generation=select_input.generation,
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            Learning._private_event(
                instance,
                event,
                _SelectedEvent,
                _SelectedEventData(
                    turn=select_input.turn,
                    decoded=select_input.decoded,
                    generate=generate,
                    prior_turns=select_input.prior_turns,
                    operation_id=operation_id,
                    generation=select_input.generation,
                ),
                operation_id=operation_id,
                generation=select_input.generation,
            ),
        )

    @staticmethod
    def _fail_select(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        # Both narrowings are unreachable: paired with hsm.guard(_matches_select_failure).
        data = event.data
        if not isinstance(data, processing.FailureData):
            return
        child_input = data.input.input
        if not isinstance(child_input, SelectInput):
            return
        processing.dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=child_input.operation_id,
            metadata=dict(event.metadata),
            failure=ability.FailureData(message=data.message),
        )

    @staticmethod
    def _has_selected(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _SelectedEventData) and Learning._matches_operation(instance, event)

    @staticmethod
    def _dispatch_revision(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        selected = event.data
        if not isinstance(selected, _SelectedEventData):
            # Unreachable: paired with hsm.guard(_has_selected), which narrows this payload.
            return
        generate = selected.generate
        try:
            intent = _intent_from_generate(generate)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                Learning._private_event(
                    instance,
                    event,
                    _StageFailedEvent,
                    ability.FailureData(message=f"Learning intent invalid: {error}"),
                    operation_id=selected.operation_id,
                    generation=selected.generation,
                ),
            )
            return
        # Revision handoff only: map memory-grounded runtime_input into Revision's turn shape.
        revision_input = revision.InputData(
            cognition_input=_revision_cognition_input(generate.runtime_input, operation_id=selected.operation_id),
            cognition_output=_expected_output(generate.runtime_input),
            instruction=revision.InstructionData(text=selected.decoded.text, kind=selected.decoded.kind),
            prior_episodes=_revision_prior_episodes(selected.prior_turns),
            intent=intent,
            parent_operation_id=selected.operation_id,
            parent_generation=selected.generation,
        )
        _ = hsm.dispatch(
            ctx,
            instance._revision,
            dataclasses.replace(
                instance._revision.input_event.with_data_and_id(revision_input, selected.operation_id),
                metadata=dict(event.metadata),
                source=hsm.id(instance),
                target=hsm.id(instance._revision),
            ),
        )

    @staticmethod
    def _matches_revision_output(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        child = instance._revision
        data = event.data
        if not isinstance(data, revision.OutputData):
            return False
        return processing.matches_child_terminal(
            instance,
            child,
            event,
            name=child.output_event.name,
            request_id=data.input.parent_operation_id,
            operation_id=data.input.parent_operation_id,
            generation=data.input.parent_generation,
        )

    @staticmethod
    def _matches_revision_failure(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        child = instance._revision
        data = event.data
        if not isinstance(data, revision.FailureData):
            return False
        return processing.matches_child_terminal(
            instance,
            child,
            event,
            name=child.failed_event.name,
            request_id=data.input.parent_operation_id,
            operation_id=data.input.parent_operation_id,
            generation=data.input.parent_generation,
        )

    @staticmethod
    def _complete_revision(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        if not isinstance(data, revision.OutputData):
            # Unreachable: paired with hsm.guard(_matches_revision_output), which narrows this payload.
            return
        runtime_input = _runtime_input_from_revision_stimulus(data.input.cognition_input)
        if runtime_input is None:
            processing.dispatch_terminal_failure(
                ctx,
                instance,
                operation_id=data.input.parent_operation_id,
                metadata=dict(event.metadata),
                failure=ability.FailureData(message="Learning lost runtime input after revision."),
            )
            return
        # Recover the expected selection from the turn's real selections.
        for item in data.input.cognition_output:
            runtime_input = runtime_input.model_copy(
                update={
                    "expected_event": item.event,
                    "expected_target": item.target,
                    "expected_data": item.data,
                    "expected_reason": item.reason,
                }
            )
            break
        decoded = _decoded_from_revision_input(data.input)
        if decoded is None:
            processing.dispatch_terminal_failure(
                ctx,
                instance,
                operation_id=data.input.parent_operation_id,
                metadata=dict(event.metadata),
                failure=ability.FailureData(message="Learning lost decoded lesson after revision."),
            )
            return
        processing.dispatch_terminal_output(
            ctx,
            instance,
            operation_id=data.input.parent_operation_id,
            metadata=dict(event.metadata),
            output=OutputData(decoded=decoded, runtime_input=runtime_input, behavior=data.applied),
        )

    @staticmethod
    def _fail_revision(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        if not isinstance(data, revision.FailureData):
            # Unreachable: paired with hsm.guard(_matches_revision_failure), which narrows this payload.
            return
        processing.dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=data.input.parent_operation_id,
            metadata=dict(event.metadata),
            failure=ability.FailureData(message=data.message),
        )

    @staticmethod
    def _request_reboot(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        processing.request_reboot(
            ctx,
            instance,
            event,
            reason=(
                "learning_detach_rollback_failed"
                if event.name == ability.Ability._composite_attachment_terminal_event.name
                else "learning_child_teardown_failed"
            ),
        )

    @staticmethod
    def _cancel_select(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        if not isinstance(data, processing.CancelData):
            # Unreachable: paired with hsm.guard(_is_cancel_request), which narrows this payload.
            return
        processing.dispatch_child_cancel(
            ctx,
            instance,
            instance._select_processing,
            event,
            request_id=Learning._child_id(instance, _SELECT_ID_SUFFIX),
            parent_operation_id=data.operation_id,
            token=data.token,
        )

    @staticmethod
    def _cancel_revision(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        if not isinstance(data, processing.CancelData):
            # Unreachable: paired with hsm.guard(_is_cancel_request), which narrows this payload.
            return
        processing.dispatch_child_cancel(
            ctx,
            instance,
            instance._revision,
            event,
            request_id=data.operation_id,
            parent_operation_id=data.operation_id,
            token=data.token,
        )

    @staticmethod
    def _cancel_select_timeout(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        operation_id = processing.active_operation_id(instance) or event.id or uuid.uuid4().hex
        processing.dispatch_child_cancel(
            ctx,
            instance,
            instance._select_processing,
            event,
            request_id=Learning._child_id(instance, _SELECT_ID_SUFFIX),
            parent_operation_id=operation_id,
            token=uuid.uuid4().hex,
        )

    @staticmethod
    def _cancel_revision_timeout(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        operation_id = processing.active_operation_id(instance) or event.id or uuid.uuid4().hex
        processing.dispatch_child_cancel(
            ctx,
            instance,
            instance._revision,
            event,
            request_id=operation_id,
            parent_operation_id=operation_id,
            token=uuid.uuid4().hex,
        )

    @staticmethod
    def _matches_cancelled(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, processing.CancelledData):
            return False
        select_id = Learning._child_id(instance, _SELECT_ID_SUFFIX)
        select_matches = event.source == hsm.id(instance._select_processing) and data.operation_id == select_id
        revision_matches = (
            event.source == hsm.id(instance._revision)
            and data.parent_operation_id is not None
            and processing.active_operation(instance, data.parent_operation_id) is not None
        )
        return (
            (select_matches or revision_matches) and event.id == data.operation_id and event.target == hsm.id(instance)
        )

    @staticmethod
    def _emit_learning_cancelled(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        if not isinstance(data, processing.CancelledData):
            # Unreachable: paired with hsm.guard(_matches_cancelled), which narrows this payload.
            return
        operation_id = data.parent_operation_id or data.operation_id
        if not instance._attachments:
            processing.finish_operation(ctx, instance, operation_id)
            return
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                processing.CancelledEvent.with_data(
                    processing.CancelledData(operation_id=operation_id, token=data.token)
                ),
                id=operation_id,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=dict(event.metadata),
            ),
        )
        processing.finish_operation(ctx, instance, operation_id)

    @staticmethod
    def _fail_operation_timeout(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        operation_id = processing.active_operation_id(instance)
        processing.dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=operation_id,
            metadata=dict(event.metadata),
            failure=ability.FailureData(message="Learning child operation timed out."),
        )

    @staticmethod
    def _fail_cancel_timeout(ctx: hsm.Context, instance: "Learning", event: hsm.Event[typing.Any]) -> None:
        operation_id = processing.active_operation_id(instance)
        processing.dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=operation_id,
            metadata=dict(event.metadata),
            failure=ability.FailureData(message="Learning child cancellation timed out."),
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Learning",
        hsm.initial(hsm.target("/Learning/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Learning/idle"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
            ),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_learning_input),
                hsm.target("/Learning/decoding"),
            ),
        ),
        hsm.state(
            "decoding",
            hsm.defer(input_event),
            hsm.activity(_decode_activity),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
                hsm.target("/Learning/idle"),
            ),
            hsm.transition(
                hsm.on(_DecodedEvent),
                hsm.guard(_has_decoded),
                hsm.target("/Learning/selecting"),
            ),
            hsm.transition(
                hsm.on(_StageFailedEvent),
                hsm.guard(_has_stage_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Learning/idle"),
            ),
            hsm.transition(
                hsm.after(_child_timeout_delay),
                hsm.effect(_fail_decode_timeout),
                hsm.target("/Learning/idle"),
            ),
        ),
        hsm.state(
            "selecting",
            hsm.defer(input_event),
            hsm.activity(_dispatch_select),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_cancel_select),
                hsm.target("/Learning/cancelling"),
            ),
            hsm.transition(
                hsm.on(processing.OutputEvent),
                hsm.guard(_matches_select_output),
                hsm.effect(_on_select_output),
            ),
            hsm.transition(
                hsm.on(_SelectedEvent),
                hsm.guard(_has_selected),
                hsm.effect(_dispatch_revision),
                hsm.target("/Learning/revising"),
            ),
            hsm.transition(
                hsm.on(_StageFailedEvent),
                hsm.guard(_has_stage_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Learning/idle"),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.guard(_matches_select_failure),
                hsm.effect(_fail_select),
                hsm.target("/Learning/idle"),
            ),
            hsm.transition(
                hsm.after(_child_timeout_delay),
                hsm.effect(_cancel_select_timeout),
                hsm.target("/Learning/timing_out"),
            ),
        ),
        hsm.state(
            "revising",
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_cancel_revision),
                hsm.target("/Learning/cancelling"),
            ),
            hsm.transition(
                hsm.on(revision.OutputEvent),
                hsm.guard(_matches_revision_output),
                hsm.effect(_complete_revision),
                hsm.target("/Learning/idle"),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.guard(_matches_revision_failure),
                hsm.effect(_fail_revision),
                hsm.target("/Learning/idle"),
            ),
            hsm.transition(
                hsm.on(_StageFailedEvent),
                hsm.guard(_has_stage_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Learning/idle"),
            ),
            hsm.transition(
                hsm.after(_child_timeout_delay),
                hsm.effect(_cancel_revision_timeout),
                hsm.target("/Learning/timing_out"),
            ),
        ),
        hsm.state(
            "cancelling",
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(processing.CancelledEvent),
                hsm.guard(_matches_cancelled),
                hsm.effect(_emit_learning_cancelled),
                hsm.target("/Learning/idle"),
            ),
            hsm.transition(
                hsm.after(_cancel_timeout_delay),
                hsm.effect(_fail_cancel_timeout, _request_reboot),
                hsm.target("/Learning/rebooting"),
            ),
        ),
        hsm.state(
            "timing_out",
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(processing.CancelledEvent),
                hsm.guard(_matches_cancelled),
                hsm.effect(_fail_operation_timeout),
                hsm.target("/Learning/idle"),
            ),
            hsm.transition(
                hsm.after(_cancel_timeout_delay),
                hsm.effect(_fail_cancel_timeout, _request_reboot),
                hsm.target("/Learning/rebooting"),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal, _request_reboot),
                hsm.target("/Learning/rebooting"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Learning/idle"),
            ),
        ),
        hsm.state("rebooting", hsm.defer(input_event)),
        hsm.observe(observer),
    )

    def __init__(
        self,
        *,
        decoder: decoding.Decoder[InputData, DecodedData],
        processor: processing.Processor | revision.ProcessorFactory,
        memory: memory.Memory,
    ) -> None:
        super().__init__()
        leaf = processor() if not isinstance(processor, processing.Processor) else processor
        self._decoder = decoder
        self._select_processing = processing.Processing(
            processor=leaf,
            instructions=type(self).instructions,
        )
        self._revision = revision.Revision(processor=leaf, memory=memory)
        self._memory = memory
        self._attachment_group: attachment.Group = attachment.Group(
            self._select_processing,
            self._revision,
            self._memory,
        )


OutputEvent = Learning.output_event

__all__ = [
    "GENERATE_INSTRUCTIONS",
    "DecodedData",
    "RuntimeInputData",
    "GenerateData",
    "GenerateEvent",
    "InputData",
    "InputEvent",
    "Learning",
    "OutputData",
    "OutputEvent",
    "RememberedSelection",
    "RememberedTurn",
    "SelectInput",
]

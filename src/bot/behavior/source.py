"""Starlark source contracts for behavior HSM behaviors."""

from . import schema

import io
import json
import re
import select
import subprocess
import sys
import threading
import time
import tokenize
import typing

import hsm
import pydantic
from starlark_go import Starlark, configure_starlark
from starlark_go.errors import StarlarkError

from bot.telemetry import span

SOURCE_MAX_BYTES = 64 * 1024
SOURCE_MAX_TOKENS = 8 * 1024
SOURCE_MAX_NESTING = 64
SOURCE_MAX_COLLECTION_ITEMS = 1024
SOURCE_MAX_JSON_NODES = 8 * 1024
SOURCE_MAX_JSON_DEPTH = 64
SOURCE_MAX_JSON_STRING_BYTES = 64 * 1024
SOURCE_MAX_DECLARED_EVENTS = 256
SOURCE_MAX_RESULT_BYTES = 512 * 1024
SOURCE_MAX_MEMORY_BYTES = 256 * 1024 * 1024
SOURCE_EVALUATION_SECONDS = 3.0
SOURCE_WORKER_READY_SECONDS = 15.0
SOURCE_WORKER_TEARDOWN_SECONDS = 1.0
SOURCE_PROCESS_POLL_SECONDS = 0.05
SOURCE_EMPTY_POLL_SECONDS = 0.0
SOURCE_WORKER_READY = b'{"ready":true}\n'

_CALLBACK_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_MODEL_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
_HSM_NAMESPACE_NAMES = frozenset(
    {
        "activity",
        "after",
        "defer",
        "define",
        "dispatch",
        "effect",
        "entry",
        "event",
        "exit",
        "final",
        "guard",
        "initial",
        "on",
        "state",
        "target",
        "transition",
    }
)
_ELEMENT_KEYS: dict[str, frozenset[str]] = {
    "activity": frozenset({"kind", "callbacks"}),
    "after": frozenset({"kind", "seconds"}),
    "defer": frozenset({"kind", "events"}),
    "define": frozenset({"kind", "name", "elements"}),
    "effect": frozenset({"kind", "callbacks"}),
    "entry": frozenset({"kind", "callbacks"}),
    "exit": frozenset({"kind", "callbacks"}),
    "final": frozenset({"kind", "name"}),
    "guard": frozenset({"kind", "callback"}),
    "initial": frozenset({"kind", "elements"}),
    "on": frozenset({"kind", "events"}),
    "state": frozenset({"kind", "name", "elements"}),
    "target": frozenset({"kind", "path"}),
    "transition": frozenset({"kind", "name", "elements"}),
}
ElementSpec: typing.TypeAlias = dict[str, object]


class SourceError(ValueError):
    """Raised when Starlark behavior source cannot be evaluated as a stateforward.bot behavior declaration.

    ``failure_kind`` is ``source_error`` for invalid source, worker evaluation
    failure, or an invalid worker payload. Ready or evaluation budget misses set
    ``failure_kind`` to ``timeout``. When produced by ``check`` / ``start``,
    ``report`` carries structured diagnostics for fix loops.
    """

    report: object | None
    failure_kind: str

    def __init__(self, message: str, *, report: object | None = None) -> None:
        super().__init__(message)
        self.report = report
        self.failure_kind = "source_error"


def _validate_json_budget(value: object, *, path: str = "$") -> None:
    """Reject unbounded source-authored JSON trees before schema validation."""

    active: set[int] = set()
    nodes = 0

    def visit(item: object, *, item_path: str, depth: int) -> None:
        nonlocal nodes
        if depth > SOURCE_MAX_JSON_DEPTH:
            raise ValueError(f"source JSON value at {item_path} exceeds maximum depth of {SOURCE_MAX_JSON_DEPTH}.")
        nodes += 1
        if nodes > SOURCE_MAX_JSON_NODES:
            raise ValueError(f"source JSON value at {item_path} exceeds its node budget.")
        if isinstance(item, str):
            if len(item.encode("utf-8")) > SOURCE_MAX_JSON_STRING_BYTES:
                raise ValueError(f"source JSON string at {item_path} exceeds its byte budget.")
            return
        if item is None or isinstance(item, bool | int | float):
            return
        if isinstance(item, dict):
            mapping = typing.cast(dict[object, object], item)
            item_id = id(mapping)
            if item_id in active:
                raise ValueError(f"source JSON value at {item_path} contains a cycle.")
            if len(mapping) > SOURCE_MAX_COLLECTION_ITEMS:
                raise ValueError(f"source JSON object at {item_path} exceeds its item budget.")
            active.add(item_id)
            try:
                for key, child in mapping.items():
                    visit(child, item_path=f"{item_path}.{key}", depth=depth + 1)
            finally:
                active.remove(item_id)
            return
        if isinstance(item, list | tuple):
            sequence = typing.cast(list[object] | tuple[object, ...], item)
            item_id = id(sequence)
            if item_id in active:
                raise ValueError(f"source JSON value at {item_path} contains a cycle.")
            if len(sequence) > SOURCE_MAX_COLLECTION_ITEMS:
                raise ValueError(f"source JSON array at {item_path} exceeds its item budget.")
            active.add(item_id)
            try:
                for index, child in enumerate(sequence):
                    visit(child, item_path=f"{item_path}[{index}]", depth=depth + 1)
            finally:
                active.remove(item_id)
            return
        raise ValueError(f"source JSON value at {item_path} has unsupported type {type(item).__name__}.")

    visit(value, item_path=path, depth=0)


class EventContract(pydantic.BaseModel):
    """HSM event contract declared by Starlark behavior source."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        json_schema_extra={
            "description": (
                "Event contract declared by Starlark behavior source. Cognition abilities write these contracts so "
                "the compiled behavior exposes typed HSM input, output, and internal events instead of an opaque "
                "script boundary."
            ),
            "examples": [
                {
                    "name": "bot.behavior.answer_greeting.input",
                    "schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                    "description": "Greeting text recognized by cognition or conversation.",
                    "examples": [{"text": "hello"}],
                }
            ],
        },
    )

    name: str = pydantic.Field(
        min_length=1,
        max_length=512,
        description="Stable HSM event name for a compiled behavior boundary or internal behavior transition.",
        examples=["bot.behavior.answer_greeting.input"],
    )
    payload_schema: schema.JsonSchema = pydantic.Field(
        default_factory=dict,
        alias="schema",
        serialization_alias="schema",
        description=(
            "JSON schema for the event payload. This is authored as source data and copied onto the HSM event."
        ),
        examples=[
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }
        ],
    )
    description: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional human-readable description added to the event schema for LLM tool use.",
        examples=["Greeting text recognized by cognition or conversation."],
    )
    examples: tuple[object, ...] = pydantic.Field(
        default=(),
        max_length=SOURCE_MAX_COLLECTION_ITEMS,
        description="Example event payloads added to the event schema for LLM tool use.",
        examples=[[{"text": "hello"}]],
    )

    @pydantic.field_validator("payload_schema")
    @classmethod
    def validate_schema_object(cls, value: schema.JsonSchema) -> schema.JsonSchema:
        """Require source-authored schemas to stay JSON-object-shaped."""

        _validate_json_budget(value, path="event.schema")
        schema.validate_supported_json_schema(value)
        return dict(value)

    @pydantic.field_validator("examples")
    @classmethod
    def validate_examples_budget(cls, value: tuple[object, ...]) -> tuple[object, ...]:
        _validate_json_budget(value, path="event.examples")
        return value

    def hsm_event(self) -> hsm.Event[object]:
        """Build the HSM event declared by this source contract."""

        schema = dict(self.payload_schema)
        if self.description is not None:
            schema["description"] = self.description
        if self.examples:
            schema["examples"] = list(self.examples)
        return hsm.Event[object](name=self.name, schema=schema)


class Source(pydantic.BaseModel):
    """Parsed Starlark program for a behavior (HSM-aligned source, not a live instance)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        json_schema_extra={
            "description": (
                "Starlark-authored HSM behavior specification for an executable behavior. Reflection, reasoning, "
                "or another cognitive ability can write this source, and stateforward.bot lowers its model into a static "
                "stateforward.hsm topology with safe Starlark callback wrappers."
            ),
            "examples": [
                {
                    "kind": "behavior_program",
                    "name": "AnswerGreeting",
                    "triggers": ["conversation.greeting.recognized"],
                    "input_event": {
                        "name": "bot.behavior.answer_greeting.input",
                        "schema": {"type": "object"},
                    },
                    "output_event": {
                        "name": "bot.behavior.answer_greeting.output",
                        "schema": {"type": "object"},
                    },
                    "model": {
                        "kind": "define",
                        "name": "AnswerGreeting",
                        "elements": [
                            {"kind": "initial", "elements": [{"kind": "target", "path": "/AnswerGreeting/idle"}]},
                            {"kind": "state", "name": "idle", "elements": []},
                        ],
                    },
                }
            ],
        },
    )

    kind: typing.Literal["behavior_program"] = pydantic.Field(
        description="SourceData kind for executable behavior.",
        examples=["behavior_program"],
    )
    name: str = pydantic.Field(
        min_length=1,
        max_length=256,
        description="PascalCase HSM model name for the compiled behavior.",
        examples=["AnswerGreeting"],
    )
    input_event: EventContract = pydantic.Field(
        description="InputData event contract accepted by the behavior.",
    )
    output_event: EventContract = pydantic.Field(
        description="OutputData event contract emitted by the behavior.",
    )
    model: ElementSpec = pydantic.Field(
        description=(
            "SourceData-authored HSM model tree. It mirrors lower-case hsm builders such as define, state, "
            "transition, on, guard, effect, entry, exit, activity, target, initial, defer, after, and final."
        ),
    )
    triggers: tuple[str, ...] = pydantic.Field(
        default=(),
        max_length=SOURCE_MAX_COLLECTION_ITEMS,
        description="External event names that fast intuition may use to propose the behavior.",
        examples=[["conversation.greeting.recognized"]],
    )
    description: str = pydantic.Field(
        default="",
        description="Human-readable behavior description.",
        examples=["Answer a recurring greeting without asking reasoning to choose every step."],
    )
    examples: tuple[str, ...] = pydantic.Field(
        default=(),
        max_length=SOURCE_MAX_COLLECTION_ITEMS,
        description="Natural-language examples of when the behavior should be proposed.",
        examples=[["answer hello with a matching greeting"]],
    )
    source: str = pydantic.Field(
        default="",
        exclude=True,
        repr=False,
        description="Original Starlark source used to recreate the safe callback runtime.",
    )

    @pydantic.field_validator("name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """Keep generated HSM model paths stable and reviewable."""

        if _MODEL_NAME_RE.fullmatch(value) is None:
            raise ValueError("must be a PascalCase HSM model name")
        return value

    @pydantic.model_validator(mode="after")
    def validate_model(self) -> typing.Self:
        """Validate the source-authored model tree before lowering."""

        _validate_model_tree(self.model, model_name=self.name)
        if self.model.get("name") != self.name:
            raise ValueError("behavior model name must match the behavior name.")
        _ = self.declared_event_specs()
        return self

    def declared_event_specs(self, *, failed_event: hsm.Event[object] | None = None) -> dict[str, EventContract]:
        """Return event specs declared at the behavior boundary or inside its model tree."""

        del failed_event
        specs: dict[str, EventContract] = {}
        _add_event_spec(specs, self.input_event)
        _add_event_spec(specs, self.output_event)
        for spec in _collect_event_specs(self.model):
            _add_event_spec(specs, spec)
        return specs


def _element(kind: str, **fields: object) -> dict[str, object]:
    return {"kind": kind, **fields}


def _event_builder(**kwargs: object) -> dict[str, object]:
    return typing.cast(dict[str, object], EventContract.model_validate(kwargs).model_dump(by_alias=True))


def _define_builder(name: str, *elements: object) -> dict[str, object]:
    return _element("define", name=name, elements=tuple(elements))


def _state_builder(name: str, *elements: object) -> dict[str, object]:
    return _element("state", name=name, elements=tuple(elements))


def _transition_builder(*elements: object) -> dict[str, object]:
    name: str | None = None
    child_elements = tuple(elements)
    if child_elements and isinstance(child_elements[0], str):
        name = child_elements[0]
        child_elements = child_elements[1:]
    return _element("transition", name=name, elements=child_elements)


def _on_builder(*events: object) -> dict[str, object]:
    return _element("on", events=tuple(events))


def _target_builder(path: str) -> dict[str, object]:
    return _element("target", path=path)


def _initial_builder(*elements: object) -> dict[str, object]:
    child_elements = tuple(_target_builder(element) if isinstance(element, str) else element for element in elements)
    return _element("initial", elements=child_elements)


def _defer_builder(*events: object) -> dict[str, object]:
    return _element("defer", events=tuple(events))


def _final_builder(name: str) -> dict[str, object]:
    return _element("final", name=name)


def _guard_builder(callback: str) -> dict[str, object]:
    return _element("guard", callback=callback)


def _callback_behavior_builder(kind: str, *callbacks: str) -> dict[str, object]:
    return _element(kind, callbacks=tuple(callbacks))


def _effect_builder(*callbacks: str) -> dict[str, object]:
    return _callback_behavior_builder("effect", *callbacks)


def _entry_builder(*callbacks: str) -> dict[str, object]:
    return _callback_behavior_builder("entry", *callbacks)


def _exit_builder(*callbacks: str) -> dict[str, object]:
    return _callback_behavior_builder("exit", *callbacks)


def _activity_builder(*callbacks: str) -> dict[str, object]:
    return _callback_behavior_builder("activity", *callbacks)


def _after_builder(*, seconds: float) -> dict[str, object]:
    return _element("after", seconds=seconds)


def _dispatch_parse_stub(event: object, data: object | None = None) -> None:
    del event, data
    raise SourceError("dispatch is only available inside behavior callbacks.")


def _behavior_program_builder(**kwargs: object) -> dict[str, object]:
    return typing.cast(dict[str, object], Source.model_validate(kwargs).model_dump(by_alias=True))


def starlark_globals() -> dict[str, object]:
    """Return the builder namespace the isolated worker binds.

    This is a parse-runtime internal, not a public evaluation entry. Callers
    must not use it to evaluate untrusted Starlark in the parent process.
    ``parse_source`` is the only public eval entry for behavior source.
    """

    return {
        "activity": _activity_builder,
        "after": _after_builder,
        "defer": _defer_builder,
        "define": _define_builder,
        "dispatch": _dispatch_parse_stub,
        "effect": _effect_builder,
        "entry": _entry_builder,
        "event": _event_builder,
        "exit": _exit_builder,
        "final": _final_builder,
        "guard": _guard_builder,
        "behavior_program": _behavior_program_builder,
        "initial": _initial_builder,
        "on": _on_builder,
        "state": _state_builder,
        "target": _target_builder,
        "transition": _transition_builder,
    }


# LLM-facing authoring contract: API only — no complete behavior programs to copy.
# Topology notes align with stateforward HSM dsl.md (Initial = target + optional effect).
STARLARK_API = """
Starlark behavior API (author source from scratch; do not paste canned full programs).
Aligned with HSM DSL (snake_case aliases of Define/State/Initial/Transition/…).

Required program shape:
1) input_event = hsm.event(name=..., schema={...}, description=..., examples=[...])
2) output_event = hsm.event(name=..., schema={...}, description=..., examples=[...])
3) optional metadata globals:
   - triggers = ["external.event.name", ...]
   - description = "..."
   - examples = ["when this should run", ...]
4) top-level callback defs: def callback_name(event): ...
5) behavior = hsm.define(
       "PascalCaseName",
       hsm.initial(hsm.target("/PascalCaseName/idle")),  # required target; optional effect
       hsm.state(
           "idle",
           hsm.transition(
               hsm.on(input_event),
               hsm.guard("callback_name"),   # optional
               hsm.effect("callback_name"),  # optional
           ),
       ),
   )
   Model name must be PascalCase and match the behavior install name.

Topology (from HSM Initial / Transition rules; behavior Starlark subset):
- hsm.initial(...) declares the default entry transition into a composite/root.
  Partials: required target (hsm.target(...) or absolute path string) and optional effect
  (hsm.effect("callback")). Example: hsm.initial(hsm.target("/Name/idle"), hsm.effect("setup")).
- Do NOT put hsm.transition(...) or hsm.on(...) under initial. Initial already owns one
  transition (auto-triggered on entry). Nested transition/on/guard there causes
  "initial vertex has more than one outgoing transition" or invalid initial transition.
- Initial transition cannot have a guard and cannot declare extra event triggers.
- Event-driven transitions belong under hsm.state(...): hsm.transition(hsm.on(...), target?, guard?, effect?).
- A transition must have a target and/or at least one effect (guard alone is not enough).
- Every composite/root needs exactly one initial entry into a declared state path.
- Absolute target paths: /ModelName/state (nested /ModelName/parent/child).

Builders available in Starlark (hsm. prefix optional after normalize):
- hsm.event(name, schema, description=None, examples=())
- hsm.define(name, *elements)
- hsm.state(name, *elements)
- hsm.transition([optional_name,] *elements)
- hsm.on(*events)          # event() results / variables — not bare name strings for model events
- hsm.target("/ModelName/statePath")
- hsm.initial(target, effect?)   # target required; effect optional
- hsm.guard("callback_name")     # string callback name only
- hsm.effect("callback_name", ...)
- hsm.entry("callback_name", ...) / hsm.exit(...) / hsm.activity(...)
- hsm.after(seconds=number)
- hsm.defer(*events)
- hsm.final(name)
- hsm.dispatch(event, data, target=None)
  Only inside effect/entry/exit/activity — never in guards.
  event is a declared name or an hsm.event(...) value (name+schema). No ability bindings.
  Every emit stamps source = behavior_id (hsm.id of the live behavior) so devices/abilities
  can reply with target=event["source"] (or target=behavior_id).
  Optional target is a live instance id; omit it to process the event on the behavior.
  behavior_id is also available as a callback global.
  Reply events processed by the behavior must appear in the model (hsm.on(reply_event)).

Behavior-only Starlark limits (not full HSM host DSL):
- Callbacks are string names to top-level def callback(event): ... (no raw function objects)
- No host-language operations, attributes, submachine models, history, choice, entry/exit points
- No direct ability bindings or ability-alias dispatch — events only
- Guards: def name(event): return bool
- Event keys: name, data, id, source, target, kind — the full modeled event,
  not data alone (Autonomy builds the behavior input event from the live turn event)
  (metadata is telemetry-only and is intentionally omitted from the Starlark event)
- Effects/activities must hsm.dispatch(declared_event, payload_dict, target=optional_id)
  and must not return a value (returning is a runtime error)
- To wait on a reply: transition on the reply event after dispatching with source stamped

JSON schema for event payloads (object-shaped only):
- Supported: type, properties, required, additionalProperties, items, enum, const,
  oneOf, anyOf, allOf, description, examples, minimum, maximum, minLength, maxLength, pattern
- Unsupported: $ref and other non-object host-language extensions

Event naming:
- Prefer stable dotted names such as bot.behavior.<snake_model>.input / .output
- Public input/output names must be unique and must not collide with generated
  bot.behavior.<snake_model>.failed

Invent input/output contracts, guards, and effects from the observed cognitive pattern
(this turn + prior_episodes / existing_behavior). The source field is the program you write.

Behavior terminal output (output_event payload) must be cognition event selection(s):
- Prefer a single object with keys: event (required), target?, data?, reason?
- Mirror the event names, targets, and data field names from prior episode / this-turn outputs
- Fill live values from event["data"] and event["id"] / event["source"] / event["target"]
  at runtime; do not hardcode example ids. Never read event["metadata"] —
  it is not on the event and must not carry coordination or selection IDs.
- Input contracts and guards must use fields present on the live behavior input
- triggers should list the stimulus names that should propose this behavior
""".strip()


def _source_tokens(program: str) -> list[tokenize.TokenInfo]:
    if len(program) > SOURCE_MAX_BYTES:
        raise SourceError(f"Starlark behavior source exceeds the {SOURCE_MAX_BYTES}-byte budget.")
    encoded_size = len(program.encode("utf-8"))
    if encoded_size > SOURCE_MAX_BYTES:
        raise SourceError(f"Starlark behavior source exceeds the {SOURCE_MAX_BYTES}-byte budget.")
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(program).readline))
    except tokenize.TokenError as error:
        raise SourceError(str(error)) from error
    if len(tokens) > SOURCE_MAX_TOKENS:
        raise SourceError(f"Starlark behavior source exceeds the {SOURCE_MAX_TOKENS}-token budget.")

    bracket_stack: list[str] = []
    comma_counts: list[int] = []
    for token in tokens:
        if token.type != tokenize.OP:
            continue
        if token.string in "([{":
            bracket_stack.append(token.string)
            _ = comma_counts.append(0)
            if len(bracket_stack) > SOURCE_MAX_NESTING:
                raise SourceError(f"Starlark behavior source exceeds the {SOURCE_MAX_NESTING}-level nesting budget.")
        elif token.string == "," and comma_counts:
            comma_counts[-1] = comma_counts[-1] + 1
            if comma_counts[-1] > SOURCE_MAX_COLLECTION_ITEMS:
                raise SourceError("Starlark behavior source exceeds its collection item budget.")
        elif token.string in ")]}":
            if bracket_stack:
                _ = bracket_stack.pop()
                _ = comma_counts.pop()
    return tokens


def normalize_source(source: str) -> str:
    """Rewrite ``hsm.foo(...)`` calls to the flat builder names the Starlark bridge binds.

    This rewrite is a parse-runtime internal for the isolated worker and for
    compiled-callback execution. It does not evaluate source and is not a
    ``parse_source`` substitute. Callers must not evaluate the rewritten text
    in the parent to parse untrusted behavior source; use ``parse_source``.
    """

    tokens = _source_tokens(source)
    rewritten: list[tuple[int, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token.type == tokenize.NAME
            and token.string == "hsm"
            and index + 2 < len(tokens)
            and tokens[index + 1].string == "."
            and tokens[index + 2].type == tokenize.NAME
            and tokens[index + 2].string in _HSM_NAMESPACE_NAMES
        ):
            rewritten.append((tokens[index + 2].type, tokens[index + 2].string))
            index += 3
            continue
        rewritten.append((token.type, token.string))
        index += 1
    return tokenize.untokenize(rewritten)


def _apply_worker_limits(*, cpu_seconds: int | None) -> None:
    """Apply best-effort OS limits inside an already isolated evaluation process."""

    try:
        import resource
    except ImportError:
        return
    try:
        address_space = typing.cast(int | None, getattr(resource, "RLIMIT_AS", None))
        if address_space is not None:
            resource.setrlimit(address_space, (SOURCE_MAX_MEMORY_BYTES, SOURCE_MAX_MEMORY_BYTES))
    except (OSError, ValueError):
        pass
    if cpu_seconds is None:
        return
    try:
        cpu_limit = typing.cast(int | None, getattr(resource, "RLIMIT_CPU", None))
        if cpu_limit is not None:
            resource.setrlimit(cpu_limit, (cpu_seconds, cpu_seconds))
    except (OSError, ValueError):
        pass


def _worker_error(error: BaseException) -> str:
    message = str(error).strip()
    return message or f"{type(error).__name__} during isolated Starlark evaluation."


def _encode_worker_payload(payload: dict[str, object]) -> bytes:
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > SOURCE_MAX_RESULT_BYTES:
        return json.dumps(
            {
                "ok": False,
                "error": "isolated Starlark evaluation result exceeds its wire-size budget.",
            },
            separators=(",", ":"),
        ).encode("utf-8")
    return encoded


def _write_worker_payload(payload: dict[str, object]) -> None:
    sys.stdout.buffer.write(_encode_worker_payload(payload))
    sys.stdout.buffer.flush()


def _write_worker_ready() -> None:
    sys.stdout.buffer.write(SOURCE_WORKER_READY)
    sys.stdout.buffer.flush()


def _source_worker_main() -> None:
    """Evaluate top-level source in a clean interpreter that can be terminated safely."""

    cpu_floor_seconds = 1
    _apply_worker_limits(cpu_seconds=None)
    try:
        _write_worker_ready()
        _apply_worker_limits(cpu_seconds=max(cpu_floor_seconds, int(SOURCE_EVALUATION_SECONDS) + cpu_floor_seconds))
        program = sys.stdin.buffer.read(SOURCE_MAX_BYTES + 1).decode("utf-8")
        configure_starlark(allow_recursion=False)
        spec = _parse_source_in_process(program)
        _write_worker_payload({"ok": True, "spec": spec.model_dump(mode="json", by_alias=True)})
    except BaseException as error:
        try:
            _write_worker_payload({"ok": False, "error": _worker_error(error)})
        except (BrokenPipeError, OSError, ValueError):
            pass


def _evaluation_timeout() -> SourceError:
    error = SourceError(f"Starlark source evaluation exceeded its {SOURCE_EVALUATION_SECONDS:g}-second budget.")
    error.failure_kind = "timeout"
    return error


def _ready_timeout() -> SourceError:
    error = SourceError(
        f"isolated Starlark worker did not become ready within {SOURCE_WORKER_READY_SECONDS:g} seconds."
    )
    error.failure_kind = "timeout"
    return error


def _restore_model_tree(value: object) -> object:
    if not isinstance(value, dict):
        return value
    mapping = typing.cast(dict[str, object], value)
    restored: dict[str, object] = dict(mapping)
    elements = mapping.get("elements")
    if isinstance(elements, list):
        restored["elements"] = tuple(_restore_model_tree(item) for item in typing.cast(list[object], elements))
    events = mapping.get("events")
    if isinstance(events, list):
        restored["events"] = tuple(typing.cast(list[object], events))
    callbacks = mapping.get("callbacks")
    if isinstance(callbacks, list):
        restored["callbacks"] = tuple(typing.cast(list[object], callbacks))
    return restored


class _WorkerConnection:
    def send_bytes(self, buffer: bytes) -> None:
        raise NotImplementedError

    def wait_ready(self, *, timeout: float) -> None:
        raise NotImplementedError

    def poll(self, timeout: float = 0.0) -> bool:
        raise NotImplementedError

    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _WorkerProcess:
    def start(self) -> None:
        raise NotImplementedError

    def is_alive(self) -> bool:
        raise NotImplementedError

    def terminate(self) -> None:
        raise NotImplementedError

    def kill(self) -> None:
        raise NotImplementedError

    def join(self, timeout: float | None = None) -> None:
        raise NotImplementedError


class _WorkerContext:
    def Pipe(self, *, duplex: bool) -> tuple[_WorkerConnection, _WorkerConnection]:
        raise NotImplementedError

    def Process(
        self,
        *,
        target: typing.Callable[..., object],
        args: tuple[object, ...],
    ) -> _WorkerProcess:
        raise NotImplementedError


class _ParentStdioConnection(_WorkerConnection):
    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._closed = False

    def bind(self, *, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    def send_bytes(self, buffer: bytes) -> None:
        process = self._process
        stdin = None if process is None else process.stdin
        if process is None or stdin is None:
            raise SourceError("isolated Starlark source evaluation failed before returning a result.")
        stdin.write(buffer)
        stdin.flush()
        stdin.close()

    def wait_ready(self, *, timeout: float) -> None:
        process = self._process
        stdout = None if process is None else process.stdout
        if process is None or stdout is None or stdout.closed:
            raise SourceError("isolated Starlark worker did not become ready.")
        deadline = time.monotonic() + timeout
        buffer = bytearray()
        empty_read_bytes = 1
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select((stdout,), (), (), remaining)
            if not ready:
                if process.poll() is not None:
                    leftover = stdout.read(SOURCE_MAX_RESULT_BYTES - len(buffer))
                    if leftover:
                        buffer.extend(leftover)
                    break
                continue
            unread = max(empty_read_bytes, len(SOURCE_WORKER_READY) - len(buffer))
            chunk = typing.cast(io.BufferedReader, stdout).read1(unread)
            if not chunk:
                break
            buffer.extend(chunk)
            if b"\n" in buffer:
                break
        payload = bytes(buffer)
        if payload == SOURCE_WORKER_READY:
            return
        if payload:
            try:
                payload_value: object = typing.cast(object, json.loads(payload.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload_value = None
            if isinstance(payload_value, dict) and payload_value.get("ok") is not True:
                message = payload_value.get("error")
                raise SourceError(
                    message if isinstance(message, str) else "isolated Starlark worker did not become ready."
                )
        raise _ready_timeout()

    def poll(self, timeout: float = 0.0) -> bool:
        process = self._process
        stdout = None if process is None else process.stdout
        if process is None or stdout is None or stdout.closed:
            return False
        ready, _, _ = select.select((stdout,), (), (), timeout)
        return bool(ready)

    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        process = self._process
        stdout = None if process is None else process.stdout
        if process is None or stdout is None:
            raise SourceError("isolated Starlark source evaluation returned an invalid result.")
        limit = SOURCE_MAX_RESULT_BYTES if maxlength is None else maxlength
        chunks: list[bytes] = []
        total = 0
        deadline = time.monotonic() + SOURCE_EVALUATION_SECONDS
        while total < limit and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select((stdout,), (), (), remaining)
            if not ready:
                if process.poll() is not None:
                    leftover = stdout.read(limit - total)
                    if leftover:
                        chunks.append(leftover)
                    break
                continue
            chunk = typing.cast(io.BufferedReader, stdout).read1(limit - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass


class _ChildStdioConnection(_WorkerConnection):
    def send_bytes(self, buffer: bytes) -> None:
        del buffer

    def wait_ready(self, *, timeout: float) -> None:
        del timeout

    def poll(self, timeout: float = 0.0) -> bool:
        del timeout
        return False

    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        del maxlength
        return b""

    def close(self) -> None:
        return


class _CleanInterpreterProcess(_WorkerProcess):
    def __init__(self, *, parent: _ParentStdioConnection) -> None:
        self._parent = parent
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        process = subprocess.Popen(
            (sys.executable, "-m", "bot.behavior.source"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._process = process
        self._parent.bind(process=process)

    def is_alive(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def terminate(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def kill(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()

    def join(self, timeout: float | None = None) -> None:
        process = self._process
        if process is None:
            return
        try:
            if timeout is None:
                _ = process.wait()
            else:
                _ = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return


class _CleanInterpreterContext(_WorkerContext):
    def __init__(self) -> None:
        self._parent: _ParentStdioConnection | None = None

    def Pipe(self, *, duplex: bool) -> tuple[_WorkerConnection, _WorkerConnection]:
        del duplex
        parent = _ParentStdioConnection()
        self._parent = parent
        return parent, _ChildStdioConnection()

    def Process(
        self,
        *,
        target: typing.Callable[..., object],
        args: tuple[object, ...],
    ) -> _WorkerProcess:
        del target, args
        parent = self._parent
        if parent is None:
            raise SourceError("isolated Starlark source evaluation failed before returning a result.")
        return _CleanInterpreterProcess(parent=parent)


def _parse_source_isolated(program: str) -> Source:
    completed = threading.Event()
    result: list[Source] = []
    errors: list[BaseException] = []

    def evaluate() -> None:
        try:
            result.append(_parse_source_process(program))
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()

    supervisor = threading.Thread(
        target=span.bind(evaluate),
        name="bot-behavior-source-evaluation",
        daemon=True,
    )
    supervisor.start()
    isolation_seconds = SOURCE_WORKER_READY_SECONDS + SOURCE_EVALUATION_SECONDS
    _ = completed.wait(timeout=isolation_seconds)
    if not completed.is_set():
        raise _evaluation_timeout()
    if errors:
        error = errors[0]
        if isinstance(error, SourceError):
            raise error
        raise SourceError("isolated Starlark source evaluation failed.") from error
    if not result:
        raise SourceError("isolated Starlark source evaluation returned no result.")
    return result[0]


def _parse_source_process(program: str) -> Source:
    context = _evaluation_context()
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(target=_source_worker_main, args=())
    started = False
    try:
        process.start()
        started = True
        child_connection.close()
        with span.operation(
            "bot.behavior.source.ready",
            scope="bot.behavior",
            component="behavior.source",
            stage="ready",
        ):
            parent_connection.wait_ready(timeout=SOURCE_WORKER_READY_SECONDS)
        deadline = time.monotonic() + SOURCE_EVALUATION_SECONDS
        with span.operation(
            "bot.behavior.source.evaluate",
            scope="bot.behavior",
            component="behavior.source",
            stage="evaluate",
        ):
            parent_connection.send_bytes(program.encode("utf-8"))
            if deadline - time.monotonic() <= SOURCE_EMPTY_POLL_SECONDS:
                raise _evaluation_timeout()
            response: bytes | None = None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= SOURCE_EMPTY_POLL_SECONDS:
                    raise _evaluation_timeout()
                if parent_connection.poll(min(SOURCE_PROCESS_POLL_SECONDS, remaining)):
                    response = parent_connection.recv_bytes(SOURCE_MAX_RESULT_BYTES)
                    break
                if not process.is_alive():
                    if parent_connection.poll(SOURCE_EMPTY_POLL_SECONDS):
                        response = parent_connection.recv_bytes(SOURCE_MAX_RESULT_BYTES)
                    break
            if response is None:
                raise SourceError("Starlark source evaluation exceeded its isolated resource budget.")
            try:
                payload_value: object = typing.cast(object, json.loads(response.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SourceError("isolated Starlark source evaluation returned an invalid result.") from error
            if not isinstance(payload_value, dict):
                raise SourceError("isolated Starlark source evaluation returned an invalid result.")
            payload = typing.cast(dict[str, object], payload_value)
            if payload.get("ok") is not True:
                message = payload.get("error")
                raise SourceError(
                    message if isinstance(message, str) else "isolated Starlark source evaluation failed."
                )
            wire_spec = payload.get("spec")
            if not isinstance(wire_spec, dict):
                raise SourceError("isolated Starlark source evaluation returned an invalid behavior spec.")
            try:
                spec = Source.model_validate(
                    {
                        **typing.cast(dict[str, object], wire_spec),
                        "model": _restore_model_tree(typing.cast(dict[str, object], wire_spec).get("model")),
                    }
                )
            except (pydantic.ValidationError, ValueError) as error:
                raise SourceError(str(error)) from error
            return spec.model_copy(update={"source": program})
    except SourceError:
        raise
    except Exception as error:
        raise SourceError("isolated Starlark source evaluation failed before returning a result.") from error
    finally:
        try:
            parent_connection.close()
        except OSError:
            pass
        try:
            child_connection.close()
        except OSError:
            pass
        teardown_deadline = time.monotonic() + SOURCE_WORKER_TEARDOWN_SECONDS
        if started and process.is_alive():
            process.terminate()
        if started:
            process.join(timeout=_remaining_seconds(teardown_deadline))
            if process.is_alive():
                process.kill()
                process.join(timeout=_remaining_seconds(teardown_deadline))


def _remaining_seconds(deadline: float) -> float:
    return max(SOURCE_EMPTY_POLL_SECONDS, deadline - time.monotonic())


def _evaluation_context() -> _WorkerContext:
    """Start an isolated clean interpreter so the evaluation budget starts after the worker is ready."""

    return _CleanInterpreterContext()


def parse_source(source: str) -> Source:
    """Evaluate Starlark behavior declaration source into a behavior spec.

    This is the only public evaluation entry for untrusted behavior source.
    Evaluation always runs in an isolated worker process; it never executes
    Starlark in the calling process. ``_parse_source_in_process`` is private
    to that worker.

    Caller obligations:
        ``source`` must be a Starlark behavior declaration that either assigns
        ``behavior = hsm.define(...)`` with global ``input_event`` and
        ``output_event`` declarations, or assigns
        ``behavior = behavior_program(...)``. The source must satisfy the
        module ``SOURCE_*`` budgets (byte, token, nesting, collection, JSON,
        declared-event, result, memory, evaluation, worker-ready, and
        teardown limits) before and during evaluation.

    Side effects:
        Each call starts one isolated worker process
        (``python -m bot.behavior.source``) and one supervisor thread. Both
        are torn down on success, failure, and timeout: pipes close, then the
        worker is terminated and, if still alive, killed within
        ``SOURCE_WORKER_TEARDOWN_SECONDS``.

    Failure:
        Raises ``SourceError`` for invalid source, Starlark or schema errors,
        worker evaluation failure, or an invalid worker payload (non-JSON,
        non-object, or missing spec). Ready or evaluation budget misses raise
        ``SourceError`` with ``failure_kind="timeout"``. Other ``SourceError``
        instances use ``failure_kind="source_error"``.

    Concurrency:
        Concurrent ``parse_source`` calls are independent and thread-safe:
        each call owns its own supervisor thread and worker process and does
        not share evaluation state with other calls.

    Returns:
        The validated ``Source`` spec, with ``source`` preserved as the
        caller-supplied program text.
    """

    _ = _source_tokens(source)
    return _parse_source_isolated(source)


def _parse_source_in_process(source: str) -> Source:
    """Evaluate source inside the isolated worker; callers use ``parse_source``."""

    normalized_source = normalize_source(source)
    try:
        runtime = Starlark(globals=starlark_globals())
        runtime.exec(normalized_source, filename="<bot-behavior>")
        result = typing.cast(object | None, runtime.get("behavior", None))
        if result is None:
            raise SourceError("Starlark behavior source must assign behavior = hsm.define(...).")
        spec = _behavior_spec_from_result(runtime, result)
        return spec.model_copy(update={"source": source})
    except StarlarkError as error:
        raise SourceError(str(error)) from error
    except pydantic.ValidationError as error:
        raise SourceError(str(error)) from error


def _behavior_spec_from_result(runtime: Starlark, result: object) -> Source:
    if not isinstance(result, dict):
        raise SourceError("Starlark behavior source must assign behavior = hsm.define(...) or behavior_program(...).")
    result_spec = typing.cast(ElementSpec, result)
    if result_spec.get("kind") == "behavior_program":
        return Source.model_validate(result_spec)
    if result_spec.get("kind") != "define":
        raise SourceError("Starlark behavior source must assign behavior = hsm.define(...) or behavior_program(...).")
    input_event = typing.cast(object | None, runtime.get("input_event", None))
    output_event = typing.cast(object | None, runtime.get("output_event", None))
    if input_event is None or output_event is None:
        raise SourceError("Starlark hsm.define behaviors must declare input_event and output_event globals.")
    name = result_spec.get("name")
    return Source.model_validate(
        {
            "kind": "behavior_program",
            "name": name,
            "input_event": input_event,
            "output_event": output_event,
            "model": result_spec,
            "triggers": _runtime_get(runtime, "triggers", ()),
            "description": _runtime_get(runtime, "description", ""),
            "examples": _runtime_get(runtime, "examples", ()),
        }
    )


def _runtime_get(runtime: Starlark, name: str, default: object) -> object:
    return typing.cast(object, runtime.get(name, default))


def _validate_model_tree(model: ElementSpec, *, model_name: str) -> None:
    _validate_element(model, path="model", depth=0)
    if model.get("kind") != "define":
        raise ValueError("behavior model must be declared with hsm.define.")
    state_paths = _collect_state_paths(model, model_path=f"/{model_name}", depth=0)
    target_paths = _collect_target_paths(model, depth=0)
    unknown_targets = tuple(sorted(path for path in target_paths if path not in state_paths))
    if unknown_targets:
        raise ValueError(f"behavior model targets unknown states: {', '.join(unknown_targets)}.")


def _validate_element(element: object, *, path: str, depth: int) -> None:
    if depth > SOURCE_MAX_NESTING:
        raise ValueError(f"{path} exceeds the {SOURCE_MAX_NESTING}-level model depth budget.")
    if not isinstance(element, dict):
        raise ValueError(f"{path} must be an hsm element.")
    element_spec = typing.cast(ElementSpec, element)
    kind = element_spec.get("kind")
    if kind not in _HSM_NAMESPACE_NAMES.difference({"dispatch", "event"}):
        raise ValueError(f"{path} has unsupported hsm element kind: {kind!r}.")
    _validate_element_keys(element_spec, kind=typing.cast(str, kind), path=path)
    if kind == "define":
        _validate_name(element_spec.get("name"), path=f"{path}.name", model=True)
        _validate_elements(element_spec.get("elements"), path=f"{path}.elements", depth=depth)
    elif kind == "state":
        _validate_name(element_spec.get("name"), path=f"{path}.name")
        _validate_elements(element_spec.get("elements"), path=f"{path}.elements", depth=depth)
    elif kind == "transition":
        name = element_spec.get("name")
        if name is not None:
            _validate_name(name, path=f"{path}.name")
        _validate_elements(element_spec.get("elements"), path=f"{path}.elements", depth=depth)
    elif kind == "initial":
        _validate_elements(element_spec.get("elements"), path=f"{path}.elements", depth=depth)
    elif kind in {"on", "defer"}:
        events = element_spec.get("events")
        if not isinstance(events, tuple) or not events:
            raise ValueError(f"{path}.events must contain at least one event.")
        event_refs = typing.cast(tuple[object, ...], events)
        for index, event_ref in enumerate(event_refs):
            _validate_event_ref(event_ref, path=f"{path}.events[{index}]")
    elif kind == "target":
        value = element_spec.get("path")
        if not isinstance(value, str) or len(value) > 512 or not value.startswith("/"):
            raise ValueError(f"{path}.path must be an absolute HSM target path.")
    elif kind == "guard":
        _validate_callback(element_spec.get("callback"), path=f"{path}.callback")
    elif kind in {"effect", "entry", "exit", "activity"}:
        callbacks = element_spec.get("callbacks")
        if not isinstance(callbacks, tuple) or not callbacks:
            raise ValueError(f"{path}.callbacks must contain at least one callback name.")
        callback_names = typing.cast(tuple[object, ...], callbacks)
        for index, callback in enumerate(callback_names):
            _validate_callback(callback, path=f"{path}.callbacks[{index}]")
    elif kind == "after":
        seconds = element_spec.get("seconds")
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            raise ValueError(f"{path}.seconds must be a positive number.")
    elif kind == "final":
        _validate_name(element_spec.get("name"), path=f"{path}.name")


def _validate_elements(value: object, *, path: str, depth: int) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{path} must be a tuple of hsm elements.")
    elements = typing.cast(tuple[object, ...], value)
    if len(elements) > SOURCE_MAX_COLLECTION_ITEMS:
        raise ValueError(f"{path} exceeds its collection item budget.")
    for index, element in enumerate(elements):
        _validate_element(element, path=f"{path}[{index}]", depth=depth + 1)


def _validate_name(value: object, *, path: str, model: bool = False) -> None:
    if not isinstance(value, str) or len(value) > 256:
        raise ValueError(f"{path} must be a string.")
    pattern = _MODEL_NAME_RE if model else _IDENTIFIER_RE
    expected = "PascalCase model name" if model else "lower-case underscore identifier"
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{path} must be a {expected}.")


def _validate_callback(value: object, *, path: str) -> None:
    if not isinstance(value, str) or len(value) > 256 or _CALLBACK_RE.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lower-case underscore callback name.")


def _validate_event_ref(value: object, *, path: str) -> None:
    if isinstance(value, str):
        raise ValueError(f"{path} must be an event spec with a JSON schema, not a string event name.")
    if isinstance(value, dict):
        _ = EventContract.model_validate(typing.cast(dict[str, object], value))
        return
    raise ValueError(f"{path} must be an event spec or event name.")


def _validate_element_keys(element: ElementSpec, *, kind: str, path: str) -> None:
    unexpected = tuple(sorted(str(key) for key in set(element).difference(_ELEMENT_KEYS[kind])))
    if unexpected:
        raise ValueError(f"{path} has unexpected hsm element fields: {', '.join(unexpected)}.")


def _add_event_spec(specs: dict[str, EventContract], event_spec: EventContract) -> None:
    existing = specs.get(event_spec.name)
    if existing == event_spec:
        return
    if existing is not None:
        raise ValueError(f"Behavior event names must be unique; {event_spec.name!r} is declared more than once.")
    if len(specs) >= SOURCE_MAX_DECLARED_EVENTS:
        raise ValueError(f"behavior declares more than {SOURCE_MAX_DECLARED_EVENTS} events.")
    specs[event_spec.name] = event_spec


def _collect_state_paths(element: ElementSpec, *, model_path: str, depth: int) -> set[str]:
    if depth > SOURCE_MAX_NESTING:
        raise ValueError(f"behavior model exceeds the {SOURCE_MAX_NESTING}-level model depth budget.")
    paths: set[str] = set()
    kind = element.get("kind")
    elements = typing.cast(tuple[object, ...], element.get("elements", ()))
    if kind == "define":
        for child in elements:
            if isinstance(child, dict):
                paths.update(
                    _collect_state_paths(typing.cast(ElementSpec, child), model_path=model_path, depth=depth + 1)
                )
        return paths
    if kind in {"state", "final"}:
        name = typing.cast(str, element["name"])
        state_path = f"{model_path}/{name}"
        paths.add(state_path)
        for child in elements:
            if isinstance(child, dict):
                paths.update(
                    _collect_state_paths(typing.cast(ElementSpec, child), model_path=state_path, depth=depth + 1)
                )
        return paths
    for child in elements:
        if isinstance(child, dict):
            paths.update(_collect_state_paths(typing.cast(ElementSpec, child), model_path=model_path, depth=depth + 1))
    return paths


def _collect_target_paths(element: ElementSpec, *, depth: int) -> set[str]:
    if depth > SOURCE_MAX_NESTING:
        raise ValueError(f"behavior model exceeds the {SOURCE_MAX_NESTING}-level model depth budget.")
    if element.get("kind") == "target":
        return {typing.cast(str, element["path"])}
    paths: set[str] = set()
    for child in typing.cast(tuple[object, ...], element.get("elements", ())):
        if isinstance(child, dict):
            paths.update(_collect_target_paths(typing.cast(ElementSpec, child), depth=depth + 1))
    return paths


def _collect_event_specs(element: object, *, depth: int = 0) -> tuple[EventContract, ...]:
    if depth > SOURCE_MAX_NESTING:
        raise ValueError(f"behavior model exceeds the {SOURCE_MAX_NESTING}-level model depth budget.")
    specs: list[EventContract] = []
    if not isinstance(element, dict):
        return ()
    element_spec = typing.cast(ElementSpec, element)
    kind = element_spec.get("kind")
    if kind in {"on", "defer"}:
        for event_ref in typing.cast(tuple[object, ...], element_spec.get("events", ())):
            if isinstance(event_ref, dict):
                if len(specs) >= SOURCE_MAX_DECLARED_EVENTS:
                    raise ValueError(f"behavior declares more than {SOURCE_MAX_DECLARED_EVENTS} events.")
                specs.append(EventContract.model_validate(typing.cast(dict[str, object], event_ref)))
    for child in typing.cast(tuple[object, ...], element_spec.get("elements", ())):
        specs.extend(_collect_event_specs(child, depth=depth + 1))
        if len(specs) > SOURCE_MAX_DECLARED_EVENTS:
            raise ValueError(f"behavior declares more than {SOURCE_MAX_DECLARED_EVENTS} events.")
    return tuple(specs)


__all__ = [
    "Source",
    "EventContract",
    "SourceError",
    "STARLARK_API",
    "normalize_source",
    "parse_source",
    "starlark_globals",
]


if __name__ == "__main__":
    _source_worker_main()

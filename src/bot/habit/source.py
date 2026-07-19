"""Starlark source contracts for habit HSM behaviors."""

from . import schema

import io
import re
import tokenize
import typing

import hsm
import pydantic
from starlark_go import Starlark
from starlark_go.errors import StarlarkError

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
    """Raised when Starlark habit source cannot be evaluated as a stateforward.bot habit declaration.

    When produced by ``check`` / ``start``, ``report`` carries structured diagnostics for fix loops.
    """

    report: object | None

    def __init__(self, message: str, *, report: object | None = None) -> None:
        super().__init__(message)
        self.report = report


class EventContract(pydantic.BaseModel):
    """HSM event contract declared by Starlark habit source."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        json_schema_extra={
            "description": (
                "Event contract declared by Starlark habit source. Cognition abilities write these contracts so "
                "the compiled habit exposes typed HSM input, output, and internal events instead of an opaque "
                "script boundary."
            ),
            "examples": [
                {
                    "name": "bot.habit.answer_greeting.input",
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
        description="Stable HSM event name for a compiled habit boundary or internal habit transition.",
        examples=["bot.habit.answer_greeting.input"],
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
        description="Example event payloads added to the event schema for LLM tool use.",
        examples=[[{"text": "hello"}]],
    )

    @pydantic.field_validator("payload_schema")
    @classmethod
    def validate_schema_object(cls, value: schema.JsonSchema) -> schema.JsonSchema:
        """Require source-authored schemas to stay JSON-object-shaped."""

        schema.validate_supported_json_schema(value)
        return dict(value)

    def hsm_event(self) -> hsm.Event[object]:
        """Build the HSM event declared by this source contract."""

        schema = dict(self.payload_schema)
        if self.description is not None:
            schema["description"] = self.description
        if self.examples:
            schema["examples"] = list(self.examples)
        return hsm.Event[object](name=self.name, schema=schema)


class Source(pydantic.BaseModel):
    """Parsed Starlark program for a habit (HSM-aligned source, not a live instance)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        json_schema_extra={
            "description": (
                "Starlark-authored HSM behavior specification for an executable habit. Reflection, reasoning, "
                "or another cognitive ability can write this source, and stateforward.bot lowers its model into a static "
                "stateforward.hsm topology with safe Starlark callback wrappers."
            ),
            "examples": [
                {
                    "kind": "habit_behavior",
                    "name": "AnswerGreeting",
                    "triggers": ["conversation.greeting.recognized"],
                    "input_event": {
                        "name": "bot.habit.answer_greeting.input",
                        "schema": {"type": "object"},
                    },
                    "output_event": {
                        "name": "bot.habit.answer_greeting.output",
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

    kind: typing.Literal["habit_behavior"] = pydantic.Field(
        description="SourceData kind for executable habit behavior.",
        examples=["habit_behavior"],
    )
    name: str = pydantic.Field(
        min_length=1,
        description="PascalCase HSM model name for the compiled habit.",
        examples=["AnswerGreeting"],
    )
    input_event: EventContract = pydantic.Field(
        description="InputData event contract accepted by the habit.",
    )
    output_event: EventContract = pydantic.Field(
        description="OutputData event contract emitted by the habit.",
    )
    model: ElementSpec = pydantic.Field(
        description=(
            "SourceData-authored HSM model tree. It mirrors lower-case hsm builders such as define, state, "
            "transition, on, guard, effect, entry, exit, activity, target, initial, defer, after, and final."
        ),
    )
    triggers: tuple[str, ...] = pydantic.Field(
        default=(),
        description="External event names that fast intuition may use to propose the habit.",
        examples=[["conversation.greeting.recognized"]],
    )
    description: str = pydantic.Field(
        default="",
        description="Human-readable habit description.",
        examples=["Answer a recurring greeting without asking reasoning to choose every step."],
    )
    examples: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Natural-language examples of when the habit should be proposed.",
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
            raise ValueError("habit model name must match the behavior name.")
        _ = self.declared_event_specs()
        return self

    def declared_event_specs(self, *, failed_event: hsm.Event[object] | None = None) -> dict[str, EventContract]:
        """Return event specs declared at the habit boundary or inside its model tree."""

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
    raise SourceError("dispatch is only available inside habit callbacks.")


def _habit_behavior_builder(**kwargs: object) -> dict[str, object]:
    return typing.cast(dict[str, object], Source.model_validate(kwargs).model_dump(by_alias=True))


def starlark_globals() -> dict[str, object]:
    """Return the safe global namespace used to evaluate habit Starlark source."""

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
        "habit_behavior": _habit_behavior_builder,
        "initial": _initial_builder,
        "on": _on_builder,
        "state": _state_builder,
        "target": _target_builder,
        "transition": _transition_builder,
    }


# LLM-facing authoring contract: API only — no complete habit programs to copy.
# Topology notes align with stateforward HSM dsl.md (Initial = target + optional effect).
STARLARK_API = """
Starlark habit API (author source from scratch; do not paste canned full programs).
Aligned with HSM DSL (snake_case aliases of Define/State/Initial/Transition/…).

Required program shape:
1) input_event = hsm.event(name=..., schema={...}, description=..., examples=[...])
2) output_event = hsm.event(name=..., schema={...}, description=..., examples=[...])
3) optional metadata globals:
   - triggers = ["external.event.name", ...]
   - description = "..."
   - examples = ["when this should run", ...]
4) top-level callback defs: def callback_name(event): ...
5) habit = hsm.define(
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
   Model name must be PascalCase and match the habit install name.

Topology (from HSM Initial / Transition rules; habit Starlark subset):
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
  Every emit stamps source = habit_id (hsm.id of the live habit) so devices/abilities
  can reply with target=event["source"] (or target=habit_id).
  Optional target is a live instance id; omit it to process the event on the habit.
  habit_id is also available as a callback global.
  Reply events processed by the habit must appear in the model (hsm.on(reply_event)).

Habit-only Starlark limits (not full HSM host DSL):
- Callbacks are string names to top-level def callback(event): ... (no raw function objects)
- No host-language operations, attributes, submachine models, history, choice, entry/exit points
- No direct ability bindings or ability-alias dispatch — events only
- Guards: def name(event): return bool; payload is event["data"]
- Event facade keys: name, data, id, source, target, kind
  (metadata is telemetry-only and is intentionally omitted from the Starlark event facade)
- Effects/activities must hsm.dispatch(declared_event, payload_dict, target=optional_id)
  and must not return a value (returning is a runtime error)
- To wait on a reply: transition on the reply event after dispatching with source stamped

JSON schema for event payloads (object-shaped only):
- Supported: type, properties, required, additionalProperties, items, enum, const,
  oneOf, anyOf, allOf, description, examples, minimum, maximum, minLength, maxLength, pattern
- Unsupported: $ref and other non-object host-language extensions

Event naming:
- Prefer stable dotted names such as bot.habit.<snake_model>.input / .output
- Public input/output names must be unique and must not collide with generated
  bot.habit.<snake_model>.failed

Invent input/output contracts, guards, and effects from the observed cognitive pattern
(this turn + prior_episodes / existing_habit). The source field is the program you write.

Habit terminal output (output_event payload) must be cognition event selection(s):
- Prefer a single object with keys: event (required), target?, data?, reason?
- Mirror the event names, targets, and data field names from prior episode / this-turn outputs
- Fill live values from event["data"] and envelope fields (event["id"], event["source"],
  event["target"]) at runtime; do not hardcode example ids. Never read event["metadata"] —
  it is not on the facade and must not carry coordination or selection IDs.
- Input contracts and guards must use fields present on the live habit input
- triggers should list the stimulus names that should propose this habit
""".strip()


def normalize_source(source: str) -> str:
    """Rewrite `hsm.foo(...)` calls to the flat builder names supported by this Starlark bridge."""

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
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


def parse_source(source: str) -> Source:
    """Evaluate Starlark habit declaration source into a behavior spec.

    SourceData may either assign `habit = hsm.define(...)` with global `input_event`
    and `output_event` declarations, or assign `habit = habit_behavior(...)`.
    """

    normalized_source = normalize_source(source)
    try:
        runtime = Starlark(globals=starlark_globals())
        runtime.exec(normalized_source, filename="<bot-habit>")
        result = typing.cast(object | None, runtime.get("habit", None))
        if result is None:
            result = typing.cast(object | None, runtime.get("behavior", None))
        if result is None:
            raise SourceError("Starlark habit source must assign habit = hsm.define(...).")
        spec = _behavior_spec_from_result(runtime, result)
        return spec.model_copy(update={"source": source})
    except StarlarkError as error:
        raise SourceError(str(error)) from error
    except pydantic.ValidationError as error:
        raise SourceError(str(error)) from error
    except tokenize.TokenError as error:
        raise SourceError(str(error)) from error


def _behavior_spec_from_result(runtime: Starlark, result: object) -> Source:
    if not isinstance(result, dict):
        raise SourceError("Starlark habit source must assign habit = hsm.define(...) or habit_behavior(...).")
    result_spec = typing.cast(ElementSpec, result)
    if result_spec.get("kind") == "habit_behavior":
        return Source.model_validate(result_spec)
    if result_spec.get("kind") != "define":
        raise SourceError("Starlark habit source must assign habit = hsm.define(...) or habit_behavior(...).")
    input_event = typing.cast(object | None, runtime.get("input_event", None))
    output_event = typing.cast(object | None, runtime.get("output_event", None))
    if input_event is None or output_event is None:
        raise SourceError("Starlark hsm.define habits must declare input_event and output_event globals.")
    name = result_spec.get("name")
    return Source.model_validate(
        {
            "kind": "habit_behavior",
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
    _validate_element(model, path="model")
    if model.get("kind") != "define":
        raise ValueError("habit model must be declared with hsm.define.")
    state_paths = _collect_state_paths(model, model_path=f"/{model_name}")
    target_paths = _collect_target_paths(model)
    unknown_targets = tuple(sorted(path for path in target_paths if path not in state_paths))
    if unknown_targets:
        raise ValueError(f"habit model targets unknown states: {', '.join(unknown_targets)}.")


def _validate_element(element: object, *, path: str) -> None:
    if not isinstance(element, dict):
        raise ValueError(f"{path} must be an hsm element.")
    element_spec = typing.cast(ElementSpec, element)
    kind = element_spec.get("kind")
    if kind not in _HSM_NAMESPACE_NAMES.difference({"dispatch", "event"}):
        raise ValueError(f"{path} has unsupported hsm element kind: {kind!r}.")
    _validate_element_keys(element_spec, kind=typing.cast(str, kind), path=path)
    if kind == "define":
        _validate_name(element_spec.get("name"), path=f"{path}.name", model=True)
        _validate_elements(element_spec.get("elements"), path=f"{path}.elements")
    elif kind == "state":
        _validate_name(element_spec.get("name"), path=f"{path}.name")
        _validate_elements(element_spec.get("elements"), path=f"{path}.elements")
    elif kind == "transition":
        name = element_spec.get("name")
        if name is not None:
            _validate_name(name, path=f"{path}.name")
        _validate_elements(element_spec.get("elements"), path=f"{path}.elements")
    elif kind == "initial":
        _validate_elements(element_spec.get("elements"), path=f"{path}.elements")
    elif kind in {"on", "defer"}:
        events = element_spec.get("events")
        if not isinstance(events, tuple) or not events:
            raise ValueError(f"{path}.events must contain at least one event.")
        event_refs = typing.cast(tuple[object, ...], events)
        for index, event_ref in enumerate(event_refs):
            _validate_event_ref(event_ref, path=f"{path}.events[{index}]")
    elif kind == "target":
        value = element_spec.get("path")
        if not isinstance(value, str) or not value.startswith("/"):
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


def _validate_elements(value: object, *, path: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{path} must be a tuple of hsm elements.")
    elements = typing.cast(tuple[object, ...], value)
    for index, element in enumerate(elements):
        _validate_element(element, path=f"{path}[{index}]")


def _validate_name(value: object, *, path: str, model: bool = False) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string.")
    pattern = _MODEL_NAME_RE if model else _IDENTIFIER_RE
    expected = "PascalCase model name" if model else "lower-case underscore identifier"
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{path} must be a {expected}.")


def _validate_callback(value: object, *, path: str) -> None:
    if not isinstance(value, str) or _CALLBACK_RE.fullmatch(value) is None:
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
        raise ValueError(f"Habit event names must be unique; {event_spec.name!r} is declared more than once.")
    specs[event_spec.name] = event_spec


def _collect_state_paths(element: ElementSpec, *, model_path: str) -> set[str]:
    paths: set[str] = set()
    kind = element.get("kind")
    elements = typing.cast(tuple[object, ...], element.get("elements", ()))
    if kind == "define":
        for child in elements:
            if isinstance(child, dict):
                paths.update(_collect_state_paths(typing.cast(ElementSpec, child), model_path=model_path))
        return paths
    if kind in {"state", "final"}:
        name = typing.cast(str, element["name"])
        state_path = f"{model_path}/{name}"
        paths.add(state_path)
        for child in elements:
            if isinstance(child, dict):
                paths.update(_collect_state_paths(typing.cast(ElementSpec, child), model_path=state_path))
        return paths
    for child in elements:
        if isinstance(child, dict):
            paths.update(_collect_state_paths(typing.cast(ElementSpec, child), model_path=model_path))
    return paths


def _collect_target_paths(element: ElementSpec) -> set[str]:
    if element.get("kind") == "target":
        return {typing.cast(str, element["path"])}
    paths: set[str] = set()
    for child in typing.cast(tuple[object, ...], element.get("elements", ())):
        if isinstance(child, dict):
            paths.update(_collect_target_paths(typing.cast(ElementSpec, child)))
    return paths


def _collect_event_specs(element: object) -> tuple[EventContract, ...]:
    specs: list[EventContract] = []
    if not isinstance(element, dict):
        return ()
    element_spec = typing.cast(ElementSpec, element)
    kind = element_spec.get("kind")
    if kind in {"on", "defer"}:
        for event_ref in typing.cast(tuple[object, ...], element_spec.get("events", ())):
            if isinstance(event_ref, dict):
                specs.append(EventContract.model_validate(typing.cast(dict[str, object], event_ref)))
    for child in typing.cast(tuple[object, ...], element_spec.get("elements", ())):
        specs.extend(_collect_event_specs(child))
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

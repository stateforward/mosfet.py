"""TypeSafe label-tier processor: intuition reflex selection over `system_one`.

The mapping is structural, not behavioral:

- **state** is the turn as the model may see it: the stimulus envelope (name, data) as a
  structured JSON document. System One evaluates structure, so the live event data rides
  as JSON instead of a serialized text template.
- The **event menu** is one Choice per turn: criteria are the offered event names (schema
  descriptions as labels) together with the docs-prescribed pass criterion ("add
  `other`/`none of the above` if coverage uncertain") that maps to the typed explicit
  decline (`Result.unhandled()`), which the host cascades to deliberative reasoning —
  not a handled-empty selection.
- **Payload fields are filled by selection over supplied options** — never authored. For
  each offered event's required payload keys, candidate values are derived mechanically
  from the turn's own evidence (the stimulus data mapping, then the schema's own enum
  options); each key gets its own Choice whose criteria are the candidate values
  themselves, so the answer label *is* the payload value and can never leave the menu
  ("Your code never has to recover a value from generated prose"). An event with a
  required key that has no grounded candidates is left unoffered — fail-closed at the
  menu, not at dispatch. Candidate questions ride the same request: the primitives docs
  batch independent questions and adding questions barely changes response time.
- **confidence** rides the SDK's ChoiceAnswer confidence normalized through the shared
  `processing.normalize_confidence`, so the host's escalate policy sees the same scale as
  every other provider. Multi-select is caller policy per the choice primitive: a
  composition may pass `dispatch_threshold`, in which case every offered event whose
  probability clears the threshold dispatches (the reserved pass criterion is excluded)
  and each winner's confidence is its own probability on the shared scale.

The client is an injected dependency (same role as `ChatClient` in `openai_compat`), opened
per turn as an async context so the SDK's transport lifecycle stays inside the call.
"""

from __future__ import annotations

from .client import AsyncSystemOneClient, SystemOneError
from . import _json as _json_adapter
from bot.abilities import processing
from bot.abilities.cognition import intuition as cognition_intuition

import collections.abc
import typing

import pydantic

_RESERVED_PASS_CRITERION = "__unhandled__"
_DEFAULT_PASS_WORDING = "None of these: leave the turn unhandled."
_SELECTION_INSTRUCTIONS = (
    "Choose the single criterion that answers this turn from the bot's own peripheral "
    "transducers: the criterion text names the situation. Choose the pass criterion when "
    "none offered fits."
)
_PAYLOAD_KEY_INSTRUCTIONS = (
    "Fill one payload field for the selected event: the criteria are the exact values the "
    "evidence offers. Pick the value the turn supports."
)


class ProcessingError(RuntimeError):
    """TypeSafe processor failures surfaced as one error type."""


def _event_wording(event: processing.Event[typing.Any]) -> str:
    schema = getattr(event, "schema", None)
    description = getattr(schema, "description", None)
    if isinstance(description, str) and description.strip():
        return description.strip().splitlines()[0]
    return event.name


def _single_enabler(
    event_name: str,
    actor_events: collections.abc.Mapping[str, collections.abc.Sequence[str]],
) -> str | None:
    enablers = actor_events.get(event_name)
    if enablers is not None and len(enablers) == 1:
        return next(iter(enablers))
    return None


def _stimulus_document(stimulus: object) -> dict[str, typing.Any]:
    """The stimulus envelope as structured state: name plus live event data."""

    name = getattr(stimulus, "name", None)
    if isinstance(name, str):
        data = getattr(stimulus, "data", None)
        return {"envelope": {"name": name}, "data": _data_document(data)}
    return {"data": _data_document(stimulus)}


def _data_document(data: object) -> typing.Any:
    return _json_adapter.jsonable(data)


def _payload_required_keys(event: processing.Event[typing.Any]) -> tuple[str, ...]:
    schema_base = getattr(event, "schema", None)
    if not isinstance(schema_base, type) or not issubclass(schema_base, pydantic.BaseModel):
        return ()
    return tuple(name for name, field in schema_base.model_fields.items() if field.is_required())


def _schema_enum_options(event: processing.Event[typing.Any], key: str) -> tuple[typing.Any, ...]:
    schema_base = getattr(event, "schema", None)
    if not isinstance(schema_base, type) or not issubclass(schema_base, pydantic.BaseModel):
        return ()
    field = schema_base.model_fields.get(key)
    if field is None:
        return ()
    adapter = pydantic.TypeAdapter(field.annotation)
    schema = typing.cast("dict[str, typing.Any]", adapter.json_schema())
    enum = schema.get("enum")
    if isinstance(enum, collections.abc.Sequence) and not isinstance(enum, str):
        return tuple(item for item in enum if item is not None)
    return ()


def _candidate_values(
    event: processing.Event[typing.Any],
    key: str,
    stimulus_data: typing.Any,
) -> tuple[typing.Any, ...]:
    """Candidate values for one payload key, derived mechanically from bot evidence.

    Sources, in order: the stimulus data mapping (the value the turn itself carries), then
    the schema's own enum options. Nothing is invented; nothing is recovered from prose.
    """

    candidates: list[typing.Any] = []
    if isinstance(stimulus_data, dict):
        value = stimulus_data.get(key)
        if value is not None:
            candidates.append(value)
    candidates.extend(_schema_enum_options(event, key))
    return tuple(candidates)


class PayloadGroundingError(ProcessingError):
    """A chosen event required a payload key with no grounded candidates."""


class Processor(processing.Processor):
    """System One label-tier selection: one Choice over the offered event menu."""

    _client_factory: collections.abc.Callable[[], AsyncSystemOneClient]
    _timeout: float | None
    _api_key: str | None
    _model: str | None
    _pass_wording: str
    _dispatch_threshold: float | None

    def __init__(
        self,
        *,
        client: AsyncSystemOneClient | None = None,
        client_factory: collections.abc.Callable[[], AsyncSystemOneClient] | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        pass_wording: str = _DEFAULT_PASS_WORDING,
        dispatch_threshold: float | None = None,
    ) -> None:
        if client_factory is not None:
            resolved_factory = client_factory
        elif client is not None:
            resolved = client

            def resolved_factory() -> AsyncSystemOneClient:
                return resolved
        elif api_key is not None or model is not None:
            key = api_key
            chosen_model = model

            def resolved_factory() -> AsyncSystemOneClient:
                return AsyncSystemOneClient(api_key=key, model=chosen_model, timeout=timeout)
        else:

            def resolved_factory() -> AsyncSystemOneClient:
                return AsyncSystemOneClient(timeout=timeout)

        if dispatch_threshold is not None and not 0.0 <= dispatch_threshold <= 1.0:
            raise ValueError("dispatch_threshold must be within [0.0, 1.0].")
        self._client_factory = resolved_factory
        self._timeout = timeout
        self._api_key = api_key
        self._model = model
        self._pass_wording = pass_wording
        self._dispatch_threshold = dispatch_threshold

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        stimulus_document = _stimulus_document(input.input)
        stimulus_data = stimulus_document.get("data")
        groundable: dict[str, processing.Event[typing.Any]] = {}
        for event in input.schemas:
            if not event.name or processing.is_deliberative_handoff_schema(getattr(event, "schema", None)):
                continue
            if not self._groundable(event, stimulus_data):
                continue
            groundable[event.name] = event
        if not groundable:
            return ()

        question_map: dict[str, typing.Any] = {}
        import typesafe_sdk

        event_criteria: dict[str, str] = {
            name: _event_wording(event) for name, event in groundable.items()
        }
        event_criteria[_RESERVED_PASS_CRITERION] = self._pass_wording
        question_map["selection"] = typesafe_sdk.Choice(
            instructions=input.instructions or _SELECTION_INSTRUCTIONS,
            criteria=event_criteria,
        )

        payload_key_specs: dict[str, tuple[str, str]] = {}
        for name, event in groundable.items():
            for key in _payload_required_keys(event):
                question_name = f"{name}::{key}"
                wording = _payload_key_wording(name, key)
                question_map[question_name] = typesafe_sdk.Choice(
                    instructions=wording + " " + _PAYLOAD_KEY_INSTRUCTIONS,
                    criteria={str(candidate): wording for candidate in self._payload_candidates(event, key, stimulus_data)},
                )
                payload_key_specs[question_name] = (name, key)

        state_document = {"turn": stimulus_document}
        try:
            async with self._client_factory() as client:
                response = await client.system_one(state=state_document, questions=question_map)
        except SystemOneError as error:
            raise ProcessingError(f"TypeSafe system_one call failed: {error}") from error

        chosen = response.choices["selection"].choice
        if chosen == _RESERVED_PASS_CRITERION:
            # Explicit unhandled, not a handled-empty selection: the pass criterion means
            # "none offered fits." Intuition's own envelope shape for that is
            # OutputData(result=None), which the host reads as unhandled and cascades to
            # deliberative reasoning. An empty selection tuple would lie — it is a handled
            # turn with no actions (consumed, never escalated).
            # Cast note: the intuition envelope is the runtime-declared shape for an
            # unhandled turn (_product_from_processor_output accepts OutputData), while the
            # abstract Processor protocol still declares Events only. The double cast marks
            # that protocol gap as intentional at this one boundary.
            envelope = cognition_intuition.OutputData(result=None, reason=self._pass_wording)
            return typing.cast("processing.Events", typing.cast(object, envelope))

        event = groundable.get(chosen)
        if event is None:
            raise ProcessingError(f"TypeSafe selected unknown criterion {chosen!r}.")

        data: dict[str, object] = {}
        for question_name, (name, key) in payload_key_specs.items():
            if name != chosen:
                continue
            answer = response.choices[question_name].choice
            expected_candidates = self._payload_candidates(event, key, stimulus_data)
            if answer not in expected_candidates:
                raise PayloadGroundingError(
                    f"TypeSafe selected unknown {key!r} value for {name}: {answer!r}."
                )
            data[key] = answer

        winner = response.choices["selection"]
        # The caller-side policy from the docs: the event menu's dispositions are the
        # distribution; a composition passes a dispatch threshold to fire several events
        # from one turn. Default is single-winner, the reflex default.
        anchors: list[str]
        if self._dispatch_threshold is None:
            anchors = [winner.choice]
        else:
            anchors = [
                criterion
                for criterion, prob in winner.probabilities.items()
                if prob >= self._dispatch_threshold and criterion != _RESERVED_PASS_CRITERION
            ]
        anchors = [item for item in anchors if item in groundable]
        confidences = {
            criterion: processing.normalize_confidence(winner.probabilities.get(criterion, 0.0))
            for criterion in anchors
        }
        selections_out: list[processing.SelectedEvent] = []
        for criterion in anchors:
            data: dict[str, object] = {}
            for question_name, (name, key) in payload_key_specs.items():
                if name != criterion:
                    continue
                answer = response.choices[question_name].choice
                expected_candidates = self._payload_candidates(groundable[criterion], key, stimulus_data)
                if answer not in expected_candidates:
                    raise PayloadGroundingError(
                        f"TypeSafe selected unknown {key!r} value for {name}: {answer!r}."
                    )
                data[key] = answer
            selections_out.append(
                processing.SelectedEvent(
                    event=criterion,
                    target=_single_enabler(criterion, input.actor_events),
                    data=data or None,
                    reason=None,
                    confidence=confidences[criterion],
                )
            )
        return tuple(selections_out)

    def _groundable(
        self,
        event: processing.Event[typing.Any],
        stimulus_data: typing.Any,
    ) -> bool:
        """True when every required payload key has mechanically derived candidates."""

        for key in _payload_required_keys(event):
            if not self._payload_candidates(event, key, stimulus_data):
                return False
        return True

    def _payload_candidates(
        self,
        event: processing.Event[typing.Any],
        key: str,
        stimulus_data: typing.Any,
    ) -> tuple[typing.Any, ...]:
        return _candidate_values(event, key, stimulus_data)


def _payload_key_wording(event_name: str, key: str) -> str:
    return f"Which value fills {key!r} for {event_name}?"


__all__ = [
    "PayloadGroundingError",
    "ProcessingError",
    "Processor",
]

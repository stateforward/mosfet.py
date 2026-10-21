"""Decision-model processor: reflex's decision-layer selection as one decision request.

`Processor` is a `processing.DecisionProcessor`: a decision model chooses among the offered
events with a calibrated probability for each, so it is a processor reflex's decision layer runs on.

The mapping is structural, not behavioral:

- **state** is the turn as the model may see it (D-R-50): core's model-facing projection of the
  stimulus (``cognition.event.model_facing_content``), the same text reasoning reads. Typed causal
  parents are left out and media renders as a placeholder, never as bytes or base64; what someone
  said or typed is kept whole (D-R-47). The state is JSON, so no image is attached to it.
- The **event menu** is one choice question per turn (``selection``): its criteria are the offered
  event names, each described by the first line of its event schema's description, together with a
  reserved pass criterion ("none of these") for each later tier the turn names (``hand_ons``; R-23):
  one to answer it in words now, with intuition, offered only when intuition is composed, and one to
  think it through first, with reasoning. Each maps to the typed explicit decline (an unhandled
  decision output naming its tier, ``hand_to``), which the host hands to that tier, never a
  handled-empty selection.
- **Payload fields are filled by selection over supplied options**, never authored. For each
  offered event's required payload keys, candidate values come mechanically from the turn's own
  evidence (the stimulus data mapping, then the schema's own enum options); each key gets its own
  choice question whose criteria are the candidate values, so the answer label *is* the payload
  value and can never leave the menu. An event with a required key that has no grounded candidate
  is left unoffered: fail-closed at the menu, before any request.
- **Only the chosen option is returned** (D-R-28): one single-choice question chooses one event,
  returned with the chosen probability as its confidence and its grounded payload. The other options'
  probabilities are not selections. The decision layer's confidence gate still applies to the one returned.
- **The request fits the model's context** (D-R-36): with ``max_context_tokens``, the processor owns a
  ``processing.ContextFit``; a request whose JSON body is over its budget (3 UTF-8 bytes a token until
  the server's counted input tokens teach it fewer, D-R-45) has the selection's
  instructions (the turn's situation, its world block then what was acted, followed by its stable
  instructions, the self block, and any repair feedback: context assembly) cut to their leading whole
  lines, down to the question's own default instructions. The state (the stimuli) and the menu are
  never cut, ranked or dropped; a request that still does not fit is sent, and the server's refusal is
  the decision layer's failure.
"""

from __future__ import annotations

import collections.abc
import time
import types
import typing

import httpx
import pydantic

import mosfet.telemetry
from mosfet import media
from mosfet.abilities import processing
from mosfet.abilities.cognition import event as cognition_event
from mosfet.abilities.cognition import decision as cognition_decision
from mosfet.abilities.language import text

from . import _json
from . import client

_PROVIDER: typing.Final = "decision"
_SELECTION: typing.Final = "selection"
# The reserved pass criteria, one per later tier a pass may hand the turn to (R-23).
_PASS_CRITERIA: typing.Final[dict[processing.HandOn, str]] = {
    "intuition": "__intuition__",
    "reasoning": "__reasoning__",
}
_DEFAULT_PASS_WORDING: typing.Final[collections.abc.Mapping[processing.HandOn, str]] = types.MappingProxyType(
    {
        "intuition": "None of these fits: answer it in words now, with intuition.",
        "reasoning": "None of these fits: it needs thought, so think it through first, with reasoning.",
    }
)
"""How each pass criterion says what passing to that tier does (D-R-42, R-23): never as leaving the turn unhandled."""
_SELECTION_INSTRUCTIONS: typing.Final = (
    "Choose the single criterion that answers this turn from the bot's own peripheral "
    "transducers: the criterion text names the situation. Choose the pass criterion when "
    "none offered fits."
)
_PAYLOAD_KEY_INSTRUCTIONS: typing.Final = (
    "Fill one payload field for the selected event: the criteria are the exact values the "
    "evidence offers. Pick the value the turn supports."
)


class ProcessingError(RuntimeError):
    """A decision could not be made: the server failed, or its answer does not fit the menu it was asked."""


class PayloadGroundingError(ProcessingError):
    """The chosen event's payload key was answered with a value that is not one of its grounded candidates."""


def _pydantic_schema(event: processing.Event[typing.Any]) -> type[pydantic.BaseModel] | None:
    schema = event.schema
    return schema if isinstance(schema, type) and issubclass(schema, pydantic.BaseModel) else None


def _event_wording(event: processing.Event[typing.Any]) -> str:
    """The first line of the event schema's JSON schema description (D-R-29), or the event's name."""

    schema = _pydantic_schema(event)
    description = None if schema is None else schema.model_json_schema().get("description")
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


def _payload_required_keys(event: processing.Event[typing.Any]) -> tuple[str, ...]:
    schema = _pydantic_schema(event)
    if schema is None:
        return ()
    return tuple(name for name, field in schema.model_fields.items() if field.is_required())


def _candidate_values(
    event: processing.Event[typing.Any],
    key: str,
    stimulus_data: typing.Any,
) -> dict[str, typing.Any]:
    """Candidate values for one payload key by label, derived mechanically from bot evidence.

    Sources, in order: the stimulus data mapping (the value the turn itself carries), then the
    schema's own enum options. Nothing is invented; nothing is recovered from prose.
    """

    candidates: list[typing.Any] = []
    if isinstance(stimulus_data, dict):
        value = typing.cast(dict[str, typing.Any], stimulus_data).get(key)
        if value is not None:
            candidates.append(value)
    schema = _pydantic_schema(event)
    field = None if schema is None else schema.model_fields.get(key)
    if field is not None and field.annotation is not None:
        enum = pydantic.TypeAdapter(field.annotation).json_schema().get("enum")
        if isinstance(enum, list):
            candidates.extend(item for item in typing.cast(list[typing.Any], enum) if item is not None)
    return {str(candidate): candidate for candidate in candidates}


def _payload_key_wording(event_name: str, key: str) -> str:
    return f"Which value fills {key!r} for {event_name}?"


def _body_bytes(request: client.Request) -> int:
    """The size of the JSON body :class:`client.Client` sends for ``request``."""

    return len(request.model_dump_json(exclude={"model"} if request.model is None else None).encode("utf-8"))


def _fitted(request: client.Request, *, budget_bytes: int) -> tuple[client.Request, processing.ContextFitOutcome]:
    """``request`` within ``budget_bytes`` of JSON body, cutting only the selection's instructions (D-R-36).

    ``budget_bytes`` is the processor's ``processing.ContextFit`` budget, measured against the JSON body
    :class:`client.Client` sends. Over it, the instructions keep as many leading whole lines as fit (the
    world block comes first), then the question's default instructions; when even those do not fit, the
    request goes with them and the outcome is ``overflow``. The state and every criterion are left as they are.
    """

    def fits(candidate: client.Request) -> bool:
        return _body_bytes(candidate) <= budget_bytes

    def instructed(instructions: str) -> client.Request:
        selection = request.questions[_SELECTION].model_copy(update={"instructions": instructions})
        return request.model_copy(update={"questions": {**request.questions, _SELECTION: selection}})

    if fits(request):
        return request, "fit"
    lines = (request.questions[_SELECTION].instructions or "").splitlines()
    best: client.Request | None = None
    low, high = 1, len(lines) - 1
    while low <= high:
        count = (low + high) // 2
        candidate = instructed("\n".join(lines[:count]))
        if fits(candidate):
            best, low = candidate, count + 1
        else:
            high = count - 1
    if best is not None:
        return best, "cut"
    default = instructed(_SELECTION_INSTRUCTIONS)
    return default, "cut" if fits(default) else "overflow"


class Processor(processing.DecisionProcessor):
    """A decision processor on the decision server at ``base_url``: one choice over the offered event menu.

    ``api_key`` is sent as a bearer only when given. ``model`` names the model a server that serves
    several should answer with. ``timeout`` (seconds) bounds each request; the cognition step
    timeout bounds the turn regardless. ``max_context_tokens`` is the decision model's context, which
    each request is fitted to by cutting the turn's instructions (D-R-36), through a
    ``processing.ContextFit`` that learns bytes per token from the input tokens each answer reports
    (D-R-45); None sends every request whole. ``pass_wording`` describes each pass criterion, by the tier it
    hands the turn to; a pass to intuition is offered only when the turn names intuition among its
    ``hand_ons``. ``transport`` replaces the network (tests).
    Construction opens nothing: each turn is one request.
    """

    _client: client.Client
    _model: str | None
    _fit: processing.ContextFit | None
    _pass_wording: dict[processing.HandOn, str]

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_context_tokens: int | None = None,
        pass_wording: collections.abc.Mapping[processing.HandOn, str] = _DEFAULT_PASS_WORDING,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Raises ``ValueError`` as :class:`client.Client` does for its arguments, for a
        ``max_context_tokens`` that is not positive, and for a ``pass_wording`` without a wording for each tier."""

        missing = [tier for tier in _PASS_CRITERIA if not pass_wording.get(tier)]
        if missing:
            raise ValueError(f"pass_wording needs a wording for each tier; it has none for {', '.join(missing)}.")
        self._fit = (
            None
            if max_context_tokens is None
            else processing.ContextFit(max_context_tokens=max_context_tokens, processor=_PROVIDER)
        )
        self._client = client.Client(base_url=base_url, api_key=api_key, timeout=timeout, transport=transport)
        self._model = model
        self._pass_wording = {tier: pass_wording[tier] for tier in _PASS_CRITERIA}

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        """Decide this turn's one event, or pass it on to the next tier.

        Raises :class:`ProcessingError` when the server fails (its :class:`client.Error` is the
        cause) or answers outside the menu, and :class:`PayloadGroundingError` when a payload
        value is not one of its candidates.
        """

        del drafts
        stimulus = input.input
        # Grounding reads the live stimulus data, media left out (D-R-50); the state is what the model may see.
        stimulus_data = _json.jsonable(
            typing.cast(processing.Event[typing.Any], stimulus).data
            if isinstance(stimulus, processing.Event)
            else stimulus
        )
        groundable: dict[str, processing.Event[typing.Any]] = {}
        candidates: dict[tuple[str, str], dict[str, typing.Any]] = {}
        for event in input.schemas:
            if not event.name or processing.is_deliberative_handoff_schema(event.schema):
                continue
            keyed = {key: _candidate_values(event, key, stimulus_data) for key in _payload_required_keys(event)}
            if not all(keyed.values()):
                continue
            groundable[event.name] = event
            candidates.update({(event.name, key): values for key, values in keyed.items()})
        if not groundable:
            return ()

        criteria: dict[str, str | None] = {name: _event_wording(event) for name, event in groundable.items()}
        # One pass per tier the turn may be handed to; a frame that names none hands a pass to reasoning (R-23).
        passes = {_PASS_CRITERIA[tier]: tier for tier in input.hand_ons or ("reasoning",)}
        criteria.update({criterion: self._pass_wording[tier] for criterion, tier in passes.items()})
        questions: dict[str, client.Choice] = {
            _SELECTION: client.Choice(
                # The turn's situation leads, where the world block always was; then the stable instructions,
                # then a repair's feedback. A fitted cut keeps leading whole lines (D-R-36).
                instructions="\n".join(
                    block for block in (input.situation, input.instructions, input.feedback) if block
                )
                or _SELECTION_INSTRUCTIONS,
                criteria=criteria,
            )
        }
        for (name, key), values in candidates.items():
            wording = _payload_key_wording(name, key)
            questions[f"{name}::{key}"] = client.Choice(
                instructions=f"{wording} {_PAYLOAD_KEY_INSTRUCTIONS}",
                criteria=dict.fromkeys(values, wording),
            )
        try:
            # Text-only capacity: the state is JSON, so an image is its placeholder and never an attached part.
            parts = cognition_event.model_facing_content(stimulus, capacity=media.ImageCapacity())
        except (TypeError, ValueError) as error:
            raise ProcessingError(f"The turn cannot be shown to the decision model: {error}") from error
        turn = "".join(part.text for part in parts if isinstance(part, text.TextPart))
        request = client.Request(state={"turn": turn}, questions=questions, model=self._model)
        if self._fit is not None:
            request, outcome = _fitted(request, budget_bytes=self._fit.budget_bytes())
            self._fit.record(outcome)

        started = time.monotonic()
        # D-R-31: the request is logged under the model that answered it, once the server named it; a failed
        # or cancelled request is still logged, under the model it asked for.
        model = self._model
        try:
            try:
                answered = await self._client.decide(request)
                read = None if answered.usage is None else answered.usage.input_tokens
                if self._fit is not None and read is not None:
                    self._fit.observe(_body_bytes(request), read)
                if answered.model is not None:
                    model = answered.model
            finally:
                mosfet.telemetry.record_generator_request(
                    provider=_PROVIDER,
                    model=model,
                    messages=[request.state],
                    tools={name: question.model_dump(mode="json") for name, question in request.questions.items()},
                )
        except client.Error as error:
            mosfet.telemetry.record_generator_response(
                provider=_PROVIDER,
                model=self._model,
                response=None,
                latency_s=time.monotonic() - started,
                error=error,
            )
            raise ProcessingError(f"The decision request failed: {error}") from error
        mosfet.telemetry.record_generator_response(
            provider=_PROVIDER,
            model=model,
            response=answered.model_dump(mode="json"),
            latency_s=time.monotonic() - started,
        )

        selection = answered.answers.get(_SELECTION)
        if selection is None:
            raise ProcessingError("The decision server answered no selection.")
        chosen = selection.choice
        tier = passes.get(chosen)
        if tier is not None:
            # Explicit unhandled, not a handled-empty selection: a pass criterion means "none offered fits",
            # and names the tier it hands the turn to. The decision layer's own envelope for that is
            # OutputData(result=None, hand_to=tier), which the host reads as unhandled and hands to that tier;
            # an empty tuple would be a handled turn with no actions. The abstract Processor protocol declares
            # Events only, so the double cast marks that runtime-declared shape as intentional at this one boundary.
            envelope = cognition_decision.OutputData(result=None, hand_to=tier, reason=self._pass_wording[tier])
            return typing.cast(processing.Events, typing.cast(object, envelope))
        event = groundable.get(chosen)
        if event is None:
            raise ProcessingError(f"The decision server chose an unknown criterion {chosen!r}.")

        data: dict[str, object] = {}
        for (name, key), values in candidates.items():
            if name != chosen:
                continue
            answer = answered.answers.get(f"{name}::{key}")
            if answer is None or answer.choice not in values:
                raise PayloadGroundingError(f"The decision server chose no grounded {key!r} value for {name}.")
            data[key] = values[answer.choice]
        return (
            processing.SelectedEvent(
                event=chosen,
                target=_single_enabler(chosen, input.actor_events),
                data=data or None,
                reason=None,
                confidence=processing.normalize_confidence(selection.probabilities.get(chosen, selection.confidence)),
            ),
        )


__all__ = [
    "PayloadGroundingError",
    "ProcessingError",
    "Processor",
]

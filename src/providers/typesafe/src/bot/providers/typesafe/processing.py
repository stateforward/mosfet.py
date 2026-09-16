"""TypeSafe label-tier processor: intuition reflex selection over `system_one`.

The mapping is structural, not behavioral:

- **state** is the turn's model-facing payload (`input.model_facing_payload()`), which is
  exactly the turn as a model may see it — stimulus with envelope, nothing added.
- **questions** are one Choice per call: criteria are the offered event names (schema
  description as each label's wording) plus one reserved pass criterion. The pass criterion
  is the modeled host affordance — an explicit no-op selection (e.g. cognition ignore) when
  the host enables one, otherwise the processor's own unhandled return that cascades the
  turn. Multi-select and payload authoring are out of this tier's contract: System One does
  not call tools and cannot write payload fields.
- **confidence** is the SDK's own ChoiceAnswer confidence, normalized through the shared
  `processing.normalize_confidence`, so the host's escalate policy sees the same scale as
  every other provider.
- **payloads stay None.** A chosen event whose schema requires model-authored payload keys
  is dispatched with `data=None`; the host's dispatch validation is the honest boundary —
  typed rejection, then the host's cascade. This provider never fabricates payload fields.

The client is an injected dependency (same role as `ChatClient` in `openai_compat`), opened
per turn as an async context so the SDK's transport lifecycle stays inside the call.
"""

from __future__ import annotations

from .client import AsyncSystemOneClient, SystemOneError
from bot.abilities import processing

import collections.abc
import typing

_RESERVED_PASS_CRITERION = "__unhandled__"
_DEFAULT_PASS_WORDING = "None of these: leave the turn unhandled."
_SELECTION_INSTRUCTIONS = (
    "Choose the single criterion that answers this turn from the bot's own peripheral "
    "transducers: the criterion text names the situation. Choose the pass criterion when "
    "none offered fits."
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


class Processor(processing.Processor):
    """System One label-tier selection: one Choice over the offered event menu."""

    _client_factory: collections.abc.Callable[[], AsyncSystemOneClient]
    _timeout: float | None
    _api_key: str | None
    _model: str | None
    _pass_wording: str

    def __init__(
        self,
        *,
        client: AsyncSystemOneClient | None = None,
        client_factory: collections.abc.Callable[[], AsyncSystemOneClient] | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        pass_wording: str = _DEFAULT_PASS_WORDING,
    ) -> None:
        if client_factory is not None:

            def resolved_factory() -> AsyncSystemOneClient:
                return client_factory()

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

        self._client_factory = resolved_factory
        self._timeout = timeout
        self._api_key = api_key
        self._model = model
        self._pass_wording = pass_wording

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        offered = tuple(
            event
            for event in input.schemas
            if event.name and not processing.is_deliberative_handoff_schema(getattr(event, "schema", None))
        )
        if not offered:
            return ()
        criteria: dict[str, str] = {event.name: _event_wording(event) for event in offered}
        criteria[_RESERVED_PASS_CRITERION] = self._pass_wording
        import typesafe_sdk

        question_map = {
            "selection": typesafe_sdk.Choice(
                instructions=input.instructions or _SELECTION_INSTRUCTIONS,
                criteria=criteria,
            ),
        }
        state = {"turn": input.model_facing_payload()}
        try:
            async with self._client_factory() as client:
                response = await client.system_one(state=state, questions=question_map)
        except SystemOneError as error:
            raise ProcessingError(f"TypeSafe system_one call failed: {error}") from error
        choice_answer = response.choices["selection"]
        chosen = choice_answer.choice
        if chosen == _RESERVED_PASS_CRITERION:
            return ()
        if not any(event.name == chosen for event in offered):
            raise ProcessingError(f"TypeSafe selected unknown criterion {chosen!r}.")
        target = _single_enabler(chosen, input.actor_events)
        confidence = processing.normalize_confidence(choice_answer.confidence)
        return (
            processing.SelectedEvent(
                event=chosen,
                target=target,
                data=None,
                reason=None,
                confidence=confidence,
            ),
        )


__all__ = [
    "ProcessingError",
    "Processor",
]

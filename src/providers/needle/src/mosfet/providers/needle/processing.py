"""Needle processor: the Intuition reflex tier as local grammar-constrained tool calling.

The mapping is structural, not behavioral:

- **tools** are the turn's offered events, one Needle tool per event: the event's
  model-facing payload schema is the tool's parameters and its schema description is the tool
  description. Needle tool names must be identifiers, so canonical event names are mapped to
  identifier-safe names and mapped back on return. Deliberative handoff events are not
  offered: the reflex tier's way of deferring is to call nothing.
- **text** is the turn exactly as a model may see it (``InputData.model_facing_payload``);
  ``instructions`` are the system prompt.
- **calls** become ``SelectedEvent`` values carrying the engine-authored arguments as payload
  data. The engine's grammar keeps arguments inside each schema; the host's dispatch
  validation remains the final boundary. Calls the engine flags as ungrounded
  (``validation.ungrounded``) are not returned — the engine's own evidence says those
  arguments were not in the turn.
- **confidence** is the engine's calibrated response confidence (0-1), normalized to the
  shared 0-100 scale and stamped on every returned call. Dispatch policy stays with
  Intuition's confidence gate.
- **no calls** (off-topic refusal, or every call withheld into ``suppressed_calls``) is the
  explicit unhandled envelope, which the host cascades to deliberative reasoning — never a
  handled-empty selection.
"""

from __future__ import annotations

import asyncio
import collections.abc
import re
import time
import typing

import hsm

import mosfet.telemetry
from mosfet.abilities import processing
from mosfet.abilities.cognition import intuition as cognition_intuition
from mosfet.telemetry import span

from . import engine as needle_engine

_SCOPE = "mosfet.providers.needle"
_COMPONENT = "needle.processing"
_DEFAULT_INSTRUCTIONS = (
    "Call the tools this turn requires from the bot's own perception. Call nothing when no tool fits."
)
_UNHANDLED_REASON = "Needle called no offered tool."


class ProcessingError(RuntimeError):
    """Needle processor failures surfaced as one error type."""


def _tool_name(event_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", event_name)


def _tool(event: hsm.Event[typing.Any], name: str) -> dict[str, object]:
    parameters: dict[str, object] = dict(processing.model_facing_event_json_schema(event))
    if not parameters:
        parameters = {"type": "object", "properties": {}}
    description = parameters.pop("description", None)
    if not isinstance(description, str) or not description.strip():
        description = event.name
    return {"name": name, "description": description.strip(), "parameters": parameters}


def _ungrounded_tools(response: needle_engine.Response) -> frozenset[str]:
    validation = response.get("validation")
    if not isinstance(validation, collections.abc.Mapping):
        return frozenset()
    flagged = typing.cast(collections.abc.Mapping[str, object], validation).get("ungrounded")
    if not isinstance(flagged, collections.abc.Sequence) or isinstance(flagged, str):
        return frozenset()
    return frozenset(str(item).partition(".")[0] for item in flagged)


class Processor(processing.Processor):
    """Local Needle 3 tool calling over the offered event menu."""

    _engine: needle_engine.Engine

    def __init__(self, *, engine: needle_engine.Engine | None = None) -> None:
        self._engine = engine if engine is not None else needle_engine.LocalEngine()

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        offered: dict[str, hsm.Event[typing.Any]] = {}
        for event in input.schemas:
            if not event.name or processing.is_deliberative_handoff_schema(getattr(event, "schema", None)):
                continue
            name = _tool_name(event.name)
            existing = offered.get(name)
            if existing is not None and existing.name != event.name:
                raise ProcessingError(f"Events {existing.name!r} and {event.name!r} map to one Needle tool {name!r}.")
            offered[name] = event
        if not offered:
            return ()

        tools = [_tool(event, name) for name, event in offered.items()]
        system = input.instructions or _DEFAULT_INSTRUCTIONS
        payload = input.model_facing_payload()
        mosfet.telemetry.record_generator_request(
            provider="needle",
            model=None,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": payload}],
            tools=tools,
        )
        started = time.monotonic()
        with span.operation(
            "bot.provider.needle.processing.complete",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="complete",
        ):
            try:
                response = await asyncio.to_thread(
                    self._engine.complete,
                    system=system,
                    tools=tools,
                    text=payload,
                )
            except needle_engine.EngineError as error:
                mosfet.telemetry.record_generator_response(
                    provider="needle",
                    model=None,
                    response=None,
                    latency_s=time.monotonic() - started,
                    error=error,
                )
                raise ProcessingError(str(error)) from error
        mosfet.telemetry.record_generator_response(
            provider="needle",
            model=None,
            response=response,
            latency_s=time.monotonic() - started,
        )

        raw_calls = response.get("function_calls")
        calls = raw_calls if isinstance(raw_calls, collections.abc.Sequence) and not isinstance(raw_calls, str) else ()
        ungrounded = _ungrounded_tools(response)
        confidence = processing.normalize_confidence(response.get("confidence"))
        selections: list[processing.SelectedEvent] = []
        for raw_call in calls:
            if not isinstance(raw_call, collections.abc.Mapping):
                raise ProcessingError("Needle returned a malformed function call.")
            call = typing.cast(collections.abc.Mapping[str, object], raw_call)
            name = call.get("name")
            event = offered.get(name) if isinstance(name, str) else None
            if event is None:
                raise ProcessingError(f"Needle called unoffered tool {name!r}.")
            if name in ungrounded:
                continue
            arguments = call.get("arguments")
            if arguments is not None and not isinstance(arguments, collections.abc.Mapping):
                raise ProcessingError(f"Needle returned non-object arguments for {event.name}.")
            data = dict(typing.cast(collections.abc.Mapping[str, object], arguments)) if arguments else None
            selections.append(processing.SelectedEvent(event=event.name, data=data, reason=None, confidence=confidence))

        if not selections:
            # Explicit unhandled, as in the TypeSafe label tier: Intuition reads
            # OutputData(result=None) as unhandled and cascades to reasoning. The abstract
            # Processor protocol declares Events only, so the envelope is cast at this boundary.
            envelope = cognition_intuition.OutputData(result=None, reason=_UNHANDLED_REASON)
            return typing.cast("processing.Events", typing.cast(object, envelope))
        return processing.fill_unique_selection_targets(tuple(selections), input.actor_events)


__all__ = ["ProcessingError", "Processor"]

"""Compiler that turns Starlark habit source into executable HSM behavior."""

from __future__ import annotations

import types
import typing

import hsm

from bot.abilities import ability

from . import behavior
from . import source


def build(program: str) -> behavior.Behavior:
    """Compile Starlark habit source into an executable HSM-backed behavior.

    Habits may only dispatch and process modeled events; they have no ability bindings.
    """

    return _build_program(program.strip())


def _build_program(program: str) -> behavior.Behavior:
    if not program:
        raise ValueError("Habit source must be non-empty starlark.")
    spec = source.parse_source(program)
    behavior.validate_event_names(spec)
    failed_event_name = behavior.generated_event_names(spec)[0]
    input_event = spec.input_event.hsm_event()
    output_event = spec.output_event.hsm_event()
    failed_event = hsm.Event[ability.FailureData](
        name=failed_event_name,
        schema=ability.FailureData,
    )
    model = behavior.define_model(
        spec,
        input_event=input_event,
        output_event=output_event,
        failed_event=failed_event,
    )
    namespace: dict[str, object] = {
        "__doc__": f"Habit behavior compiled from Starlark source for {spec.name}.",
        "__module__": "bot.habit",
        "input_event": input_event,
        "output_event": output_event,
        "failed_event": failed_event,
        "submodel": model,
    }
    behavior_type = types.new_class(
        f"{spec.name}Behavior",
        (behavior.Behavior,),
        {},
        lambda namespace_dict: namespace_dict.update(namespace),
    )
    return typing.cast(behavior.Behavior, behavior_type(spec=spec))


__all__ = ["build"]

"""Executable HSM behavior for compiled behaviors.

Behaviors compile to event-only HSM machines: Starlark callbacks may dispatch and process
declared events, never bound abilities.
"""

from . import source
from . import schema
from . import runtime
from bot.abilities import ability

import collections.abc
import dataclasses
import datetime
import re
import typing

import hsm

from bot import abilities

from bot.telemetry import observer


class Behavior(abilities.Ability[object, object]):
    """Executable behavior compiled from Starlark source into HSM-visible behavior."""

    input_event: typing.ClassVar[hsm.Event[object]]
    output_event: typing.ClassVar[hsm.Event[object]]
    failed_event: typing.ClassVar[hsm.Event[ability.FailureData]]
    _spec: source.Source
    _callback_runtime: runtime.CallbackRuntime
    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Behavior",
        hsm.initial(hsm.target("/Behavior/idle")),
        hsm.state("idle"),
        hsm.observe(observer),
    )

    @staticmethod
    def starlark_guard(
        callback: str,
    ) -> collections.abc.Callable[[hsm.Context, "Behavior", hsm.Event[object]], bool]:
        def guard(ctx: hsm.Context, instance: "Behavior", event: hsm.Event[object]) -> bool:
            return instance._callback_runtime.guard(callback, ctx, instance, event)

        guard.__name__ = f"_behavior_guard_{callback}"
        return guard

    @staticmethod
    def input_valid(ctx: hsm.Context, instance: "Behavior", event: hsm.Event[object]) -> bool:
        del ctx
        return schema.matches_json_schema(event.data, instance._spec.input_event.payload_schema)

    @staticmethod
    def input_invalid(ctx: hsm.Context, instance: "Behavior", event: hsm.Event[object]) -> bool:
        return not Behavior.input_valid(ctx, instance, event)

    @staticmethod
    def dispatch_input_schema_failure(
        ctx: hsm.Context,
        instance: "Behavior",
        event: hsm.Event[object],
    ) -> None:
        _dispatch_callback_failure(
            ctx, instance, event, f"{instance._spec.name} input does not match its input schema."
        )

    @staticmethod
    def starlark_effect(
        callback: str,
    ) -> collections.abc.Callable[[hsm.Context, "Behavior", hsm.Event[object]], None]:
        def effect(ctx: hsm.Context, instance: "Behavior", event: hsm.Event[object]) -> None:
            try:
                instance._callback_runtime.effect(callback, ctx, instance, event)
            except runtime.CallbackError as error:
                _dispatch_callback_failure(ctx, instance, event, str(error))

        effect.__name__ = f"_behavior_effect_{callback}"
        return effect

    @staticmethod
    def starlark_activity(
        callback: str,
    ) -> collections.abc.Callable[[hsm.Context, "Behavior", hsm.Event[object]], collections.abc.Awaitable[None]]:
        async def activity(ctx: hsm.Context, instance: "Behavior", event: hsm.Event[object]) -> None:
            try:
                await instance._callback_runtime.activity(callback, ctx, instance, event)
            except runtime.CallbackError as error:
                _dispatch_callback_failure(ctx, instance, event, str(error))

        activity.__name__ = f"_behavior_activity_{callback}"
        return activity

    def __init__(self, *, spec: source.Source) -> None:
        super().__init__()
        callback_runtime = runtime.CallbackRuntime(
            source=spec.source,
            declared_events=spec.declared_event_specs(
                failed_event=typing.cast(hsm.Event[object], self.failed_event)
            ),
        )
        self._spec = spec
        self._callback_runtime = callback_runtime


def _snake_case_model_name(name: str) -> str:
    with_boundaries = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return with_boundaries.lower()


def generated_event_names(spec: source.Source) -> tuple[str, ...]:
    """Return generated failure event names for a behavior spec."""

    event_prefix = f"bot.behavior.{_snake_case_model_name(spec.name)}"
    return (f"{event_prefix}.failed",)


def validate_event_names(spec: source.Source) -> None:
    """Reject public behavior event names that collide with each other or generated events."""

    source_names = (spec.input_event.name, spec.output_event.name)
    if len(set(source_names)) != len(source_names):
        raise ValueError("Behavior source event names must be unique.")
    generated_names = generated_event_names(spec)
    collisions = tuple(sorted(set(spec.declared_event_specs()).intersection(generated_names)))
    if collisions:
        raise ValueError(f"Behavior source event names cannot use generated behavior event names: {', '.join(collisions)}.")


def define_model(
    spec: source.Source,
    *,
    input_event: hsm.Event[object],
    output_event: hsm.Event[object],
    failed_event: hsm.Event[ability.FailureData],
) -> hsm.Model:
    """Lower the source-authored Starlark HSM model into a static HSM topology."""

    event_specs = spec.declared_event_specs(failed_event=typing.cast(hsm.Event[object], failed_event))
    event_objects: dict[str, hsm.Event[object] | str] = {
        name: eventspec.hsm_event() for name, eventspec in event_specs.items()
    }
    event_objects[input_event.name] = input_event
    event_objects[output_event.name] = output_event
    event_objects[failed_event.name] = typing.cast(hsm.Event[object], failed_event)
    input_validation_failure = hsm.transition(
        hsm.on(input_event),
        hsm.guard(Behavior.input_invalid),
        hsm.effect(Behavior.dispatch_input_schema_failure),
    )
    return typing.cast(
        hsm.Model,
        _lower_element(
            spec.model,
            event_objects=event_objects,
            input_event_name=input_event.name,
            generated_failed_event=typing.cast(hsm.Event[object], failed_event),
            generated_failure_elements=_root_initial_transition_elements(spec.model),
            extra_root_elements=(input_validation_failure,),
        ),
    )


def _lower_element(
    element: dict[str, object],
    *,
    event_objects: collections.abc.Mapping[str, hsm.Event[object] | str],
    input_event_name: str,
    generated_failed_event: hsm.Event[object] | None = None,
    generated_failure_elements: tuple[object, ...] | None = None,
    extra_root_elements: tuple[hsm.Element, ...] = (),
) -> hsm.Element:
    kind = typing.cast(str, element["kind"])
    if kind == "define":
        current_root_name = typing.cast(str, element["name"])
        children = tuple(
            _lower_element(
                typing.cast(dict[str, object], child),
                event_objects=event_objects,
                input_event_name=input_event_name,
            )
            for child in typing.cast(tuple[object, ...], element["elements"])
        )
        generated_failure = ()
        if generated_failed_event is not None and generated_failure_elements is not None:
            generated_failure = (
                hsm.transition(
                    hsm.on(generated_failed_event),
                    *(
                        _lower_element(
                            typing.cast(dict[str, object], child),
                            event_objects=event_objects,
                            input_event_name=input_event_name,
                        )
                        for child in generated_failure_elements
                    ),
                ),
            )
        return hsm.define(
            current_root_name,
            *extra_root_elements,
            *children,
            *generated_failure,
            hsm.observe(observer),
        )
    if kind == "state":
        return hsm.state(
            typing.cast(str, element["name"]),
            *(
                _lower_element(
                    typing.cast(dict[str, object], child),
                    event_objects=event_objects,
                    input_event_name=input_event_name,
                )
                for child in typing.cast(tuple[object, ...], element["elements"])
            ),
        )
    if kind == "transition":
        source_elements = typing.cast(tuple[object, ...], element["elements"])
        children = tuple(
            _lower_element(
                typing.cast(dict[str, object], child),
                event_objects=event_objects,
                input_event_name=input_event_name,
            )
            for child in source_elements
        )
        if _transition_handles_event(source_elements, input_event_name):
            children = (*children, hsm.guard(Behavior.input_valid))
        name = typing.cast(str | None, element.get("name"))
        if name is None:
            return hsm.transition(*children)
        return hsm.transition(name, *children)
    if kind == "initial":
        return hsm.initial(
            *(
                _lower_element(
                    typing.cast(dict[str, object], child),
                    event_objects=event_objects,
                    input_event_name=input_event_name,
                )
                for child in typing.cast(tuple[object, ...], element["elements"])
            )
        )
    if kind == "on":
        return hsm.on(*(_lower_event_ref(event, event_objects=event_objects) for event in _event_refs(element)))
    if kind == "defer":
        return hsm.defer(*(_lower_event_ref(event, event_objects=event_objects) for event in _event_refs(element)))
    if kind == "target":
        return hsm.target(typing.cast(str, element["path"]))
    if kind == "guard":
        return hsm.guard(Behavior.starlark_guard(typing.cast(str, element["callback"])))
    if kind == "effect":
        return hsm.effect(*(Behavior.starlark_effect(callback) for callback in _callback_names(element)))
    if kind == "entry":
        return hsm.entry(*(Behavior.starlark_effect(callback) for callback in _callback_names(element)))
    if kind == "exit":
        return hsm.exit(*(Behavior.starlark_effect(callback) for callback in _callback_names(element)))
    if kind == "activity":
        return hsm.activity(*(Behavior.starlark_activity(callback) for callback in _callback_names(element)))
    if kind == "after":
        return hsm.after(_static_after(typing.cast(float | int, element["seconds"])))
    if kind == "final":
        return hsm.final(typing.cast(str, element["name"]))
    raise ValueError(f"Unsupported behavior hsm element kind: {kind}.")


def _event_refs(element: dict[str, object]) -> tuple[object, ...]:
    return typing.cast(tuple[object, ...], element["events"])


def _transition_handles_event(elements: tuple[object, ...], event_name: str) -> bool:
    for child in elements:
        if not isinstance(child, dict):
            continue
        childspec = typing.cast(dict[str, object], child)
        if childspec.get("kind") != "on":
            continue
        for event_ref in _event_refs(childspec):
            if _event_ref_name(event_ref) == event_name:
                return True
    return False


def _event_ref_name(event_ref: object) -> str | None:
    if isinstance(event_ref, str):
        return event_ref
    if isinstance(event_ref, dict):
        event_ref_data = typing.cast(dict[str, object], event_ref)
        name = event_ref_data.get("name")
        if isinstance(name, str):
            return name
    return None


def _callback_names(element: dict[str, object]) -> tuple[str, ...]:
    return typing.cast(tuple[str, ...], element["callbacks"])


def _root_initial_transition_elements(model: dict[str, object]) -> tuple[object, ...]:
    for child in typing.cast(tuple[object, ...], model.get("elements", ())):
        if not isinstance(child, dict):
            continue
        childspec = typing.cast(dict[str, object], child)
        if childspec.get("kind") != "initial":
            continue
        return typing.cast(tuple[object, ...], childspec.get("elements", ()))
    raise ValueError("Behavior model must declare a root initial target for generated failure recovery.")


def _lower_event_ref(
    event: object,
    *,
    event_objects: collections.abc.Mapping[str, hsm.Event[object] | str],
) -> hsm.Event[object] | str:
    if isinstance(event, str):
        event_object = event_objects.get(event)
        if event_object is None:
            raise ValueError(f"Behavior event reference {event!r} is not declared with a schema.")
        return event_object
    eventspec = source.EventContract.model_validate(event)
    return event_objects.get(eventspec.name, eventspec.hsm_event())


def _dispatch_callback_failure(
    ctx: hsm.Context,
    instance: Behavior,
    event: hsm.Event[object],
    message: str,
) -> None:
    failure = ability.FailureData(message=message)
    failed_event = dataclasses.replace(
        instance.failed_event.with_data(failure),
        id=event.id or None,
        metadata=dict(event.metadata),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, failed_event)
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(failed_event))


def _static_after(
    seconds: float | int,
) -> collections.abc.Callable[[hsm.Context, Behavior, hsm.Event[object]], datetime.timedelta]:
    def after(ctx: hsm.Context, instance: Behavior, event: hsm.Event[object]) -> datetime.timedelta:
        del ctx, instance, event
        return datetime.timedelta(seconds=float(seconds))

    after.__name__ = f"_behavior_after_{seconds:g}_seconds"
    return after


__all__ = [
    "Behavior",
    "define_model",
    "generated_event_names",
    "validate_event_names",
]

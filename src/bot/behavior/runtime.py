"""Safe Starlark callback runtime for compiled behavior HSM models.

Behaviors may only dispatch and process declared events — never bound abilities.

Every event a behavior emits is stamped with ``source=hsm.id(behavior)`` so abilities,
devices, and other machines can dispatch replies back to that id. Optional
``target`` on ``dispatch`` delivers the event to a live instance in the world
scope; without a target, the event is processed on the behavior itself.
"""

from . import source
from . import schema
from bot.abilities import ability

import collections.abc
import dataclasses
import typing

import hsm
import pydantic
from starlark_go import Starlark
from starlark_go.errors import StarlarkError

from bot import abilities
from bot import lifecycle


class CallbackError(RuntimeError):
    """Raised when a Starlark behavior callback violates the safe callback contract."""


class _Instance(typing.Protocol):
    output_event: typing.ClassVar[hsm.Event[object]]
    failed_event: typing.ClassVar[hsm.Event[ability.FailureData]]

    def context(self) -> hsm.Context: ...

    def get(self, name: str) -> tuple[object, bool]: ...

    def set(self, name: str, value: object) -> collections.abc.Awaitable[None]: ...

    def dispatch(
        self,
        ctx: hsm.Context,
        event: hsm.Event[typing.Any],
    ) -> collections.abc.Awaitable[None]: ...


@dataclasses.dataclass(frozen=True)
class _EventDispatch:
    eventspec: source.EventContract
    data: object
    target: str | None


@dataclasses.dataclass(frozen=True)
class _PreparedEventDispatch:
    eventspec: source.EventContract
    data: object
    target: str | None


@dataclasses.dataclass(frozen=True)
class _PreflightFailure:
    message: str


@dataclasses.dataclass
class _CallbackContext:
    ctx: hsm.Context
    instance: _Instance
    event: hsm.Event[object]
    dispatch_allowed: bool
    queue_dispatches: bool
    dispatches: list[_EventDispatch] = dataclasses.field(default_factory=list)


class _DispatchHost:
    _current: _CallbackContext | None
    _declared_events: collections.abc.Mapping[str, source.EventContract]

    def __init__(
        self,
        *,
        declared_events: collections.abc.Mapping[str, source.EventContract],
    ) -> None:
        self._current = None
        self._declared_events = dict(declared_events)

    def bind(self, context: _CallbackContext) -> None:
        self._current = context

    def unbind(self) -> None:
        self._current = None

    def dispatch(
        self,
        event: object,
        data: object | None = None,
        target: object | None = None,
    ) -> None:
        """Dispatch a declared HSM event from a Starlark effect or activity.

        Every emitted event is stamped with ``source=hsm.id(behavior)``. Pass
        ``target`` (an instance id string) to deliver to another live machine so
        it can reply with ``target=event.source``; omit target to process on the
        behavior itself.
        """

        current = self._current
        if current is None:
            raise CallbackError("dispatch is only available while a behavior callback is running.")
        if not current.dispatch_allowed:
            raise CallbackError("guards cannot dispatch events.")
        eventspec = self._eventspec(event)
        target_id = _optional_target_id(target)
        if current.queue_dispatches:
            current.dispatches.append(_EventDispatch(eventspec=eventspec, data=_json_like(data), target=target_id))
            return
        self._dispatch_event(current, eventspec, data, target=target_id)

    def dispatch_prepared_event(self, current: _CallbackContext, dispatch: _PreparedEventDispatch) -> None:
        """Dispatch an activity event after the activity batch has passed preflight."""

        self._dispatch_event(current, dispatch.eventspec, dispatch.data, target=dispatch.target)

    def dispatch_failure(self, current: _CallbackContext, message: str) -> None:
        """Reject the behavior operation when a callback cannot complete safely."""

        self._dispatch_callback_failure(current, message)

    def _dispatch_event(
        self,
        current: _CallbackContext,
        eventspec: source.EventContract,
        data: object,
        *,
        target: str | None,
    ) -> None:
        if not schema.matches_json_schema(data, eventspec.payload_schema):
            self._dispatch_callback_failure(
                current,
                f"{eventspec.name} payload does not match its event schema.",
            )
            return
        if eventspec.name == current.instance.output_event.name:
            if target is not None:
                self._dispatch_callback_failure(
                    current,
                    "behavior output events cannot set an external dispatch target.",
                )
                return
            self._dispatch_output(current, data)
            return
        if eventspec.name == current.instance.failed_event.name:
            if target is not None:
                self._dispatch_callback_failure(
                    current,
                    "behavior failure events cannot set an external dispatch target.",
                )
                return
            failure = ability.FailureData.model_validate(data)
            self._dispatch_failure(current, failure)
            return
        hsm_event = _behavior_event(
            eventspec.hsm_event().with_data(data),
            current=current,
            target=target,
        )
        if target is None:
            _ = hsm.dispatch(
                current.ctx,
                typing.cast(hsm.Dispatchable, current.instance),
                hsm_event,
            )
            return
        if not _target_is_live(current.ctx, target):
            self._dispatch_callback_failure(
                current,
                f"behavior dispatch target {target!r} is not a live instance.",
            )
            return
        _ = hsm.dispatch_to(current.ctx, hsm_event, target)

    def _eventspec(self, event: object) -> source.EventContract:
        if isinstance(event, str):
            eventspec = self._declared_events.get(event)
            if eventspec is None:
                raise CallbackError(f"behavior callback cannot dispatch undeclared event {event!r}.")
            return eventspec
        if isinstance(event, dict):
            # Full hsm.event(...) results are self-describing contracts (name + schema).
            # Prefer the model-declared contract when the name is already registered.
            try:
                contract = source.EventContract.model_validate(typing.cast(dict[str, object], event))
            except Exception as error:
                raise CallbackError("dispatch event specs must be valid event contracts.") from error
            declared = self._declared_events.get(contract.name)
            if declared is not None:
                return declared
            return contract
        raise CallbackError("dispatch requires a declared event name or event spec.")

    def _dispatch_output(self, current: _CallbackContext, data: object) -> None:
        output_event = _behavior_event(
            current.instance.output_event.with_data(data),
            current=current,
            target=None,
        )
        behavior = typing.cast(hsm.Dispatchable, current.instance)
        _ = hsm.dispatch(current.ctx, behavior, output_event)
        _ = hsm.dispatch(
            current.ctx,
            typing.cast(abilities.Ability[typing.Any, typing.Any], typing.cast(object, current.instance)),
            ability.TerminalOutputEvent.with_data(output_event),
        )

    def _dispatch_failure(self, current: _CallbackContext, failure: ability.FailureData) -> None:
        failed_event = _behavior_event(
            current.instance.failed_event.with_data(failure),
            current=current,
            target=None,
        )
        behavior = typing.cast(hsm.Dispatchable, current.instance)
        _ = hsm.dispatch(current.ctx, behavior, failed_event)
        _ = hsm.dispatch(
            current.ctx,
            typing.cast(abilities.Ability[typing.Any, typing.Any], typing.cast(object, current.instance)),
            ability.TerminalErrorEvent.with_data(failed_event),
        )

    def _dispatch_callback_failure(self, current: _CallbackContext, message: str) -> None:
        self._dispatch_failure(current, ability.FailureData(message=message))


def _starlark_runtime(program: str) -> Starlark:
    """Build the callback Starlark runtime (program param avoids shadowing ``source`` package)."""

    runtime = Starlark(globals=source.starlark_globals())
    runtime.exec(source.normalize_source(program), filename="<bot-behavior-callbacks>")
    return runtime


class CallbackRuntime:
    """Executes named Starlark callbacks through a narrow event and dispatch facade."""

    _runtime: Starlark
    _host: _DispatchHost

    def __init__(
        self,
        *,
        source: str,
        declared_events: collections.abc.Mapping[str, source.EventContract],
    ) -> None:
        self._host = _DispatchHost(declared_events=declared_events)
        self._runtime = _starlark_runtime(source)
        self._runtime.set(dispatch=self._host.dispatch)

    def guard(
        self,
        name: str,
        ctx: hsm.Context,
        instance: _Instance,
        event: hsm.Event[object],
    ) -> bool:
        """Run a named Starlark guard and require a boolean result."""

        result, dispatches, _ = self._call(
            name,
            ctx,
            instance,
            event,
            dispatch_allowed=False,
            queue_dispatches=False,
        )
        if dispatches:
            raise CallbackError("guards cannot dispatch.")
        if not isinstance(result, bool):
            raise CallbackError(f"behavior guard {name!r} must return a bool.")
        return result

    def effect(
        self,
        name: str,
        ctx: hsm.Context,
        instance: _Instance,
        event: hsm.Event[object],
    ) -> None:
        """Run a named Starlark effect (immediate event dispatch only)."""

        _ = self._call_behavior(name, ctx, instance, event, queue_dispatches=False)

    async def activity(
        self,
        name: str,
        ctx: hsm.Context,
        instance: _Instance,
        event: hsm.Event[object],
    ) -> None:
        """Run a named Starlark activity callback.

        The callback remains synchronous Starlark; dispatches are queued, preflighted,
        then applied so a batch fails together before any event side effects.
        """

        call_context, dispatches = self._call_behavior(
            name,
            ctx,
            instance,
            event,
            queue_dispatches=True,
        )
        prepared_dispatches = self._prepare_activity_dispatches(call_context, dispatches)
        if prepared_dispatches is None:
            return
        for dispatch in prepared_dispatches:
            self._host.dispatch_prepared_event(call_context, dispatch)

    def _call_behavior(
        self,
        name: str,
        ctx: hsm.Context,
        instance: _Instance,
        event: hsm.Event[object],
        *,
        queue_dispatches: bool,
    ) -> tuple[_CallbackContext, tuple[_EventDispatch, ...]]:
        result, dispatches, call_context = self._call(
            name,
            ctx,
            instance,
            event,
            dispatch_allowed=True,
            queue_dispatches=queue_dispatches,
        )
        if result is not None:
            raise CallbackError(f"behavior callback {name!r} must not return a value.")
        return call_context, dispatches

    def _call(
        self,
        name: str,
        ctx: hsm.Context,
        instance: _Instance,
        event: hsm.Event[object],
        *,
        dispatch_allowed: bool,
        queue_dispatches: bool,
    ) -> tuple[object, tuple[_EventDispatch, ...], _CallbackContext]:
        call_context = _CallbackContext(
            ctx=ctx,
            instance=instance,
            event=event,
            dispatch_allowed=dispatch_allowed,
            queue_dispatches=queue_dispatches,
        )
        self._host.bind(call_context)
        try:
            facade = _event_facade(event)
            self._runtime.set(__bot_event=facade, event=facade)
            # Live behavior id for directed replies: devices set target=this id.
            self._runtime.set(
                behavior_id=_behavior_id(instance),
            )
            result = typing.cast(
                object,
                self._runtime.eval(f"{name}(__bot_event)", filename="<bot-behavior-callback>"),
            )
            return result, tuple(call_context.dispatches), call_context
        except StarlarkError as error:
            raise CallbackError(str(error)) from error
        finally:
            self._host.unbind()

    def _prepare_activity_dispatches(
        self,
        current: _CallbackContext,
        dispatches: tuple[_EventDispatch, ...],
    ) -> tuple[_PreparedEventDispatch, ...] | None:
        prepared_dispatches: list[_PreparedEventDispatch] = []
        failure: _PreflightFailure | None = None
        for dispatch in dispatches:
            result = self._prepare_event_dispatch(dispatch)
            if isinstance(result, _PreflightFailure):
                if failure is None:
                    failure = result
                continue
            prepared_dispatches.append(result)
        if failure is not None:
            self._host.dispatch_failure(current, failure.message)
            return None
        return tuple(prepared_dispatches)

    def _prepare_event_dispatch(
        self,
        dispatch: _EventDispatch,
    ) -> _PreparedEventDispatch | _PreflightFailure:
        if not schema.matches_json_schema(dispatch.data, dispatch.eventspec.payload_schema):
            return _PreflightFailure(
                f"{dispatch.eventspec.name} payload does not match its event schema.",
            )
        return _PreparedEventDispatch(
            eventspec=dispatch.eventspec,
            data=dispatch.data,
            target=dispatch.target,
        )


def _optional_target_id(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    raise CallbackError("dispatch target must be a non-empty instance id string.")


def _behavior_id(instance: _Instance) -> str:
    return hsm.id(typing.cast(hsm.Instance, typing.cast(object, instance)))


def _target_is_live(ctx: hsm.Context, target: str) -> bool:
    """True when ``target`` resolves to a started instance that can accept HSM dispatch."""

    instances = ctx.value(hsm.Keys.Instances)
    if not isinstance(instances, collections.abc.Mapping):
        return False
    found = typing.cast(collections.abc.Mapping[object, object], instances).get(target)
    if not isinstance(found, hsm.Instance):
        return False
    # Map membership alone is not delivery liveness: a stopped instance can still be mapped.
    return lifecycle.is_started(found)


def _behavior_event(
    event: hsm.Event[typing.Any],
    *,
    current: _CallbackContext,
    target: str | None,
) -> hsm.Event[typing.Any]:
    """Stamp operation correlation and the behavior's addressable id as event source."""

    behavior_id = _behavior_id(current.instance)
    from bot.abilities import processing

    operation_id = (
        processing.active_operation_id(current.instance) if isinstance(current.instance, hsm.Instance) else None
    )
    return dataclasses.replace(
        event,
        id=operation_id or current.event.id or event.id,
        metadata={**current.event.metadata, **event.metadata},
        source=behavior_id,
        target=target or "",
    )


def _event_facade(event: hsm.Event[object]) -> dict[str, object]:
    """Expose modeled event fields to behavior without exposing telemetry metadata."""

    return {
        "name": event.name,
        "data": _json_like(event.data),
        "id": event.id,
        "source": event.source,
        "target": event.target,
        "kind": str(event.kind),
    }


def _json_like(value: object) -> object:
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_like(dataclasses.asdict(value))
    if isinstance(value, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        return {str(key): _json_like(item) for key, item in mapping.items()}
    if isinstance(value, tuple | list):
        sequence = typing.cast(collections.abc.Sequence[object], value)
        return [_json_like(item) for item in sequence]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


__all__ = [
    "CallbackError",
    "CallbackRuntime",
]

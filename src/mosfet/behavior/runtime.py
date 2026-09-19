"""Safe Starlark callback runtime for compiled behavior HSM models.

Behaviors may only dispatch and process declared events — never bound abilities.

Every event a behavior emits is stamped with ``source=hsm.id(behavior)`` so abilities,
devices, and other machines can dispatch replies back to that id. Optional
``target`` on ``dispatch`` delivers the event to a live instance in the environment
scope; without a target, the event is processed on the behavior itself.
"""

from . import source
from . import schema
from mosfet.abilities import ability

import collections.abc
import asyncio
import dataclasses
import json
import multiprocessing
import re
import threading
import time
import typing

import hsm
from mosfet import abilities
from mosfet import telemetry
from mosfet.telemetry import span
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from starlark_go import Starlark, configure_starlark

# Process-start serialization: the warm worker boots the shared forkserver; a callback
# worker that forks while that bootstrap is still running dies with the interpreter's
# "bootstrapping phase" failure, which surfaced as an E0008 apply timeout. All starts in
# this module hold one lock so bootstraps never race. Starts are cheap once the server
# exists, so the lock costs nothing after warm.
_FORK_LOCK = threading.Lock()

CALLBACK_EVALUATION_SECONDS = 2.0
CALLBACK_WARMUP_SECONDS = 10.0
CALLBACK_PROCESS_POLL_SECONDS = 0.05
CALLBACK_EMPTY_POLL_SECONDS = 0.0
CALLBACK_MAX_DISPATCHES = 32
CALLBACK_MAX_JSON_NODES = 8 * 1024
CALLBACK_MAX_JSON_DEPTH = 64
CALLBACK_MAX_COLLECTION_ITEMS = 1024
CALLBACK_MAX_JSON_STRING_BYTES = 64 * 1024
CALLBACK_MAX_RESPONSE_BYTES = 512 * 1024
CALLBACK_MAX_MEMORY_BYTES = 256 * 1024 * 1024
_CALLBACK_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_OTHER_CALLBACK_RESULT = object()


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
    ) -> collections.abc.Awaitable[bool]: ...


@dataclasses.dataclass(frozen=True)
class _EventDispatch:
    eventspec: source.EventContract
    data: object
    target: str | None


@dataclasses.dataclass(frozen=True)
class _RawEventDispatch:
    event: object
    data: object | None
    target: object | None


@dataclasses.dataclass(frozen=True)
class _PreparedEventDispatch:
    eventspec: source.EventContract
    data: object
    target: str | None


@dataclasses.dataclass(frozen=True)
class _PreflightFailure:
    message: str


@dataclasses.dataclass(frozen=True)
class _WorkerCallbackResult:
    result: object
    dispatches: tuple[_RawEventDispatch, ...]
    declared_events: tuple[source.EventContract, ...]


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
        # Normalize through the owning Data model's canonical JSON tree on every selection/dispatch hop.
        projected = _json_like({} if data is None else data)
        if current.queue_dispatches:
            current.dispatches.append(_EventDispatch(eventspec=eventspec, data=projected, target=target_id))
            return
        self._dispatch_event(current, eventspec, projected, target=target_id)

    def dispatch_prepared_event(self, current: _CallbackContext, dispatch: _PreparedEventDispatch) -> None:
        """Dispatch an activity event after the activity batch has passed preflight."""

        self._dispatch_event(current, dispatch.eventspec, dispatch.data, target=dispatch.target)

    def prepare_dispatch(
        self,
        dispatch: _RawEventDispatch,
        *,
        additional_events: collections.abc.Mapping[str, source.EventContract] | None = None,
    ) -> _EventDispatch:
        """Validate a worker dispatch against this behavior's declared contracts."""

        eventspec = self._eventspec(dispatch.event, additional_events=additional_events)
        target = _optional_target_id(dispatch.target)
        data = _json_like({} if dispatch.data is None else dispatch.data)
        return _EventDispatch(eventspec=eventspec, data=data, target=target)

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
        # Delivery is the recipient capability boundary. The HSM environment owns
        # recipient lookup and topology admission; this runtime must not probe or
        # duplicate that state before dispatching.
        _ = hsm.dispatch_to(current.ctx, hsm_event, target)

    def _eventspec(
        self,
        event: object,
        *,
        additional_events: collections.abc.Mapping[str, source.EventContract] | None = None,
    ) -> source.EventContract:
        if isinstance(event, str):
            eventspec = self._declared_events.get(event)
            if eventspec is None and additional_events is not None:
                eventspec = additional_events.get(event)
            if eventspec is None:
                raise CallbackError(f"behavior callback cannot dispatch undeclared event {event!r}.")
            return eventspec
        if isinstance(event, dict):
            # Full hsm.event(...) results are self-describing, but the behavior still
            # needs to declare the contract before it can emit it.
            try:
                contract = source.EventContract.model_validate(typing.cast(dict[str, object], event))
            except Exception as error:
                raise CallbackError("dispatch event specs must be valid event contracts.") from error
            declared = self._declared_events.get(contract.name)
            if declared is None and additional_events is not None:
                declared = additional_events.get(contract.name)
            if declared is None:
                raise CallbackError(f"behavior callback cannot dispatch undeclared event {contract.name!r}.")
            return declared
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


class _CallbackWorkerError(RuntimeError):
    """Internal error returned by an isolated Starlark callback worker."""


def _apply_callback_worker_limits() -> None:
    try:
        import resource
    except ImportError:
        return
    try:
        address_space = typing.cast(int | None, getattr(resource, "RLIMIT_AS", None))
        if address_space is not None:
            resource.setrlimit(address_space, (CALLBACK_MAX_MEMORY_BYTES, CALLBACK_MAX_MEMORY_BYTES))
    except (OSError, ValueError):
        pass
    try:
        cpu_limit = typing.cast(int | None, getattr(resource, "RLIMIT_CPU", None))
        if cpu_limit is not None:
            limit = max(1, int(CALLBACK_EVALUATION_SECONDS) + 1)
            resource.setrlimit(cpu_limit, (limit, limit))
    except (OSError, ValueError):
        pass


def _worker_json_value(value: object, *, path: str = "$") -> object:
    """Convert a callback value with finite collection, depth, and scalar budgets."""

    nodes = 0

    def visit(item: object, *, item_path: str, depth: int) -> object:
        nonlocal nodes
        if depth > CALLBACK_MAX_JSON_DEPTH:
            raise _CallbackWorkerError(f"callback value at {item_path} exceeds its depth budget.")
        nodes += 1
        if nodes > CALLBACK_MAX_JSON_NODES:
            raise _CallbackWorkerError(f"callback value at {item_path} exceeds its node budget.")
        if isinstance(item, str):
            if len(item.encode("utf-8")) > CALLBACK_MAX_JSON_STRING_BYTES:
                raise _CallbackWorkerError(f"callback string at {item_path} exceeds its byte budget.")
            return item
        if item is None or isinstance(item, bool | int | float):
            return item
        if isinstance(item, dict):
            mapping = typing.cast(dict[object, object], item)
            if len(mapping) > CALLBACK_MAX_COLLECTION_ITEMS:
                raise _CallbackWorkerError(f"callback object at {item_path} exceeds its item budget.")
            return {
                str(key): visit(child, item_path=f"{item_path}.{key}", depth=depth + 1)
                for key, child in mapping.items()
            }
        if isinstance(item, list | tuple):
            sequence = typing.cast(list[object] | tuple[object, ...], item)
            if len(sequence) > CALLBACK_MAX_COLLECTION_ITEMS:
                raise _CallbackWorkerError(f"callback array at {item_path} exceeds its item budget.")
            return [
                visit(child, item_path=f"{item_path}[{index}]", depth=depth + 1) for index, child in enumerate(sequence)
            ]
        raise _CallbackWorkerError(f"callback value at {item_path} has unsupported type {type(item).__name__}.")

    return visit(value, item_path=path, depth=0)


def _worker_declared_events(runtime: Starlark) -> tuple[source.EventContract, ...]:
    """Collect event contracts declared as top-level source globals for this run."""

    contracts: dict[str, source.EventContract] = {}
    global_names = runtime.globals()
    for name in global_names:
        try:
            candidate: object = typing.cast(object, runtime.get(name, None))
        except Exception:
            continue
        if not isinstance(candidate, dict):
            continue
        try:
            contract = source.EventContract.model_validate(typing.cast(dict[str, object], candidate))
        except Exception:
            continue
        existing = contracts.get(contract.name)
        if existing is not None and existing != contract:
            raise _CallbackWorkerError(f"behavior declares event {contract.name!r} with conflicting contracts.")
        if existing is None:
            if len(contracts) >= source.SOURCE_MAX_DECLARED_EVENTS:
                raise _CallbackWorkerError("behavior declares too many events.")
            contracts[contract.name] = contract
    return tuple(contracts.values())


def _callback_worker_payload(connection: Connection, payload: dict[str, object]) -> None:
    try:
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        encoded = json.dumps(
            {"ok": False, "error": f"callback result is not JSON-compatible: {error}"},
            separators=(",", ":"),
        ).encode("utf-8")
    if len(encoded) > CALLBACK_MAX_RESPONSE_BYTES:
        encoded = json.dumps(
            {"ok": False, "error": "callback result exceeds its wire-size budget."},
            separators=(",", ":"),
        ).encode("utf-8")
    try:
        connection.send_bytes(encoded)
    except (BrokenPipeError, OSError, ValueError):
        pass


def _callback_worker(connection: Connection) -> None:
    """Run exactly one callback in a killable process."""

    _apply_callback_worker_limits()
    try:
        request_value: object = typing.cast(
            object, json.loads(connection.recv_bytes(CALLBACK_MAX_RESPONSE_BYTES).decode("utf-8"))
        )
        if not isinstance(request_value, dict):
            raise _CallbackWorkerError("callback worker request is invalid.")
        request = typing.cast(dict[str, object], request_value)
        program = request.get("program")
        callback = request.get("callback")
        event = request.get("event")
        behavior_id = request.get("behavior_id")
        dispatch_allowed = request.get("dispatch_allowed")
        if (
            not isinstance(program, str)
            or len(program) > source.SOURCE_MAX_BYTES
            or not isinstance(callback, str)
            or _CALLBACK_RE.fullmatch(callback) is None
            or not isinstance(event, dict)
            or not isinstance(behavior_id, str)
            or not isinstance(dispatch_allowed, bool)
        ):
            raise _CallbackWorkerError("callback worker request is invalid.")
        configure_starlark(allow_recursion=False)
        runtime = Starlark(globals=source.starlark_globals())
        runtime.exec(source.normalize_source(program), filename="<bot-behavior-callbacks>")
        declared_events = _worker_declared_events(runtime)
        dispatches: list[dict[str, object]] = []

        def dispatch(event_spec: object, data: object | None = None, target: object | None = None) -> None:
            if not dispatch_allowed:
                raise _CallbackWorkerError("guards cannot dispatch events.")
            if len(dispatches) >= CALLBACK_MAX_DISPATCHES:
                raise _CallbackWorkerError("callback exceeds its dispatch count budget.")
            dispatches.append(
                {
                    "event": _worker_json_value(event_spec, path="$.event"),
                    "data": _worker_json_value({} if data is None else data, path="$.data"),
                    "target": _worker_json_value(target, path="$.target") if target is not None else None,
                }
            )

        runtime.set(
            dispatch=dispatch,
            __bot_event=event,
            event=event,
            behavior_id=behavior_id,
        )
        result: object = typing.cast(
            object, runtime.eval(f"{callback}(__bot_event)", filename="<bot-behavior-callback>")
        )
        result_kind = "bool" if isinstance(result, bool) else "none" if result is None else "other"
        _callback_worker_payload(
            connection,
            {
                "ok": True,
                "result": {"kind": result_kind, "value": result if isinstance(result, bool) else None},
                "dispatches": dispatches,
                "declared_events": [event.model_dump(mode="json", by_alias=True) for event in declared_events],
            },
        )
    except BaseException as error:
        _callback_worker_payload(connection, {"ok": False, "error": str(error) or type(error).__name__})


def _callback_warm_worker() -> None:
    """Start the isolated process server before latency-sensitive callback dispatch."""


def _run_callback_worker(request: dict[str, object]) -> _WorkerCallbackResult:
    deadline = time.monotonic() + CALLBACK_EVALUATION_SECONDS
    completed = threading.Event()
    result: list[_WorkerCallbackResult] = []
    errors: list[BaseException] = []

    def evaluate() -> None:
        try:
            result.append(_run_callback_process(request, deadline=deadline))
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()

    supervisor = threading.Thread(
        target=span.bind(evaluate),
        name="bot-behavior-callback-evaluation",
        daemon=True,
    )
    supervisor.start()
    _ = completed.wait(timeout=_remaining_seconds(deadline))
    if not completed.is_set():
        raise CallbackError(f"Starlark callback evaluation exceeded its {CALLBACK_EVALUATION_SECONDS:g}-second budget.")
    if errors:
        error = errors[0]
        if isinstance(error, CallbackError):
            raise error
        raise CallbackError("isolated Starlark callback evaluation failed.") from error
    if not result:
        raise CallbackError("isolated Starlark callback evaluation returned no result.")
    return result[0]


def _run_callback_process(request: dict[str, object], *, deadline: float) -> _WorkerCallbackResult:
    try:
        if _remaining_seconds(deadline) <= CALLBACK_EMPTY_POLL_SECONDS:
            raise CallbackError(
                f"Starlark callback evaluation exceeded its {CALLBACK_EVALUATION_SECONDS:g}-second budget."
            )
        encoded_request = json.dumps(request, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as error:
        raise CallbackError("callback worker request is not JSON-compatible.") from error
    if len(encoded_request) > CALLBACK_MAX_RESPONSE_BYTES:
        raise CallbackError("callback worker request exceeds its wire-size budget.")
    context: BaseContext = _evaluation_context()
    parent_connection, child_connection = typing.cast(tuple[Connection, Connection], context.Pipe(duplex=True))
    process_factory = typing.cast(typing.Callable[..., BaseProcess], getattr(context, "Process"))
    process = process_factory(target=_callback_worker, args=(child_connection,))
    started = False
    response: bytes | None = None
    timed_out = False
    try:
        with _FORK_LOCK:
            process.start()
        started = True
        child_connection.close()
        parent_connection.send_bytes(encoded_request)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if parent_connection.poll(min(CALLBACK_PROCESS_POLL_SECONDS, remaining)):
                response = parent_connection.recv_bytes(CALLBACK_MAX_RESPONSE_BYTES)
                break
            if not process.is_alive():
                if parent_connection.poll(CALLBACK_EMPTY_POLL_SECONDS):
                    response = parent_connection.recv_bytes(CALLBACK_MAX_RESPONSE_BYTES)
                break
    except Exception as error:
        raise CallbackError("isolated Starlark callback evaluation failed before returning a result.") from error
    finally:
        try:
            parent_connection.close()
        except OSError:
            pass
        try:
            child_connection.close()
        except OSError:
            pass
        if started and process.is_alive():
            process.terminate()
        if started:
            process.join(timeout=_remaining_seconds(deadline))
            if process.is_alive():
                process.kill()
                process.join(timeout=_remaining_seconds(deadline))

    if timed_out:
        raise CallbackError(f"Starlark callback evaluation exceeded its {CALLBACK_EVALUATION_SECONDS:g}-second budget.")
    if response is None:
        raise CallbackError("Starlark callback evaluation exceeded its isolated resource budget.")
    try:
        payload_value: object = typing.cast(object, json.loads(response.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CallbackError("isolated Starlark callback evaluation returned an invalid result.") from error
    if not isinstance(payload_value, dict):
        raise CallbackError("isolated Starlark callback evaluation returned an invalid result.")
    payload = typing.cast(dict[str, object], payload_value)
    if payload.get("ok") is not True:
        message = payload.get("error")
        raise CallbackError(message if isinstance(message, str) else "isolated Starlark callback evaluation failed.")
    result_payload = payload.get("result")
    if not isinstance(result_payload, dict):
        raise CallbackError("isolated Starlark callback evaluation returned an invalid result.")
    result_payload = typing.cast(dict[str, object], result_payload)
    result_kind = result_payload.get("kind")
    if result_kind == "bool":
        result: object = result_payload.get("value")
    elif result_kind == "none":
        result = None
    elif result_kind == "other":
        result = _OTHER_CALLBACK_RESULT
    else:
        raise CallbackError("isolated Starlark callback evaluation returned an invalid result kind.")
    raw_dispatches = payload.get("dispatches")
    if not isinstance(raw_dispatches, list):
        raise CallbackError("isolated Starlark callback evaluation returned an invalid dispatch batch.")
    raw_dispatches = typing.cast(list[object], raw_dispatches)
    if len(raw_dispatches) > CALLBACK_MAX_DISPATCHES:
        raise CallbackError("isolated Starlark callback evaluation returned an invalid dispatch batch.")
    raw_declared_events = payload.get("declared_events")
    if not isinstance(raw_declared_events, list):
        raise CallbackError("isolated Starlark callback evaluation returned invalid event declarations.")
    raw_declared_events = typing.cast(list[object], raw_declared_events)
    declared_events: list[source.EventContract] = []
    for raw_event in raw_declared_events:
        if not isinstance(raw_event, dict):
            raise CallbackError("isolated Starlark callback evaluation returned invalid event declarations.")
        try:
            declared_events.append(source.EventContract.model_validate(typing.cast(dict[str, object], raw_event)))
        except Exception as error:
            raise CallbackError("isolated Starlark callback evaluation returned invalid event declarations.") from error
    dispatches: list[_RawEventDispatch] = []
    for raw in raw_dispatches:
        if not isinstance(raw, dict) or "event" not in raw:
            raise CallbackError("isolated Starlark callback evaluation returned an invalid dispatch.")
        raw = typing.cast(dict[str, object], raw)
        dispatches.append(
            _RawEventDispatch(
                event=raw["event"],
                data=raw.get("data"),
                target=raw.get("target"),
            )
        )
    return _WorkerCallbackResult(
        result=result,
        dispatches=tuple(dispatches),
        declared_events=tuple(declared_events),
    )


def _remaining_seconds(deadline: float) -> float:
    return max(CALLBACK_EMPTY_POLL_SECONDS, deadline - time.monotonic())


def _warm_callback_process() -> None:
    deadline = time.monotonic() + CALLBACK_WARMUP_SECONDS
    context = _evaluation_context()
    process_factory = typing.cast(typing.Callable[..., BaseProcess], getattr(context, "Process"))
    process = process_factory(target=_callback_warm_worker)
    with _FORK_LOCK:
        process.start()
    process.join(timeout=_remaining_seconds(deadline))
    if process.is_alive():
        process.terminate()
        process.join(timeout=_remaining_seconds(deadline))
    if process.is_alive():
        process.kill()
        process.join(timeout=_remaining_seconds(deadline))
    if process.is_alive():
        raise CallbackError("isolated Starlark callback worker warmup exceeded its deadline.")


def _evaluation_context() -> BaseContext:
    """Use a low-startup isolated process where the platform supports it."""

    available = multiprocessing.get_all_start_methods()
    if "forkserver" in available:
        multiprocessing.set_forkserver_preload([f"{__package__}.source", f"{__package__}.runtime"])
        return multiprocessing.get_context("forkserver")
    for start_method in ("spawn", "fork"):
        if start_method in available:
            return multiprocessing.get_context(start_method)
    raise CallbackError("the platform provides no supported isolated process start method.")


class CallbackRuntime:
    """Executes named Starlark callbacks through a narrow event and dispatch facade."""

    _program: str
    _host: _DispatchHost

    def __init__(
        self,
        *,
        source: str,
        declared_events: collections.abc.Mapping[str, source.EventContract],
    ) -> None:
        self._host = _DispatchHost(declared_events=declared_events)
        self._program = source

    async def warm(self) -> None:
        """Warm the isolated process server without blocking the event loop."""

        await asyncio.to_thread(_warm_callback_process)

    async def evaluate_guard(
        self,
        *,
        callback: str,
        event: hsm.Event[object],
        behavior_id: str,
    ) -> bool:
        """Evaluate one guard off-loop and return its typed boolean outcome."""

        deadline = time.monotonic() + CALLBACK_EVALUATION_SECONDS
        try:
            worker_result = await asyncio.wait_for(
                asyncio.to_thread(
                    _run_callback_process,
                    {
                        "program": self._program,
                        "callback": callback,
                        "event": _event_facade(event),
                        "behavior_id": behavior_id,
                        "dispatch_allowed": False,
                    },
                    deadline=deadline,
                ),
                timeout=_remaining_seconds(deadline),
            )
        except TimeoutError as error:
            raise CallbackError(
                f"Starlark guard evaluation exceeded its {CALLBACK_EVALUATION_SECONDS:g}-second budget."
            ) from error
        if worker_result.dispatches:
            raise CallbackError("guards cannot dispatch.")
        if not isinstance(worker_result.result, bool):
            raise CallbackError(f"behavior guard {callback!r} must return a bool.")
        with span.operation(
            "bot.behavior.guard",
            scope="bot.behavior",
            component="behavior.callback",
            stage="guard",
            context=telemetry.event_context(event),
        ) as active:
            active.set_attribute("bot.guard.admitted", worker_result.result)
        return worker_result.result

    def lifecycle(
        self,
        name: str,
        ctx: hsm.Context,
        instance: _Instance,
        event: hsm.Event[object],
        *,
        stage: typing.Literal["entry", "exit"],
    ) -> None:
        """Run a named Starlark entry or exit callback to completion.

        HSM entry and exit are synchronous RTC actions. Isolation still happens in
        the callback process; this call returns only after that process finishes and
        any queued dispatches have been applied, so a later activity cannot observe
        a still-pending entry.
        """

        with span.operation(
            "bot.behavior.lifecycle",
            scope="bot.behavior",
            component="behavior.callback",
            stage=stage,
            context=telemetry.event_context(event),
        ):
            self._apply_isolated_callback(name, ctx, instance, event)

    async def effect(
        self,
        name: str,
        ctx: hsm.Context,
        instance: _Instance,
        event: hsm.Event[object],
    ) -> None:
        """Run a named Starlark effect off-loop, then apply its dispatches in order."""

        with span.operation(
            "bot.behavior.effect",
            scope="bot.behavior",
            component="behavior.callback",
            stage="effect",
            context=telemetry.event_context(event),
        ):
            call_context, dispatches = await asyncio.to_thread(
                self._call_behavior,
                name,
                ctx,
                instance,
                event,
                queue_dispatches=True,
            )
            self._dispatch_prepared_callback(call_context, dispatches)

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

        try:
            call_context, dispatches = await asyncio.wait_for(
                asyncio.to_thread(
                    self._call_behavior,
                    name,
                    ctx,
                    instance,
                    event,
                    queue_dispatches=True,
                ),
                timeout=CALLBACK_EVALUATION_SECONDS,
            )
        except TimeoutError as error:
            raise CallbackError(
                f"Starlark callback evaluation exceeded its {CALLBACK_EVALUATION_SECONDS:g}-second budget."
            ) from error
        self._dispatch_prepared_callback(call_context, dispatches)

    def _apply_isolated_callback(
        self,
        name: str,
        ctx: hsm.Context,
        instance: _Instance,
        event: hsm.Event[object],
    ) -> None:
        call_context, dispatches = self._call_behavior(
            name,
            ctx,
            instance,
            event,
            queue_dispatches=True,
        )
        self._dispatch_prepared_callback(call_context, dispatches)

    def _dispatch_prepared_callback(
        self,
        call_context: _CallbackContext,
        dispatches: tuple[_EventDispatch, ...],
    ) -> None:
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
        if not queue_dispatches:
            for dispatch in dispatches:
                self._host.dispatch_prepared_event(
                    call_context,
                    _PreparedEventDispatch(
                        eventspec=dispatch.eventspec,
                        data=dispatch.data,
                        target=dispatch.target,
                    ),
                )
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
            worker_result = _run_callback_worker(
                {
                    "program": self._program,
                    "callback": name,
                    "event": facade,
                    "behavior_id": _behavior_id(instance),
                    "dispatch_allowed": dispatch_allowed,
                }
            )
            additional_events = {event.name: event for event in worker_result.declared_events}
            dispatches = tuple(
                self._host.prepare_dispatch(dispatch, additional_events=additional_events)
                for dispatch in worker_result.dispatches
            )
            return worker_result.result, dispatches, call_context
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


def _behavior_event(
    event: hsm.Event[typing.Any],
    *,
    current: _CallbackContext,
    target: str | None,
) -> hsm.Event[typing.Any]:
    """Stamp operation correlation and the behavior's addressable id as event source."""

    behavior_id = _behavior_id(current.instance)
    from mosfet.abilities import processing

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
    """Expose the canonical JSON-compatible event/Data tree to Starlark."""

    from mosfet.event import event_json_value

    projected = event_json_value(value)
    return _check_json_budget(projected)


def _check_json_budget(value: object) -> object:
    nodes = 0

    def visit(item: object, *, depth: int) -> object:
        nonlocal nodes
        if depth > CALLBACK_MAX_JSON_DEPTH:
            raise CallbackError("event JSON value exceeds the callback depth budget.")
        nodes += 1
        if nodes > CALLBACK_MAX_JSON_NODES:
            raise CallbackError("event JSON value exceeds the callback node budget.")
        if isinstance(item, str):
            if len(item.encode("utf-8")) > CALLBACK_MAX_JSON_STRING_BYTES:
                raise CallbackError("event JSON string exceeds the callback byte budget.")
            return item
        if item is None or isinstance(item, bool | int | float):
            return item
        if isinstance(item, dict):
            mapping = typing.cast(dict[object, object], item)
            if len(mapping) > CALLBACK_MAX_COLLECTION_ITEMS:
                raise CallbackError("event JSON object exceeds the callback item budget.")
            return {str(key): visit(child, depth=depth + 1) for key, child in mapping.items()}
        if isinstance(item, list | tuple):
            sequence = typing.cast(list[object] | tuple[object, ...], item)
            if len(sequence) > CALLBACK_MAX_COLLECTION_ITEMS:
                raise CallbackError("event JSON array exceeds the callback item budget.")
            return [visit(child, depth=depth + 1) for child in sequence]
        return item

    return visit(value, depth=0)


__all__ = [
    "CallbackError",
    "CallbackRuntime",
]

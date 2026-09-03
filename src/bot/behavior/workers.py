"""Pooled forkserver workers for isolated Starlark behavior callback evaluation.

Learned-behavior callbacks are untrusted Starlark. This module owns the process
boundary that contains them: a fixed pool of long-lived forkserver workers, each
worker owned by a ``WorkerSlot`` machine, all coordinated by a ``CallbackWorkers``
pool machine. Evaluation keeps every guarantee of the one-shot spawn it replaces:

- Untrusted code never executes in the host process; each evaluation runs in a
  killable forkserver child reached over a JSON-only pipe.
- A timed-out evaluation ceases to exist: the slot kills its worker and forks a
  replacement rather than abandoning a hung task.
- Workers stay stateless between evaluations: every request rebuilds a fresh
  Starlark globals environment, so no learned program can observe another's
  residue.

The pool amortizes process creation across evaluations, retires workers after a
bounded number of evaluations, and caps concurrent isolated processes. Requests
beyond free capacity are deferred in machine topology and released FIFO when a
slot announces availability; deferral is a deliberate saturation-control choice,
kept bounded by the pool size and observable through the pool snapshot.
"""

import asyncio
import collections.abc
import dataclasses
import datetime
import json
import multiprocessing
import threading
import re
import time
import typing
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess

import pydantic
from starlark_go import Starlark, configure_starlark

import bot
import hsm

from . import runtime
from . import source

_CALLBACK_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

_OTHER_CALLBACK_RESULT = object()

_MAX_SPAWN_FAILURES = 2


@dataclasses.dataclass(frozen=True)
class CallbackWorkersSnapshot:
    """Point-in-time operational observation of one pool (diagnostics only)."""

    ready: bool
    free_indices: tuple[int, ...]
    settled_operations: int
    stale_results: int


_POOL_READY_STATES = frozenset(
    {
        "/BehaviorCallbackWorkers/ready/accepting",
        "/BehaviorCallbackWorkers/ready/parked",
    }
)


def _is_pool_ready_state(state: str) -> bool:
    """Explicit ready predicate for pool snapshots (never for coordination).

    Matches only the concrete ready substate paths, so a future state whose path merely
    contains ``ready`` as a substring can never read as ready.
    """

    return state in _POOL_READY_STATES


class SlotBudgets(pydantic.BaseModel):
    """Injected resource and lifecycle budgets for pooled callback workers.

    Tests inject small values so deadline paths run in milliseconds instead of
    wall-clock seconds; production defaults mirror the one-shot spawn budgets.

    :examples:

    >>> SlotBudgets(evaluation_seconds=0.25, max_evaluations=2)
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    evaluation_seconds: float = pydantic.Field(
        default=runtime.CALLBACK_EVALUATION_SECONDS,
        description=(
            "Wall-clock seconds one callback evaluation may run before the slot "
            "fails the request and kills its worker. Must be positive."
        ),
        examples=[2.0, 0.25],
    )
    poll_seconds: float = pydantic.Field(
        default=runtime.CALLBACK_PROCESS_POLL_SECONDS,
        description="Seconds between pipe liveness polls while awaiting a worker response.",
        examples=[0.05],
    )
    max_evaluations: int = pydantic.Field(
        default=200,
        description=(
            "Evaluations one worker process serves before the slot retires it "
            "between requests and forks a fresh replacement."
        ),
        examples=[200, 2],
    )
    warmup_seconds: float = pydantic.Field(
        default=runtime.CALLBACK_WARMUP_SECONDS,
        description="Seconds the pool may spend starting children and priming the forkserver before failing.",
        examples=[10.0],
    )
    shutdown_seconds: float = pydantic.Field(
        default=5.0,
        description="Seconds the pool waits for slot stop confirmations before reporting a failed shutdown.",
        examples=[5.0],
    )

    def kill_after_seconds(self) -> float:
        """Busy-state kill backstop: strictly beyond the transport deadline."""

        return self.evaluation_seconds * 1.5


@dataclasses.dataclass(frozen=True)
class RawDispatch:
    """One raw event dispatch requested by an evaluated callback."""

    event: object
    data: object | None
    target: object | None


@dataclasses.dataclass(frozen=True)
class WorkerEvaluationResult:
    """Decoded outcome of one isolated callback evaluation."""

    result: object
    dispatches: tuple[RawDispatch, ...]
    declared_events: tuple[source.EventContract, ...]


@dataclasses.dataclass(frozen=True)
class _RoundTripOutcome:
    kind: str
    value: bool | None
    dispatches: tuple[RawDispatch, ...]
    declared_events: tuple[source.EventContract, ...]


class _WorkerLoopError(RuntimeError):
    """Internal error raised inside the worker loop and reported over the pipe."""


# ---------------------------------------------------------------------------
# Worker-process side (runs inside the isolated child)
# ---------------------------------------------------------------------------


def apply_worker_limits() -> None:
    """Apply address-space limits to the current (worker) process where supported.

    Per-evaluation CPU is bounded authoritatively by the slot transport deadline and
    kill ladder; a cumulative RLIMIT_CPU would retire persistent workers early.
    """

    try:
        import resource
    except ImportError:
        return
    try:
        address_space = typing.cast(int | None, getattr(resource, "RLIMIT_AS", None))
        if address_space is not None:
            memory = runtime.CALLBACK_MAX_MEMORY_BYTES
            resource.setrlimit(address_space, (memory, memory))
    except (OSError, ValueError):
        pass


def bounded_json_value(value: object, *, path: str = "$") -> object:
    """Convert a callback value with finite collection, depth, and scalar budgets."""

    nodes = 0

    def visit(item: object, *, item_path: str, depth: int) -> object:
        nonlocal nodes
        if depth > runtime.CALLBACK_MAX_JSON_DEPTH:
            raise _WorkerLoopError(f"callback value at {item_path} exceeds its depth budget.")
        nodes += 1
        if nodes > runtime.CALLBACK_MAX_JSON_NODES:
            raise _WorkerLoopError(f"callback value at {item_path} exceeds its node budget.")
        if isinstance(item, str):
            if len(item.encode("utf-8")) > runtime.CALLBACK_MAX_JSON_STRING_BYTES:
                raise _WorkerLoopError(f"callback string at {item_path} exceeds its byte budget.")
            return item
        if item is None or isinstance(item, bool | int | float):
            return item
        if isinstance(item, dict):
            mapping = typing.cast(dict[object, object], item)
            if len(mapping) > runtime.CALLBACK_MAX_COLLECTION_ITEMS:
                raise _WorkerLoopError(f"callback object at {item_path} exceeds its item budget.")
            return {
                str(key): visit(child, item_path=f"{item_path}.{key}", depth=depth + 1)
                for key, child in mapping.items()
            }
        if isinstance(item, list | tuple):
            sequence = typing.cast(list[object] | tuple[object, ...], item)
            if len(sequence) > runtime.CALLBACK_MAX_COLLECTION_ITEMS:
                raise _WorkerLoopError(f"callback array at {item_path} exceeds its item budget.")
            return [
                visit(child, item_path=f"{item_path}[{index}]", depth=depth + 1) for index, child in enumerate(sequence)
            ]
        raise _WorkerLoopError(f"callback value at {item_path} has unsupported type {type(item).__name__}.")

    return visit(value, item_path=path, depth=0)


def _send_worker_payload(connection: Connection, payload: dict[str, object]) -> None:
    """Encode and send one response frame, dropping the frame on a broken pipe."""

    try:
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        encoded = json.dumps(
            {"ok": False, "error": f"callback result is not JSON-compatible: {error}"},
            separators=(",", ":"),
        ).encode("utf-8")
    if len(encoded) > runtime.CALLBACK_MAX_RESPONSE_BYTES:
        encoded = json.dumps(
            {"ok": False, "error": "callback result exceeds its wire-size budget."},
            separators=(",", ":"),
        ).encode("utf-8")
    try:
        connection.send_bytes(encoded)
    except (BrokenPipeError, OSError, ValueError):
        pass


def _declared_events(starlark: Starlark) -> tuple[source.EventContract, ...]:
    """Collect event contracts declared as top-level source globals for this run."""

    contracts: dict[str, source.EventContract] = {}
    for name in starlark.globals():
        try:
            candidate: object = typing.cast(object, starlark.get(name, None))
        except Exception:  # noqa: BLE001 - probe failures simply skip the global
            continue
        if not isinstance(candidate, dict):
            continue
        try:
            contract = source.EventContract.model_validate(typing.cast(dict[str, object], candidate))
        except Exception:  # noqa: BLE001 - non-contract globals are skipped
            continue
        existing = contracts.get(contract.name)
        if existing is not None and existing != contract:
            raise _WorkerLoopError(f"behavior declares event {contract.name!r} with conflicting contracts.")
        if existing is None:
            if len(contracts) >= source.SOURCE_MAX_DECLARED_EVENTS:
                raise _WorkerLoopError("behavior declares too many events.")
            contracts[contract.name] = contract
    return tuple(contracts.values())


def _evaluate_frame(request: dict[str, object]) -> dict[str, object]:
    """Evaluate one validated callback request in a fresh Starlark environment."""

    program = request.get("program")
    callback = request.get("callback")
    event = request.get("event")
    behavior_id = request.get("behavior_id")
    dispatch_allowed = request.get("dispatch_allowed")
    if (
        not isinstance(program, str)
        or len(program) > source.SOURCE_MAX_BYTES
        or not isinstance(callback, str)
        or _CALLBACK_NAME_RE.fullmatch(callback) is None
        or not isinstance(event, dict)
        or not isinstance(behavior_id, str)
        or not isinstance(dispatch_allowed, bool)
    ):
        raise _WorkerLoopError("callback worker request is invalid.")
    configure_starlark(allow_recursion=False)
    starlark = Starlark(globals=source.starlark_globals())
    starlark.exec(source.normalize_source(program), filename="<bot-behavior-callbacks>")
    declared = _declared_events(starlark)
    dispatches: list[dict[str, object]] = []

    def dispatch(event_spec: object, data: object | None = None, target: object | None = None) -> None:
        if not dispatch_allowed:
            raise _WorkerLoopError("guards cannot dispatch events.")
        if len(dispatches) >= runtime.CALLBACK_MAX_DISPATCHES:
            raise _WorkerLoopError("callback exceeds its dispatch count budget.")
        dispatches.append(
            {
                "event": bounded_json_value(event_spec, path="$.event"),
                "data": bounded_json_value({} if data is None else data, path="$.data"),
                "target": bounded_json_value(target, path="$.target") if target is not None else None,
            }
        )

    starlark.set(
        dispatch=dispatch,
        __bot_event=event,
        event=event,
        behavior_id=behavior_id,
    )
    result: object = typing.cast(object, starlark.eval(f"{callback}(__bot_event)", filename="<bot-behavior-callback>"))
    result_kind = "bool" if isinstance(result, bool) else "none" if result is None else "other"
    return {
        "ok": True,
        "result": {"kind": result_kind, "value": result if isinstance(result, bool) else None},
        "dispatches": dispatches,
        "declared_events": [contract.model_dump(mode="json", by_alias=True) for contract in declared],
    }


def worker_loop(connection: Connection) -> None:
    """Serve callback evaluations on one pipe until told to stop.

    This is the persistent forkserver child target. Each request evaluates in a
    fresh Starlark environment so a worker carries no state between programs.
    """

    apply_worker_limits()
    while True:
        try:
            raw = connection.recv_bytes(runtime.CALLBACK_MAX_RESPONSE_BYTES)
        except (EOFError, OSError):
            return
        try:
            frame_value: object = typing.cast(object, json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            _send_worker_payload(connection, {"ok": False, "error": "callback worker request is invalid."})
            continue
        if not isinstance(frame_value, dict):
            _send_worker_payload(connection, {"ok": False, "error": "callback worker request is invalid."})
            continue
        frame = typing.cast(dict[str, object], frame_value)
        if frame.get("stop") is True:
            return
        try:
            _send_worker_payload(connection, _evaluate_frame(frame))
        except _WorkerLoopError as error:
            _send_worker_payload(connection, {"ok": False, "error": str(error)})
        except BaseException as error:  # noqa: BLE001 - the child must always answer its pipe
            _send_worker_payload(connection, {"ok": False, "error": str(error) or type(error).__name__})


def evaluation_context() -> BaseContext:
    """Return the isolated low-startup multiprocess context used for workers."""

    available = multiprocessing.get_all_start_methods()
    if "forkserver" in available:
        multiprocessing.set_forkserver_preload(["bot.behavior.source", "bot.behavior.runtime", "bot.behavior.workers"])
        return multiprocessing.get_context("forkserver")
    for start_method in ("spawn", "fork"):
        if start_method in available:
            return multiprocessing.get_context(start_method)
    raise RuntimeError("no supported multiprocessing start method.")


def warm_worker_process() -> None:
    """Prime the forkserver once so later worker forks skip interpreter startup."""

    context = evaluation_context()
    process_factory = typing.cast(typing.Callable[..., BaseProcess], getattr(context, "Process"))
    process = process_factory(target=_warm_noop)
    process.start()
    grace = 1.0
    process.join(timeout=grace)
    if process.is_alive():
        process.terminate()
        process.join(timeout=grace)
    if process.is_alive():
        process.kill()
        process.join(timeout=grace)
    if process.is_alive():
        raise runtime.CallbackError("isolated Starlark callback worker warmup exceeded its grace period.")


def _warm_noop() -> None:
    """Forkserver priming payload: force server startup and module preload."""


def decode_worker_response(response: bytes) -> WorkerEvaluationResult:
    """Decode and validate one worker response frame into a typed result."""

    try:
        payload_value: object = typing.cast(object, json.loads(response.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise runtime.CallbackError("isolated Starlark callback evaluation returned an invalid result.") from error
    if not isinstance(payload_value, dict):
        raise runtime.CallbackError("isolated Starlark callback evaluation returned an invalid result.")
    payload = typing.cast(dict[str, object], payload_value)
    if payload.get("ok") is not True:
        message = payload.get("error")
        raise runtime.CallbackError(
            message if isinstance(message, str) else "isolated Starlark callback evaluation failed."
        )
    result_payload = payload.get("result")
    if not isinstance(result_payload, dict):
        raise runtime.CallbackError("isolated Starlark callback evaluation returned an invalid result.")
    result_payload = typing.cast(dict[str, object], result_payload)
    result_kind = result_payload.get("kind")
    if result_kind == "bool":
        result: object = result_payload.get("value")
    elif result_kind == "none":
        result = None
    elif result_kind == "other":
        result = _OTHER_CALLBACK_RESULT
    else:
        raise runtime.CallbackError("isolated Starlark callback evaluation returned an invalid result kind.")
    raw_dispatches = payload.get("dispatches")
    if not isinstance(raw_dispatches, list):
        raise runtime.CallbackError("isolated Starlark callback evaluation returned an invalid dispatch batch.")
    raw_dispatches = typing.cast(list[object], raw_dispatches)
    if len(raw_dispatches) > runtime.CALLBACK_MAX_DISPATCHES:
        raise runtime.CallbackError("isolated Starlark callback evaluation returned an invalid dispatch batch.")
    raw_declared = payload.get("declared_events")
    if not isinstance(raw_declared, list):
        raise runtime.CallbackError("isolated Starlark callback evaluation returned invalid event declarations.")
    declared_events: list[source.EventContract] = []
    for raw_event in typing.cast(list[object], raw_declared):
        if not isinstance(raw_event, dict):
            raise runtime.CallbackError("isolated Starlark callback evaluation returned invalid event declarations.")
        try:
            declared_events.append(source.EventContract.model_validate(typing.cast(dict[str, object], raw_event)))
        except Exception as error:
            raise runtime.CallbackError(
                "isolated Starlark callback evaluation returned invalid event declarations."
            ) from error
    dispatches: list[RawDispatch] = []
    for raw in raw_dispatches:
        if not isinstance(raw, dict) or "event" not in raw:
            raise runtime.CallbackError("isolated Starlark callback evaluation returned an invalid dispatch.")
        raw_dispatch = typing.cast(dict[str, object], raw)
        dispatches.append(
            RawDispatch(
                event=raw_dispatch["event"],
                data=typing.cast(object | None, raw_dispatch.get("data")),
                target=typing.cast(object | None, raw_dispatch.get("target")),
            )
        )
    return WorkerEvaluationResult(result=result, dispatches=tuple(dispatches), declared_events=tuple(declared_events))


# ---------------------------------------------------------------------------
# Parent-side blocking transport helpers (run in executor threads)
# ---------------------------------------------------------------------------


def _shutdown_sync(process: BaseProcess | None, connection: Connection | None, *, graceful: bool) -> None:
    """Tear down one worker: polite stop frame first, then terminate, then kill."""

    if graceful and connection is not None:
        try:
            connection.send_bytes(json.dumps({"stop": True}, separators=(",", ":")).encode("utf-8"))
        except (BrokenPipeError, OSError, ValueError):
            pass
    if process is not None and process.is_alive():
        process.join(timeout=0.25)
    if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
    if process is not None and process.is_alive():
        process.kill()
        process.join(timeout=0.5)
    if connection is not None:
        try:
            connection.close()
        except OSError:
            pass


def _round_trip_sync(
    connection: Connection,
    encoded_request: bytes,
    *,
    deadline: float,
    poll_seconds: float,
) -> WorkerEvaluationResult:
    """Blocking send/poll/recv/decode for one evaluation on a live worker pipe."""

    try:
        connection.send_bytes(encoded_request)
    except (BrokenPipeError, OSError, ValueError) as error:
        raise runtime.CallbackError(f"callback worker connection failed: {error}") from error
    response: bytes | None = None
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        if connection.poll(min(poll_seconds, remaining)):
            try:
                response = connection.recv_bytes(runtime.CALLBACK_MAX_RESPONSE_BYTES)
            except (EOFError, OSError) as error:
                raise runtime.CallbackError("callback worker connection failed.") from error
            break
    if timed_out:
        raise runtime.CallbackError("Starlark callback evaluation exceeded its budget.")
    if response is None:
        raise runtime.CallbackError("Starlark callback evaluation produced no response.")
    return decode_worker_response(response)


# ---------------------------------------------------------------------------
# Typed event contracts
# ---------------------------------------------------------------------------


class SlotEvaluationRequestData(pydantic.BaseModel):
    """One isolated callback evaluation request routed to a pooled worker slot.

    :examples:

    >>> SlotEvaluationRequestData(
    ...     program="behavior = None",
    ...     callback="always_true",
    ...     event={"name": "bot.behavior.x", "data": {}},
    ...     behavior_id="answer_greeting",
    ...     dispatch_allowed=False,
    ... )
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    program: str = pydantic.Field(
        description="Full Starlark behavior source owning the callback.",
        examples=["behavior = None"],
    )
    callback: str = pydantic.Field(
        description="Top-level function name in the program to evaluate with the event.",
        examples=["has_text"],
    )
    event: dict[str, object] = pydantic.Field(
        description="JSON facade of the triggering event exposed to the callback.",
        examples=[{"name": "conversation.greeting.recognized", "data": {"text": "hello"}}],
    )
    behavior_id: str = pydantic.Field(
        description="Identifier of the behavior instance being evaluated.",
        examples=["answer_greeting"],
    )
    dispatch_allowed: bool = pydantic.Field(
        description="Whether the callback may request event dispatches; guards run with this False.",
        examples=[False, True],
    )


class SlotEvaluationResultData(pydantic.BaseModel):
    """Settled outcome of one slot evaluation, correlated by operation id.

    :examples:

    >>> SlotEvaluationResultData(operation_id="callbackworkers:1", ok=True)
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    operation_id: str = pydantic.Field(
        description="Envelope id of the pool request this outcome settles.",
        examples=["callbackworkers:1"],
    )
    ok: bool = pydantic.Field(description="Whether the evaluation completed within budget.", examples=[True])
    result_kind: str | None = pydantic.Field(
        default=None,
        description="Kind of returned value: bool, none, or other.",
        examples=["bool", "none", "other"],
    )
    result_value: bool | None = pydantic.Field(
        default=None,
        description="Returned boolean when kind is bool; otherwise None.",
        examples=[True],
    )
    dispatches: tuple[dict[str, object], ...] = pydantic.Field(
        default=(),
        description=(
            "Raw event dispatch batches requested by the callback; each item carries "
            "event, data, and optional target JSON values validated against declared schemas upstream."
        ),
        examples=[({"event": {"name": "bot.behavior.answer.output"}, "data": {}, "target": None},)],
    )
    declared_events: tuple[dict[str, object], ...] = pydantic.Field(
        default=(),
        description="Event contracts declared by the evaluated program as JSON documents.",
        examples=[({"name": "bot.behavior.answer.output", "schema": {}},)],
    )
    error: str | None = pydantic.Field(
        default=None,
        description="Failure reason when ok is False.",
        examples=["Starlark callback evaluation exceeded its budget."],
    )


class SlotIndexData(pydantic.BaseModel):
    """Identifies one pooled slot by its fixed index.

    :examples:

    >>> SlotIndexData(index=0)
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    index: int = pydantic.Field(description="Stable index of the slot within its pool.", examples=[0])


class _WorkerSignalData(pydantic.BaseModel):
    """Typed empty signal for pooled-worker lifecycle transitions.

    Slot spawn/kill/retire/shutdown/stop and pool warm/stop transitions carry no payload;
    this type gives those signals a modeled schema instead of an untyped ``Event[None]``.
    Delivery still matches on the event name, so the schema change is wire-compatible.

    :examples:

    >>> _WorkerSignalData()
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Empty lifecycle signal for pooled Starlark callback workers; carries no payload.",
            "examples": [{}],
        },
    )


_SlotSpawnedEvent = hsm.Event[_WorkerSignalData](
    name="bot.behavior.workers.slot.spawned",
    schema=_WorkerSignalData,
)
_SlotSpawnFailedEvent = hsm.Event[SlotEvaluationResultData](
    name="bot.behavior.workers.slot.spawn_failed",
    schema=SlotEvaluationResultData,
)
_SlotEvaluateRequestEvent = hsm.Event[SlotEvaluationRequestData](
    name="bot.behavior.workers.slot.evaluate",
    schema=SlotEvaluationRequestData,
)
_SlotResultEvent = hsm.Event[SlotEvaluationResultData](
    name="bot.behavior.workers.slot.result",
    schema=SlotEvaluationResultData,
)
_SlotKilledEvent = hsm.Event[_WorkerSignalData](
    name="bot.behavior.workers.slot.killed",
    schema=_WorkerSignalData,
)
_SlotRetiredEvent = hsm.Event[_WorkerSignalData](
    name="bot.behavior.workers.slot.retired",
    schema=_WorkerSignalData,
)
_SlotShutdownCompleteEvent = hsm.Event[_WorkerSignalData](
    name="bot.behavior.workers.slot.shutdown_complete",
    schema=_WorkerSignalData,
)
_SlotStopRequestEvent = hsm.Event[_WorkerSignalData](
    name="bot.behavior.workers.slot.stop_request",
    schema=_WorkerSignalData,
)
_SlotAvailableEvent = hsm.Event[SlotIndexData](
    name="bot.behavior.workers.slot.available",
    schema=SlotIndexData,
)
_SlotStoppedEvent = hsm.Event[_WorkerSignalData](
    name="bot.behavior.workers.slot.stopped",
    schema=_WorkerSignalData,
)

_WarmedEvent = hsm.Event[_WorkerSignalData](
    name="bot.behavior.workers.warmed",
    schema=_WorkerSignalData,
)
_WarmFailedEvent = hsm.Event[SlotEvaluationResultData](
    name="bot.behavior.workers.warm_failed",
    schema=SlotEvaluationResultData,
)
_StopRequestEvent = hsm.Event[_WorkerSignalData](
    name="bot.behavior.workers.stop_request",
    schema=_WorkerSignalData,
)
_PoolEvaluateRequestedEvent = hsm.Event[SlotEvaluationRequestData](
    name="bot.behavior.workers.evaluate.request",
    schema=SlotEvaluationRequestData,
)


# ---------------------------------------------------------------------------
# WorkerSlot: one killable worker lifecycle
# ---------------------------------------------------------------------------


class WorkerSlot(hsm.Instance):
    """One pooled worker lifecycle: spawn, serve evaluations, kill, respawn.

    The slot owns exactly one forkserver worker process at a time. Only this
    class touches its own fields; peers coordinate through typed events.
    """

    model: hsm.Model

    def __init__(self, *, index: int, budgets: SlotBudgets, owner: "CallbackWorkers") -> None:
        super().__init__()
        self._index = index
        self._budgets = budgets
        self._owner = owner
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._active_operation: str | None = None
        self._evaluations_done = 0
        self._spawn_failures = 0
        self._spawn_done = threading.Event()
        self._spawn_done.set()
        self.model = WorkerSlot._build_model(budgets)

    def worker_pid(self) -> int | None:
        """Process id of the current isolated worker, for diagnostics and tests."""

        return self._process.pid if self._process is not None else None

    @staticmethod
    def _build_model(budgets: SlotBudgets) -> hsm.Model:
        """Build the slot model with deadlines closed over injected budgets."""

        kill_after = datetime.timedelta(seconds=budgets.kill_after_seconds())

        def kill_delay(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> datetime.timedelta:
            del ctx, instance, event
            return kill_after

        return bot.define(
            "BehaviorWorkerSlot",
            hsm.initial(hsm.target("/BehaviorWorkerSlot/spawning")),
            hsm.state(
                "spawning",
                hsm.activity(WorkerSlot._spawn_activity.__get__(None, object)),
                hsm.transition(hsm.on(_SlotSpawnedEvent), hsm.target("/BehaviorWorkerSlot/idle")),
                hsm.transition(
                    hsm.on(_SlotSpawnFailedEvent),
                    hsm.guard(WorkerSlot._spawn_retryable.__get__(None, object)),
                    hsm.effect(WorkerSlot._count_spawn_failure.__get__(None, object)),
                    hsm.target("/BehaviorWorkerSlot/spawning"),
                ),
                hsm.transition(hsm.on(_SlotSpawnFailedEvent), hsm.target("/BehaviorWorkerSlot/degraded")),
                hsm.transition(hsm.on(_SlotStopRequestEvent), hsm.target("/BehaviorWorkerSlot/stopping")),
            ),
            hsm.state(
                "idle",
                hsm.entry(WorkerSlot._announce_available.__get__(None, object)),
                hsm.transition(
                    hsm.on(_SlotEvaluateRequestEvent),
                    hsm.effect(WorkerSlot._begin_operation),
                    hsm.target("/BehaviorWorkerSlot/busy"),
                ),
                hsm.transition(hsm.on(_SlotStopRequestEvent), hsm.target("/BehaviorWorkerSlot/stopping")),
            ),
            hsm.state(
                "busy",
                hsm.activity(WorkerSlot._round_trip_activity.__get__(None, object)),
                hsm.defer(_SlotEvaluateRequestEvent),
                # Stop waits out the active evaluation (bounded by kill_after) and is
                # replayed on exit; direct busy->stopping here would orphan the worker.
                hsm.defer(_SlotStopRequestEvent),
                # Precedence (declaration order): retire-on-limit, else failure->kill,
                # else normal idle return. Same trigger; deterministic in this DSL.
                hsm.transition(
                    hsm.on(_SlotResultEvent),
                    hsm.guard(WorkerSlot._retire_after_result.__get__(None, object)),
                    hsm.effect(WorkerSlot._forward_result),
                    hsm.target("/BehaviorWorkerSlot/retiring"),
                ),
                hsm.transition(
                    hsm.on(_SlotResultEvent),
                    hsm.guard(WorkerSlot._failed_result.__get__(None, object)),
                    hsm.effect(WorkerSlot._forward_result),
                    hsm.target("/BehaviorWorkerSlot/killing"),
                ),
                hsm.transition(
                    hsm.on(_SlotResultEvent),
                    hsm.effect(WorkerSlot._forward_result),
                    hsm.target("/BehaviorWorkerSlot/idle"),
                ),
                hsm.transition(
                    hsm.after(kill_delay),
                    hsm.effect(WorkerSlot._note_kill_timeout),
                    hsm.target("/BehaviorWorkerSlot/killing"),
                ),
            ),
            hsm.state(
                "killing",
                hsm.activity(WorkerSlot._kill_activity.__get__(None, object)),
                hsm.transition(hsm.on(_SlotKilledEvent), hsm.target("/BehaviorWorkerSlot/spawning")),
                hsm.transition(hsm.on(_SlotStopRequestEvent), hsm.target("/BehaviorWorkerSlot/stopping")),
            ),
            hsm.state(
                "retiring",
                hsm.activity(WorkerSlot._retire_activity.__get__(None, object)),
                hsm.transition(hsm.on(_SlotRetiredEvent), hsm.target("/BehaviorWorkerSlot/spawning")),
                hsm.transition(hsm.on(_SlotStopRequestEvent), hsm.target("/BehaviorWorkerSlot/stopping")),
            ),
            hsm.state(
                "stopping",
                hsm.activity(WorkerSlot._shutdown_activity.__get__(None, object)),
                hsm.transition(
                    hsm.on(_SlotShutdownCompleteEvent),
                    hsm.effect(WorkerSlot._notify_stopped),
                    hsm.target("/BehaviorWorkerSlot/stopped"),
                ),
            ),
            hsm.final("stopped"),
            hsm.state(
                "degraded",
                hsm.transition(hsm.on(_SlotStopRequestEvent), hsm.target("/BehaviorWorkerSlot/stopping")),
            ),
        )

    # -- spawning ---------------------------------------------------------

    @staticmethod
    async def _spawn_activity(ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[typing.Any]) -> None:
        instance._spawn_done.clear()

        def fork() -> tuple[BaseProcess, Connection]:
            # The fork thread is the ONLY writer of _spawn_done: teardown waits on it,
            # so a premature set would let stop reap an unpublished fork.
            try:
                context = evaluation_context()
                parent_connection, child_connection = typing.cast(
                    tuple[Connection, Connection], context.Pipe(duplex=True)
                )
                process_factory = typing.cast(typing.Callable[..., BaseProcess], getattr(context, "Process"))
                process = process_factory(target=worker_loop, args=(child_connection,))
                try:
                    process.start()
                except BaseException:
                    parent_connection.close()
                    child_connection.close()
                    raise
                try:
                    child_connection.close()
                except BaseException:
                    _shutdown_sync(process, parent_connection, graceful=False)
                    raise
                # Publish eagerly so later cancellation/failure still leaves this
                # worker reachable by every teardown path.
                instance._process = process
                instance._connection = parent_connection
                return process, parent_connection
            finally:
                instance._spawn_done.set()

        try:
            process, connection = await asyncio.to_thread(fork)
        except asyncio.CancelledError:
            # fork()'s finally owns _spawn_done: the executor thread may still be
            # finishing, and teardown waits on that signal before reaping.
            raise
        except Exception as error:  # noqa: BLE001 - surfaced downstream as typed failure
            _shutdown_sync(instance._process, instance._connection, graceful=False)
            instance._process = None
            instance._connection = None
            await instance.dispatch(
                ctx,
                dataclasses.replace(
                    _SlotSpawnFailedEvent.with_data(
                        SlotEvaluationResultData(
                            operation_id="", ok=False, error=f"callback worker spawn failed: {error}"
                        )
                    ),
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                ),
            )
            return
        instance._process = process
        instance._connection = connection
        instance._active_operation = None
        await instance.dispatch(
            ctx,
            dataclasses.replace(
                _SlotSpawnedEvent.with_data(_WorkerSignalData()),
                source=hsm.id(instance),
                target=hsm.id(instance),
            ),
        )

    @staticmethod
    def _spawn_retryable(ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[SlotEvaluationResultData]) -> bool:
        return instance._spawn_failures < _MAX_SPAWN_FAILURES

    @staticmethod
    def _count_spawn_failure(
        ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[SlotEvaluationResultData]
    ) -> None:
        instance._spawn_failures += 1

    # -- evaluation -------------------------------------------------------

    @staticmethod
    async def _round_trip_activity(
        ctx: hsm.Context,
        instance: "WorkerSlot",
        event: hsm.Event[SlotEvaluationRequestData],
    ) -> None:
        request = event.data
        assert isinstance(request, SlotEvaluationRequestData)
        connection = instance._connection
        operation_id = instance._active_operation or ""
        result_data: SlotEvaluationResultData
        if connection is None or instance._process is None or not instance._process.is_alive():
            result_data = SlotEvaluationResultData(
                operation_id=operation_id, ok=False, error="callback worker is not connected."
            )
        else:
            encoded = json.dumps(
                {
                    "program": request.program,
                    "callback": request.callback,
                    "event": request.event,
                    "behavior_id": request.behavior_id,
                    "dispatch_allowed": request.dispatch_allowed,
                },
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            if len(encoded) > runtime.CALLBACK_MAX_RESPONSE_BYTES:
                raise runtime.CallbackError("callback worker request exceeds its wire-size budget.")
            deadline = time.monotonic() + instance._budgets.evaluation_seconds
            try:
                decoded = await asyncio.to_thread(
                    _round_trip_sync,
                    connection,
                    encoded,
                    deadline=deadline,
                    poll_seconds=instance._budgets.poll_seconds,
                )
                kind, value = _kind_and_value(decoded.result)
                result_data = SlotEvaluationResultData(
                    operation_id=operation_id,
                    ok=True,
                    result_kind=kind,
                    result_value=value,
                    dispatches=tuple(
                        {"event": item.event, "data": item.data, "target": item.target} for item in decoded.dispatches
                    ),
                    declared_events=tuple(
                        contract.model_dump(mode="json", by_alias=True) for contract in decoded.declared_events
                    ),
                )
            except runtime.CallbackError as error:
                result_data = SlotEvaluationResultData(operation_id=operation_id, ok=False, error=str(error))
        await instance.dispatch(
            ctx,
            dataclasses.replace(
                _SlotResultEvent.with_data(result_data),
                id=operation_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _begin_operation(ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[SlotEvaluationRequestData]) -> None:
        instance._active_operation = event.id

    @staticmethod
    def _retire_after_result(
        ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[SlotEvaluationResultData]
    ) -> bool:
        data = event.data
        assert isinstance(data, SlotEvaluationResultData)
        return data.ok and instance._evaluations_done + 1 >= instance._budgets.max_evaluations

    @staticmethod
    def _failed_result(ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[SlotEvaluationResultData]) -> bool:
        data = event.data
        assert isinstance(data, SlotEvaluationResultData)
        return not data.ok

    @staticmethod
    def _forward_result(ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[SlotEvaluationResultData]) -> None:
        data = event.data
        assert isinstance(data, SlotEvaluationResultData)
        if data.ok:
            instance._evaluations_done += 1
        instance._forward_to_owner(ctx, data, event.metadata)

    @staticmethod
    def _note_kill_timeout(ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[typing.Any]) -> None:
        operation = instance._active_operation or ""
        instance._active_operation = None
        instance._forward_to_owner(
            ctx,
            SlotEvaluationResultData(
                operation_id=operation,
                ok=False,
                error="Starlark callback worker was killed after exceeding its evaluation budget.",
            ),
            {},
        )

    def _forward_to_owner(
        self,
        ctx: hsm.Context,
        data: SlotEvaluationResultData,
        metadata: collections.abc.Mapping[str, object],
    ) -> None:
        _ = self._owner.dispatch(
            ctx,
            dataclasses.replace(
                _SlotResultEvent.with_data(data),
                id=data.operation_id,
                source=hsm.id(self),
                target=hsm.id(self._owner),
                metadata=dict(metadata),
            ),
        )

    # -- teardown ---------------------------------------------------------

    @staticmethod
    async def _kill_activity(ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[typing.Any]) -> None:
        await asyncio.to_thread(instance._spawn_done.wait, 5.0)
        await asyncio.to_thread(_shutdown_sync, instance._process, instance._connection, graceful=False)
        instance._process = None
        instance._connection = None
        await instance.dispatch(
            ctx,
            dataclasses.replace(
                _SlotKilledEvent.with_data(_WorkerSignalData()),
                source=hsm.id(instance),
                target=hsm.id(instance),
            ),
        )

    @staticmethod
    async def _retire_activity(ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[typing.Any]) -> None:
        await asyncio.to_thread(instance._spawn_done.wait, 5.0)
        await asyncio.to_thread(_shutdown_sync, instance._process, instance._connection, graceful=True)
        instance._process = None
        instance._connection = None
        instance._evaluations_done = 0
        await instance.dispatch(
            ctx,
            dataclasses.replace(
                _SlotRetiredEvent.with_data(_WorkerSignalData()),
                source=hsm.id(instance),
                target=hsm.id(instance),
            ),
        )

    @staticmethod
    async def _shutdown_activity(ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[typing.Any]) -> None:
        await asyncio.to_thread(instance._spawn_done.wait, 5.0)
        await asyncio.to_thread(_shutdown_sync, instance._process, instance._connection, graceful=True)
        instance._process = None
        instance._connection = None
        await instance.dispatch(
            ctx,
            dataclasses.replace(
                _SlotShutdownCompleteEvent.with_data(_WorkerSignalData()),
                source=hsm.id(instance),
                target=hsm.id(instance),
            ),
        )

    @staticmethod
    def _announce_available(ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[typing.Any]) -> None:
        _ = instance._owner.dispatch(
            ctx,
            dataclasses.replace(
                _SlotAvailableEvent.with_data(SlotIndexData(index=instance._index)),
                source=hsm.id(instance),
                target=hsm.id(instance._owner),
            ),
        )

    @staticmethod
    def _notify_stopped(ctx: hsm.Context, instance: "WorkerSlot", event: hsm.Event[typing.Any]) -> None:
        _ = instance._owner.dispatch(
            ctx,
            dataclasses.replace(
                _SlotStoppedEvent.with_data(_WorkerSignalData()),
                source=hsm.id(instance),
                target=hsm.id(instance._owner),
            ),
        )


def _kind_and_value(result: object) -> tuple[str, bool | None]:
    kind = "bool" if isinstance(result, bool) else "none" if result is None else "other"
    return kind, result if isinstance(result, bool) else None


# ---------------------------------------------------------------------------
# CallbackWorkers: the pool machine
# ---------------------------------------------------------------------------


class CallbackWorkers(hsm.Instance):
    """Fixed-size pool of ``WorkerSlot`` machines serving isolated evaluations.

    Requests beyond free capacity are deferred on the ready state and released
    FIFO when a slot announces availability; capacity is bounded by pool size.
    """

    model: hsm.Model

    def __init__(self, *, size: int = 4, budgets: SlotBudgets | None = None) -> None:
        super().__init__()
        self._budgets = budgets if budgets is not None else SlotBudgets()
        self._slots = [WorkerSlot(index=index, budgets=self._budgets, owner=self) for index in range(max(1, size))]
        self._free_indices: set[int] = set()
        self._pending: dict[str, asyncio.Future[WorkerEvaluationResult]] = {}
        self._confirmations = 0
        self._stale_results = 0
        self._operations = 0
        self.model = CallbackWorkers._build_model(self._budgets)

    @staticmethod
    def _build_model(budgets: SlotBudgets) -> hsm.Model:
        """Build the pool model with lifecycle deadlines closed over injected budgets."""

        warmup_after = datetime.timedelta(seconds=budgets.warmup_seconds)
        shutdown_after = datetime.timedelta(seconds=budgets.shutdown_seconds)

        def warmup_delay(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> datetime.timedelta:
            del ctx, instance, event
            return warmup_after

        def shutdown_delay(
            ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]
        ) -> datetime.timedelta:
            del ctx, instance, event
            return shutdown_after

        return bot.define(
            "BehaviorCallbackWorkers",
            hsm.initial(hsm.target("/BehaviorCallbackWorkers/priming")),
            hsm.state(
                "priming",
                hsm.activity(CallbackWorkers._prime_activity.__get__(None, object)),
                hsm.defer(_PoolEvaluateRequestedEvent),
                hsm.transition(hsm.on(_WarmedEvent), hsm.target("/BehaviorCallbackWorkers/ready/accepting")),
                hsm.transition(hsm.on(_WarmFailedEvent), hsm.target("/BehaviorCallbackWorkers/failed")),
                hsm.transition(hsm.after(warmup_delay), hsm.target("/BehaviorCallbackWorkers/failed")),
                hsm.transition(hsm.on(_StopRequestEvent), hsm.target("/BehaviorCallbackWorkers/stopping")),
                hsm.transition(
                    hsm.on(_SlotAvailableEvent),
                    hsm.effect(CallbackWorkers._mark_slot_available.__get__(None, object)),
                ),
            ),
            hsm.state(
                "ready",
                hsm.initial(hsm.target("/BehaviorCallbackWorkers/ready/parked")),
                hsm.state(
                    "parked",
                    hsm.defer(_PoolEvaluateRequestedEvent),
                    hsm.transition(
                        hsm.on(_SlotAvailableEvent),
                        hsm.effect(CallbackWorkers._mark_slot_available.__get__(None, object)),
                        hsm.target("/BehaviorCallbackWorkers/ready/accepting"),
                    ),
                    hsm.transition(hsm.on(_StopRequestEvent), hsm.target("/BehaviorCallbackWorkers/stopping")),
                ),
                hsm.state(
                    "accepting",
                    hsm.transition(
                        hsm.on(_PoolEvaluateRequestedEvent),
                        hsm.guard(CallbackWorkers._has_free_slot.__get__(None, object)),
                        hsm.effect(CallbackWorkers._assign_slot.__get__(None, object)),
                    ),
                    hsm.transition(
                        hsm.on(_PoolEvaluateRequestedEvent),
                        hsm.effect(CallbackWorkers._requeue_evaluate.__get__(None, object)),
                        hsm.target("/BehaviorCallbackWorkers/ready/parked"),
                    ),
                    hsm.transition(
                        hsm.on(_SlotAvailableEvent),
                        hsm.effect(CallbackWorkers._mark_slot_available.__get__(None, object)),
                    ),
                    hsm.transition(hsm.on(_StopRequestEvent), hsm.target("/BehaviorCallbackWorkers/stopping")),
                ),
                hsm.transition(
                    hsm.on(_SlotResultEvent),
                    hsm.guard(CallbackWorkers._is_live_result.__get__(None, object)),
                    hsm.effect(CallbackWorkers._complete_result.__get__(None, object)),
                ),
                hsm.transition(
                    hsm.on(_SlotResultEvent),
                    hsm.effect(CallbackWorkers._count_stale_result.__get__(None, object)),
                ),
            ),
            hsm.state(
                "stopping",
                hsm.entry(CallbackWorkers._dispatch_stop_requests.__get__(None, object)),
                hsm.transition(
                    hsm.on(_SlotStoppedEvent),
                    hsm.guard(CallbackWorkers._all_slots_confirmed.__get__(None, object)),
                    hsm.target("/BehaviorCallbackWorkers/stopped"),
                ),
                hsm.transition(
                    hsm.on(_SlotStoppedEvent),
                    hsm.effect(CallbackWorkers._count_confirmation.__get__(None, object)),
                ),
                hsm.transition(hsm.after(shutdown_delay), hsm.target("/BehaviorCallbackWorkers/failed")),
            ),
            hsm.final("stopped"),
            hsm.state(
                "failed",
                hsm.entry(CallbackWorkers._force_stop_slots.__get__(None, object)),
            ),
        )

    async def evaluate(self, request: SlotEvaluationRequestData) -> WorkerEvaluationResult:
        """Evaluate one callback request on a pooled worker and await its outcome.

        Requires a started pool; call once ``snapshot().ready`` is true (for example
        after attaching). Requests beyond free capacity defer until a slot frees.
        """

        loop = asyncio.get_running_loop()
        self._operations += 1
        operation_id = f"{type(self).__name__.lower()}:{self._operations}"
        future: asyncio.Future[WorkerEvaluationResult] = loop.create_future()
        self._pending[operation_id] = future
        try:
            await hsm.dispatch(
                self.context(),
                self,
                dataclasses.replace(
                    _PoolEvaluateRequestedEvent.with_data(request),
                    id=operation_id,
                    source=hsm.id(self),
                    target=hsm.id(self),
                ),
            )
        except RuntimeError as error:
            self._pending.pop(operation_id, None)
            raise runtime.CallbackError("callback worker pool is not running.") from error
        try:
            return await asyncio.wait_for(future, self._budgets.evaluation_seconds * 3 + 1.0)
        except asyncio.CancelledError:
            self._pending.pop(operation_id, None)
            raise
        except TimeoutError as error:
            self._pending.pop(operation_id, None)
            raise runtime.CallbackError("callback worker pool did not settle the evaluation.") from error

    def worker_pid(self, index: int) -> int | None:
        """Process id of one slot's current isolated worker, for diagnostics and tests."""

        return self._slots[index].worker_pid()

    def snapshot(self) -> "CallbackWorkersSnapshot":
        """Point-in-time operational observation of the pool (never for coordination)."""

        return CallbackWorkersSnapshot(
            ready=_is_pool_ready_state(self.state()),
            free_indices=tuple(sorted(self._free_indices)),
            settled_operations=self._operations,
            stale_results=self._stale_results,
        )

    # -- priming ----------------------------------------------------------

    @staticmethod
    async def _prime_activity(ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[typing.Any]) -> None:
        try:
            for slot in instance._slots:
                assert slot.model is not None
                await bot.started(instance.context(), slot, slot.model)
            await asyncio.to_thread(warm_worker_process)
        except Exception as error:  # noqa: BLE001 - surfaced downstream as typed failure
            await instance.dispatch(
                ctx,
                dataclasses.replace(
                    _WarmFailedEvent.with_data(
                        SlotEvaluationResultData(
                            operation_id="", ok=False, error=f"callback worker pool warmup failed: {error}"
                        )
                    ),
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                ),
            )
            return
        await instance.dispatch(
            ctx,
            dataclasses.replace(
                _WarmedEvent.with_data(_WorkerSignalData()),
                source=hsm.id(instance),
                target=hsm.id(instance),
            ),
        )

    # -- request routing --------------------------------------------------

    @staticmethod
    def _has_free_slot(
        ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[SlotEvaluationRequestData]
    ) -> bool:
        return bool(instance._free_indices)

    @staticmethod
    def _assign_slot(
        ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[SlotEvaluationRequestData]
    ) -> None:
        request = event.data
        assert isinstance(request, SlotEvaluationRequestData)
        index = min(instance._free_indices)
        instance._free_indices.discard(index)
        slot = instance._slots[index]
        _ = slot.dispatch(
            ctx,
            dataclasses.replace(
                _SlotEvaluateRequestEvent.with_data(request),
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(slot),
            ),
        )

    @staticmethod
    def _requeue_evaluate(
        ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[SlotEvaluationRequestData]
    ) -> None:
        _ = instance.dispatch(
            ctx,
            dataclasses.replace(
                event,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    # -- settlement -------------------------------------------------------

    @staticmethod
    def _is_live_result(
        ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[SlotEvaluationResultData]
    ) -> bool:
        data = event.data
        assert isinstance(data, SlotEvaluationResultData)
        future = instance._pending.get(data.operation_id)
        return future is not None and not future.done()

    @staticmethod
    def _complete_result(
        ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[SlotEvaluationResultData]
    ) -> None:
        data = event.data
        assert isinstance(data, SlotEvaluationResultData)
        future = instance._pending.pop(data.operation_id, None)
        if future is None or future.done():
            instance._stale_results += 1
            return
        if data.ok:
            dispatches = tuple(
                RawDispatch(
                    event=item["event"],
                    data=typing.cast(object | None, item.get("data")),
                    target=typing.cast(object | None, item.get("target")),
                )
                for item in data.dispatches
            )
            declared = tuple(
                source.EventContract.model_validate(typing.cast(dict[str, object], raw)) for raw in data.declared_events
            )
            if data.result_kind == "bool":
                result: object = data.result_value
            elif data.result_kind == "none":
                result = None
            else:
                result = _OTHER_CALLBACK_RESULT
            future.set_result(WorkerEvaluationResult(result=result, dispatches=dispatches, declared_events=declared))
        else:
            future.set_exception(runtime.CallbackError(data.error or "isolated Starlark callback failed."))

    @staticmethod
    def _count_stale_result(
        ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[SlotEvaluationResultData]
    ) -> None:
        instance._stale_results += 1

    @staticmethod
    def _mark_slot_available(ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[SlotIndexData]) -> None:
        index_data = event.data
        assert isinstance(index_data, SlotIndexData)
        instance._free_indices.add(index_data.index)

    # -- shutdown ---------------------------------------------------------

    @staticmethod
    def _force_stop_slots(ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[typing.Any]) -> None:
        """Best-effort reaping when the pool gives up waiting for clean confirmation."""

        for operation_id, future in list(instance._pending.items()):
            if not future.done():
                future.set_exception(runtime.CallbackError("callback worker pool failed."))
            instance._pending.pop(operation_id, None)
        for slot in instance._slots:
            _ = slot.dispatch(
                ctx,
                dataclasses.replace(
                    _SlotStopRequestEvent.with_data(_WorkerSignalData()),
                    source=hsm.id(instance),
                    target=hsm.id(slot),
                ),
            )

    @staticmethod
    def _dispatch_stop_requests(ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[typing.Any]) -> None:
        instance._confirmations = 0
        for operation_id, future in list(instance._pending.items()):
            if not future.done():
                future.set_exception(runtime.CallbackError("callback worker pool stopped."))
            instance._pending.pop(operation_id, None)
        for slot in instance._slots:
            _ = slot.dispatch(
                ctx,
                dataclasses.replace(
                    _SlotStopRequestEvent.with_data(_WorkerSignalData()),
                    source=hsm.id(instance),
                    target=hsm.id(slot),
                ),
            )

    @staticmethod
    def _all_slots_confirmed(ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[typing.Any]) -> bool:
        return instance._confirmations + 1 >= len(instance._slots)

    @staticmethod
    def _count_confirmation(ctx: hsm.Context, instance: "CallbackWorkers", event: hsm.Event[typing.Any]) -> None:
        instance._confirmations += 1


__all__ = [
    "CallbackWorkers",
    "CallbackWorkersSnapshot",
    "RawDispatch",
    "SlotBudgets",
    "SlotEvaluationRequestData",
    "SlotEvaluationResultData",
    "WorkerEvaluationResult",
    "WorkerSlot",
    "apply_worker_limits",
    "decode_worker_response",
    "evaluation_context",
    "warm_worker_process",
    "worker_loop",
]

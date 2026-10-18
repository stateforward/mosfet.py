"""``Host``: a sandbox host on the Docker Engine API over a rootless socket, answering ``sandbox.host``'s protocol.

Every Engine API exchange is one leaf state whose activity makes exactly one request on an ``httpx`` client built
for it (``timeout=None``, no retries, no redirects) and dispatches a typed completion to the host. Nothing in
``httpx`` decides behavior: each leaf is bounded by its own ``hsm.after``, whose exit cancels the activity. Ids the
Engine API mints (the container, the exec instance, the containers a sweep found) travel in the completion data;
the only durable fields are the constructor values, the line of requests (the first is the one being served), the
reopen count and which refusals were already logged.

Topology (``DockerSandboxHost``)::

    opening                 every request answered failed{unavailable}
      pinging               GET /_ping
      versioning            GET /version: ApiVersion at least api_version, else refused api_too_old
      inspecting            GET /info: rootless, cgroup v2, memory/pids limits, runsc when asked, else refused
      networking            (choice) egress_network? -> network_inspecting : open
      network_inspecting    GET /networks/{name}; 404 -> network_creating
      network_creating      POST /networks/create (a bridge without inter-container traffic)
      every leaf: after(open_timeout), a failed exchange or a refusal -> down
    down                    entry: announces down to the owner; every request answered failed{unavailable}
                            after(reopen_initial doubling to reopen_max; reopen_max after a refusal) -> opening
    open                    entry: announces serving to the owner
                            a request that arrives while one is served waits in line (at most 16, else busy)
      next                  (choice) the first request in line picks its chain; none -> idle
      idle                  a request joins the line -> next
      serving
        posting             POST /containers/create            -> starting
        starting            POST /containers/{id}/start        -> verifying ; failure -> cleaning
        verifying           exec: cat cpu.max pids.max memory.max; applied -> answered created ;
                            not applied -> cleaning (answered unsupported) ; failure -> cleaning
        cleaning            DELETE /containers/{id}?force      -> answered failed
        staging             exec: the write wrapper stages a run's stdin in /tmp      -> running
        running             exec: timeout -s KILL <s> argv; its start is within its timeout until
                            after(the run's timeout) moves it past; 124, or 137 past it -> timed_out ;
                            137 within it -> checking ; otherwise -> answered ran{exited}
        checking            exec: cat memory.events; an oom_kill -> out_of_memory, else killed
        writing             exec: the write wrapper (base64 argv chunks)              -> answered written
        reading             exec: the read wrapper (regular file, not a link)         -> answered contents
        destroying          DELETE /containers/{id}?force&v (404 counts)              -> answered destroyed
        listing             GET /containers/json by the owner's labels                -> deleting
        deleting            (choice) ids left? -> removing : answered swept{count}
        removing            DELETE the first, carry the rest                         -> deleting
        every leaf: after(step_timeout) -> answered failed{timeout}; the run's start is bounded by its timeout
        plus kill_grace instead. A status that is not the step's success is answered with its kind; a failed
        transport is answered unavailable and takes the host down.

Each exec chain is ``creating`` (``POST /containers/{id}/exec``), ``starting`` (``POST /exec/{id}/start``, the
multiplexed body demuxed with head and tail bounds) and ``inspecting`` (``GET /exec/{id}/json``); ``verifying`` and
``checking`` judge the output alone and need no ``inspecting``.

``/info`` alone cannot tell whether the limits hold (Podman reports ``CpuCfsQuota`` false while its machines
enforce ``cpu.max``), so each new machine reads the limits its cgroup applies before it is answered (W5-14).

Telemetry (scope ``bot.providers.docker``; every attribute value comes from a closed set in code):

* ``bot.provider.docker.exchange.count`` — ``step`` (the 13 values of ``Step``) × ``status`` {1xx, 2xx, 3xx,
  4xx, 5xx, transport, timeout}: at most 91 series.
* ``bot.provider.docker.exchange.duration`` — seconds per exchange, by ``step``: 13 series.
* ``bot.provider.docker.open.count`` — ``outcome`` {open, down_unavailable, down_unsupported, down_timeout}: 4.
* ``bot.provider.docker.deferred.count`` — requests that waited in line, no attributes: 1.
* One ``bot.docker.exchange`` span per exchange, with ``bot.docker.step`` and, on failure, ``bot.failure.kind``.
* A refusal at open is logged once per kind (``mapping.Refusal``).
* A daemon lost mid-request is logged with the step and the types of the transport error and its causes
  (``ReadError from ConnectionResetError``), never their messages.

Never used as attributes, logs or messages: argv, paths, file text, output, environment values, image
references, container, exec or request ids, or exit codes.
"""

import collections
import collections.abc
import dataclasses
import datetime
import functools
import logging
import pathlib
import re
import time
import typing

import hsm
import httpx
import pydantic
from opentelemetry import metrics

from mosfet.abilities import sandbox
from mosfet.define import define
from mosfet.telemetry import span
from mosfet.telemetry.hsm import Traced

from . import mapping, stream
from .capacity import Runtime, capacity as runtime_capacity  # aliased: Host's ``capacity`` parameter shadows it

Step = typing.Literal[
    "ping",
    "version",
    "info",
    "network_inspect",
    "network_create",
    "create",
    "start",
    "clean",
    "exec_create",
    "exec_start",
    "exec_inspect",
    "delete",
    "list",
]
"""Which Engine API exchange a leaf state makes; a closed set, used as a telemetry attribute."""

Purpose = typing.Literal["verify", "stage", "run", "check", "write", "read"]
"""What an exec chain is for."""

_SCOPE: typing.Final = "bot.providers.docker"
_COMPONENT: typing.Final = "docker"
_STEP_ATTR: typing.Final = "bot.docker.step"
_MAX_BODY_BYTES: typing.Final = 1024 * 1024
_OK: typing.Final = frozenset({200})
_MAX_LINE: typing.Final = 16
_MAX_BACKOFF_DOUBLINGS: typing.Final = 30
_MESSAGE_CHARS: typing.Final = 1024
_MAX_CAUSES: typing.Final = 8
_TIMED_OUT_EXITS: typing.Final = frozenset({124, 137})
_KILLED_EXIT: typing.Final = 137
_MIN_DURATION: typing.Final = datetime.timedelta(0)
_OWNER_PATTERN: typing.Final = r"^[^\x00]{1,128}$"
_NETWORK_PATTERN: typing.Final = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
_RESERVED_NETWORKS: typing.Final = frozenset({"none", "host", "bridge", "default"})
_API_VERSION_PATTERN: typing.Final = r"^v[0-9]+\.[0-9]+$"
_ISOLATION: typing.Final[dict[str, sandbox.Isolation]] = {
    "runc": sandbox.Isolation.CONTAINER,
    "runsc": sandbox.Isolation.USER_KERNEL,
}
_DEFAULT_OPEN_TIMEOUT: typing.Final = datetime.timedelta(seconds=10)
_DEFAULT_STEP_TIMEOUT: typing.Final = datetime.timedelta(seconds=30)
_DEFAULT_KILL_GRACE: typing.Final = datetime.timedelta(seconds=10)
_DEFAULT_REOPEN_INITIAL: typing.Final = datetime.timedelta(seconds=1)
_DEFAULT_REOPEN_MAX: typing.Final = datetime.timedelta(seconds=60)

_IMAGE: typing.Final = pydantic.TypeAdapter[str](sandbox.ImageRef)

_LOGGER = logging.getLogger(__name__)

_METER = metrics.get_meter(_SCOPE)
_EXCHANGES = _METER.create_counter(
    "bot.provider.docker.exchange.count",
    unit="{exchange}",
    description="Docker Engine API exchanges, by step and status class (or transport failure, or timeout).",
)
_EXCHANGE_DURATION = _METER.create_histogram(
    "bot.provider.docker.exchange.duration",
    unit="s",
    description="How long one Docker Engine API exchange took, by step.",
)
_OPENS = _METER.create_counter(
    "bot.provider.docker.open.count",
    unit="{attempt}",
    description="Docker sandbox host open attempts, by outcome.",
)
_DEFERRED = _METER.create_counter(
    "bot.provider.docker.deferred.count",
    unit="{request}",
    description="Sandbox host requests that waited in line behind another.",
)

_ROOT: typing.Final = "/DockerSandboxHost"
_OPENING: typing.Final = f"{_ROOT}/opening"
_DOWN: typing.Final = f"{_ROOT}/down"
_OPEN: typing.Final = f"{_ROOT}/open"
_NEXT: typing.Final = f"{_OPEN}/next"
_IDLE: typing.Final = f"{_OPEN}/idle"
_SERVING: typing.Final = f"{_OPEN}/serving"

_REQUESTS: typing.Final = (
    sandbox.host.CreateEvent,
    sandbox.host.RunEvent,
    sandbox.host.WriteEvent,
    sandbox.host.ReadEvent,
    sandbox.host.DestroyEvent,
    sandbox.host.SweepEvent,
)

type _Guard = collections.abc.Callable[[hsm.Context, "Host", hsm.Event[typing.Any]], bool]
type _Effect = collections.abc.Callable[[hsm.Context, "Host", hsm.Event[typing.Any]], None]
type _Activity = collections.abc.Callable[
    [hsm.Context, "Host", hsm.Event[typing.Any]], collections.abc.Coroutine[typing.Any, typing.Any, None]
]
type _Delay = collections.abc.Callable[[hsm.Context, "Host", hsm.Event[typing.Any]], datetime.timedelta]


class _Output(pydantic.BaseModel):
    """One exec's bounded output."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_bytes: int = 0


class _Carry(pydantic.BaseModel):
    """What one request's chain has learned so far, handed from each leaf's completion to the next leaf."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    container: str | None = None
    exec_id: str | None = None
    output: _Output | None = None
    duration: datetime.timedelta | None = None
    exit_code: int | None = None
    oom_kills: int = 0
    remaining: tuple[str, ...] = ()
    count: int = 0
    failure: sandbox.FailureKind | None = None


class _ExchangedData(pydantic.BaseModel):
    """One exchange got an HTTP answer; ``refusal`` says why an answer at open refuses the daemon."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    request_id: str | None
    step: Step
    status: int
    refusal: mapping.Refusal | None = None
    too_large: bool = False
    carry: _Carry = pydantic.Field(default_factory=_Carry)


class _ExchangeFailedData(pydantic.BaseModel):
    """One exchange got no usable answer: the transport failed (``unavailable``) or the answer was broken."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    request_id: str | None
    step: Step
    kind: sandbox.FailureKind
    transport_error: tuple[str, ...] = pydantic.Field(
        default=(),
        description="The failed transport's error type and its causes' types, outermost first; never messages.",
    )
    carry: _Carry = pydantic.Field(default_factory=_Carry)


_ExchangedEvent = hsm.Event[_ExchangedData](
    name="bot.provider.docker.exchanged", kind=hsm.CompletionEventKind, schema=_ExchangedData
)
_ExchangeFailedEvent = hsm.Event[_ExchangeFailedData](
    name="bot.provider.docker.exchange_failed", kind=hsm.ErrorEventKind, schema=_ExchangeFailedData
)


class _BodyTooLargeError(Exception):
    """An Engine API answer was larger than the host reads."""


@dataclasses.dataclass(frozen=True, slots=True)
class _Call:
    """One Engine API request: method, path under the API version, query and JSON body."""

    method: str
    path: str
    params: collections.abc.Mapping[str, str] | None = None
    body: object = None


type _Read = collections.abc.Callable[[int, bytes, _Carry], _Carry]
type _Refuse = collections.abc.Callable[[bytes], mapping.Refusal | None]


def _positive(**durations: datetime.timedelta) -> None:
    for name, duration in durations.items():
        if duration <= _MIN_DURATION:
            raise ValueError(f"docker.Host {name} must be positive.")


def _failed(kind: sandbox.FailureKind, message: str) -> hsm.Event[sandbox.HostFailureData]:
    return sandbox.host.FailedEvent.with_data(sandbox.HostFailureData(kind=kind, message=message[:_MESSAGE_CHARS]))


def _identified(status: int, body: bytes, carry: _Carry, field: typing.Literal["container", "exec_id"]) -> _Carry:
    del status
    return carry.model_copy(update={field: mapping.Identified.model_validate_json(body).id})


class Host(Traced):
    """A sandbox host on a rootless Docker Engine (or a Podman compatible one), answering ``sandbox.host`` events.

    The owner starts it, stops it, and otherwise reaches it only by ``sandbox.host`` request events; each is
    answered with exactly one terminal, from this host to the request's source. Requests are served one at a time,
    in order of arrival. The host announces ``serving`` to its owner (the machine that started it) each time it
    opens, and ``down`` each time it fails to open or loses the daemon.
    """

    _socket: pathlib.Path
    _capacity: sandbox.Capacity
    _images: frozenset[str]
    _owner: str
    _egress_network: str | None
    _runtime: Runtime
    _api_version: str
    _open_timeout: datetime.timedelta
    _step_timeout: datetime.timedelta
    _kill_grace: datetime.timedelta
    _reopen_initial: datetime.timedelta
    _reopen_max: datetime.timedelta
    _transport: collections.abc.Callable[[], httpx.AsyncBaseTransport]
    # Durable machine data: the requests in line (the first is being served), how many opens in a row failed,
    # whether the last one was a refusal (retried at reopen_max), and which refusals were logged.
    _line: collections.deque[hsm.Event[typing.Any]]
    _reopens: int
    _refused: bool
    _logged: set[str]

    def __init__(
        self,
        *,
        socket: pathlib.Path,
        capacity: sandbox.Capacity | None = None,
        images: frozenset[str],
        owner: str,
        egress_network: str | None = None,
        runtime: Runtime = "runc",
        api_version: str = "v1.41",
        open_timeout: datetime.timedelta = _DEFAULT_OPEN_TIMEOUT,
        step_timeout: datetime.timedelta = _DEFAULT_STEP_TIMEOUT,
        kill_grace: datetime.timedelta = _DEFAULT_KILL_GRACE,
        reopen_initial: datetime.timedelta = _DEFAULT_REOPEN_INITIAL,
        reopen_max: datetime.timedelta = _DEFAULT_REOPEN_MAX,
        transport: collections.abc.Callable[[], httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        """Serve sandboxes on the Engine API at the unix ``socket``, which must be a rootless daemon's.

        ``capacity`` is the value the bot file declared (``docker.capacity``) and injected into the ability as well;
        by default it is ``docker.capacity(max_resources=sandbox.Resources(), runtime=runtime)``, this host's own
        runtime with the default ceilings.
        ``images`` is the owner's allowlist: nothing is ever pulled, and a create for another image is refused
        ``unsupported``. ``owner`` labels every container so a later host of the same owner sweeps what a dead one
        left. ``egress_network`` is the owner's bridge for ``Network.EGRESS``; without it egress is refused.
        ``runtime`` is the OCI runtime (``runsc`` for gVisor). Opening is bounded by ``open_timeout`` per step, each
        request step by ``step_timeout``, and a run by its own timeout plus ``kill_grace``. A host that could not
        open retries after ``reopen_initial``, doubling up to ``reopen_max``. ``transport`` makes the transport of
        each exchange's client; by default an ``httpx.AsyncHTTPTransport`` on ``socket`` with no retries.

        Raises:
            ValueError: If ``images`` is empty or holds a malformed reference, ``owner`` is empty, longer than 128
                characters or holds a NUL, ``egress_network`` is not a plain network name of the owner's,
                ``runtime`` is unknown, ``api_version`` is not ``v<major>.<minor>``, ``capacity`` does not declare
                this host (its isolation for ``runtime``, and files), a duration is not positive, or
                ``reopen_max`` is shorter than ``reopen_initial``.
        """

        if not images:
            raise ValueError("docker.Host images must allow at least one image.")
        for image in images:
            _ = _IMAGE.validate_python(image)
        if re.fullmatch(_OWNER_PATTERN, owner) is None:
            raise ValueError("docker.Host owner must be 1 to 128 characters without NUL.")
        if egress_network is not None and (
            re.fullmatch(_NETWORK_PATTERN, egress_network) is None or egress_network in _RESERVED_NETWORKS
        ):
            raise ValueError("docker.Host egress_network must name the owner's own bridge network.")
        if runtime not in _ISOLATION:
            raise ValueError("docker.Host runtime must be 'runc' or 'runsc'.")
        if re.fullmatch(_API_VERSION_PATTERN, api_version) is None:
            raise ValueError("docker.Host api_version must look like 'v1.41'.")
        declared = (
            capacity if capacity is not None else runtime_capacity(max_resources=sandbox.Resources(), runtime=runtime)
        )
        if declared.isolation is not _ISOLATION[runtime] or not declared.files:
            raise ValueError("docker.Host capacity must be the one docker.capacity declares for its runtime.")
        _positive(
            open_timeout=open_timeout,
            step_timeout=step_timeout,
            kill_grace=kill_grace,
            reopen_initial=reopen_initial,
        )
        if reopen_max < reopen_initial:
            raise ValueError("docker.Host reopen_max cannot be shorter than reopen_initial.")
        super().__init__()
        self._socket = socket
        self._capacity = declared
        self._images = images
        self._owner = owner
        self._egress_network = egress_network
        self._runtime = runtime
        self._api_version = api_version
        self._open_timeout = open_timeout
        self._step_timeout = step_timeout
        self._kill_grace = kill_grace
        self._reopen_initial = reopen_initial
        self._reopen_max = reopen_max
        self._transport = (
            transport
            if transport is not None
            else functools.partial(httpx.AsyncHTTPTransport, uds=str(socket), retries=0)
        )
        self._line = collections.deque()
        self._reopens = 0
        self._refused = False
        self._logged = set()

    # Exchanges: one request per activity, its completion dispatched to this host.

    @staticmethod
    def _client(instance: "Host") -> httpx.AsyncClient:
        """A client for one exchange: no timeout (the state's ``hsm.after`` bounds it), no redirects."""

        return httpx.AsyncClient(
            transport=instance._transport(),
            timeout=None,
            base_url=f"http://docker/{instance._api_version}",
            follow_redirects=False,
        )

    @staticmethod
    async def _capped(response: httpx.Response) -> bytes:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body += chunk
            if len(body) > _MAX_BODY_BYTES:
                raise _BodyTooLargeError
        return bytes(body)

    @staticmethod
    def _count(step: Step, status: str, started: float) -> None:
        _EXCHANGES.add(1, {"step": step, "status": status})
        _EXCHANGE_DURATION.record(time.monotonic() - started, {"step": step})

    @staticmethod
    def _complete(
        ctx: hsm.Context,
        instance: "Host",
        step: Step,
        request_id: str | None,
        carry: _Carry,
        *,
        status: int | None = None,
        kind: sandbox.FailureKind | None = None,
        refusal: mapping.Refusal | None = None,
        too_large: bool = False,
        error: BaseException | None = None,
    ) -> None:
        if status is None:
            transport_error: tuple[str, ...] = ()
            cause = error
            while cause is not None and len(transport_error) < _MAX_CAUSES:
                transport_error += (type(cause).__name__,)
                cause = cause.__cause__ if cause.__cause__ is not None else cause.__context__
            failure = _ExchangeFailedData(
                request_id=request_id,
                step=step,
                kind=kind or sandbox.FailureKind.FAILED,
                transport_error=transport_error,
                carry=carry,
            )
            _ = hsm.dispatch(ctx, instance, _ExchangeFailedEvent.with_data(failure))
            return
        exchanged = _ExchangedData(
            request_id=request_id, step=step, status=status, refusal=refusal, too_large=too_large, carry=carry
        )
        _ = hsm.dispatch(ctx, instance, _ExchangedEvent.with_data(exchanged))

    @staticmethod
    async def _exchange(
        ctx: hsm.Context,
        instance: "Host",
        step: Step,
        call: _Call,
        carry: _Carry,
        *,
        ok: frozenset[int] = _OK,
        read: _Read | None = None,
        refuse: _Refuse | None = None,
    ) -> None:
        """Make ``call`` and dispatch its completion; ``read`` and ``refuse`` parse a success answer's body."""

        request_id = instance._line[0].id if instance._line else None
        started = time.monotonic()  # telemetry only; hsm.after owns the deadline
        answered: int | None = None
        kind: sandbox.FailureKind | None = None
        failed: httpx.TransportError | None = None
        refusal: mapping.Refusal | None = None
        with span.operation(
            "bot.docker.exchange", scope=_SCOPE, component=_COMPONENT, stage=step, attributes={_STEP_ATTR: step}
        ) as active:
            try:
                async with (
                    Host._client(instance) as client,
                    client.stream(call.method, call.path, params=call.params, json=call.body) as response,
                ):
                    answered = response.status_code
                    body = await Host._capped(response)
                if answered in ok:
                    if read is not None:
                        carry = read(answered, body, carry)
                    if refuse is not None:
                        refusal = refuse(body)
            except httpx.TransportError as error:
                kind = sandbox.FailureKind.UNAVAILABLE
                failed = error
            except (_BodyTooLargeError, httpx.HTTPError, httpx.StreamError, ValueError):
                # An answer that could not be read or parsed: the step failed, and the host stays up.
                kind = sandbox.FailureKind.FAILED
            if kind is not None:
                span.record_failure(active, kind.value)
        Host._count(step, Host._label(answered, kind), started)
        status = answered if kind is None else None
        Host._complete(ctx, instance, step, request_id, carry, status=status, kind=kind, refusal=refusal, error=failed)

    @staticmethod
    def _label(answered: int | None, kind: sandbox.FailureKind | None) -> str:
        """The exchange's status class, or ``transport`` when no answer came back."""

        if answered is None or kind is sandbox.FailureKind.UNAVAILABLE:
            return "transport"
        return mapping.status_class(answered)

    @staticmethod
    def _carry_of(event: hsm.Event[typing.Any]) -> _Carry:
        data = event.data
        if isinstance(data, (_ExchangedData, _ExchangeFailedData)):
            return data.carry
        return _Carry()

    @staticmethod
    def _head(instance: "Host") -> object:
        return instance._line[0].data if instance._line else None

    @staticmethod
    def _machine(instance: "Host") -> str:
        data = Host._head(instance)
        if isinstance(data, (sandbox.HostRunData, sandbox.HostWriteData, sandbox.HostReadData, sandbox.DestroyData)):
            return data.machine.id
        raise ValueError("the request in line names no machine")

    # Opening.

    @staticmethod
    async def _ping(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        await Host._exchange(ctx, instance, "ping", _Call("GET", "/_ping"), _Carry())

    @staticmethod
    async def _version(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event

        def refuse(body: bytes) -> mapping.Refusal | None:
            version = mapping.Version.model_validate_json(body)
            return mapping.version_refusal(version, api_version=instance._api_version)

        await Host._exchange(ctx, instance, "version", _Call("GET", "/version"), _Carry(), refuse=refuse)

    @staticmethod
    async def _info(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event

        def refuse(body: bytes) -> mapping.Refusal | None:
            return mapping.info_refusal(mapping.Info.model_validate_json(body), runtime=instance._runtime)

        await Host._exchange(ctx, instance, "info", _Call("GET", "/info"), _Carry(), refuse=refuse)

    @staticmethod
    async def _network_inspect(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        call = _Call("GET", f"/networks/{instance._egress_network}")
        await Host._exchange(ctx, instance, "network_inspect", call, _Carry())

    @staticmethod
    async def _network_create(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        call = _Call("POST", "/networks/create", body=mapping.network_body(instance._egress_network or ""))
        await Host._exchange(ctx, instance, "network_create", call, _Carry(), ok=frozenset({201, 409}))

    @staticmethod
    def _wants_network(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._egress_network is not None

    @staticmethod
    def _opening(step: Step, *statuses: int) -> _Guard:
        """An answer to ``step`` at open with one of ``statuses`` that refuses nothing."""

        def guard(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
            del ctx, instance
            data = event.data
            return (
                isinstance(data, _ExchangedData)
                and data.request_id is None
                and data.step == step
                and data.status in statuses
                and data.refusal is None
            )

        return guard

    @staticmethod
    def _is_refusal(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _ExchangedData) and data.request_id is None and data.refusal is not None

    @staticmethod
    def _is_open_failure(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, (_ExchangedData, _ExchangeFailedData)) and data.request_id is None

    @staticmethod
    def _refuse_daemon(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _ExchangedData) and data.refusal is not None
        instance._refused = True
        _OPENS.add(1, {"outcome": "down_unsupported"})
        if data.refusal not in instance._logged:
            instance._logged.add(data.refusal)
            _LOGGER.warning(
                "The Docker Engine is refused for sandboxes (%s); retrying every %s.",
                data.refusal,
                instance._reopen_max,
            )

    @staticmethod
    def _open_failed(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._refused = False
        _OPENS.add(1, {"outcome": "down_unavailable"})

    @staticmethod
    def _open_overran(step: Step) -> _Effect:
        def effect(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
            del ctx, event
            instance._refused = False
            _EXCHANGES.add(1, {"step": step, "status": "timeout"})
            _OPENS.add(1, {"outcome": "down_timeout"})

        return effect

    @staticmethod
    def _open_delay(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._open_timeout

    @staticmethod
    def _opened(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        instance._reopens = 0
        instance._refused = False
        _OPENS.add(1, {"outcome": "open"})
        sandbox.host.announce(ctx, instance, sandbox.host.ServingEvent.with_data(sandbox.ServingData()))

    @staticmethod
    def _fell(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        instance._reopens += 1
        sandbox.host.announce(ctx, instance, sandbox.host.DownEvent.with_data(sandbox.DownData()))

    @staticmethod
    def _reopen_delay(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        if instance._refused:
            return instance._reopen_max
        doublings = min(max(instance._reopens - 1, 0), _MAX_BACKOFF_DOUBLINGS)
        return min(instance._reopen_initial * (1 << doublings), instance._reopen_max)

    # Answering requests.

    @staticmethod
    def _reply(
        ctx: hsm.Context, instance: "Host", request: hsm.Event[typing.Any], terminal: hsm.Event[typing.Any]
    ) -> None:
        """Send ``terminal`` for ``request`` from this host to the request's source."""

        source = request.source
        if not source:
            _LOGGER.warning("A sandbox host request came without a source; its %s answer was dropped.", terminal.name)
            return
        reply = dataclasses.replace(terminal, id=request.id, source=hsm.id(instance), target=source)
        _ = hsm.dispatch_to(ctx, reply, source)

    @staticmethod
    def _answer(ctx: hsm.Context, instance: "Host", terminal: hsm.Event[typing.Any]) -> None:
        """Answer the request first in line with ``terminal``; the next one moves up."""

        Host._reply(ctx, instance, instance._line.popleft(), terminal)

    @staticmethod
    def _refuse_unavailable(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        Host._reply(
            ctx,
            instance,
            event,
            _failed(sandbox.FailureKind.UNAVAILABLE, "The Docker Engine sandbox host is not serving."),
        )

    @staticmethod
    def _drain(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        while instance._line:
            Host._answer(
                ctx,
                instance,
                _failed(sandbox.FailureKind.UNAVAILABLE, "The Docker Engine sandbox host stopped serving."),
            )

    @staticmethod
    def _take(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._line.append(event)

    @staticmethod
    def _hold(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        if len(instance._line) >= _MAX_LINE:
            Host._reply(
                ctx,
                instance,
                event,
                _failed(
                    sandbox.FailureKind.BUSY, f"The Docker Engine sandbox host already holds {_MAX_LINE} requests."
                ),
            )
            return
        instance._line.append(event)
        _DEFERRED.add(1)

    # Choosing what the request first in line needs.

    @staticmethod
    def _next_is(*types: type[pydantic.BaseModel]) -> _Guard:
        def guard(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
            del ctx, event
            return isinstance(Host._head(instance), types)

        return guard

    @staticmethod
    def _names_bad_machine(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        try:
            return not mapping.is_id(Host._machine(instance))
        except ValueError:
            return False

    @staticmethod
    def _allows(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        """The create first in line asks for an allowed image, within capacity, on a network this host has."""

        del ctx, event
        data = Host._head(instance)
        if not isinstance(data, sandbox.CreateData):
            return False
        bench = data.spec
        if bench.image not in instance._images:
            return False
        if bench.network is sandbox.Network.EGRESS and instance._egress_network is None:
            return False
        try:
            instance._capacity.check(bench)
        except ValueError:
            return False
        return True

    @staticmethod
    def _has_stdin(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        data = Host._head(instance)
        return isinstance(data, sandbox.HostRunData) and data.command.stdin is not None

    @staticmethod
    def _answer_bad_machine(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        Host._answer(
            ctx, instance, _failed(sandbox.FailureKind.INVALID_INPUT, "The machine is not one this host could name.")
        )

    @staticmethod
    def _answer_unsupported(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        Host._answer(
            ctx,
            instance,
            _failed(
                sandbox.FailureKind.UNSUPPORTED,
                "The spec asks for an image not allowed, a limit beyond capacity, or egress without a network.",
            ),
        )

    # Serving: correlation, and the answers every step shares.

    @staticmethod
    def _answers(instance: "Host", event: hsm.Event[typing.Any], step: Step, statuses: tuple[int, ...]) -> bool:
        """``event`` answers ``step`` for the request first in line, with one of ``statuses`` (any when empty)."""

        data = event.data
        return (
            isinstance(data, _ExchangedData)
            and Host._correlated(instance, data.request_id)
            and data.step == step
            and (not statuses or data.status in statuses)
        )

    @staticmethod
    def _done(step: Step, *statuses: int) -> _Guard:
        """A guard: an answer to ``step`` for the request first in line, with one of ``statuses``."""

        def guard(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
            del ctx
            return Host._answers(instance, event, step, statuses)

        return guard

    @staticmethod
    def _correlated(instance: "Host", request_id: str | None) -> bool:
        return request_id is not None and bool(instance._line) and instance._line[0].id == request_id

    @staticmethod
    def _is_answered(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, _ExchangedData) and Host._correlated(instance, data.request_id)

    @staticmethod
    def _is_lost(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _ExchangeFailedData)
            and Host._correlated(instance, data.request_id)
            and data.kind is sandbox.FailureKind.UNAVAILABLE
        )

    @staticmethod
    def _is_broken(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, _ExchangeFailedData) and Host._correlated(instance, data.request_id)

    @staticmethod
    def _answer_status(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ExchangedData)
        kind = mapping.failure_kind(data.status)
        Host._answer(
            ctx, instance, _failed(kind, f"The Docker Engine answered the {data.step} step with HTTP {data.status}.")
        )

    @staticmethod
    def _lose(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        """The daemon was lost mid-request; leaving ``open`` answers the request and the rest of the line."""

        del ctx, instance
        data = event.data
        assert isinstance(data, _ExchangeFailedData)
        _OPENS.add(1, {"outcome": "down_unavailable"})
        _LOGGER.warning(
            "The Docker Engine sandbox host lost the daemon during its %s step (%s); it answers unavailable and reopens.",
            data.step,
            " from ".join(data.transport_error),
        )

    @staticmethod
    def _answer_broken(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ExchangeFailedData)
        Host._answer(
            ctx, instance, _failed(data.kind, f"The Docker Engine's answer to the {data.step} step was unusable.")
        )

    @staticmethod
    def _overran(step: Step) -> _Effect:
        def effect(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
            del event
            _EXCHANGES.add(1, {"step": step, "status": "timeout"})
            Host._answer(
                ctx, instance, _failed(sandbox.FailureKind.TIMEOUT, f"The {step} step did not finish in time.")
            )

        return effect

    @staticmethod
    def _step_delay(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._step_timeout

    @staticmethod
    def _run_timeout(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        """The run's own timeout: once it passes, a kill is the timeout's, not the memory limit's."""

        del ctx, event
        data = Host._head(instance)
        if not isinstance(data, sandbox.HostRunData):
            return instance._step_timeout
        return datetime.timedelta(seconds=mapping.timeout_seconds(data.command))

    @staticmethod
    def _run_delay(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        return Host._run_timeout(ctx, instance, event) + instance._kill_grace

    # Creating and destroying machines.

    @staticmethod
    async def _post(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        data = Host._head(instance)
        assert isinstance(data, sandbox.CreateData)
        body = mapping.create_body(
            data.spec, owner=instance._owner, runtime=instance._runtime, egress_network=instance._egress_network
        )
        read = functools.partial(_identified, field="container")
        await Host._exchange(
            ctx,
            instance,
            "create",
            _Call("POST", "/containers/create", body=body),
            _Carry(),
            ok=frozenset({201}),
            read=read,
        )

    @staticmethod
    async def _start(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        carry = Host._carry_of(event)
        call = _Call("POST", f"/containers/{carry.container}/start")
        await Host._exchange(ctx, instance, "start", call, carry, ok=frozenset({204, 304}))

    @staticmethod
    async def _clean(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        """Remove a container that would not start or does not apply its limits, keeping why for the answer."""

        data = event.data
        carry = Host._carry_of(event)
        if carry.failure is not None:
            failure = carry.failure
        elif isinstance(data, _ExchangedData):
            failure = mapping.failure_kind(data.status)
        elif isinstance(data, _ExchangeFailedData):
            failure = data.kind
        else:
            failure = sandbox.FailureKind.FAILED
        carry = carry.model_copy(update={"failure": failure})
        call = _Call("DELETE", f"/containers/{carry.container}", params={"force": "true", "v": "true"})
        await Host._exchange(ctx, instance, "clean", call, carry, ok=frozenset({204, 404}))

    @staticmethod
    def _answer_created(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        container = Host._carry_of(event).container or ""
        created = sandbox.CreatedData(machine=sandbox.Machine(id=container))
        Host._answer(ctx, instance, sandbox.host.CreatedEvent.with_data(created))

    @staticmethod
    def _answer_cleaned(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        kind = Host._carry_of(event).failure or sandbox.FailureKind.FAILED
        if kind is sandbox.FailureKind.UNSUPPORTED:
            message = "The machine does not apply its CPU, pids or memory limits; it was removed."
        else:
            message = "The machine was created but would not start or be checked; it was removed."
        Host._answer(ctx, instance, _failed(kind, message))

    @staticmethod
    async def _destroy(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        call = _Call("DELETE", f"/containers/{Host._machine(instance)}", params={"force": "true", "v": "true"})
        await Host._exchange(ctx, instance, "delete", call, _Carry(), ok=frozenset({204, 404}))

    @staticmethod
    def _answer_destroyed(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        Host._answer(ctx, instance, sandbox.host.DestroyedEvent.with_data(sandbox.DestroyedData()))

    @staticmethod
    async def _list(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event

        def read(status: int, body: bytes, carry: _Carry) -> _Carry:
            del status
            ids = tuple(found.id for found in mapping.Listed.validate_json(body))
            return carry.model_copy(update={"remaining": ids})

        call = _Call(
            "GET", "/containers/json", params={"all": "true", "filters": mapping.sweep_filters(instance._owner)}
        )
        await Host._exchange(ctx, instance, "list", call, _Carry(), read=read)

    @staticmethod
    async def _remove(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        carry = Host._carry_of(event)

        def read(status: int, body: bytes, carry: _Carry) -> _Carry:
            del body
            return carry.model_copy(
                update={"remaining": carry.remaining[1:], "count": carry.count + (1 if status == 204 else 0)}
            )

        call = _Call("DELETE", f"/containers/{carry.remaining[0]}", params={"force": "true", "v": "true"})
        await Host._exchange(ctx, instance, "delete", call, carry, ok=frozenset({204, 404}), read=read)

    @staticmethod
    def _has_remaining(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return bool(Host._carry_of(event).remaining)

    @staticmethod
    def _answer_swept(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        swept = sandbox.SweptData(count=Host._carry_of(event).count)
        Host._answer(ctx, instance, sandbox.host.SweptEvent.with_data(swept))

    # Exec chains: create the exec instance, start it and demux its output, then read its exit code.

    @staticmethod
    def _argv(
        instance: "Host", purpose: Purpose
    ) -> tuple[tuple[str, ...], collections.abc.Mapping[str, str], str | None]:
        """The fixed argv, the environment and the working directory of ``purpose`` for the request in line."""

        request = instance._line[0]
        data = request.data
        if isinstance(data, sandbox.CreateData) and purpose == "verify":
            return mapping.LIMITS_ARGV, {}, None
        if isinstance(data, sandbox.HostRunData):
            staged = mapping.stdin_path(request.id or "")
            if purpose == "stage":
                return mapping.write_argv(staged, data.command.stdin or ""), {}, None
            if purpose == "check":
                return mapping.OOM_ARGV, {}, None
            argv = mapping.run_argv(data.command, stdin_path=staged if data.command.stdin is not None else None)
            return argv, data.command.env, data.command.cwd
        if isinstance(data, sandbox.HostWriteData):
            return mapping.write_argv(data.write.path, data.write.text), {}, None
        if isinstance(data, sandbox.HostReadData):
            return mapping.read_argv(data.read.path), {}, None
        raise ValueError("the request in line runs nothing")

    @staticmethod
    def _output_chars(instance: "Host") -> int:
        data = Host._head(instance)
        if isinstance(data, sandbox.HostRunData):
            return data.max_output_chars
        if isinstance(data, sandbox.HostReadData):
            return data.max_chars
        return sandbox.MIN_OUTPUT_CHARS

    @staticmethod
    def _exec_create(purpose: Purpose) -> _Activity:
        async def activity(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
            argv, env, cwd = Host._argv(instance, purpose)
            carry = Host._carry_of(event)
            # A machine being verified is not answered yet: its id is the one the create carried.
            machine = carry.container if purpose == "verify" else Host._machine(instance)
            call = _Call("POST", f"/containers/{machine}/exec", body=mapping.exec_body(argv, env=env, cwd=cwd))
            read = functools.partial(_identified, field="exec_id")
            await Host._exchange(ctx, instance, "exec_create", call, carry, ok=frozenset({201}), read=read)

        return activity

    @staticmethod
    def _exec_start(purpose: Purpose) -> _Activity:
        async def activity(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
            request_id = instance._line[0].id
            carry = Host._carry_of(event)
            limit = Host._output_chars(instance)
            demux = stream.Demux(limit=limit)
            started = time.monotonic()  # measures RanData.duration only; hsm.after owns every deadline
            answered: int | None = None
            kind: sandbox.FailureKind | None = None
            failed: httpx.TransportError | None = None
            too_large = False
            with span.operation(
                "bot.docker.exchange",
                scope=_SCOPE,
                component=_COMPONENT,
                stage="exec_start",
                attributes={_STEP_ATTR: "exec_start"},
            ) as active:
                try:
                    async with (
                        Host._client(instance) as client,
                        client.stream(
                            "POST", f"/exec/{carry.exec_id}/start", json={"Detach": False, "Tty": False}
                        ) as response,
                    ):
                        answered = response.status_code
                        if answered == 200:
                            async for chunk in response.aiter_bytes():
                                demux.feed(chunk)
                                if purpose == "read" and demux.stdout.total > mapping.MAX_READ_BYTES:
                                    too_large = True
                                    break
                            if not too_large:
                                demux.close()
                        else:
                            _ = await Host._capped(response)
                except httpx.TransportError as error:
                    kind = sandbox.FailureKind.UNAVAILABLE
                    failed = error
                except (stream.StreamError, _BodyTooLargeError, httpx.HTTPError, httpx.StreamError):
                    kind = sandbox.FailureKind.FAILED
                if kind is not None:
                    span.record_failure(active, kind.value)
            duration = datetime.timedelta(seconds=time.monotonic() - started)
            Host._count("exec_start", Host._label(answered, kind), started)
            status = answered if kind is None else None
            if status == 200 and not too_large:
                if purpose == "verify":
                    data = Host._head(instance)
                    applied = isinstance(data, sandbox.CreateData) and mapping.limits_applied(
                        demux.stdout.text(limit)[0], data.spec.resources
                    )
                    carry = carry.model_copy(update={"failure": None if applied else sandbox.FailureKind.UNSUPPORTED})
                elif purpose == "check":
                    carry = carry.model_copy(update={"oom_kills": mapping.oom_kills(demux.stdout.text(limit)[0])})
                else:
                    stdout, stdout_truncated = demux.stdout.text(limit)
                    stderr, stderr_truncated = demux.stderr.text(limit)
                    output = _Output(
                        stdout=stdout,
                        stderr=stderr,
                        stdout_truncated=stdout_truncated,
                        stderr_truncated=stderr_truncated,
                        stdout_bytes=demux.stdout.total,
                    )
                    carry = carry.model_copy(update={"output": output, "duration": duration})
            Host._complete(
                ctx,
                instance,
                "exec_start",
                request_id,
                carry,
                status=status,
                kind=kind,
                too_large=too_large,
                error=failed,
            )

        return activity

    @staticmethod
    async def _exec_inspect(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        carry = Host._carry_of(event)

        def read(status: int, body: bytes, carry: _Carry) -> _Carry:
            del status
            inspected = mapping.ExecInspected.model_validate_json(body)
            if inspected.running or inspected.exit_code is None:
                raise ValueError("the exec instance has no exit code yet")
            return carry.model_copy(update={"exit_code": inspected.exit_code})

        await Host._exchange(
            ctx, instance, "exec_inspect", _Call("GET", f"/exec/{carry.exec_id}/json"), carry, read=read
        )

    @staticmethod
    def _exited_with(*codes: int) -> _Guard:
        def guard(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
            del ctx
            return Host._answers(instance, event, "exec_inspect", (200,)) and Host._carry_of(event).exit_code in codes

        return guard

    @staticmethod
    def _applies_limits(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        """The new machine's cgroup applies the limits its spec asked for."""

        del ctx
        return Host._answers(instance, event, "exec_start", (200,)) and Host._carry_of(event).failure is None

    @staticmethod
    def _is_too_large(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            Host._answers(instance, event, "exec_start", (200,)) and isinstance(data, _ExchangedData) and data.too_large
        )

    @staticmethod
    def _ran(
        carry: _Carry, outcome: typing.Literal["exited", "timed_out", "out_of_memory", "killed"]
    ) -> sandbox.RanData:
        output = carry.output or _Output()
        return sandbox.RanData(
            outcome=outcome,
            exit_code=carry.exit_code if outcome == "exited" else None,
            stdout=output.stdout,
            stderr=output.stderr,
            stdout_truncated=output.stdout_truncated,
            stderr_truncated=output.stderr_truncated,
            duration=carry.duration or _MIN_DURATION,
        )

    @staticmethod
    def _answer_ran(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        carry = Host._carry_of(event)
        outcome: typing.Literal["exited", "timed_out"] = (
            "timed_out" if carry.exit_code in _TIMED_OUT_EXITS else "exited"
        )
        Host._answer(ctx, instance, sandbox.host.RanEvent.with_data(Host._ran(carry, outcome)))

    @staticmethod
    def _answer_checked(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        carry = Host._carry_of(event)
        outcome: typing.Literal["out_of_memory", "killed"] = "out_of_memory" if carry.oom_kills > 0 else "killed"
        Host._answer(ctx, instance, sandbox.host.RanEvent.with_data(Host._ran(carry, outcome)))

    @staticmethod
    def _answer_unstaged(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        Host._answer(ctx, instance, _failed(sandbox.FailureKind.FAILED, "The command's stdin could not be staged."))

    @staticmethod
    def _answer_written(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        data = Host._head(instance)
        assert isinstance(data, sandbox.HostWriteData)
        written = sandbox.WrittenData(path=data.write.path, bytes=len(data.write.text.encode()))
        Host._answer(ctx, instance, sandbox.host.WrittenEvent.with_data(written))

    @staticmethod
    def _answer_unwritten(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        Host._answer(ctx, instance, _failed(sandbox.FailureKind.FAILED, "The file could not be written."))

    @staticmethod
    def _answer_contents(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        data = Host._head(instance)
        assert isinstance(data, sandbox.HostReadData)
        output = Host._carry_of(event).output or _Output()
        contents = sandbox.ContentsData(
            path=data.read.path, text=output.stdout, truncated=output.stdout_truncated, bytes=output.stdout_bytes
        )
        Host._answer(ctx, instance, sandbox.host.ContentsEvent.with_data(contents))

    @staticmethod
    def _answer_not_found(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        Host._answer(ctx, instance, _failed(sandbox.FailureKind.NOT_FOUND, "No regular file at that path."))

    @staticmethod
    def _answer_too_large(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        Host._answer(
            ctx,
            instance,
            _failed(sandbox.FailureKind.TOO_LARGE, f"The file is larger than {mapping.MAX_READ_BYTES} bytes."),
        )

    @staticmethod
    def _answer_unread(ctx: hsm.Context, instance: "Host", event: hsm.Event[typing.Any]) -> None:
        del event
        Host._answer(ctx, instance, _failed(sandbox.FailureKind.FAILED, "The file could not be read."))

    # Topology builders: called while the class body runs, so they reference no ``Host`` attribute themselves.

    @staticmethod
    def _bounded(delay: _Delay, overran: _Effect, target: str = _NEXT) -> hsm.Element:
        """A leaf's own timer: after ``delay``, ``overran`` answers (or counts) and the host moves to ``target``."""

        return hsm.transition(hsm.after(delay), hsm.effect(overran), hsm.target(target))

    @staticmethod
    def _leaf(name: str, activity: _Activity, *transitions: hsm.Element) -> hsm.Element:
        """A leaf that makes one exchange; its last transition is its own timer (``_bounded``)."""

        return hsm.state(name, hsm.activity(activity), *transitions)

    @staticmethod
    def _chain(name: str, *elements: hsm.Element) -> hsm.Element:
        """An exec chain under ``serving``: ``creating``, ``starting`` and, unless it needs no exit code, ``inspecting``."""

        return hsm.state(name, hsm.initial(hsm.target(f"{_SERVING}/{name}/creating")), *elements)

    model: typing.ClassVar[hsm.Model] = define(
        "DockerSandboxHost",
        hsm.initial(hsm.target(_OPENING)),
        hsm.state(
            "opening",
            hsm.initial(hsm.target(f"{_OPENING}/pinging")),
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_refuse_unavailable)),
            hsm.transition(
                hsm.on(_ExchangedEvent), hsm.guard(_is_refusal), hsm.effect(_refuse_daemon), hsm.target(_DOWN)
            ),
            hsm.transition(
                hsm.on(_ExchangedEvent, _ExchangeFailedEvent),
                hsm.guard(_is_open_failure),
                hsm.effect(_open_failed),
                hsm.target(_DOWN),
            ),
            _leaf(
                "pinging",
                _ping,
                hsm.transition(
                    hsm.on(_ExchangedEvent), hsm.guard(_opening("ping", 200)), hsm.target(f"{_OPENING}/versioning")
                ),
                _bounded(_open_delay, _open_overran("ping"), _DOWN),
            ),
            _leaf(
                "versioning",
                _version,
                hsm.transition(
                    hsm.on(_ExchangedEvent), hsm.guard(_opening("version", 200)), hsm.target(f"{_OPENING}/inspecting")
                ),
                _bounded(_open_delay, _open_overran("version"), _DOWN),
            ),
            _leaf(
                "inspecting",
                _info,
                hsm.transition(
                    hsm.on(_ExchangedEvent), hsm.guard(_opening("info", 200)), hsm.target(f"{_OPENING}/networking")
                ),
                _bounded(_open_delay, _open_overran("info"), _DOWN),
            ),
            hsm.choice(
                "networking",
                hsm.transition(hsm.guard(_wants_network), hsm.target(f"{_OPENING}/network_inspecting")),
                hsm.transition(hsm.target(_OPEN)),
            ),
            _leaf(
                "network_inspecting",
                _network_inspect,
                hsm.transition(hsm.on(_ExchangedEvent), hsm.guard(_opening("network_inspect", 200)), hsm.target(_OPEN)),
                hsm.transition(
                    hsm.on(_ExchangedEvent),
                    hsm.guard(_opening("network_inspect", 404)),
                    hsm.target(f"{_OPENING}/network_creating"),
                ),
                _bounded(_open_delay, _open_overran("network_inspect"), _DOWN),
            ),
            _leaf(
                "network_creating",
                _network_create,
                hsm.transition(
                    hsm.on(_ExchangedEvent), hsm.guard(_opening("network_create", 201, 409)), hsm.target(_OPEN)
                ),
                _bounded(_open_delay, _open_overran("network_create"), _DOWN),
            ),
        ),
        hsm.state(
            "down",
            hsm.entry(_fell),
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_refuse_unavailable)),
            hsm.transition(hsm.after(_reopen_delay), hsm.target(_OPENING)),
        ),
        hsm.state(
            "open",
            hsm.entry(_opened),
            hsm.exit(_drain),
            hsm.initial(hsm.target(_NEXT)),
            # A request that arrives while another is served waits in line.
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_hold)),
            hsm.choice(
                "next",
                hsm.transition(hsm.guard(_names_bad_machine), hsm.effect(_answer_bad_machine), hsm.target(_NEXT)),
                hsm.transition(hsm.guard(_allows), hsm.target(f"{_SERVING}/posting")),
                hsm.transition(
                    hsm.guard(_next_is(sandbox.CreateData)), hsm.effect(_answer_unsupported), hsm.target(_NEXT)
                ),
                hsm.transition(hsm.guard(_has_stdin), hsm.target(f"{_SERVING}/staging")),
                hsm.transition(hsm.guard(_next_is(sandbox.HostRunData)), hsm.target(f"{_SERVING}/running")),
                hsm.transition(hsm.guard(_next_is(sandbox.HostWriteData)), hsm.target(f"{_SERVING}/writing")),
                hsm.transition(hsm.guard(_next_is(sandbox.HostReadData)), hsm.target(f"{_SERVING}/reading")),
                hsm.transition(hsm.guard(_next_is(sandbox.DestroyData)), hsm.target(f"{_SERVING}/destroying")),
                hsm.transition(hsm.guard(_next_is(sandbox.SweepData)), hsm.target(f"{_SERVING}/listing")),
                hsm.transition(hsm.target(_IDLE)),
            ),
            hsm.state("idle", hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_take), hsm.target(_NEXT))),
            hsm.state(
                "serving",
                # What a leaf does not take itself: a lost daemon, a broken answer, or a status that is not its success.
                hsm.transition(hsm.on(_ExchangeFailedEvent), hsm.guard(_is_lost), hsm.effect(_lose), hsm.target(_DOWN)),
                hsm.transition(
                    hsm.on(_ExchangeFailedEvent), hsm.guard(_is_broken), hsm.effect(_answer_broken), hsm.target(_NEXT)
                ),
                hsm.transition(
                    hsm.on(_ExchangedEvent), hsm.guard(_is_answered), hsm.effect(_answer_status), hsm.target(_NEXT)
                ),
                _leaf(
                    "posting",
                    _post,
                    hsm.transition(
                        hsm.on(_ExchangedEvent), hsm.guard(_done("create", 201)), hsm.target(f"{_SERVING}/starting")
                    ),
                    _bounded(_step_delay, _overran("create")),
                ),
                _leaf(
                    "starting",
                    _start,
                    hsm.transition(
                        hsm.on(_ExchangedEvent),
                        hsm.guard(_done("start", 204, 304)),
                        hsm.target(f"{_SERVING}/verifying"),
                    ),
                    hsm.transition(
                        hsm.on(_ExchangedEvent), hsm.guard(_done("start")), hsm.target(f"{_SERVING}/cleaning")
                    ),
                    hsm.transition(
                        hsm.on(_ExchangeFailedEvent), hsm.guard(_is_broken), hsm.target(f"{_SERVING}/cleaning")
                    ),
                    _bounded(_step_delay, _overran("start")),
                ),
                _chain(
                    "verifying",
                    # A machine whose limits are not applied, or that could not be checked, is removed.
                    hsm.transition(
                        hsm.on(_ExchangedEvent), hsm.guard(_is_answered), hsm.target(f"{_SERVING}/cleaning")
                    ),
                    hsm.transition(
                        hsm.on(_ExchangeFailedEvent), hsm.guard(_is_broken), hsm.target(f"{_SERVING}/cleaning")
                    ),
                    _leaf(
                        "creating",
                        _exec_create("verify"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_create", 201)),
                            hsm.target(f"{_SERVING}/verifying/starting"),
                        ),
                        _bounded(_step_delay, _overran("exec_create")),
                    ),
                    _leaf(
                        "starting",
                        _exec_start("verify"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_applies_limits),
                            hsm.effect(_answer_created),
                            hsm.target(_NEXT),
                        ),
                        _bounded(_step_delay, _overran("exec_start")),
                    ),
                ),
                _leaf(
                    "cleaning",
                    _clean,
                    hsm.transition(
                        hsm.on(_ExchangedEvent),
                        hsm.guard(_done("clean")),
                        hsm.effect(_answer_cleaned),
                        hsm.target(_NEXT),
                    ),
                    _bounded(_step_delay, _overran("clean")),
                ),
                _chain(
                    "staging",
                    _leaf(
                        "creating",
                        _exec_create("stage"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_create", 201)),
                            hsm.target(f"{_SERVING}/staging/starting"),
                        ),
                        _bounded(_step_delay, _overran("exec_create")),
                    ),
                    _leaf(
                        "starting",
                        _exec_start("stage"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_start", 200)),
                            hsm.target(f"{_SERVING}/staging/inspecting"),
                        ),
                        _bounded(_step_delay, _overran("exec_start")),
                    ),
                    _leaf(
                        "inspecting",
                        _exec_inspect,
                        hsm.transition(
                            hsm.on(_ExchangedEvent), hsm.guard(_exited_with(0)), hsm.target(f"{_SERVING}/running")
                        ),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_inspect", 200)),
                            hsm.effect(_answer_unstaged),
                            hsm.target(_NEXT),
                        ),
                        _bounded(_step_delay, _overran("exec_inspect")),
                    ),
                ),
                _chain(
                    "running",
                    _leaf(
                        "creating",
                        _exec_create("run"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_create", 201)),
                            hsm.target(f"{_SERVING}/running/starting"),
                        ),
                        _bounded(_step_delay, _overran("exec_create")),
                    ),
                    # The run's start says, on the host's clock, whether it ended within its timeout or past it,
                    # and hands that to the inspecting leaf of the same name.
                    hsm.state(
                        "starting",
                        hsm.activity(_exec_start("run")),
                        hsm.initial(hsm.target(f"{_SERVING}/running/starting/within")),
                        hsm.state(
                            "within",
                            hsm.transition(
                                hsm.on(_ExchangedEvent),
                                hsm.guard(_done("exec_start", 200)),
                                hsm.target(f"{_SERVING}/running/inspecting/within"),
                            ),
                            hsm.transition(hsm.after(_run_timeout), hsm.target(f"{_SERVING}/running/starting/past")),
                        ),
                        hsm.state(
                            "past",
                            hsm.transition(
                                hsm.on(_ExchangedEvent),
                                hsm.guard(_done("exec_start", 200)),
                                hsm.target(f"{_SERVING}/running/inspecting/past"),
                            ),
                        ),
                        # The run's own bound: its timeout, plus the grace for timeout's kill to land.
                        _bounded(_run_delay, _overran("exec_start")),
                    ),
                    hsm.state(
                        "inspecting",
                        hsm.activity(_exec_inspect),
                        hsm.initial(hsm.target(f"{_SERVING}/running/inspecting/past")),
                        # Killed within its timeout: the memory limit may have killed it.
                        hsm.state(
                            "within",
                            hsm.transition(
                                hsm.on(_ExchangedEvent),
                                hsm.guard(_exited_with(_KILLED_EXIT)),
                                hsm.target(f"{_SERVING}/checking"),
                            ),
                        ),
                        hsm.state("past"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_inspect", 200)),
                            hsm.effect(_answer_ran),
                            hsm.target(_NEXT),
                        ),
                        _bounded(_step_delay, _overran("exec_inspect")),
                    ),
                ),
                _chain(
                    "checking",
                    _leaf(
                        "creating",
                        _exec_create("check"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_create", 201)),
                            hsm.target(f"{_SERVING}/checking/starting"),
                        ),
                        _bounded(_step_delay, _overran("exec_create")),
                    ),
                    _leaf(
                        "starting",
                        _exec_start("check"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_start", 200)),
                            hsm.effect(_answer_checked),
                            hsm.target(_NEXT),
                        ),
                        _bounded(_step_delay, _overran("exec_start")),
                    ),
                ),
                _chain(
                    "writing",
                    _leaf(
                        "creating",
                        _exec_create("write"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_create", 201)),
                            hsm.target(f"{_SERVING}/writing/starting"),
                        ),
                        _bounded(_step_delay, _overran("exec_create")),
                    ),
                    _leaf(
                        "starting",
                        _exec_start("write"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_start", 200)),
                            hsm.target(f"{_SERVING}/writing/inspecting"),
                        ),
                        _bounded(_step_delay, _overran("exec_start")),
                    ),
                    _leaf(
                        "inspecting",
                        _exec_inspect,
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_exited_with(0)),
                            hsm.effect(_answer_written),
                            hsm.target(_NEXT),
                        ),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_inspect", 200)),
                            hsm.effect(_answer_unwritten),
                            hsm.target(_NEXT),
                        ),
                        _bounded(_step_delay, _overran("exec_inspect")),
                    ),
                ),
                _chain(
                    "reading",
                    _leaf(
                        "creating",
                        _exec_create("read"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_create", 201)),
                            hsm.target(f"{_SERVING}/reading/starting"),
                        ),
                        _bounded(_step_delay, _overran("exec_create")),
                    ),
                    _leaf(
                        "starting",
                        _exec_start("read"),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_is_too_large),
                            hsm.effect(_answer_too_large),
                            hsm.target(_NEXT),
                        ),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_start", 200)),
                            hsm.target(f"{_SERVING}/reading/inspecting"),
                        ),
                        _bounded(_step_delay, _overran("exec_start")),
                    ),
                    _leaf(
                        "inspecting",
                        _exec_inspect,
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_exited_with(0)),
                            hsm.effect(_answer_contents),
                            hsm.target(_NEXT),
                        ),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_exited_with(mapping.NOT_FOUND_EXIT)),
                            hsm.effect(_answer_not_found),
                            hsm.target(_NEXT),
                        ),
                        hsm.transition(
                            hsm.on(_ExchangedEvent),
                            hsm.guard(_done("exec_inspect", 200)),
                            hsm.effect(_answer_unread),
                            hsm.target(_NEXT),
                        ),
                        _bounded(_step_delay, _overran("exec_inspect")),
                    ),
                ),
                _leaf(
                    "destroying",
                    _destroy,
                    hsm.transition(
                        hsm.on(_ExchangedEvent),
                        hsm.guard(_done("delete", 204, 404)),
                        hsm.effect(_answer_destroyed),
                        hsm.target(_NEXT),
                    ),
                    _bounded(_step_delay, _overran("delete")),
                ),
                _leaf(
                    "listing",
                    _list,
                    hsm.transition(
                        hsm.on(_ExchangedEvent), hsm.guard(_done("list", 200)), hsm.target(f"{_SERVING}/deleting")
                    ),
                    _bounded(_step_delay, _overran("list")),
                ),
                hsm.choice(
                    "deleting",
                    hsm.transition(hsm.guard(_has_remaining), hsm.target(f"{_SERVING}/removing")),
                    hsm.transition(hsm.effect(_answer_swept), hsm.target(_NEXT)),
                ),
                _leaf(
                    "removing",
                    _remove,
                    hsm.transition(
                        hsm.on(_ExchangedEvent),
                        hsm.guard(_done("delete", 204, 404)),
                        hsm.target(f"{_SERVING}/deleting"),
                    ),
                    _bounded(_step_delay, _overran("delete")),
                ),
            ),
        ),
    )


__all__ = ["Host", "Purpose", "Step"]

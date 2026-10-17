"""The sandbox ability: a bench where the bot runs, writes and reads, on a machine a host keeps for it.

The bot chooses what to run, what to write and read, and when to start over (``run``, ``write``, ``read``,
``reset``); nothing here runs tests after a write, retries a failing build or resets after a failure. The owner
fixes the bench (``spec.Spec``) and declares what the host can offer (``spec.Capacity``); a bench beyond it is
refused at construction.

The ability owns an unstarted sandbox host HSM (any HSM that answers the ``host`` event protocol). It starts the
host under its own lifetime context when it is attached and stops it after releasing; those are the only direct
operations on the host. Everything else is a ``host`` request event, sent with ``ability.run_terminal_operation``
from an activity and answered by exactly one terminal.

Topology (``Sandbox`` behavior inside the composite ability lifecycle)::

    initializing    attach the (empty) composite group                         -> starting_host
    starting_host   owner start of the host; host serving -> provisioning ; host down -> unavailable
                                                      after(host_start_timeout) -> unavailable
    provisioning    sweep, then create the machine    after(create_timeout)     -> unavailable
    unavailable     every held request answered; host serving -> provisioning
                                                      after(backoff)            -> starting_host
    serving         a machine is ready                after(max_lifetime)       -> aged (served once idle)
      next          (choice) a reset first in line -> renewing ; a request in line -> working ; else ready
      ready         run/write/read/reset joins the line; aged -> renewing
      working       the first request in line, on the host   after(backstop) -> renewing
    renewing        destroy the machine, then create one      after(destroy + create) -> unavailable
                    (a failed or cut renewal loses the machine; the next one is told fresh)
    provisioning, serving and renewing: host down -> unavailable (the machine is lost; the next one is told fresh)
    detaching       every held request answered unavailable
      releasing     destroy the machine (within destroy_timeout), then owner stop of the host -> unlinking
      unlinking     detach the composite group; the owner is then told it is detached

The host announces ``serving`` and ``down`` to this ability, the machine that started it (``host.announce``), and
the ability keeps what it last heard as its own state. It makes a machine only on a host that announced
``serving``, so a boot never asks a host that is still opening. On ``down`` it is unavailable at once and resumes
on the next ``serving`` with no backoff in between; its own retry (``host_start_timeout``, then a backoff from
``retry_initial`` doubling to ``retry_max``) still bounds a host that never serves, and a create the host refuses.

Tools are offered only from ``ready`` and ``working``, so a turn's tool menu comes from live topology. Elsewhere
the four requests are not offered, yet one already selected is never dropped: ``starting_host``,
``provisioning`` and ``renewing`` hold it until the bench is ready (``hsm.defer``), and ``unavailable``,
``detaching`` and ``degraded`` answer it ``failed{unavailable}`` at once, selected by its typed payload. A
detached sandbox answers one at the request's source (``detached_requests``).

Every request gets exactly one terminal. Each observation (``ran``, ``written``, ``contents``, ``fresh``,
``failed``) reaches the body as the stimulus of a ``cognition.InputEvent`` product under a fresh id, as
Seeing's look does; the stimulus carries the request's ``id``. A machine renewed after an overrun or for age
is also told: ``fresh``, under an id of its own. Overruns renew the machine because a host cannot kill one
command: the host's own bound answers ``failed{timeout}``, and when the host does not answer at all, the
ability's backstop (``timeout + kill_grace + reply_grace``) does the same. Renewal is mechanical recovery of the
bench, not a choice made for the bot. A machine lost to a host ``down`` is told the same way: the next machine
the bench makes is observed ``fresh``, under an id of its own, so the bot never goes on against a lost workspace
unaware.

The ability never reads ``runtime.Secrets`` or the process environment.

Telemetry (scope ``bot.abilities.sandbox``; every attribute value is a closed set from code):

* ``bot.ability.sandbox.request.count`` — ``request`` {run, write, read, reset} × ``bot.outcome`` {ok, failed}
  × ``bot.failure.kind`` (``none`` when ok, else the closed ``host.FailureKind``): at most 36 series.
* ``bot.ability.sandbox.run.count`` — ``outcome`` {exited_zero, exited_nonzero, timed_out, out_of_memory,
  killed}: 5 series.
* ``bot.ability.sandbox.run.duration`` — seconds a command ran, by the same ``outcome``: 5 series.
* ``bot.ability.sandbox.output.truncated.count`` — ``stream`` {stdout, stderr, file}: 3 series.
* ``bot.ability.sandbox.machine.count`` — ``event`` {swept, sweep_failed, created, create_failed, destroyed,
  destroy_failed, destroy_timeout, lost}: 8 series.
* ``bot.ability.sandbox.machine.live`` — machines this ability holds (0 or 1), no attributes: 1 series.
* ``bot.ability.sandbox.queue.depth`` — requests held in line, no attributes: 1 series.
* Spans ``bot.sandbox.sweep``, ``create``, ``destroy``, ``run``, ``write`` and ``read``, one per host request,
  carry the constant component and stage and, on failure, the closed ``bot.failure.kind``.

Never used as attributes, logs or captured payloads: argv, paths, output, file text, environment values, image
references, machine ids, request ids or exit codes.
"""

import asyncio
import collections
import collections.abc
import dataclasses
import datetime
import logging
import typing
import uuid

import hsm
import mosfet
import pydantic
from opentelemetry import metrics
from pydantic.json_schema import SkipJsonSchema

from mosfet import event, lifecycle
from mosfet.abilities import cognition
from mosfet.protocols import attachment
from mosfet.start import bound_model
from mosfet.telemetry import span

from .. import ability
from . import command, host

# The constructor's ``spec`` parameter shadows the module inside ``__init__``.
from .spec import Capacity, Spec

Request = typing.Literal["run", "write", "read", "reset"]
"""Which of the bot's four requests a failure answers."""

_SCOPE: typing.Final = "bot.abilities.sandbox"
_COMPONENT: typing.Final = "sandbox"
_MAX_BACKOFF_DOUBLINGS: typing.Final = 30
_MESSAGE_CHARS: typing.Final = 1024
_MIN_DURATION: typing.Final = datetime.timedelta(0)
_DEFAULT_MAX_RUN_TIMEOUT: typing.Final = datetime.timedelta(minutes=10)
_DEFAULT_KILL_GRACE: typing.Final = datetime.timedelta(seconds=10)
_DEFAULT_REPLY_GRACE: typing.Final = datetime.timedelta(seconds=5)
_DEFAULT_CREATE_TIMEOUT: typing.Final = datetime.timedelta(seconds=120)
_DEFAULT_DESTROY_TIMEOUT: typing.Final = datetime.timedelta(seconds=30)
_DEFAULT_IO_TIMEOUT: typing.Final = datetime.timedelta(seconds=30)
_DEFAULT_HOST_START_TIMEOUT: typing.Final = datetime.timedelta(seconds=30)
_DEFAULT_RETRY_INITIAL: typing.Final = datetime.timedelta(seconds=5)
_DEFAULT_RETRY_MAX: typing.Final = datetime.timedelta(minutes=5)

_LOGGER = logging.getLogger(__name__)

_METER = metrics.get_meter(_SCOPE)
_REQUEST_COUNT = _METER.create_counter(
    "bot.ability.sandbox.request.count",
    unit="{request}",
    description="Sandbox requests answered, by request, outcome and failure kind.",
)
_RUNS = _METER.create_counter(
    "bot.ability.sandbox.run.count",
    unit="{run}",
    description="Commands that finished in a sandbox machine, by how they ended.",
)
_RUN_DURATION = _METER.create_histogram(
    "bot.ability.sandbox.run.duration",
    unit="s",
    description="How long a command ran in a sandbox machine, by how it ended.",
)
_TRUNCATED = _METER.create_counter(
    "bot.ability.sandbox.output.truncated.count",
    unit="{stream}",
    description="Output streams and file reads whose middle was left out to fit the bound, by stream.",
)
_MACHINES = _METER.create_counter(
    "bot.ability.sandbox.machine.count",
    unit="{event}",
    description="Sandbox machine lifecycle events: sweeps, creates, destroys and losses.",
)
_LIVE = _METER.create_up_down_counter(
    "bot.ability.sandbox.machine.live",
    unit="{machine}",
    description="Sandbox machines a sandbox ability holds.",
)
_QUEUE_DEPTH = _METER.create_up_down_counter(
    "bot.ability.sandbox.queue.depth",
    unit="{request}",
    description="Sandbox requests held in line.",
)


class FailureData(ability.FailureData):
    """A request to the sandbox did not do what was asked; ``kind`` says why and ``request`` which one it was."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "A request to the sandbox did not do what was asked; kind says why and request which one.",
            "examples": [
                {"kind": "busy", "message": "The sandbox is already holding 4 requests.", "request": "run"},
                {"kind": "timeout", "message": "The command overran; the machine is being replaced.", "request": "run"},
                {"kind": "not_found", "message": "No regular file at that path.", "request": "read"},
            ],
        },
    )

    kind: host.FailureKind = pydantic.Field(
        description=(
            "Why: 'invalid_input' (a path outside the workspace and /tmp, or a timeout beyond the sandbox's "
            "ceiling), 'unavailable' (no machine right now), 'unsupported', 'not_found', 'too_large', 'timeout' "
            "(the command overran and the machine is replaced, losing the workspace), 'failed', or 'busy' "
            "(too many requests waiting)."
        ),
        examples=["busy"],
    )
    message: str = pydantic.Field(
        max_length=_MESSAGE_CHARS,
        description="What happened, for the bot; never argv, an environment value, file content or output.",
        examples=["The sandbox is already holding 4 requests."],
    )
    request: Request = pydantic.Field(
        description="Which request this answers: 'run', 'write', 'read' or 'reset'.",
        examples=["run"],
    )


RunEvent = hsm.Event[command.RunData](name="bot.ability.sandbox.run", kind=event.EventKind, schema=command.RunData)
WriteEvent = hsm.Event[command.WriteData](
    name="bot.ability.sandbox.write", kind=event.EventKind, schema=command.WriteData
)
ReadEvent = hsm.Event[command.ReadData](name="bot.ability.sandbox.read", kind=event.EventKind, schema=command.ReadData)
ResetEvent = hsm.Event[command.ResetData](
    name="bot.ability.sandbox.reset", kind=event.EventKind, schema=command.ResetData
)
RanEvent = hsm.Event[command.RanData](name="bot.ability.sandbox.ran", schema=command.RanData)
WrittenEvent = hsm.Event[command.WrittenData](name="bot.ability.sandbox.written", schema=command.WrittenData)
ContentsEvent = hsm.Event[command.ContentsData](name="bot.ability.sandbox.contents", schema=command.ContentsData)
FreshEvent = hsm.Event[command.FreshData](name="bot.ability.sandbox.fresh", schema=command.FreshData)
FailedEvent = hsm.Event[FailureData](name="bot.ability.sandbox.failed", schema=FailureData)

_REQUEST_EVENTS: typing.Final = (RunEvent, WriteEvent, ReadEvent, ResetEvent)
_REQUEST_DATA: typing.Final = (command.RunData, command.WriteData, command.ReadData, command.ResetData)
_KINDS: typing.Final[dict[type[pydantic.BaseModel], Request]] = {
    command.RunData: "run",
    command.WriteData: "write",
    command.ReadData: "read",
    command.ResetData: "reset",
}


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class _Pending:
    """One request the bot made, held in line until it is answered. Compared by identity."""

    kind: Request
    request: hsm.Event[typing.Any]


class _MachineData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    machine: host.Machine


class _AnsweredData(pydantic.BaseModel):
    """The host's terminal for the request first in line (or a failure the ability stood in for)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True, frozen=True, extra="forbid"
    )

    pending: SkipJsonSchema[pydantic.SkipValidation[_Pending]] = pydantic.Field(exclude=True)
    terminal: SkipJsonSchema[pydantic.SkipValidation[hsm.Event[typing.Any]]] = pydantic.Field(exclude=True)


class _ReleasedData(pydantic.BaseModel):
    """The machine is destroyed (or left to the next sweep) and the host stopped, for this detach request."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True, frozen=True, extra="forbid"
    )

    detach: SkipJsonSchema[pydantic.SkipValidation[hsm.Event[typing.Any]]] = pydantic.Field(exclude=True)


class _NoData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")


_HostStartedEvent = hsm.Event[_NoData](
    name="bot.ability.sandbox.host_started", kind=hsm.CompletionEventKind, schema=_NoData
)
_HostStartFailedEvent = hsm.Event[host.HostFailureData](
    name="bot.ability.sandbox.host_start_failed", kind=hsm.ErrorEventKind, schema=host.HostFailureData
)
_MachineReadyEvent = hsm.Event[_MachineData](
    name="bot.ability.sandbox.machine_ready", kind=hsm.CompletionEventKind, schema=_MachineData
)
_MachineFailedEvent = hsm.Event[host.HostFailureData](
    name="bot.ability.sandbox.machine_failed", kind=hsm.ErrorEventKind, schema=host.HostFailureData
)
_AnsweredEvent = hsm.Event[_AnsweredData](
    name="bot.ability.sandbox.answered", kind=hsm.CompletionEventKind, schema=_AnsweredData
)
_AgedEvent = hsm.Event[_NoData](name="bot.ability.sandbox.aged", schema=_NoData)
_ReleasedEvent = hsm.Event[_ReleasedData](
    name="bot.ability.sandbox.released", kind=hsm.CompletionEventKind, schema=_ReleasedData
)


def _host_failure(kind: host.FailureKind, message: str) -> hsm.Event[host.HostFailureData]:
    return host.FailedEvent.with_data(host.HostFailureData(kind=kind, message=message))


def _failure_of(terminal: hsm.Event[typing.Any]) -> host.HostFailureData | None:
    data = terminal.data
    return data if isinstance(data, host.HostFailureData) else None


def _run_outcome(ran: command.RanData) -> str:
    if ran.outcome == "exited":
        return "exited_zero" if ran.exit_code == 0 else "exited_nonzero"
    return ran.outcome


def _positive(**durations: datetime.timedelta) -> None:
    for name, duration in durations.items():
        if duration <= _MIN_DURATION:
            raise ValueError(f"Sandbox {name} must be positive.")


class Sandbox(ability.Ability[command.RunData, cognition.InputData]):
    """A bench where the bot runs commands and moves whole files, on a machine an owned host keeps for it.

    ``run``, ``write``, ``read`` and ``reset`` are the model-callable requests; each is answered once, as the
    stimulus of the cognition input this ability hands its owner. Requests are served one at a time, first in
    first out; at most ``queue_limit`` are held at once (the one being served included), and the rest are
    answered ``busy``.
    """

    input_event: typing.ClassVar[hsm.Event[command.RunData]] = RunEvent
    output_event: typing.ClassVar[hsm.Event[cognition.InputData]] = cognition.InputEvent
    failed_event: typing.ClassVar[hsm.Event[FailureData]] = FailedEvent
    effort: typing.ClassVar[ability.Effort] = ability.Effort.M
    detached_requests: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = _REQUEST_EVENTS
    # The composite form gives the bench its own detaching state, so it can destroy its machine and stop its host
    # before the owner is told it is detached.
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True

    _attachment_group: attachment.Group | None
    _host: hsm.Instance
    _host_model: hsm.Model
    _capacity: Capacity
    _spec: Spec
    _max_run_timeout: datetime.timedelta
    _kill_grace: datetime.timedelta
    _reply_grace: datetime.timedelta
    _create_timeout: datetime.timedelta
    _destroy_timeout: datetime.timedelta
    _io_timeout: datetime.timedelta
    _host_start_timeout: datetime.timedelta
    _retry_initial: datetime.timedelta
    _retry_max: datetime.timedelta
    _queue_limit: int
    _max_output_chars: int
    # Durable machine data: the machine this bench holds, the requests in line (the first is being served), how
    # many times in a row the bench has been unavailable, whether the host last announced serving, and whether a
    # machine was lost to a host down that the bot is still to be told of.
    _machine: host.Machine | None
    _line: collections.deque[_Pending]
    _retries: int
    _host_serving: bool
    _lost_untold: bool

    def __init__(
        self,
        *,
        host: hsm.Instance,
        capacity: Capacity,
        spec: Spec,
        max_run_timeout: datetime.timedelta = _DEFAULT_MAX_RUN_TIMEOUT,
        kill_grace: datetime.timedelta = _DEFAULT_KILL_GRACE,
        reply_grace: datetime.timedelta = _DEFAULT_REPLY_GRACE,
        create_timeout: datetime.timedelta = _DEFAULT_CREATE_TIMEOUT,
        destroy_timeout: datetime.timedelta = _DEFAULT_DESTROY_TIMEOUT,
        io_timeout: datetime.timedelta = _DEFAULT_IO_TIMEOUT,
        host_start_timeout: datetime.timedelta = _DEFAULT_HOST_START_TIMEOUT,
        retry_initial: datetime.timedelta = _DEFAULT_RETRY_INITIAL,
        retry_max: datetime.timedelta = _DEFAULT_RETRY_MAX,
        queue_limit: int = 4,
        max_output_chars: int = 32_768,
    ) -> None:
        """Keep a bench from ``spec`` on ``host``, an unstarted sandbox host this ability owns from here on.

        ``capacity`` is the value the factory declared for that host (and injected into it as well).
        ``max_run_timeout`` is the longest timeout a ``run`` may ask for. A run is backstopped at its timeout plus
        ``kill_grace`` (the host's own bound) plus ``reply_grace``; a write or read at ``io_timeout``. Making a
        machine is bounded by ``create_timeout``, destroying one by ``destroy_timeout``, and starting the host by
        ``host_start_timeout``, which also bounds the wait for the host to announce ``serving``. An unavailable bench
        resumes when the host announces ``serving``, and is otherwise retried after ``retry_initial``, doubling up to
        ``retry_max``. ``max_output_chars`` bounds each output stream and file read.

        Raises:
            TypeError: If ``host`` declares no lifecycle model.
            ValueError: If ``spec`` asks for a network or limit beyond ``capacity``, ``host`` is already started, a
                duration is not positive, ``max_run_timeout`` exceeds ``command.MAX_RUN_TIMEOUT``, ``retry_max``
                is shorter than ``retry_initial``, ``queue_limit`` is below 1, or ``max_output_chars`` lies
                outside ``command.MIN_OUTPUT_CHARS``..``command.MAX_OUTPUT_CHARS``.
        """

        model = bound_model(host)
        if model is None:
            raise TypeError(f"{type(host).__name__} declares no lifecycle model; a sandbox host must declare one.")
        if lifecycle.is_started(host):
            raise ValueError("Sandbox host must be unstarted: the sandbox starts and stops the host it owns.")
        capacity.check(spec)
        _positive(
            max_run_timeout=max_run_timeout,
            kill_grace=kill_grace,
            reply_grace=reply_grace,
            create_timeout=create_timeout,
            destroy_timeout=destroy_timeout,
            io_timeout=io_timeout,
            host_start_timeout=host_start_timeout,
            retry_initial=retry_initial,
        )
        if max_run_timeout > command.MAX_RUN_TIMEOUT:
            raise ValueError(f"Sandbox max_run_timeout must be at most {command.MAX_RUN_TIMEOUT}.")
        if retry_max < retry_initial:
            raise ValueError("Sandbox retry_max cannot be shorter than retry_initial.")
        if queue_limit < 1:
            raise ValueError("Sandbox queue_limit must be at least 1.")
        if not command.MIN_OUTPUT_CHARS <= max_output_chars <= command.MAX_OUTPUT_CHARS:
            raise ValueError(
                f"Sandbox max_output_chars must lie in {command.MIN_OUTPUT_CHARS}..{command.MAX_OUTPUT_CHARS}."
            )
        super().__init__()
        self._host = host
        self._host_model = model
        self._capacity = capacity
        self._spec = spec
        self._max_run_timeout = max_run_timeout
        self._kill_grace = kill_grace
        self._reply_grace = reply_grace
        self._create_timeout = create_timeout
        self._destroy_timeout = destroy_timeout
        self._io_timeout = io_timeout
        self._host_start_timeout = host_start_timeout
        self._retry_initial = retry_initial
        self._retry_max = retry_max
        self._queue_limit = queue_limit
        self._max_output_chars = max_output_chars
        self._machine = None
        self._line = collections.deque()
        self._retries = 0
        self._host_serving = False
        self._lost_untold = False
        self._attachment_group = attachment.Group()

    @classmethod
    @typing.override
    def detached_failure(cls, event: hsm.Event[typing.Any]) -> pydantic.BaseModel:
        data = event.data
        kind = _KINDS.get(type(data), "run") if isinstance(data, pydantic.BaseModel) else "run"
        return FailureData(
            kind=host.FailureKind.UNAVAILABLE, message="The sandbox is not attached; it has no machine.", request=kind
        )

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop this ability, then the host it started, when releasing did not already stop it."""

        await super().stop(ctx)
        if lifecycle.is_started(self._host):
            await hsm.stop(self._host, ctx)

    # Answering the bot.

    @staticmethod
    def _observe(
        ctx: hsm.Context,
        instance: "Sandbox",
        observation: hsm.Event[typing.Any],
        request: hsm.Event[typing.Any] | None,
    ) -> None:
        """Hand ``observation`` to the owner as the stimulus of a cognition input under a fresh id.

        The stimulus carries the id of the ``request`` it answers, or an id of its own when it answers none.
        """

        stimulus = dataclasses.replace(
            observation,
            id=uuid.uuid4().hex if request is None else request.id,
            source=hsm.id(instance),
            target=None,
            metadata={} if request is None else dict(request.metadata),
        )
        handoff = dataclasses.replace(
            cognition.InputEvent.with_data_and_id(cognition.InputData(stimulus=stimulus), uuid.uuid4().hex),
            source=hsm.id(instance),
            target=None,
            metadata=dict(stimulus.metadata),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(handoff))

    @staticmethod
    def _answer(ctx: hsm.Context, instance: "Sandbox", pending: _Pending, observation: hsm.Event[typing.Any]) -> None:
        """Answer ``pending`` with ``observation``, its one terminal, and count it."""

        failure = observation.data if isinstance(observation.data, FailureData) else None
        _REQUEST_COUNT.add(
            1,
            {
                "request": pending.kind,
                "bot.outcome": "ok" if failure is None else "failed",
                "bot.failure.kind": "none" if failure is None else failure.kind.value,
            },
        )
        Sandbox._observe(ctx, instance, observation, pending.request)

    @staticmethod
    def _fail(ctx: hsm.Context, instance: "Sandbox", pending: _Pending, kind: host.FailureKind, message: str) -> None:
        failure = FailureData(kind=kind, message=message[:_MESSAGE_CHARS], request=pending.kind)
        Sandbox._answer(ctx, instance, pending, FailedEvent.with_data(failure))

    @staticmethod
    def _pending(event: hsm.Event[typing.Any]) -> _Pending:
        data = event.data
        assert isinstance(data, _REQUEST_DATA)
        return _Pending(kind=_KINDS[type(data)], request=event)

    @staticmethod
    def _refusal(instance: "Sandbox", data: object) -> tuple[host.FailureKind, str] | None:
        """Why the bench cannot take this request at all, before anything is asked of the host."""

        bench = instance._spec
        outside = f"The path must lie under {bench.workspace} or /tmp."
        if isinstance(data, command.RunData):
            if data.cwd is not None and not bench.admits(data.cwd):
                return host.FailureKind.INVALID_INPUT, f"cwd: {outside}"
            if data.timeout > instance._max_run_timeout:
                return host.FailureKind.INVALID_INPUT, f"The timeout may be at most {instance._max_run_timeout}."
            return None
        if isinstance(data, command.WriteData | command.ReadData):
            if not instance._capacity.files:
                return host.FailureKind.UNSUPPORTED, "This sandbox's host does not move files."
            if not bench.admits(data.path):
                return host.FailureKind.INVALID_INPUT, outside
        return None

    @staticmethod
    def _drain(ctx: hsm.Context, instance: "Sandbox", message: str) -> None:
        """Answer every request in line ``failed{unavailable}``."""

        while instance._line:
            pending = instance._line.popleft()
            _QUEUE_DEPTH.add(-1)
            Sandbox._fail(ctx, instance, pending, host.FailureKind.UNAVAILABLE, message)

    # Guards.

    @staticmethod
    def _is_request(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _REQUEST_DATA)

    @staticmethod
    def _resets_next(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return bool(instance._line) and instance._line[0].kind == "reset"

    @staticmethod
    def _has_line(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return bool(instance._line)

    @staticmethod
    def _host_is_serving(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._host_serving

    @staticmethod
    def _answers_first(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> bool:
        """The host's terminal belongs to the request still first in line (late or stale ones do not)."""

        del ctx
        data = event.data
        return isinstance(data, _AnsweredData) and bool(instance._line) and instance._line[0] is data.pending

    @staticmethod
    def _answered_ok(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        return (
            Sandbox._answers_first(ctx, instance, event)
            and isinstance(data, _AnsweredData)
            and _failure_of(data.terminal) is None
        )

    @staticmethod
    def _answered_with(
        kind: host.FailureKind,
    ) -> collections.abc.Callable[[hsm.Context, "Sandbox", hsm.Event[typing.Any]], bool]:
        def guard(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> bool:
            data = event.data
            if not Sandbox._answers_first(ctx, instance, event) or not isinstance(data, _AnsweredData):
                return False
            failure = _failure_of(data.terminal)
            return failure is not None and failure.kind is kind

        return guard

    # Delays.

    @staticmethod
    def _host_start_delay(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._host_start_timeout

    @staticmethod
    def _create_delay(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._create_timeout

    @staticmethod
    def _renew_delay(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._destroy_timeout + instance._create_timeout

    @staticmethod
    def _retry_delay(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        """``retry_initial`` doubled for every unavailable in a row before this one, never past ``retry_max``."""

        del ctx, event
        doublings = min(max(instance._retries - 1, 0), _MAX_BACKOFF_DOUBLINGS)
        return min(instance._retry_initial * (1 << doublings), instance._retry_max)

    @staticmethod
    def _lifetime_delay(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._spec.max_lifetime

    @staticmethod
    def _backstop(instance: "Sandbox", data: object) -> datetime.timedelta:
        if isinstance(data, command.RunData):
            return data.timeout + instance._kill_grace + instance._reply_grace
        return instance._io_timeout

    @staticmethod
    def _work_delay(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return Sandbox._backstop(instance, instance._line[0].request.data)

    # Talking to the host: owner start and stop, then one request per call, answered by its terminal.

    @staticmethod
    async def _ask(
        ctx: hsm.Context,
        instance: "Sandbox",
        name: typing.Literal["sweep", "create", "destroy", "run", "write", "read"],
        request: hsm.Event[typing.Any],
        data: pydantic.BaseModel,
        success: hsm.Event[typing.Any],
        timeout: datetime.timedelta,
    ) -> hsm.Event[typing.Any]:
        """Send one host request under a fresh id and return its terminal, or the failure that stands in for one."""

        with span.operation(f"bot.sandbox.{name}", scope=_SCOPE, component=_COMPONENT, stage=name) as active:
            try:
                terminal = await ability.run_terminal_operation(
                    ctx,
                    child=instance._host,
                    request=request.with_data_and_id(data, uuid.uuid4().hex),
                    terminals=(success, host.FailedEvent),
                    timeout=timeout,
                )
            except TimeoutError:
                terminal = _host_failure(host.FailureKind.TIMEOUT, "The sandbox host did not answer in time.")
            except RuntimeError:
                terminal = _host_failure(host.FailureKind.UNAVAILABLE, "The sandbox host is not running.")
            failure = _failure_of(terminal)
            if failure is not None:
                span.record_failure(active, failure.kind.value)
            return terminal

    @staticmethod
    async def _start_host(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """Owner start of the host under this ability's lifetime context (HSM-CONTEXT-001), unless it is running."""

        del event
        try:
            if not lifecycle.is_started(instance._host):
                _ = await mosfet.started(instance.context(), instance._host, instance._host_model)
        except Exception as error:
            _LOGGER.warning("The sandbox host did not start (%s).", type(error).__name__)
            _ = hsm.dispatch(
                ctx,
                instance,
                _HostStartFailedEvent.with_data(
                    host.HostFailureData(kind=host.FailureKind.UNAVAILABLE, message="The sandbox host did not start.")
                ),
            )
            return
        _ = hsm.dispatch(ctx, instance, _HostStartedEvent.with_data(_NoData()))

    @staticmethod
    async def _create(ctx: hsm.Context, instance: "Sandbox") -> None:
        """Create one machine from the bench's spec and report it to this ability."""

        created = await Sandbox._ask(
            ctx,
            instance,
            "create",
            host.CreateEvent,
            host.CreateData(spec=instance._spec),
            host.CreatedEvent,
            instance._create_timeout,
        )
        data = created.data
        if not isinstance(data, host.CreatedData):
            _MACHINES.add(1, {"event": "create_failed"})
            failure = _failure_of(created) or host.HostFailureData(
                kind=host.FailureKind.FAILED, message="The sandbox host answered create without a machine."
            )
            _ = hsm.dispatch(ctx, instance, _MachineFailedEvent.with_data(failure))
            return
        _MACHINES.add(1, {"event": "created"})
        _ = hsm.dispatch(ctx, instance, _MachineReadyEvent.with_data(_MachineData(machine=data.machine)))

    @staticmethod
    async def _destroy(ctx: hsm.Context, instance: "Sandbox") -> None:
        """Destroy the machine this bench holds, if any; a failure leaves it to the next sweep."""

        machine = instance._machine
        if machine is None:
            return
        destroyed = await Sandbox._ask(
            ctx,
            instance,
            "destroy",
            host.DestroyEvent,
            host.DestroyData(machine=machine),
            host.DestroyedEvent,
            instance._destroy_timeout,
        )
        failure = _failure_of(destroyed)
        if failure is None:
            _MACHINES.add(1, {"event": "destroyed"})
        elif failure.kind is host.FailureKind.TIMEOUT:
            _MACHINES.add(1, {"event": "destroy_timeout"})
            _LOGGER.warning("The sandbox machine was not destroyed in time; the next sweep removes it.")
        else:
            _MACHINES.add(1, {"event": "destroy_failed"})

    @staticmethod
    async def _provision(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """Sweep what an earlier run of this owner left behind, then create the bench's machine."""

        del event
        swept = await Sandbox._ask(
            ctx, instance, "sweep", host.SweepEvent, host.SweepData(), host.SweptEvent, instance._create_timeout
        )
        failure = _failure_of(swept)
        if failure is not None:
            _MACHINES.add(1, {"event": "sweep_failed"})
            _ = hsm.dispatch(ctx, instance, _MachineFailedEvent.with_data(failure))
            return
        _MACHINES.add(1, {"event": "swept"})
        await Sandbox._create(ctx, instance)

    @staticmethod
    async def _renew(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """Throw the machine away and make a fresh one from the same spec."""

        del event
        await Sandbox._destroy(ctx, instance)
        await Sandbox._create(ctx, instance)

    @staticmethod
    async def _work(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """Ask the host for the request first in line, on the bench's machine, and report its terminal."""

        del event
        pending = instance._line[0]
        machine = instance._machine
        data = pending.request.data
        if machine is None:
            terminal = _host_failure(host.FailureKind.UNAVAILABLE, "The sandbox has no machine.")
        elif isinstance(data, command.RunData):
            terminal = await Sandbox._ask(
                ctx,
                instance,
                "run",
                host.RunEvent,
                host.HostRunData(machine=machine, command=data, max_output_chars=instance._max_output_chars),
                host.RanEvent,
                Sandbox._backstop(instance, data),
            )
        elif isinstance(data, command.WriteData):
            terminal = await Sandbox._ask(
                ctx,
                instance,
                "write",
                host.WriteEvent,
                host.HostWriteData(machine=machine, write=data),
                host.WrittenEvent,
                instance._io_timeout,
            )
        else:
            assert isinstance(data, command.ReadData)
            terminal = await Sandbox._ask(
                ctx,
                instance,
                "read",
                host.ReadEvent,
                host.HostReadData(machine=machine, read=data, max_chars=instance._max_output_chars),
                host.ContentsEvent,
                instance._io_timeout,
            )
        _ = hsm.dispatch(ctx, instance, _AnsweredEvent.with_data(_AnsweredData(pending=pending, terminal=terminal)))

    @staticmethod
    async def _release(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """Destroy the machine (bounded by ``destroy_timeout``), then owner stop of the host, for this detach.

        The stop is shielded, so a stop that has begun is never cut in half.
        """

        await Sandbox._destroy(ctx, instance)
        if lifecycle.is_started(instance._host):
            await asyncio.shield(hsm.stop(instance._host, hsm.Context()))
        _ = hsm.dispatch(ctx, instance, _ReleasedEvent.with_data(_ReleasedData(detach=event)))

    @staticmethod
    async def _unlink(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """Detach the composite group for the detach request released; its terminal tells the owner."""

        data = event.data
        assert isinstance(data, _ReleasedData)
        await ability.Ability._detach_composite_group(ctx, instance, data.detach)

    # Effects.

    @staticmethod
    def _join(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """Put a request in line; one the bench cannot take, or one past the limit, is answered at once."""

        pending = Sandbox._pending(event)
        refusal = Sandbox._refusal(instance, pending.request.data)
        if refusal is not None:
            Sandbox._fail(ctx, instance, pending, *refusal)
            return
        if len(instance._line) >= instance._queue_limit:
            Sandbox._fail(
                ctx,
                instance,
                pending,
                host.FailureKind.BUSY,
                f"The sandbox is already holding {instance._queue_limit} requests; ask again when one is answered.",
            )
            return
        instance._line.append(pending)
        _QUEUE_DEPTH.add(1)

    @staticmethod
    def _refuse_unavailable(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        Sandbox._fail(
            ctx,
            instance,
            Sandbox._pending(event),
            host.FailureKind.UNAVAILABLE,
            "The sandbox has no machine right now; it is retrying on its own.",
        )

    @staticmethod
    def _refuse_detaching(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        Sandbox._fail(
            ctx, instance, Sandbox._pending(event), host.FailureKind.UNAVAILABLE, "The sandbox is being taken away."
        )

    @staticmethod
    def _fall_unavailable(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """Entering ``unavailable``: count the retry and answer everything in line."""

        del event
        instance._retries += 1
        Sandbox._drain(ctx, instance, "The sandbox has no machine right now; it is retrying on its own.")

    @staticmethod
    def _drain_detaching(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        del event
        Sandbox._drain(ctx, instance, "The sandbox is being taken away.")

    @staticmethod
    def _keep_machine(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _MachineData)
        if instance._machine is None:
            _LIVE.add(1)
        instance._machine = data.machine
        instance._retries = 0

    @staticmethod
    def _provisioned(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """A machine is ready; when it replaces one lost to a host down or a failed renewal, the bot is told fresh."""

        Sandbox._keep_machine(ctx, instance, event)
        if instance._lost_untold:
            instance._lost_untold = False
            Sandbox._observe(ctx, instance, FreshEvent.with_data(command.FreshData()), None)

    @staticmethod
    def _renewed(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """A fresh machine replaced the old one: answer a reset first in line, or tell the bot all the same."""

        Sandbox._keep_machine(ctx, instance, event)
        if instance._line and instance._line[0].kind == "reset":
            pending = instance._line.popleft()
            _QUEUE_DEPTH.add(-1)
            Sandbox._answer(ctx, instance, pending, FreshEvent.with_data(command.FreshData()))
            return
        Sandbox._observe(ctx, instance, FreshEvent.with_data(command.FreshData()), None)

    @staticmethod
    def _forget_machine(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        if instance._machine is not None:
            _LIVE.add(-1)
        instance._machine = None

    @staticmethod
    def _renewal_failed(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """Renewal failed or overran its bound: the old machine is gone, and the next one is told fresh."""

        if instance._machine is not None:
            instance._lost_untold = True
        Sandbox._forget_machine(ctx, instance, event)

    @staticmethod
    def _heard_serving(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._host_serving = True

    @staticmethod
    def _heard_down(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._host_serving = False

    @staticmethod
    def _host_went_down(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """The host stopped serving while this bench held a machine: it is lost, and the next one is told fresh."""

        Sandbox._heard_down(ctx, instance, event)
        if instance._machine is not None:
            _MACHINES.add(1, {"event": "lost"})
            instance._lost_untold = True
        Sandbox._forget_machine(ctx, instance, event)

    @staticmethod
    def _host_stopped(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """Releasing stopped the host (or it was never started): nothing it announced still holds."""

        instance._lost_untold = False
        Sandbox._forget_machine(ctx, instance, event)
        Sandbox._heard_down(ctx, instance, event)

    @staticmethod
    def _log_host_start_timeout(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _LOGGER.warning("The sandbox host did not announce serving in time.")

    @staticmethod
    def _age(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """The machine reached ``max_lifetime``; it is renewed the next time the bench is idle."""

        del event
        _ = hsm.dispatch(ctx, instance, _AgedEvent.with_data(_NoData()))

    @staticmethod
    def _first(instance: "Sandbox") -> _Pending:
        pending = instance._line.popleft()
        _QUEUE_DEPTH.add(-1)
        return pending

    @staticmethod
    def _served(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """The host answered the request first in line: hand on what happened."""

        data = event.data
        assert isinstance(data, _AnsweredData)
        pending = Sandbox._first(instance)
        result = data.terminal.data
        if isinstance(result, command.RanData):
            outcome = _run_outcome(result)
            _RUNS.add(1, {"outcome": outcome})
            _RUN_DURATION.record(result.duration.total_seconds(), {"outcome": outcome})
            for stream, truncated in (("stdout", result.stdout_truncated), ("stderr", result.stderr_truncated)):
                if truncated:
                    _TRUNCATED.add(1, {"stream": stream})
            observation: hsm.Event[typing.Any] = RanEvent.with_data(result)
        elif isinstance(result, command.WrittenData):
            observation = WrittenEvent.with_data(result)
        else:
            assert isinstance(result, command.ContentsData)
            if result.truncated:
                _TRUNCATED.add(1, {"stream": "file"})
            observation = ContentsEvent.with_data(result)
        Sandbox._answer(ctx, instance, pending, observation)

    @staticmethod
    def _host_failed(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """The host failed the request first in line: hand on why."""

        data = event.data
        assert isinstance(data, _AnsweredData)
        failure = _failure_of(data.terminal)
        assert failure is not None
        Sandbox._fail(ctx, instance, Sandbox._first(instance), failure.kind, failure.message)

    @staticmethod
    def _overran(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """The host could not stop the request first in line; the machine is replaced."""

        del event
        Sandbox._fail(
            ctx,
            instance,
            Sandbox._first(instance),
            host.FailureKind.TIMEOUT,
            "The request overran and could not be stopped; the machine is being replaced and the workspace is lost.",
        )

    @staticmethod
    def _lost(ctx: hsm.Context, instance: "Sandbox", event: hsm.Event[typing.Any]) -> None:
        """The machine or its host is gone mid-request; a new machine is provisioned."""

        Sandbox._fail(
            ctx,
            instance,
            Sandbox._first(instance),
            host.FailureKind.UNAVAILABLE,
            "The machine or its host is gone; a new machine is being made and the workspace is lost.",
        )
        _MACHINES.add(1, {"event": "lost"})
        Sandbox._forget_machine(ctx, instance, event)

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Sandbox",
        hsm.initial(hsm.target("/Sandbox/initializing")),
        hsm.state(
            "initializing",
            hsm.entry(_heard_down),
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent, *_REQUEST_EVENTS),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Sandbox/starting_host"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                # Unreachable with the empty group; kept so the composite lifecycle's targets exist.
                hsm.guard(ability.Ability._is_composite_detached_terminal),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Sandbox/degraded"),
            ),
        ),
        hsm.state(
            "starting_host",
            hsm.defer(*_REQUEST_EVENTS),
            hsm.activity(_start_host),
            # A host already serving (a retry after a create it refused) is not announced again.
            hsm.transition(hsm.on(_HostStartedEvent), hsm.guard(_host_is_serving), hsm.target("/Sandbox/provisioning")),
            hsm.transition(hsm.on(host.ServingEvent), hsm.effect(_heard_serving), hsm.target("/Sandbox/provisioning")),
            hsm.transition(hsm.on(host.DownEvent), hsm.effect(_heard_down), hsm.target("/Sandbox/unavailable")),
            hsm.transition(hsm.on(_HostStartFailedEvent), hsm.target("/Sandbox/unavailable")),
            hsm.transition(
                hsm.after(_host_start_delay),
                hsm.effect(_log_host_start_timeout),
                hsm.target("/Sandbox/unavailable"),
            ),
        ),
        hsm.state(
            "provisioning",
            hsm.defer(*_REQUEST_EVENTS),
            hsm.activity(_provision),
            hsm.transition(hsm.on(_MachineReadyEvent), hsm.effect(_provisioned), hsm.target("/Sandbox/serving")),
            hsm.transition(hsm.on(_MachineFailedEvent), hsm.target("/Sandbox/unavailable")),
            hsm.transition(hsm.on(host.ServingEvent), hsm.effect(_heard_serving)),
            hsm.transition(hsm.on(host.DownEvent), hsm.effect(_heard_down), hsm.target("/Sandbox/unavailable")),
            hsm.transition(hsm.after(_create_delay), hsm.target("/Sandbox/unavailable")),
        ),
        hsm.state(
            "unavailable",
            hsm.entry(_fall_unavailable),
            # Not offered here, yet answered at once: the bot learns now that there is no machine.
            hsm.transition(hsm.on(hsm.AnyEvent), hsm.guard(_is_request), hsm.effect(_refuse_unavailable)),
            # The host's news ends the wait at once; the backoff only bounds a host that does not announce.
            hsm.transition(hsm.on(host.ServingEvent), hsm.effect(_heard_serving), hsm.target("/Sandbox/provisioning")),
            hsm.transition(hsm.on(host.DownEvent), hsm.effect(_heard_down)),
            hsm.transition(hsm.after(_retry_delay), hsm.target("/Sandbox/starting_host")),
        ),
        hsm.state(
            "serving",
            hsm.initial(hsm.target("/Sandbox/serving/next")),
            # Internal: the machine's age is noted wherever the bench is, and acted on only while it is idle.
            hsm.transition(hsm.after(_lifetime_delay), hsm.effect(_age)),
            hsm.transition(hsm.on(host.ServingEvent), hsm.effect(_heard_serving)),
            # The request in flight, if any, is answered by unavailable's entry; the host's late reply is ignored.
            # The machine is lost either way, so the next one is told fresh.
            hsm.transition(hsm.on(host.DownEvent), hsm.effect(_host_went_down), hsm.target("/Sandbox/unavailable")),
            hsm.choice(
                "next",
                hsm.transition(hsm.guard(_resets_next), hsm.target("/Sandbox/renewing")),
                hsm.transition(hsm.guard(_has_line), hsm.target("/Sandbox/serving/working")),
                hsm.transition(hsm.target("/Sandbox/serving/ready")),
            ),
            hsm.state(
                "ready",
                hsm.transition(
                    hsm.on(*_REQUEST_EVENTS),
                    hsm.guard(_is_request),
                    hsm.effect(_join),
                    hsm.target("/Sandbox/serving/next"),
                ),
                hsm.transition(hsm.on(_AgedEvent), hsm.target("/Sandbox/renewing")),
            ),
            hsm.state(
                "working",
                hsm.defer(_AgedEvent),
                hsm.activity(_work),
                hsm.transition(hsm.on(*_REQUEST_EVENTS), hsm.guard(_is_request), hsm.effect(_join)),
                hsm.transition(
                    hsm.on(_AnsweredEvent),
                    hsm.guard(_answered_ok),
                    hsm.effect(_served),
                    hsm.target("/Sandbox/serving/next"),
                ),
                hsm.transition(
                    hsm.on(_AnsweredEvent),
                    hsm.guard(_answered_with(host.FailureKind.UNAVAILABLE)),
                    hsm.effect(_lost),
                    hsm.target("/Sandbox/provisioning"),
                ),
                hsm.transition(
                    hsm.on(_AnsweredEvent),
                    hsm.guard(_answered_with(host.FailureKind.TIMEOUT)),
                    hsm.effect(_overran),
                    hsm.target("/Sandbox/renewing"),
                ),
                hsm.transition(
                    hsm.on(_AnsweredEvent),
                    hsm.guard(_answers_first),
                    hsm.effect(_host_failed),
                    hsm.target("/Sandbox/serving/next"),
                ),
                hsm.transition(hsm.after(_work_delay), hsm.effect(_overran), hsm.target("/Sandbox/renewing")),
            ),
        ),
        hsm.state(
            "renewing",
            hsm.defer(*_REQUEST_EVENTS),
            hsm.activity(_renew),
            hsm.transition(hsm.on(_MachineReadyEvent), hsm.effect(_renewed), hsm.target("/Sandbox/serving")),
            hsm.transition(hsm.on(host.ServingEvent), hsm.effect(_heard_serving)),
            hsm.transition(hsm.on(host.DownEvent), hsm.effect(_host_went_down), hsm.target("/Sandbox/unavailable")),
            # Renewal destroyed (or abandoned) the machine either way, so the next one is told fresh.
            hsm.transition(
                hsm.on(_MachineFailedEvent), hsm.effect(_renewal_failed), hsm.target("/Sandbox/unavailable")
            ),
            hsm.transition(hsm.after(_renew_delay), hsm.effect(_renewal_failed), hsm.target("/Sandbox/unavailable")),
        ),
        hsm.state(
            "detaching",
            hsm.entry(_drain_detaching),
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.initial(hsm.target("/Sandbox/detaching/releasing")),
            hsm.transition(hsm.on(hsm.AnyEvent), hsm.guard(_is_request), hsm.effect(_refuse_detaching)),
            # Its destroy is bounded by the host operation's own modeled destroy_timeout.
            hsm.state(
                "releasing",
                hsm.activity(_release),
                hsm.exit(_host_stopped),
                hsm.transition(hsm.on(_ReleasedEvent), hsm.target("/Sandbox/detaching/unlinking")),
            ),
            hsm.state(
                "unlinking",
                hsm.activity(_unlink),
                hsm.transition(
                    hsm.on(ability.Ability._composite_attachment_terminal_event),
                    hsm.guard(ability.Ability._is_composite_rollback_failure),
                    hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                    hsm.target("/Sandbox/degraded"),
                ),
                hsm.transition(
                    hsm.on(ability.Ability._composite_attachment_terminal_event),
                    hsm.guard(ability.Ability._is_composite_detach_failed),
                    hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                    hsm.target("/Sandbox/starting_host"),
                ),
            ),
        ),
        hsm.state(
            "degraded",
            hsm.transition(hsm.on(hsm.AnyEvent), hsm.guard(_is_request), hsm.effect(_refuse_detaching)),
        ),
    )


__all__ = [
    "ContentsEvent",
    "FailedEvent",
    "FailureData",
    "FreshEvent",
    "RanEvent",
    "ReadEvent",
    "Request",
    "ResetEvent",
    "RunEvent",
    "Sandbox",
    "WriteEvent",
    "WrittenEvent",
]

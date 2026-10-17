"""The process lifecycle ``bot.run`` drives: bring-up, supervision, and one bounded graceful stop."""

import asyncio
import collections.abc
import datetime
import logging
import pathlib
import types
import typing

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from mosfet import lifecycle
from mosfet.environment import Environment
from mosfet.telemetry import observer, span
from mosfet.telemetry.hsm import Traced

from . import events
from . import policy
from . import supervisor

_LOG = logging.getLogger(__name__)
_SCOPE = "bot.runtime"
_COMPONENT = "runtime"
_METER = metrics.get_meter(_SCOPE)
_SIGNALS = _METER.create_counter(
    "bot.runtime.signal.count",
    unit="{signal}",
    description="Stop signals the process received, by signal.",
)

_StopReason = typing.Literal["stopped", "exhausted", "boot_failed"]


class _UnreachableData(pydantic.BaseModel):
    """A recorded supervisor could not be asked to stop: it never started, or the request failed."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    name: policy.BotName


# Dispatched only by the Runtime's own stop activity to itself: the typed drop or failure outcome of a
# stop request that could not be delivered.
_UnreachableEvent = hsm.Event[_UnreachableData](
    name="runtime.supervisor.unreachable", kind=hsm.ErrorEventKind, schema=_UnreachableData
)


class Runtime(Traced):
    """One ``bot.run`` process: brings its bots and shared actors up, keeps the bots running, and stops it.

    ``bots`` are the bodies ``bot.run`` built from its models, under their names; ``actors`` are the shared
    actors their parts use. ``bot.run`` has checked both: one to ``MAX_BOTS`` bots, and each actor an
    ``hsm.Instance`` whose class declares ``model``. Bring-up first removes a ready file an earlier run left,
    then starts the actors in order under this machine's own (non-private) context, a child of the process
    ``Environment``, then one :class:`~mosfet.runtime.supervisor.Supervisor` per bot; every bot attaches to
    that same environment. ``running`` waits for every supervisor to report its bot running, then touches
    the ready file. One ``stopping`` composite with a single ``stop_grace`` bound stops every supervisor,
    then every actor in reverse order.

    ``on_exit`` is called exactly once, from the transition into a final state, with the exit code and
    reason. A stop request during bring-up leaves at once; nothing is deferred. A second *signal* while
    stopping forces the stop; a programmatic stop request never does.

    Time enters only through this machine's injected ``hsm`` clock: the bring-up and stop bounds are
    ``hsm.after`` timers. The wall-clock length of a stop belongs to the process boundary, not this machine.
    """

    _bots: collections.abc.Mapping[str, "mosfet.Bot"]
    _shared: tuple[hsm.Instance, ...]
    _policy: policy.Policy
    _environment: Environment
    _ready_file: pathlib.Path | None
    _on_exit: collections.abc.Callable[[events.ExitedData], None]
    # Recorded before each start is invoked, so a start that bring-up's cancellation interrupts is still
    # stopped. Only this machine's own activities write them.
    _actors: list[hsm.Instance]
    _supervisors: dict[str, supervisor.Supervisor]
    # Supervisors that reported their bot running at least once; readiness waits for all of them.
    _running: set[str]
    # Supervisors whose stop this machine is still waiting on.
    _pending: set[str]
    # Why the process is stopping; written once by the transition that enters stopping.
    _reason: _StopReason
    _signal_seen: bool

    def __init__(
        self,
        *,
        bots: collections.abc.Mapping[str, "mosfet.Bot"],
        actors: collections.abc.Sequence[hsm.Instance],
        policy: policy.Policy,
        environment: Environment,
        ready_file: pathlib.Path | None,
        on_exit: collections.abc.Callable[[events.ExitedData], None],
    ) -> None:
        super().__init__()
        self._bots = types.MappingProxyType(dict(bots))
        self._shared = tuple(actors)
        self._policy = policy
        self._environment = environment
        self._ready_file = ready_file
        self._on_exit = on_exit
        self._actors = []
        self._supervisors = {}
        self._running = set()
        self._pending = set()
        self._reason = "stopped"
        self._signal_seen = False

    @staticmethod
    async def _start_actors(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        """Start the shared actors in order, each with its class model under this machine's lifetime."""

        del event
        try:
            for actor in instance._shared:
                model = getattr(type(actor), "model", None)
                assert isinstance(model, hsm.Model), "bot.run validates every actor's class model"
                instance._actors.append(actor)
                _ = await mosfet.started(instance.context(), actor, model)
        except Exception as error:
            _LOG.error("runtime actor start failed kind=%s", span.failure_kind(error, type(error).__name__))
            _ = hsm.dispatch(
                ctx,
                instance,
                events.ActorsFailedEvent.with_data(events.ActorsData(message=f"{type(error).__name__}: {error}")),
            )
            return
        _ = hsm.dispatch(ctx, instance, events.ActorsStartedEvent.with_data(events.ActorsData()))

    @staticmethod
    async def _start_supervisors(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        """Start one supervisor per bot under this machine's lifetime, all in the process environment.

        Each supervisor is recorded before its start is invoked. Once invoked, hsm runs that start to
        completion in the supervisor's own task, so a supervisor whose start this activity's cancellation
        interrupts is still started and still answers ``runtime.supervisor.stop``. A recorded supervisor
        whose start failed never started; stopping turns the undeliverable stop request into a typed
        ``runtime.supervisor.unreachable`` outcome instead of waiting on it.
        """

        del event
        try:
            for name, bot in instance._bots.items():
                keeper = supervisor.Supervisor(
                    name=name,
                    bot=bot,
                    environment=instance._environment,
                    policy=instance._policy,
                    owner=instance,
                )
                instance._supervisors[name] = keeper
                _ = await mosfet.started(instance.context(), keeper, supervisor.Supervisor.model)
        except Exception as error:
            _LOG.error("runtime supervisor start failed kind=%s", span.failure_kind(error, type(error).__name__))
            _ = hsm.dispatch(
                ctx,
                instance,
                events.SupervisorsFailedEvent.with_data(events.ActorsData(message=f"{type(error).__name__}: {error}")),
            )
            return
        _ = hsm.dispatch(ctx, instance, events.SupervisorsStartedEvent.with_data(events.ActorsData()))

    @staticmethod
    def _start_timeout(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._policy.start_timeout

    @staticmethod
    def _stop_grace(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._policy.stop_grace

    @staticmethod
    def _note_stop(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        """Remember a received signal, so the next one is recognised as a second signal."""

        del ctx
        data = event.data
        assert isinstance(data, events.StopData)
        if data.signal is not None:
            _SIGNALS.add(1, {"signal": data.signal})
            instance._signal_seen = True

    @staticmethod
    def _is_second_signal(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, events.StopData) and data.reason == "signal" and instance._signal_seen

    @staticmethod
    def _stop_requested(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        Runtime._note_stop(ctx, instance, event)
        instance._reason = "stopped"

    @staticmethod
    def _bring_up_failed(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _LOG.error("runtime bring-up failed bots=%d actors=%d", len(instance._bots), len(instance._shared))
        instance._reason = "boot_failed"

    @staticmethod
    def _bring_up_timed_out(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _LOG.error("runtime bring-up timed out bots=%d actors=%d", len(instance._bots), len(instance._shared))
        instance._reason = "boot_failed"

    @staticmethod
    def _bot_exhausted(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._reason = "exhausted"

    @staticmethod
    def _record_running(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, events.SupervisorData)
        instance._running.add(data.name)

    @staticmethod
    def _completes_readiness(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> bool:
        """With this report every supervisor has reported its bot running."""

        del ctx
        data = event.data
        return isinstance(data, events.SupervisorData) and instance._running | {data.name} >= set(instance._supervisors)

    @staticmethod
    def _touch_ready_file(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        if instance._ready_file is None:
            return
        try:
            instance._ready_file.touch()
        except OSError as error:
            _LOG.error("runtime ready file not written kind=%s", span.failure_kind(error, type(error).__name__))

    @staticmethod
    def _remove_ready_file(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        if instance._ready_file is None:
            return
        try:
            instance._ready_file.unlink(missing_ok=True)
        except OSError as error:
            _LOG.error("runtime ready file not removed kind=%s", span.failure_kind(error, type(error).__name__))

    @staticmethod
    def _has_supervisors(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return bool(instance._supervisors)

    @staticmethod
    def _await_supervisors(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._pending = set(instance._supervisors)

    @staticmethod
    async def _request_supervisor_stops(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        """Ask every recorded supervisor to stop; one the request cannot reach is reported unreachable.

        A supervisor that never started, or whose delivery fails, will never answer, so this activity reports
        it with the typed ``runtime.supervisor.unreachable`` outcome and the stop goes on to the actors.
        """

        del event
        deliveries = [
            (name, hsm.dispatch(ctx, keeper, events.SupervisorStopEvent.with_data(events.SupervisorStopData())))
            for name, keeper in instance._supervisors.items()
        ]
        for name, delivery in deliveries:
            try:
                _ = await delivery
            except Exception as error:
                if lifecycle.is_not_started_error(error):
                    _LOG.warning("runtime supervisor unreachable at stop bot=%s", name)
                else:
                    _LOG.error(
                        "runtime supervisor stop request failed bot=%s kind=%s",
                        name,
                        span.failure_kind(error, type(error).__name__),
                    )
                _ = hsm.dispatch(ctx, instance, _UnreachableEvent.with_data(_UnreachableData(name=name)))

    @staticmethod
    def _settles_last_supervisor(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, events.SupervisorStoppedData | _UnreachableData) and instance._pending == {data.name}

    @staticmethod
    def _settle_supervisor(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, events.SupervisorStoppedData | _UnreachableData)
        instance._pending.discard(data.name)

    @staticmethod
    async def _stop_actors(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        """Stop every recorded supervisor, then every recorded actor in reverse start order.

        Each stop is awaited through ``asyncio.shield``, so this activity ends as soon as hsm cancels it
        (the grace timer or a second signal) even when a stop ignores cancellation. A stop that outlives
        the activity keeps running until the process boundary cancels what remains. A stop that raises is
        logged and counted, and the rest are still stopped; the count rides ``runtime.actors.stopped``.
        """

        del event
        failures = 0
        with span.operation(
            "runtime.stop",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="stop",
            attributes={"bot.runtime.stop.reason": instance._reason},
        ) as active:
            # Stopping is idempotent: every supervisor has stopped already or never started.
            for name, keeper in instance._supervisors.items():
                try:
                    await asyncio.shield(hsm.stop(keeper))
                except Exception as error:
                    failures += 1
                    _LOG.warning(
                        "runtime supervisor stop failed bot=%s kind=%s",
                        name,
                        span.failure_kind(error, type(error).__name__),
                    )
            for actor in reversed(instance._actors):
                try:
                    await asyncio.shield(hsm.stop(actor))
                except Exception as error:
                    failures += 1
                    _LOG.warning(
                        "runtime actor stop failed actor=%s kind=%s",
                        type(actor).__name__,
                        span.failure_kind(error, type(error).__name__),
                    )
            active.set_attribute("process.exit.code", events.ExitedData.for_reason(instance._reason).code)
        _ = hsm.dispatch(
            ctx,
            instance,
            events.ActorsStoppedEvent.with_data(
                events.ActorsData(message=f"{failures} supervisors or actors failed to stop" if failures else None)
            ),
        )

    @staticmethod
    def _exit_stopped(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._on_exit(events.ExitedData.for_reason(instance._reason))

    @staticmethod
    def _exit_forced(ctx: hsm.Context, instance: "Runtime", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _LOG.error("runtime stop forced reason=%s", instance._reason)
        instance._on_exit(events.ExitedData.for_reason("forced"))

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Runtime",
        hsm.initial(hsm.target("starting_actors")),
        hsm.state(
            "starting_actors",
            # A run killed without leaving ready (SIGKILL, OOM, host crash) left its ready file behind: remove
            # it before anything else, so the file exists exactly while this run is ready.
            hsm.entry(_remove_ready_file),
            hsm.activity(_start_actors),
            hsm.transition(hsm.on(events.ActorsStartedEvent), hsm.target("/Runtime/starting_supervisors")),
            hsm.transition(
                hsm.on(events.ActorsFailedEvent),
                hsm.effect(_bring_up_failed),
                hsm.target("/Runtime/stopping"),
            ),
            hsm.transition(
                hsm.after(_start_timeout),
                hsm.effect(_bring_up_timed_out),
                hsm.target("/Runtime/stopping"),
            ),
            hsm.transition(
                hsm.on(events.StopEvent),
                hsm.effect(_stop_requested),
                hsm.target("/Runtime/stopping"),
            ),
        ),
        hsm.state(
            "starting_supervisors",
            hsm.activity(_start_supervisors),
            # A bot can activate, or exhaust its budget, before the last supervisor has started; its report
            # waits here for running or stopping. A stop request is never deferred: it leaves bring-up.
            hsm.defer(events.SupervisorRunningEvent, events.SupervisorExhaustedEvent),
            hsm.transition(hsm.on(events.SupervisorsStartedEvent), hsm.target("/Runtime/running")),
            hsm.transition(
                hsm.on(events.SupervisorsFailedEvent),
                hsm.effect(_bring_up_failed),
                hsm.target("/Runtime/stopping"),
            ),
            hsm.transition(
                hsm.after(_start_timeout),
                hsm.effect(_bring_up_timed_out),
                hsm.target("/Runtime/stopping"),
            ),
            hsm.transition(
                hsm.on(events.StopEvent),
                hsm.effect(_stop_requested),
                hsm.target("/Runtime/stopping"),
            ),
        ),
        hsm.state(
            "running",
            hsm.initial(hsm.target("settling")),
            hsm.state(
                "settling",
                hsm.transition(
                    hsm.on(events.SupervisorRunningEvent),
                    hsm.guard(_completes_readiness),
                    hsm.effect(_record_running),
                    hsm.target("/Runtime/running/ready"),
                ),
                hsm.transition(hsm.on(events.SupervisorRunningEvent), hsm.effect(_record_running)),
            ),
            hsm.state(
                "ready",
                hsm.entry(_touch_ready_file),
                hsm.exit(_remove_ready_file),
                # A restarted bot reports running again; the process is already ready, so it is unhandled.
            ),
            hsm.transition(
                hsm.on(events.SupervisorExhaustedEvent),
                hsm.effect(_bot_exhausted),
                hsm.target("/Runtime/stopping"),
            ),
            hsm.transition(
                hsm.on(events.StopEvent),
                hsm.effect(_stop_requested),
                hsm.target("/Runtime/stopping"),
            ),
        ),
        # One grace bound for the whole stop: supervisors first, then actors.
        hsm.state(
            "stopping",
            hsm.initial(hsm.target("draining")),
            hsm.choice(
                "draining",
                hsm.transition(hsm.guard(_has_supervisors), hsm.target("stopping_supervisors")),
                # Nothing was recorded: no supervisor.stopped is awaited.
                hsm.transition(hsm.target("stopping_actors")),
            ),
            hsm.state(
                "stopping_supervisors",
                hsm.entry(_await_supervisors),
                hsm.activity(_request_supervisor_stops),
                hsm.transition(
                    hsm.on(events.SupervisorStoppedEvent, _UnreachableEvent),
                    hsm.guard(_settles_last_supervisor),
                    hsm.effect(_settle_supervisor),
                    hsm.target("/Runtime/stopping/stopping_actors"),
                ),
                hsm.transition(
                    hsm.on(events.SupervisorStoppedEvent, _UnreachableEvent), hsm.effect(_settle_supervisor)
                ),
            ),
            hsm.state(
                "stopping_actors",
                hsm.activity(_stop_actors),
                hsm.transition(
                    hsm.on(events.ActorsStoppedEvent),
                    hsm.effect(_exit_stopped),
                    hsm.target("/Runtime/stopped"),
                ),
            ),
            hsm.transition(hsm.after(_stop_grace), hsm.effect(_exit_forced), hsm.target("/Runtime/forced")),
            hsm.transition(
                hsm.on(events.StopEvent),
                hsm.guard(_is_second_signal),
                hsm.effect(_note_stop, _exit_forced),
                hsm.target("/Runtime/forced"),
            ),
            # A first signal is remembered and the stop already under way continues with its reason; a
            # programmatic request is consumed.
            hsm.transition(hsm.on(events.StopEvent), hsm.effect(_note_stop)),
            # Running and exhausted reports from supervisors being stopped are unhandled: they change nothing.
        ),
        hsm.final("stopped"),
        hsm.final("forced"),
        hsm.observe(observer),
    )


__all__ = ["Runtime"]

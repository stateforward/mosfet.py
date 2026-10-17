"""One bot's keeper inside a ``bot.run`` process."""

import datetime
import logging
import typing

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from mosfet.environment import Environment
from mosfet.telemetry import observer, span
from mosfet.telemetry.hsm import Traced

from . import events
from . import policy

_LOG = logging.getLogger(__name__)
_SCOPE = "bot.runtime"
_COMPONENT = "runtime.supervisor"
_METER = metrics.get_meter(_SCOPE)
_STARTS = _METER.create_counter(
    "bot.runtime.bot.start.count",
    unit="{attempt}",
    description="Bot activation attempts by supervised bot name and how each attempt ended.",
)
_RESTARTS = _METER.create_counter(
    "bot.runtime.bot.restart.count",
    unit="{restart}",
    description="Bot recoveries by supervised bot name and the cause that started each one.",
)
_BACKOFF = _METER.create_histogram(
    "bot.runtime.bot.backoff.duration",
    unit="s",
    description="Delay before each bot restart, by supervised bot name.",
)
_EXHAUSTED = _METER.create_counter(
    "bot.runtime.bot.exhausted.count",
    unit="{bot}",
    description="Supervised bots whose restart budget was exhausted, by bot name.",
)
_STOPS = _METER.create_counter(
    "bot.runtime.bot.stop.count",
    unit="{stop}",
    description="Supervised bot stops by bot name and whether the bot answered in time.",
)
_RUNNING = _METER.create_up_down_counter(
    "bot.runtime.bot.running",
    unit="{bot}",
    description="Supervised bots currently running.",
)

_StartOutcome = typing.Literal["activated", "activation_failed", "start_raised", "start_timeout", "degraded"]
_RestartCause = typing.Literal[
    "activation_failed", "start_raised", "start_timeout", "degraded", "release_failed", "release_timeout"
]


class _OperationData(pydantic.BaseModel):
    """A Supervisor activity's own result, correlated with the bot request it was made for."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    operation_id: str = pydantic.Field(min_length=1)
    message: str | None = None


# Dispatched only by the Supervisor's own activities to itself.
_ReleasedEvent = hsm.Event[_OperationData](
    name="runtime.supervisor.released", kind=hsm.CompletionEventKind, schema=_OperationData
)
_StartFailedEvent = hsm.Event[_OperationData](
    name="runtime.supervisor.start_failed", kind=hsm.ErrorEventKind, schema=_OperationData
)
_StopFailedEvent = hsm.Event[_OperationData](
    name="runtime.supervisor.stop_failed", kind=hsm.ErrorEventKind, schema=_OperationData
)
_ReleaseFailedEvent = hsm.Event[_OperationData](
    name="runtime.supervisor.release_failed", kind=hsm.ErrorEventKind, schema=_OperationData
)


class Supervisor(Traced):
    """Keeps one bot running in the process environment within a restart budget, and stops it on request.

    It activates the bot and reports ``runtime.supervisor.running`` to its owner. When activation fails,
    times out, raises, or the bot degrades, it deactivates and releases the bot and restarts it after an
    exponential backoff; when the budget of consecutive failed attempts is spent it reports
    ``runtime.supervisor.exhausted`` instead. On ``runtime.supervisor.stop`` it deactivates and releases
    the bot and reports ``runtime.supervisor.stopped``.

    Every request it makes of the bot carries the operation id ``<name>:<serial>``, and only the bot's
    terminal for the current request moves it, so a late terminal from an earlier request is ignored.
    It applies the injected mechanical policy only; what the bot does while running is the bot's own.
    """

    _name: str
    _bot: "mosfet.Bot"
    _environment: Environment
    _policy: policy.Policy
    _owner: hsm.Instance
    # Monotonic count of bot requests and the operation id of the latest one; every guard that accepts a
    # bot terminal or an own activity result compares against this id.
    _serial: int
    _operation_id: str
    # Consecutive failed attempts since the bot last ran stably.
    _attempts: int
    # How the one stop this supervisor performs ended; set to timed_out by a stop bound expiring.
    _stop_outcome: typing.Literal["graceful", "timed_out"]

    def __init__(
        self,
        *,
        name: str,
        bot: "mosfet.Bot",
        environment: Environment,
        policy: policy.Policy,
        owner: hsm.Instance,
    ) -> None:
        super().__init__()
        self._name = name
        self._bot = bot
        self._environment = environment
        self._policy = policy
        self._owner = owner
        self._serial = 0
        self._operation_id = ""
        self._attempts = 0
        self._stop_outcome = "graceful"

    @staticmethod
    def _begin_request(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """Mint the operation id of the bot request the state being entered makes."""

        del ctx, event
        instance._serial += 1
        instance._operation_id = f"{instance._name}:{instance._serial}"

    @staticmethod
    async def _attach(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """Ask the bot to activate in the process environment, answering to this supervisor."""

        del event
        operation_id = instance._operation_id
        with span.operation(
            "runtime.supervisor.attempt",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="attach",
            attributes={"bot.name": instance._name, "bot.runtime.attempt": instance._attempts + 1},
        ):
            try:
                _ = await instance._bot.attach(
                    instance._environment,
                    reply_to=instance,
                    operation_id=operation_id,
                )
            except Exception as error:
                kind = span.failure_kind(error, type(error).__name__)
                span.record_current_failure(kind)
                _LOG.warning("supervised bot start raised bot=%s kind=%s", instance._name, kind)
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    _StartFailedEvent.with_data(
                        _OperationData(operation_id=operation_id, message=f"{type(error).__name__}: {error}")
                    ),
                )

    @staticmethod
    async def _detach(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """Ask the bot to deactivate and leave the process environment, answering to this supervisor."""

        del event
        operation_id = instance._operation_id
        try:
            _ = await instance._bot.detach(instance._environment, reply_to=instance, operation_id=operation_id)
        except Exception as error:
            kind = span.failure_kind(error, type(error).__name__)
            _LOG.warning("supervised bot stop raised bot=%s kind=%s", instance._name, kind)
            _ = hsm.dispatch(
                ctx,
                instance,
                _StopFailedEvent.with_data(
                    _OperationData(operation_id=operation_id, message=f"{type(error).__name__}: {error}")
                ),
            )

    @staticmethod
    async def _release(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """Stop the bot's machine and report the release, or its failure, for the current request."""

        del event
        operation_id = instance._operation_id
        try:
            await hsm.stop(instance._bot)
        except Exception as error:
            _LOG.warning(
                "supervised bot release raised bot=%s kind=%s",
                instance._name,
                span.failure_kind(error, type(error).__name__),
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                _ReleaseFailedEvent.with_data(
                    _OperationData(operation_id=operation_id, message=f"{type(error).__name__}: {error}")
                ),
            )
            return
        _ = hsm.dispatch(ctx, instance, _ReleasedEvent.with_data(_OperationData(operation_id=operation_id)))

    @staticmethod
    def _answers_request(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> bool:
        """The bot's terminal answers the request this supervisor is currently waiting on."""

        del ctx
        return event.id == instance._operation_id

    @staticmethod
    def _completes_request(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> bool:
        """An own activity result belongs to the request this supervisor is currently waiting on."""

        del ctx
        data = event.data
        return isinstance(data, _OperationData) and data.operation_id == instance._operation_id

    @staticmethod
    def _within_budget(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> bool:
        """Another attempt fits the budget of consecutive failed attempts."""

        del ctx, event
        return instance._attempts + 1 < instance._policy.max_restarts

    @staticmethod
    def _start_timeout(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._policy.start_timeout

    @staticmethod
    def _stable_after(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._policy.stable_after

    @staticmethod
    def _bot_stop_timeout(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._policy.bot_stop_timeout

    @staticmethod
    def _bot_release_timeout(
        ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, event
        return instance._policy.bot_release_timeout

    @staticmethod
    def _backoff_delay(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        """Doubling delay for the current run of consecutive failed attempts, capped at backoff_max.

        Doubling stops at the cap and never multiplies past it, so no attempt count or backoff the policy
        allows can overflow ``timedelta``.
        """

        del ctx, event
        cap = instance._policy.backoff_max
        delay = instance._policy.backoff_initial
        for _ in range(instance._attempts - 1):
            if delay >= cap / 2:
                return cap
            delay *= 2
        return delay

    @staticmethod
    def _record_start(instance: "Supervisor", outcome: _StartOutcome) -> None:
        _STARTS.add(1, {"bot.name": instance._name, "outcome": outcome})

    @staticmethod
    def _record_restart(instance: "Supervisor", cause: _RestartCause) -> None:
        _RESTARTS.add(1, {"bot.name": instance._name, "cause": cause})

    @staticmethod
    def _announce_running(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """The bot activated: report it to the owner."""

        del event
        Supervisor._record_start(instance, "activated")
        _ = hsm.dispatch(
            ctx,
            instance._owner,
            events.SupervisorRunningEvent.with_data(
                events.SupervisorData(name=instance._name, attempt=instance._attempts + 1)
            ),
        )

    @staticmethod
    def _activation_failed(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        Supervisor._record_start(instance, "activation_failed")
        Supervisor._record_restart(instance, "activation_failed")

    @staticmethod
    def _start_raised(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        Supervisor._record_start(instance, "start_raised")
        Supervisor._record_restart(instance, "start_raised")

    @staticmethod
    def _start_degraded(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        Supervisor._record_start(instance, "degraded")
        Supervisor._record_restart(instance, "degraded")

    @staticmethod
    def _start_timed_out(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        Supervisor._record_start(instance, "start_timeout")
        Supervisor._record_restart(instance, "start_timeout")

    @staticmethod
    def _running_activation_failed(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        Supervisor._record_restart(instance, "activation_failed")

    @staticmethod
    def _running_degraded(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        Supervisor._record_restart(instance, "degraded")

    @staticmethod
    def _enter_running(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _RUNNING.add(1)

    @staticmethod
    def _exit_running(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _RUNNING.add(-1)

    @staticmethod
    def _reset_attempts(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """The bot has run stably: its run of consecutive failed attempts is over."""

        del ctx, event
        instance._attempts = 0

    @staticmethod
    def _back_off(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """Count the failed attempt; backoff then waits the delay for this run of failures."""

        instance._attempts += 1
        _BACKOFF.record(
            Supervisor._backoff_delay(ctx, instance, event).total_seconds(),
            {"bot.name": instance._name},
        )

    @staticmethod
    def _announce_exhausted(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """The restart budget is spent: report it to the owner."""

        del event
        _EXHAUSTED.add(1, {"bot.name": instance._name})
        _LOG.error(
            "supervised bot exhausted its restart budget bot=%s attempts=%d", instance._name, instance._attempts + 1
        )
        _ = hsm.dispatch(
            ctx,
            instance._owner,
            events.SupervisorExhaustedEvent.with_data(
                events.SupervisorData(name=instance._name, attempt=instance._attempts + 1)
            ),
        )

    @staticmethod
    def _release_timed_out(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        Supervisor._record_restart(instance, "release_timeout")
        Supervisor._announce_exhausted(ctx, instance, event)

    @staticmethod
    def _release_failed(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """The bot could not be released, so it cannot be restarted: report the budget exhausted."""

        Supervisor._record_restart(instance, "release_failed")
        Supervisor._announce_exhausted(ctx, instance, event)

    @staticmethod
    def _stop_timed_out(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """A stop bound expired: the bot is force-released and the stop reported as timed out."""

        del ctx, event
        instance._stop_outcome = "timed_out"

    @staticmethod
    def _stop_release_failed(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """The bot was not released cleanly: the stop ends now and is reported as not graceful."""

        del ctx, event
        instance._stop_outcome = "timed_out"

    @staticmethod
    def _announce_stopped(ctx: hsm.Context, instance: "Supervisor", event: hsm.Event[typing.Any]) -> None:
        """The bot is released: report the stop and how it ended to the owner."""

        del event
        _STOPS.add(1, {"bot.name": instance._name, "outcome": instance._stop_outcome})
        _ = hsm.dispatch(
            ctx,
            instance._owner,
            events.SupervisorStoppedEvent.with_data(
                events.SupervisorStoppedData(name=instance._name, outcome=instance._stop_outcome)
            ),
        )

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Supervisor",
        hsm.initial(hsm.target("starting")),
        hsm.state(
            "starting",
            hsm.entry(_begin_request),
            hsm.activity(_attach),
            hsm.transition(
                hsm.on(mosfet.ActivatingDoneEvent),
                hsm.guard(_answers_request),
                hsm.effect(_announce_running),
                hsm.target("/Supervisor/running"),
            ),
            hsm.transition(
                hsm.on(mosfet.ActivatingFailedEvent),
                hsm.guard(_answers_request),
                hsm.effect(_activation_failed),
                hsm.target("/Supervisor/recovering"),
            ),
            hsm.transition(
                hsm.on(_StartFailedEvent),
                hsm.guard(_completes_request),
                hsm.effect(_start_raised),
                hsm.target("/Supervisor/recovering"),
            ),
            hsm.transition(
                hsm.on(mosfet.DegradedEvent),
                hsm.guard(_answers_request),
                hsm.effect(_start_degraded),
                hsm.target("/Supervisor/recovering"),
            ),
            hsm.transition(
                hsm.after(_start_timeout),
                hsm.effect(_start_timed_out),
                hsm.target("/Supervisor/recovering"),
            ),
            hsm.transition(hsm.on(events.SupervisorStopEvent), hsm.target("/Supervisor/stopping")),
        ),
        hsm.state(
            "running",
            hsm.entry(_enter_running),
            hsm.exit(_exit_running),
            hsm.transition(hsm.after(_stable_after), hsm.effect(_reset_attempts)),
            # A bot's own cognition reboot re-announces bot.activated under the same operation id; the
            # supervisor is already running, so that report is left unhandled.
            hsm.transition(
                hsm.on(mosfet.ActivatingFailedEvent),
                hsm.guard(_answers_request),
                hsm.effect(_running_activation_failed),
                hsm.target("/Supervisor/recovering"),
            ),
            hsm.transition(
                hsm.on(mosfet.DegradedEvent),
                hsm.guard(_answers_request),
                hsm.effect(_running_degraded),
                hsm.target("/Supervisor/recovering"),
            ),
            hsm.transition(hsm.on(events.SupervisorStopEvent), hsm.target("/Supervisor/stopping")),
        ),
        # Deactivate and release the bot before deciding on a restart, so its own cleanup is never cut
        # short. A stop request waits here and is handled at once by backoff or exhausted.
        hsm.state(
            "recovering",
            hsm.defer(events.SupervisorStopEvent),
            hsm.initial(hsm.target("detaching")),
            hsm.state(
                "detaching",
                hsm.entry(_begin_request),
                hsm.activity(_detach),
                hsm.transition(
                    hsm.on(mosfet.DeactivatingDoneEvent, mosfet.DegradedEvent),
                    hsm.guard(_answers_request),
                    hsm.target("/Supervisor/recovering/releasing"),
                ),
                hsm.transition(
                    hsm.on(_StopFailedEvent),
                    hsm.guard(_completes_request),
                    hsm.target("/Supervisor/recovering/releasing"),
                ),
                hsm.transition(hsm.after(_bot_stop_timeout), hsm.target("/Supervisor/recovering/releasing")),
            ),
            hsm.state(
                "releasing",
                hsm.activity(_release),
                hsm.transition(
                    hsm.on(_ReleasedEvent),
                    hsm.guard(_completes_request),
                    hsm.target("/Supervisor/recovering/retry"),
                ),
                hsm.transition(
                    hsm.on(_ReleaseFailedEvent),
                    hsm.guard(_completes_request),
                    hsm.effect(_release_failed),
                    hsm.target("/Supervisor/exhausted"),
                ),
                hsm.transition(
                    hsm.after(_bot_release_timeout),
                    hsm.effect(_release_timed_out),
                    hsm.target("/Supervisor/exhausted"),
                ),
            ),
            hsm.choice(
                "retry",
                hsm.transition(hsm.guard(_within_budget), hsm.effect(_back_off), hsm.target("/Supervisor/backoff")),
                hsm.transition(hsm.effect(_announce_exhausted), hsm.target("/Supervisor/exhausted")),
            ),
        ),
        hsm.state(
            "backoff",
            hsm.transition(hsm.after(_backoff_delay), hsm.target("/Supervisor/starting")),
            hsm.transition(
                hsm.on(events.SupervisorStopEvent),
                hsm.effect(_announce_stopped),
                hsm.target("/Supervisor/stopped"),
            ),
        ),
        hsm.state(
            "exhausted",
            hsm.transition(
                hsm.on(events.SupervisorStopEvent),
                hsm.effect(_announce_stopped),
                hsm.target("/Supervisor/stopped"),
            ),
        ),
        # The bot answers a deactivate from every lifecycle state, so detaching normally ends on its
        # terminal; the bounds force-release a bot that does not answer, and the whole stop stays within
        # bot_stop_timeout + bot_release_timeout, which the policy keeps below the process grace.
        hsm.state(
            "stopping",
            hsm.initial(hsm.target("detaching")),
            hsm.state(
                "detaching",
                hsm.entry(_begin_request),
                hsm.activity(_detach),
                hsm.transition(
                    hsm.on(mosfet.DeactivatingDoneEvent, mosfet.DegradedEvent),
                    hsm.guard(_answers_request),
                    hsm.target("/Supervisor/stopping/releasing"),
                ),
                hsm.transition(
                    hsm.on(_StopFailedEvent),
                    hsm.guard(_completes_request),
                    hsm.target("/Supervisor/stopping/releasing"),
                ),
                hsm.transition(
                    hsm.after(_bot_stop_timeout),
                    hsm.effect(_stop_timed_out),
                    hsm.target("/Supervisor/stopping/releasing"),
                ),
            ),
            hsm.state(
                "releasing",
                hsm.activity(_release),
                hsm.transition(
                    hsm.on(_ReleasedEvent),
                    hsm.guard(_completes_request),
                    hsm.effect(_announce_stopped),
                    hsm.target("/Supervisor/stopped"),
                ),
                hsm.transition(
                    hsm.on(_ReleaseFailedEvent),
                    hsm.guard(_completes_request),
                    hsm.effect(_stop_release_failed, _announce_stopped),
                    hsm.target("/Supervisor/stopped"),
                ),
                hsm.transition(
                    hsm.after(_bot_release_timeout),
                    hsm.effect(_stop_timed_out, _announce_stopped),
                    hsm.target("/Supervisor/stopped"),
                ),
            ),
        ),
        hsm.final("stopped"),
        hsm.observe(observer),
    )


__all__ = ["Supervisor"]

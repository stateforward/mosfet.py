import asyncio
import collections.abc
import dataclasses
import datetime
import typing

import hsm
import mosfet
from mosfet import lifecycle, runtime
from mosfet.environment import Environment

from tests.runtime.fixtures import doubles

_MS = datetime.timedelta(milliseconds=1)


def _record(ctx: hsm.Context, instance: "Owner", event: hsm.Event[typing.Any]) -> None:
    del ctx
    instance.reports.append(event)


class Owner(hsm.Instance):
    """Stand-in for the Runtime: records every report a supervisor sends it."""

    reports: list[hsm.Event[typing.Any]]

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Owner",
        hsm.initial(hsm.target("listening")),
        hsm.state(
            "listening",
            hsm.transition(
                hsm.on(
                    runtime.SupervisorRunningEvent,
                    runtime.SupervisorExhaustedEvent,
                    runtime.SupervisorStoppedEvent,
                ),
                hsm.effect(_record),
            ),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.reports = []

    def names(self) -> list[str]:
        return [report.name for report in self.reports]

    def stopped(self) -> runtime.SupervisorStoppedData | None:
        for report in self.reports:
            if isinstance(report.data, runtime.SupervisorStoppedData):
                return report.data
        return None


def _policy(**overrides: object) -> runtime.Policy:
    values: dict[str, object] = {
        "stop_grace": datetime.timedelta(seconds=5),
        "bot_stop_timeout": datetime.timedelta(seconds=1),
        "bot_release_timeout": datetime.timedelta(seconds=1),
        "start_timeout": datetime.timedelta(seconds=5),
        "max_restarts": 5,
        "backoff_initial": 10 * _MS,
        "backoff_max": 40 * _MS,
        "stable_after": datetime.timedelta(seconds=30),
    }
    values.update(overrides)
    return runtime.Policy.model_validate(values)


@dataclasses.dataclass
class _Supervised:
    supervisor: runtime.Supervisor
    owner: Owner
    environment: Environment
    # Every timer duration the supervisor asked its clock for, in order.
    delays: list[datetime.timedelta]

    async def stop(self) -> None:
        _ = await hsm.dispatch(
            self.environment,
            self.supervisor,
            runtime.SupervisorStopEvent.with_data(runtime.SupervisorStopData()),
        )


def _recording_clock(delays: list[datetime.timedelta]) -> hsm.Clock:
    def after(duration: datetime.timedelta) -> collections.abc.Awaitable[None]:
        delays.append(duration)
        return asyncio.sleep(duration.total_seconds())

    return hsm.Clock(after=after)


async def _supervise(bot: mosfet.Bot, policy: runtime.Policy, *, name: str = "alpha") -> _Supervised:
    environment = Environment()
    owner = await mosfet.started(hsm.Context(), Owner(), Owner.model)
    supervisor = runtime.Supervisor(
        name=name,
        bot=bot,
        environment=environment,
        policy=policy,
        owner=owner,
    )
    delays: list[datetime.timedelta] = []
    _ = await mosfet.started(
        environment, supervisor, runtime.Supervisor.model, hsm.Config(clock=_recording_clock(delays))
    )
    return _Supervised(supervisor=supervisor, owner=owner, environment=environment, delays=delays)


def test_activation_reports_running_under_the_first_operation_id() -> None:
    async def run() -> tuple[list[tuple[str, str]], list[str], runtime.SupervisorData, bool]:
        bot = doubles.QuietBot()
        supervised = await _supervise(bot, _policy())
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        report = supervised.owner.reports[0].data
        assert isinstance(report, runtime.SupervisorData)
        return bot.terminals, supervised.owner.names(), report, supervised.environment.contains(bot)

    terminals, reports, report, attached_to_environment = asyncio.run(run())

    assert terminals == [("bot.activated", "alpha:1")]
    assert reports == ["runtime.supervisor.running"]
    assert report == runtime.SupervisorData(name="alpha", attempt=1)
    assert attached_to_environment


def test_failed_activations_restart_with_doubling_capped_backoff_until_exhausted() -> None:
    async def run() -> tuple[list[tuple[str, str]], list[hsm.Event[typing.Any]], list[datetime.timedelta]]:
        bot = doubles.QuietBot((doubles.FailingDevice(),))
        supervised = await _supervise(bot, _policy(max_restarts=4, backoff_initial=10 * _MS, backoff_max=25 * _MS))
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        backoffs = [delay for delay in supervised.delays if delay < datetime.timedelta(seconds=1)]
        return bot.terminals, supervised.owner.reports, backoffs

    terminals, reports, backoffs = asyncio.run(run())

    # Each attempt is answered under its own id, and each recovery's deactivation under the next one.
    assert terminals == [
        ("bot.activating.failed", "alpha:1"),
        ("bot.deactivated", "alpha:2"),
        ("bot.activating.failed", "alpha:3"),
        ("bot.deactivated", "alpha:4"),
        ("bot.activating.failed", "alpha:5"),
        ("bot.deactivated", "alpha:6"),
        ("bot.activating.failed", "alpha:7"),
        ("bot.deactivated", "alpha:8"),
    ]
    assert backoffs == [10 * _MS, 20 * _MS, 25 * _MS]
    assert [(report.name, report.data) for report in reports] == [
        ("runtime.supervisor.exhausted", runtime.SupervisorData(name="alpha", attempt=4)),
    ]


def test_restart_after_activation_failure_brings_the_bot_and_its_devices_back() -> None:
    async def run() -> tuple[list[tuple[str, str]], runtime.SupervisorData, int, str]:
        device = doubles.FailingDevice(failures=1)
        bot = doubles.QuietBot((device,))
        supervised = await _supervise(bot, _policy())
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        report = supervised.owner.reports[0].data
        assert isinstance(report, runtime.SupervisorData)
        return bot.terminals, report, device.starts, device.state()

    terminals, report, device_starts, device_state = asyncio.run(run())

    assert terminals == [
        ("bot.activating.failed", "alpha:1"),
        ("bot.deactivated", "alpha:2"),
        ("bot.activated", "alpha:3"),
    ]
    assert report == runtime.SupervisorData(name="alpha", attempt=2)
    assert device_starts == 2
    assert device_state.startswith("/Device/attached")


def test_a_stale_activation_terminal_is_ignored() -> None:
    async def run() -> list[str]:
        gate = asyncio.Event()
        bot = doubles.QuietBot((doubles.GatedDevice(gate),))
        supervised = await _supervise(bot, _policy())
        await doubles.wait_until(lambda: bot.state() == "/Bot/activating")
        _ = await hsm.dispatch(
            supervised.environment,
            supervised.supervisor,
            mosfet.ActivatingDoneEvent.with_data_and_id(mosfet.ActivatingDoneEventData(), "alpha:0"),
        )
        await asyncio.sleep(0.02)
        stale = supervised.owner.names()
        gate.set()
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        return [*stale, "--", *supervised.owner.names()]

    assert asyncio.run(run()) == ["--", "runtime.supervisor.running"]


def test_degraded_after_a_stable_run_restarts_from_the_initial_backoff() -> None:
    async def run() -> tuple[list[datetime.timedelta], list[tuple[str, str]], list[str]]:
        bot = doubles.QuietBot((doubles.FailingDevice(failures=1),))
        supervised = await _supervise(bot, _policy(stable_after=30 * _MS))
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        # Running stably for stable_after clears the failed attempt counted before this run.
        await doubles.wait_until(lambda: 30 * _MS in supervised.delays)
        await asyncio.sleep(0.06)
        _ = await hsm.dispatch(
            supervised.environment,
            supervised.supervisor,
            mosfet.DegradedEvent.with_data_and_id(mosfet.DegradedEventData(stage="processing_cancel"), "alpha:3"),
        )
        await doubles.wait_until(lambda: len(supervised.owner.reports) == 2)
        backoffs = [delay for delay in supervised.delays if delay < 30 * _MS]
        return backoffs, bot.terminals, supervised.owner.names()

    backoffs, terminals, reports = asyncio.run(run())

    assert backoffs == [10 * _MS, 10 * _MS]
    assert terminals[-2:] == [("bot.deactivated", "alpha:4"), ("bot.activated", "alpha:5")]
    assert reports == ["runtime.supervisor.running", "runtime.supervisor.running"]


def test_attach_raising_is_recovered_and_counts_against_the_budget() -> None:
    async def run() -> list[str]:
        bot = doubles.QuietBot()
        # Started in another environment first, so the supervisor's attach raises.
        _ = await bot.attach(Environment())
        supervised = await _supervise(bot, _policy(max_restarts=1))
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        return supervised.owner.names()

    assert asyncio.run(run()) == ["runtime.supervisor.exhausted"]


def test_stop_while_the_bot_activates_ends_graceful_and_releases_it() -> None:
    async def run() -> tuple[runtime.SupervisorStoppedData | None, list[tuple[str, str]], bool]:
        bot = doubles.QuietBot((doubles.GatedDevice(asyncio.Event()),))
        supervised = await _supervise(bot, _policy())
        await doubles.wait_until(lambda: bot.state() == "/Bot/activating")
        await supervised.stop()
        await doubles.wait_until(lambda: supervised.owner.stopped() is not None)
        return supervised.owner.stopped(), bot.terminals, lifecycle.is_started(bot)

    stopped, terminals, bot_started = asyncio.run(run())

    assert stopped == runtime.SupervisorStoppedData(name="alpha", outcome="graceful")
    assert terminals == [("bot.deactivated", "alpha:2")]
    assert not bot_started


def test_stop_while_running_deactivates_and_releases_the_bot() -> None:
    async def run() -> tuple[list[str], runtime.SupervisorStoppedData | None, list[tuple[str, str]], bool]:
        bot = doubles.QuietBot()
        supervised = await _supervise(bot, _policy())
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        await supervised.stop()
        await doubles.wait_until(lambda: supervised.owner.stopped() is not None)
        return supervised.owner.names(), supervised.owner.stopped(), bot.terminals, lifecycle.is_started(bot)

    reports, stopped, terminals, bot_started = asyncio.run(run())

    assert reports == ["runtime.supervisor.running", "runtime.supervisor.stopped"]
    assert stopped == runtime.SupervisorStoppedData(name="alpha", outcome="graceful")
    assert terminals == [("bot.activated", "alpha:1"), ("bot.deactivated", "alpha:2")]
    assert not bot_started


def test_stop_while_the_bot_reboots_ends_graceful() -> None:
    async def run() -> tuple[str, runtime.SupervisorStoppedData | None, list[tuple[str, str]]]:
        cognition = doubles.StubCognition()
        bot = doubles.QuietBot(cognition=cognition)
        supervised = await _supervise(bot, _policy())
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        _ = await hsm.dispatch(
            supervised.environment,
            bot,
            dataclasses.replace(
                mosfet.RebootEvent.with_data(mosfet.RebootEventData(reason="cognition_child_teardown_failed")),
                source=hsm.id(cognition),
                target=hsm.id(bot),
            ),
        )
        rebooting = bot.state()
        await supervised.stop()
        await doubles.wait_until(lambda: supervised.owner.stopped() is not None)
        return rebooting, supervised.owner.stopped(), bot.terminals

    rebooting, stopped, terminals = asyncio.run(run())

    assert rebooting == "/Bot/reboot_deactivating"
    assert stopped == runtime.SupervisorStoppedData(name="alpha", outcome="graceful")
    assert terminals == [("bot.activated", "alpha:1"), ("bot.deactivated", "alpha:2")]


def test_stop_while_recovering_waits_for_the_release_and_never_restarts() -> None:
    async def run() -> tuple[runtime.SupervisorStoppedData | None, list[tuple[str, str]], bool]:
        # The bot fails to activate and then never answers the recovery's deactivate, so the supervisor
        # sits in recovering until its stop bound.
        bot = doubles.MuteBot((doubles.FailingDevice(),))
        supervised = await _supervise(bot, _policy(bot_stop_timeout=100 * _MS))
        await doubles.wait_until(lambda: supervised.supervisor.state() == "/Supervisor/recovering/detaching")
        await supervised.stop()
        await doubles.wait_until(lambda: supervised.owner.stopped() is not None)
        return supervised.owner.stopped(), bot.terminals, lifecycle.is_started(bot)

    stopped, terminals, bot_started = asyncio.run(run())

    assert stopped == runtime.SupervisorStoppedData(name="alpha", outcome="graceful")
    assert terminals == [("bot.activating.failed", "alpha:1")]
    assert not bot_started


def test_stop_from_backoff_is_immediate() -> None:
    async def run() -> tuple[runtime.SupervisorStoppedData | None, float]:
        bot = doubles.QuietBot((doubles.FailingDevice(),))
        supervised = await _supervise(
            bot,
            _policy(backoff_initial=datetime.timedelta(seconds=30), backoff_max=datetime.timedelta(seconds=30)),
        )
        await doubles.wait_until(lambda: supervised.supervisor.state() == "/Supervisor/backoff")
        started = asyncio.get_running_loop().time()
        await supervised.stop()
        await doubles.wait_until(lambda: supervised.owner.stopped() is not None)
        return supervised.owner.stopped(), asyncio.get_running_loop().time() - started

    stopped, elapsed = asyncio.run(run())

    assert stopped == runtime.SupervisorStoppedData(name="alpha", outcome="graceful")
    assert elapsed < 1.0


def test_stop_when_exhausted_is_immediate() -> None:
    async def run() -> list[str]:
        bot = doubles.QuietBot((doubles.FailingDevice(),))
        supervised = await _supervise(bot, _policy(max_restarts=1))
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        await supervised.stop()
        await doubles.wait_until(lambda: supervised.owner.stopped() is not None)
        return supervised.owner.names()

    assert asyncio.run(run()) == ["runtime.supervisor.exhausted", "runtime.supervisor.stopped"]


def test_an_unanswered_stop_is_force_released_within_its_bounds() -> None:
    async def run() -> tuple[runtime.SupervisorStoppedData | None, float, bool]:
        bot = doubles.MuteBot()
        supervised = await _supervise(bot, _policy(bot_stop_timeout=50 * _MS, bot_release_timeout=50 * _MS))
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        started = asyncio.get_running_loop().time()
        await supervised.stop()
        await doubles.wait_until(lambda: supervised.owner.stopped() is not None)
        return supervised.owner.stopped(), asyncio.get_running_loop().time() - started, lifecycle.is_started(bot)

    stopped, elapsed, bot_started = asyncio.run(run())

    assert stopped == runtime.SupervisorStoppedData(name="alpha", outcome="timed_out")
    assert elapsed < 1.0
    assert not bot_started


def test_stop_matches_the_terminal_of_the_bots_own_detach_timeout() -> None:
    async def run() -> tuple[runtime.SupervisorStoppedData | None, list[tuple[str, str]]]:
        # The device never answers the detach, so the bot's own deactivation bound ends its detach and
        # its cleanup answers on that timer path.
        bot = doubles.QuietBot((doubles.DeafDevice(),), deactivation_timeout=30 * _MS)
        supervised = await _supervise(bot, _policy(bot_stop_timeout=datetime.timedelta(seconds=2)))
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        await supervised.stop()
        await doubles.wait_until(lambda: supervised.owner.stopped() is not None)
        return supervised.owner.stopped(), bot.terminals

    stopped, terminals = asyncio.run(run())

    assert stopped == runtime.SupervisorStoppedData(name="alpha", outcome="graceful")
    assert terminals == [("bot.activated", "alpha:1"), ("bot.deactivated", "alpha:2")]


def test_the_largest_restart_budget_with_the_smallest_backoff_still_reports_exhaustion() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], datetime.timedelta]:
        bot = doubles.QuietBot((doubles.FailingDevice(),))
        tiny = datetime.timedelta(microseconds=1)
        supervised = await _supervise(
            bot, _policy(max_restarts=runtime.MAX_RESTARTS, backoff_initial=tiny, backoff_max=tiny)
        )
        await doubles.wait_until(lambda: supervised.owner.reports != [], timeout=20.0)
        return supervised.owner.reports, max(delay for delay in supervised.delays if delay <= tiny)

    reports, longest_backoff = asyncio.run(run())

    assert [(report.name, report.data) for report in reports] == [
        (
            "runtime.supervisor.exhausted",
            runtime.SupervisorData(name="alpha", attempt=runtime.MAX_RESTARTS),
        ),
    ]
    assert longest_backoff == datetime.timedelta(microseconds=1)


def test_a_release_that_raises_while_recovering_reports_exhaustion_without_waiting_for_its_bound() -> None:
    async def run() -> tuple[list[str], float]:
        bot = doubles.UnreleasableBot((doubles.FailingDevice(),))
        supervised = await _supervise(
            bot,
            _policy(stop_grace=datetime.timedelta(seconds=60), bot_release_timeout=datetime.timedelta(seconds=30)),
        )
        started = asyncio.get_running_loop().time()
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        return supervised.owner.names(), asyncio.get_running_loop().time() - started

    reports, elapsed = asyncio.run(run())

    # A bot that cannot be released cannot be restarted.
    assert reports == ["runtime.supervisor.exhausted"]
    assert elapsed < 1.0


def test_a_release_that_raises_while_stopping_ends_the_stop_without_waiting_for_its_bound() -> None:
    async def run() -> tuple[runtime.SupervisorStoppedData | None, float]:
        bot = doubles.UnreleasableBot()
        supervised = await _supervise(
            bot,
            _policy(stop_grace=datetime.timedelta(seconds=60), bot_release_timeout=datetime.timedelta(seconds=30)),
        )
        await doubles.wait_until(lambda: supervised.owner.reports != [])
        started = asyncio.get_running_loop().time()
        await supervised.stop()
        await doubles.wait_until(lambda: supervised.owner.stopped() is not None)
        return supervised.owner.stopped(), asyncio.get_running_loop().time() - started

    stopped, elapsed = asyncio.run(run())

    # The bot answered its stop but was not released cleanly, so the stop is not graceful.
    assert stopped == runtime.SupervisorStoppedData(name="alpha", outcome="timed_out")
    assert elapsed < 1.0

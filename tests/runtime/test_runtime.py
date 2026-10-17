import asyncio
import collections.abc
import dataclasses
import datetime
import logging
import pathlib
import threading
import typing

import hsm
import mosfet
import pytest
from mosfet import lifecycle, runtime, scope
from mosfet.environment import Environment

from tests import metric_reader
from tests.runtime.fixtures import factories

_MS = datetime.timedelta(milliseconds=1)
_TARGET_MODULE = "tests.runtime.fixtures.factories"


def _policy(**overrides: object) -> runtime.Policy:
    values: dict[str, object] = {
        "stop_grace": datetime.timedelta(seconds=30),
        "bot_stop_timeout": datetime.timedelta(seconds=1),
        "bot_release_timeout": datetime.timedelta(seconds=1),
        "start_timeout": datetime.timedelta(seconds=10),
        "max_restarts": 3,
        "backoff_initial": 10 * _MS,
        "backoff_max": 40 * _MS,
        "stable_after": datetime.timedelta(seconds=30),
    }
    values.update(overrides)
    return runtime.Policy.model_validate(values)


@dataclasses.dataclass
class _Run:
    process: runtime.Runtime
    environment: Environment
    exited: asyncio.Future[runtime.ExitedData]

    async def stop(self, reason: typing.Literal["signal", "request"] = "request") -> None:
        data = (
            runtime.StopData(reason="signal", signal="SIGTERM")
            if reason == "signal"
            else runtime.StopData(reason=reason)
        )
        _ = await hsm.dispatch(self.environment, self.process, runtime.StopEvent.with_data(data))

    async def exit(self, timeout: float = 5.0) -> runtime.ExitedData:
        return await asyncio.wait_for(asyncio.shield(self.exited), timeout)


async def _run(
    tmp_path: pathlib.Path,
    attribute: str = "configured",
    *,
    policy: runtime.Policy | None = None,
    ready_file: pathlib.Path | None = None,
    module: str = _TARGET_MODULE,
    clock: hsm.Clock | None = None,
) -> _Run:
    environment = Environment()
    exited: asyncio.Future[runtime.ExitedData] = asyncio.get_running_loop().create_future()
    process = runtime.Runtime(
        target=runtime.Target(module=module, attribute=attribute),
        config=runtime.Config(
            name="test",
            instance_id="test-1",
            data_dir=tmp_path,
            environ={},
            secrets=runtime.Secrets(directory=tmp_path / "secrets", environ={}),
        ),
        policy=policy or _policy(),
        environment=environment,
        ready_file=ready_file,
        # Exactly once: a second call would raise InvalidStateError.
        on_exit=exited.set_result,
    )
    config = None if clock is None else hsm.Config(clock=clock)
    _ = await mosfet.started(environment, process, runtime.Runtime.model, config)
    return _Run(process=process, environment=environment, exited=exited)


class _ManualClock(hsm.Clock):
    """An injected clock whose timers fire only when the test advances it past their duration."""

    timers: list[tuple[datetime.timedelta, asyncio.Future[datetime.datetime]]]

    def __init__(self) -> None:
        super().__init__(after=self._timer)
        self.timers = []

    def _timer(self, duration: datetime.timedelta) -> asyncio.Future[datetime.datetime]:
        timer: asyncio.Future[datetime.datetime] = asyncio.get_running_loop().create_future()
        self.timers.append((duration, timer))
        return timer

    def advance(self, duration: datetime.timedelta) -> None:
        """Fire every pending timer of exactly ``duration``."""

        fired = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC) + duration
        for requested, timer in self.timers:
            if requested == duration and not timer.done():
                timer.set_result(fired)


def _assemble(
    monkeypatch: pytest.MonkeyPatch, build: collections.abc.Callable[[runtime.Config], runtime.Assembly]
) -> None:
    monkeypatch.setattr(factories, "configured", build)


def _gate_second_supervisor_start(monkeypatch: pytest.MonkeyPatch, *, gate: asyncio.Event, started_first: bool) -> None:
    """Hold the second supervisor's start inside ``mosfet.started`` until ``gate`` opens.

    With ``started_first`` the supervisor is started before the hold, so a stop interrupts a start that
    already ran; otherwise the hold comes first, so a stop interrupts a start that never ran.
    """

    original = mosfet.started
    supervisors: list[hsm.Instance] = []

    async def started(
        ctx: hsm.Context | None, instance: hsm.Instance, model: hsm.Model, config: hsm.Config | None = None
    ) -> hsm.Instance:
        if not isinstance(instance, runtime.Supervisor):
            return await original(ctx, instance, model, config)
        supervisors.append(instance)
        if len(supervisors) != 2:
            return await original(ctx, instance, model, config)
        if started_first:
            result = await original(ctx, instance, model, config)
            _ = await gate.wait()
            return result
        _ = await gate.wait()
        return await original(ctx, instance, model, config)

    monkeypatch.setattr(mosfet, "started", started)


@pytest.mark.parametrize(
    ("module", "attribute", "kind"),
    [
        ("tests.runtime.fixtures.absent", "build", "import_failed"),
        (_TARGET_MODULE, "absent", "factory_missing"),
        (_TARGET_MODULE, "not_callable", "factory_missing"),
        (_TARGET_MODULE, "raising", "factory_raised"),
        (_TARGET_MODULE, "wrong_type", "invalid_assembly"),
        (_TARGET_MODULE, "needs_secret", "missing_secret"),
    ],
)
def test_boot_failure_exits_78_and_logs_its_kind(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture, module: str, attribute: str, kind: str
) -> None:
    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path, attribute, module=module)
        return await started.exit()

    with caplog.at_level(logging.ERROR, logger="mosfet.runtime.runtime"):
        exited = asyncio.run(run())

    assert exited == runtime.ExitedData(code=78, reason="boot_failed")
    assert f"runtime boot failed kind={kind} target={module}:{attribute}" in caplog.text


def test_graceful_stop_stops_every_bot_before_the_actors_in_reverse_order(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal: list[str] = []
    bots = (factories.QuietBot(), factories.QuietBot())

    class CheckingActor(factories.RecordingActor):
        @typing.override
        def stop(self, ctx: hsm.Context) -> collections.abc.Awaitable[None]:
            journal.append(f"bots started: {any(lifecycle.is_started(bot) for bot in bots)}")
            return super().stop(ctx)

    _assemble(
        monkeypatch,
        lambda config: runtime.Assembly(
            bots={"first": bots[0], "second": bots[1]},
            actors=(CheckingActor("a", journal), CheckingActor("b", journal)),
        ),
    )
    ready = tmp_path / "ready"

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path, ready_file=ready)
        await factories.wait_until(ready.exists)
        await started.stop()
        return await started.exit()

    exited = asyncio.run(run())

    assert exited == runtime.ExitedData(code=0, reason="stopped")
    assert journal == ["start:a", "start:b", "bots started: False", "stop:b", "bots started: False", "stop:a"]
    assert bots[0].terminals == [("bot.activated", "first:1"), ("bot.deactivated", "first:2")]
    assert bots[1].terminals == [("bot.activated", "second:1"), ("bot.deactivated", "second:2")]


def test_actors_and_bots_share_the_one_process_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actor = factories.RecordingActor("shared", [])
    bot = factories.QuietBot()
    _assemble(monkeypatch, lambda config: runtime.Assembly(bots={"only": bot}, actors=(actor,)))
    ready = tmp_path / "ready"

    async def run() -> tuple[bool, bool, bool]:
        started = await _run(tmp_path, ready_file=ready)
        await factories.wait_until(ready.exists)
        observed = (
            Environment.from_context(actor.context()) is started.environment,
            scope.is_private(actor.context()),
            started.environment.contains(bot),
        )
        await started.stop()
        _ = await started.exit()
        return observed

    shares_environment, private, bot_in_environment = asyncio.run(run())

    assert shares_environment
    assert not private
    assert bot_in_environment


def test_stop_while_booting_exits_0(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    building = threading.Event()
    release = threading.Event()

    def slow(config: runtime.Config) -> runtime.Assembly:
        building.set()
        _ = release.wait(5)
        return factories.quiet(config)

    _assemble(monkeypatch, slow)

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path)
        await factories.wait_until(building.is_set)
        await started.stop()
        try:
            return await started.exit()
        finally:
            release.set()

    assert asyncio.run(run()) == runtime.ExitedData(code=0, reason="stopped")


def test_stop_while_actors_start_exits_0_and_stops_what_bring_up_recorded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal: list[str] = []
    gate = asyncio.Event()
    bot = factories.QuietBot()
    _assemble(
        monkeypatch,
        lambda config: runtime.Assembly(
            bots={"only": bot},
            actors=(factories.RecordingActor("a", journal), factories.GatedActor("b", journal, gate)),
        ),
    )

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path)
        await factories.wait_until(lambda: journal == ["start:a"])
        await started.stop()
        return await started.exit()

    exited = asyncio.run(run())

    assert exited == runtime.ExitedData(code=0, reason="stopped")
    assert journal == ["start:a", "stop:b", "stop:a"]
    assert bot.terminals == []


def test_a_bot_that_activates_before_the_last_supervisor_starts_still_makes_the_process_ready(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = factories.QuietBot()
    _assemble(monkeypatch, lambda config: runtime.Assembly(bots={"first": first, "second": factories.QuietBot()}))
    gate = asyncio.Event()
    _gate_second_supervisor_start(monkeypatch, gate=gate, started_first=False)
    ready = tmp_path / "ready"

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path, ready_file=ready)
        # The first bot activates and its supervisor reports running while bring-up is still starting
        # the second supervisor.
        await factories.wait_until(lambda: first.terminals == [("bot.activated", "first:1")])
        await asyncio.sleep(0.05)
        gate.set()
        await factories.wait_until(ready.exists)
        await started.stop()
        return await started.exit()

    assert asyncio.run(run()) == runtime.ExitedData(code=0, reason="stopped")


def test_stop_before_a_supervisor_start_ran_does_not_wait_for_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = factories.QuietBot(), factories.QuietBot()
    _assemble(monkeypatch, lambda config: runtime.Assembly(bots={"first": first, "second": second}))
    _gate_second_supervisor_start(monkeypatch, gate=asyncio.Event(), started_first=False)

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path)
        await factories.wait_until(lambda: first.terminals == [("bot.activated", "first:1")])
        await started.stop()
        # The grace is 30 s: exiting within the default 5 s wait proves nothing waited on the second.
        return await started.exit()

    exited = asyncio.run(run())

    assert exited == runtime.ExitedData(code=0, reason="stopped")
    assert first.terminals == [("bot.activated", "first:1"), ("bot.deactivated", "first:2")]
    assert second.terminals == []


def test_stop_that_interrupts_a_supervisor_start_still_stops_its_bot(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second = factories.QuietBot()
    _assemble(monkeypatch, lambda config: runtime.Assembly(bots={"first": factories.QuietBot(), "second": second}))
    _gate_second_supervisor_start(monkeypatch, gate=asyncio.Event(), started_first=True)

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path)
        await factories.wait_until(lambda: second.terminals == [("bot.activated", "second:1")])
        await started.stop()
        return await started.exit()

    exited = asyncio.run(run())

    assert exited == runtime.ExitedData(code=0, reason="stopped")
    assert second.terminals == [("bot.activated", "second:1"), ("bot.deactivated", "second:2")]
    assert not lifecycle.is_started(second)


def test_supervisor_start_failure_with_none_started_exits_78_without_waiting_for_the_grace(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assemble(monkeypatch, factories.quiet)
    original = mosfet.started

    async def started(
        ctx: hsm.Context | None, instance: hsm.Instance, model: hsm.Model, config: hsm.Config | None = None
    ) -> hsm.Instance:
        if isinstance(instance, runtime.Supervisor):
            raise RuntimeError("supervisor failed to start")
        return await original(ctx, instance, model, config)

    monkeypatch.setattr(mosfet, "started", started)

    async def run() -> runtime.ExitedData:
        return await (await _run(tmp_path)).exit()

    assert asyncio.run(run()) == runtime.ExitedData(code=78, reason="boot_failed")


def test_actor_start_failure_exits_78_without_waiting_for_the_grace(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal: list[str] = []
    _assemble(
        monkeypatch,
        lambda config: runtime.Assembly(
            bots={"only": factories.QuietBot()},
            actors=(factories.RecordingActor("a", journal), factories.FailingActor("b", journal)),
        ),
    )

    async def run() -> runtime.ExitedData:
        return await (await _run(tmp_path)).exit()

    assert asyncio.run(run()) == runtime.ExitedData(code=78, reason="boot_failed")
    assert journal == ["start:a", "stop:b", "stop:a"]


def test_a_bot_that_never_answers_is_force_released_and_the_process_still_exits_0(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = factories.MuteBot()
    _assemble(monkeypatch, lambda config: runtime.Assembly(bots={"mute": bot}))
    policy = _policy(stop_grace=800 * _MS, bot_stop_timeout=200 * _MS, bot_release_timeout=200 * _MS)

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path, policy=policy)
        await factories.wait_until(lambda: bot.terminals == [("bot.activated", "mute:1")])
        await started.stop()
        return await started.exit()

    assert asyncio.run(run()) == runtime.ExitedData(code=0, reason="stopped")
    assert not lifecycle.is_started(bot)


def test_an_actor_stop_that_hangs_forces_exit_1_at_the_single_grace(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = factories.QuietBot()
    _assemble(
        monkeypatch,
        lambda config: runtime.Assembly(bots={"only": bot}, actors=(factories.HangingActor("stuck", []),)),
    )
    policy = _policy(stop_grace=400 * _MS, bot_stop_timeout=100 * _MS, bot_release_timeout=100 * _MS)

    async def run() -> tuple[runtime.ExitedData, float]:
        started = await _run(tmp_path, policy=policy)
        await factories.wait_until(lambda: bot.terminals == [("bot.activated", "only:1")])
        stopping = asyncio.get_running_loop().time()
        await started.stop()
        exited = await started.exit()
        return exited, asyncio.get_running_loop().time() - stopping

    exited, elapsed = asyncio.run(run())

    assert exited == runtime.ExitedData(code=1, reason="forced")
    # One timer bounds the whole stop: well under twice the grace.
    assert 0.35 <= elapsed < 0.75


def test_the_stop_grace_runs_on_the_injected_clock_and_the_runtime_reads_no_wall_clock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = factories.QuietBot()
    _assemble(
        monkeypatch,
        lambda config: runtime.Assembly(bots={"only": bot}, actors=(factories.HangingActor("stuck", []),)),
    )
    grace = datetime.timedelta(seconds=30)
    policy = _policy(stop_grace=grace, bot_stop_timeout=100 * _MS, bot_release_timeout=100 * _MS)
    clock = _ManualClock()

    async def run() -> tuple[bool, runtime.ExitedData]:
        started = await _run(tmp_path, policy=policy, clock=clock)
        await factories.wait_until(lambda: bot.terminals == [("bot.activated", "only:1")])
        await started.stop()
        await factories.wait_until(lambda: any(requested == grace for requested, _ in clock.timers))
        # Wall time passes; only the injected clock ends the stuck stop.
        _ = await asyncio.wait([started.exited], timeout=0.2)
        exited_early = started.exited.done()
        clock.advance(grace)
        return exited_early, await started.exit()

    def stop_durations(metrics: metric_reader.Reader) -> int:
        return sum(point.count for point in metrics.points("bot.runtime.stop.duration"))

    with metric_reader.recording(tmp_path, monkeypatch) as metrics:
        before = stop_durations(metrics)
        exited_early, exited = asyncio.run(run())
        after = stop_durations(metrics)

    assert not exited_early
    assert exited == runtime.ExitedData(code=1, reason="forced")
    # Time enters the machine only through its injected clock: it measures no wall-clock duration of its own.
    assert after == before


def test_an_actor_stop_that_ignores_cancellation_still_forces_exit_1_at_the_single_grace(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = factories.QuietBot()
    actor = factories.StubbornActor("stubborn", [])
    _assemble(monkeypatch, lambda config: runtime.Assembly(bots={"only": bot}, actors=(actor,)))
    policy = _policy(stop_grace=400 * _MS, bot_stop_timeout=100 * _MS, bot_release_timeout=100 * _MS)

    async def run() -> tuple[runtime.ExitedData, float]:
        started = await _run(tmp_path, policy=policy)
        await factories.wait_until(lambda: bot.terminals == [("bot.activated", "only:1")])
        stopping = asyncio.get_running_loop().time()
        await started.stop()
        try:
            exited = await started.exit(timeout=2.0)
        finally:
            # The actor's stop outlives the runtime; only its release lets this event loop close.
            actor.released.set()
        return exited, asyncio.get_running_loop().time() - stopping

    exited, elapsed = asyncio.run(run())

    assert exited == runtime.ExitedData(code=1, reason="forced")
    assert 0.35 <= elapsed < 0.75


def test_a_stop_request_while_stopping_is_consumed(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bot = factories.MuteBot()
    _assemble(monkeypatch, lambda config: runtime.Assembly(bots={"mute": bot}))
    policy = _policy(
        stop_grace=datetime.timedelta(seconds=2), bot_stop_timeout=300 * _MS, bot_release_timeout=100 * _MS
    )

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path, policy=policy)
        await factories.wait_until(lambda: bot.terminals == [("bot.activated", "mute:1")])
        await started.stop()
        await started.stop()
        return await started.exit()

    assert asyncio.run(run()) == runtime.ExitedData(code=0, reason="stopped")


def test_a_second_signal_while_stopping_forces_exit_1(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bot = factories.MuteBot()
    _assemble(monkeypatch, lambda config: runtime.Assembly(bots={"mute": bot}))
    policy = _policy(
        stop_grace=datetime.timedelta(seconds=2), bot_stop_timeout=500 * _MS, bot_release_timeout=100 * _MS
    )

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path, policy=policy)
        await factories.wait_until(lambda: bot.terminals == [("bot.activated", "mute:1")])
        await started.stop("signal")
        await started.stop("signal")
        return await started.exit()

    assert asyncio.run(run()) == runtime.ExitedData(code=1, reason="forced")


def test_exhaustion_stops_the_other_bots_and_exits_70_even_after_a_first_signal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mute = factories.MuteBot()
    _assemble(
        monkeypatch,
        lambda config: runtime.Assembly(
            bots={"failing": factories.QuietBot((factories.FailingDevice(),)), "mute": mute}
        ),
    )
    policy = _policy(
        stop_grace=datetime.timedelta(seconds=2),
        bot_stop_timeout=300 * _MS,
        bot_release_timeout=100 * _MS,
        max_restarts=1,
    )
    ready = tmp_path / "ready"

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path, policy=policy, ready_file=ready)
        # The signal must arrive after the exhaustion has begun the stop; the mute bot holds that stop open
        # for bot_stop_timeout, so the first signal lands inside it.
        await factories.wait_until(lambda: started.process.state().startswith("/Runtime/stopping"))
        await started.stop("signal")
        return await started.exit()

    exited = asyncio.run(run())

    assert exited == runtime.ExitedData(code=70, reason="exhausted")
    assert not lifecycle.is_started(mute)
    assert not ready.exists()


def test_ready_file_exists_only_while_ready(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _assemble(monkeypatch, factories.quiet)
    ready = tmp_path / "ready"

    async def run() -> tuple[bool, bool]:
        started = await _run(tmp_path, ready_file=ready)
        before = ready.exists()
        await factories.wait_until(ready.exists)
        await started.stop()
        _ = await started.exit()
        return before, ready.exists()

    before, after = asyncio.run(run())

    assert not before
    assert not after


def test_a_ready_file_left_by_an_earlier_run_is_removed_at_boot(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = tmp_path / "ready"
    # A SIGKILL, OOM kill or host crash skips the ready state's exit, so the previous run's file remains.
    ready.touch()
    seen_while_booting: list[bool] = []

    def build(config: runtime.Config) -> runtime.Assembly:
        seen_while_booting.append(ready.exists())
        return factories.quiet(config)

    _assemble(monkeypatch, build)

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path, ready_file=ready, clock=_ManualClock())
        # Boot has looked, and the file exists again: touched by this run's ready entry.
        await factories.wait_until(lambda: bool(seen_while_booting) and ready.exists())
        await started.stop()
        return await started.exit()

    exited = asyncio.run(run())

    assert seen_while_booting == [False]
    assert exited == runtime.ExitedData(code=0, reason="stopped")
    assert not ready.exists()


def test_a_stop_request_a_supervisor_fails_to_take_still_stops_the_actors_and_exits_0(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal: list[str] = []
    bot = factories.QuietBot()
    _assemble(
        monkeypatch,
        lambda config: runtime.Assembly(bots={"only": bot}, actors=(factories.RecordingActor("a", journal),)),
    )
    original = runtime.Supervisor.dispatch

    def dispatch(
        self: runtime.Supervisor, ctx: hsm.Context, event: hsm.Event[typing.Any]
    ) -> collections.abc.Awaitable[bool]:
        if isinstance(event.data, runtime.SupervisorStopData):

            async def lost() -> bool:
                raise RuntimeError("stop request lost")

            return lost()
        return original(self, ctx, event)

    monkeypatch.setattr(runtime.Supervisor, "dispatch", dispatch)

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path)
        await factories.wait_until(lambda: bot.terminals == [("bot.activated", "only:1")])
        await started.stop()
        return await started.exit()

    assert asyncio.run(run()) == runtime.ExitedData(code=0, reason="stopped")
    assert journal == ["start:a", "stop:a"]


def test_a_supervisor_whose_stop_raises_still_lets_the_actors_stop_and_exits_0(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal: list[str] = []
    bot = factories.QuietBot()
    _assemble(
        monkeypatch,
        lambda config: runtime.Assembly(bots={"only": bot}, actors=(factories.RecordingActor("a", journal),)),
    )

    def stop(self: runtime.Supervisor, ctx: hsm.Context) -> collections.abc.Awaitable[None]:
        del self, ctx

        async def refuse() -> None:
            raise RuntimeError("supervisor refused to stop")

        return refuse()

    monkeypatch.setattr(runtime.Supervisor, "stop", stop)

    async def run() -> runtime.ExitedData:
        started = await _run(tmp_path)
        await factories.wait_until(lambda: bot.terminals == [("bot.activated", "only:1")])
        await started.stop()
        return await started.exit()

    assert asyncio.run(run()) == runtime.ExitedData(code=0, reason="stopped")
    assert journal == ["start:a", "stop:a"]

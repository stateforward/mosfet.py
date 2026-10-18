"""``bot.run``: the process a bot file runs, its checks (B0104..B0106), its exit codes and its signals.

DSL-21 of the wave 7 bot DSL design (W7-9, W7-10; OQ-26..OQ-34). What ends before the event loop
starts (a bad model, an actor that is not one, a bot that cannot be built) runs ``bot.run`` in this
process and catches its ``SystemExit``. Signals and the bounded exit run real bot files, the packages
under ``tests/bot/run/fixtures``, as ``python -m`` children: only the child gets the repository root on
``PYTHONPATH``, and it runs in a temporary directory with its telemetry written there.
"""

import ast
import collections.abc
import os
import pathlib
import signal
import subprocess
import sys
import time
import typing

import hsm
import pytest

import mosfet
from mosfet import bot
from mosfet.abilities import cognition
from mosfet.bot import diagnostics, testing
from mosfet.devices import phone
from tests import metric_reader

_REPO = pathlib.Path(__file__).resolve().parents[3]
_FIXTURES = "tests.bot.run.fixtures"
# A child imports mosfet and the test doubles before it can be signalled; that start-up is not what is
# under test, so it gets a generous bound.
_START_BOUND = 60.0
# Wall-clock slack for scheduling the child and reaping it, on top of the bound under test.
_SLACK = 1.0
# The process exit after on_exit is bounded by 1.5 s (telemetry flush 0.5 s, tasks and threads 1 s).
_EXIT_BOUND = 1.5
_RUN_CODES = ("B0104", "B0105", "B0106", "B0107")


@pytest.fixture
def here(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """In-process runs keep telemetry disabled and leave no files outside ``tmp_path``."""

    monkeypatch.setenv("BOT_OTEL_DISABLED", "1")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _process(directory: pathlib.Path, *argv: str) -> bot.Process:
    return bot.process(argv=["--data-dir", str(directory / "data"), *argv], environ={})


def _exit(run: collections.abc.Callable[[], object]) -> int:
    with pytest.raises(SystemExit) as exited:
        _ = run()
    code = exited.value.code
    assert isinstance(code, int)
    return code


def _raising_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    def __init__(self: object, **given: object) -> None:
        del self, given
        raise ValueError("this processor cannot reason")

    monkeypatch.setattr(cognition.Reasoning, "__init__", __init__)


def test_a_model_passed_twice_is_b0105_and_exits_78(here: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = bot.define("quiet", bot.cognition(testing.Scripted()), bot.allow("no_devices"))

    assert _exit(lambda: bot.run(model, model, process=_process(here))) == 78

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error[B0105]: bot `quiet` is passed to `bot.run` twice")
    assert "passed again here" in captured.err
    assert captured.err.rstrip().endswith("error: could not run bot `quiet`: 1 error")
    # Nothing ran: the data dir is made only once the models are known good.
    assert not (here / "data").exists()


def test_two_models_with_one_name_are_b0104(here: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    first, second = (
        bot.define("quiet", bot.cognition(testing.Scripted()), bot.allow("no_devices")),
        bot.define("quiet", bot.cognition(testing.Scripted()), bot.allow("no_devices")),
    )

    assert _exit(lambda: bot.run(first, second, process=_process(here))) == 78

    err = capsys.readouterr().err
    assert err.startswith("error[B0104]: two bots passed to `bot.run` are both named `quiet`")
    assert "first defined here" in err
    assert "help: give the second bot its own name" in err


def test_an_actor_that_is_not_one_is_b0106(here: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = bot.define("quiet", bot.cognition(testing.Scripted()), bot.allow("no_devices"))
    processor = testing.Scripted()

    assert (
        _exit(
            lambda: bot.run(
                model, actors=(typing.cast(hsm.Instance, typing.cast(object, processor)),), process=_process(here)
            )
        )
        == 78
    )

    err = capsys.readouterr().err
    assert err.startswith("error[B0106]: `bot.run` is given a `Scripted` as a shared actor")
    assert "this is a `Scripted`, not an actor" in err


def test_every_mistake_is_reported_at_once(here: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = bot.define("quiet", bot.cognition(testing.Scripted()), bot.allow("no_devices"))

    code = _exit(lambda: bot.run(model, model, actors=(typing.cast(hsm.Instance, object()),), process=_process(here)))

    assert code == 78
    err = capsys.readouterr().err
    assert "error[B0105]" in err and "error[B0106]" in err
    assert err.rstrip().endswith("error: could not run bot `quiet`: 2 errors")


@pytest.mark.parametrize("count", [0, 33])
def test_a_process_runs_one_to_32_bots(here: pathlib.Path, capsys: pytest.CaptureFixture[str], count: int) -> None:
    models = [
        bot.define(f"bot_{index}", bot.cognition(testing.Scripted()), bot.allow("no_devices")) for index in range(count)
    ]

    assert _exit(lambda: bot.run(*models, process=_process(here))) == 78
    assert capsys.readouterr().err == f"error: bot.run runs 1 to 32 bots, not {count}\n"


def test_something_that_is_not_a_model_is_a_type_error(here: pathlib.Path) -> None:
    with pytest.raises(
        TypeError, match=r"^bot.run takes bot.Models, not function: pass the model `bot.define` returned$"
    ):
        bot.run(typing.cast(bot.Model, bot.define), process=_process(here))


def test_a_bot_that_cannot_be_built_is_b0102_and_exits_78_after_making_the_data_dir_private(
    here: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = bot.define("quiet", bot.cognition(testing.Scripted()), bot.allow("no_devices"))
    _raising_reasoning(monkeypatch)

    assert _exit(lambda: bot.run(model, process=_process(here))) == 78

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error[B0102]: bot `quiet` could not be built")
    assert "ValueError: this processor cannot reason" in captured.err
    assert "Traceback" not in captured.err
    assert (here / "data").stat().st_mode & 0o777 == 0o700


def test_mosfet_traceback_shows_the_cause(
    here: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = bot.define("quiet", bot.cognition(testing.Scripted()), bot.allow("no_devices"))
    _raising_reasoning(monkeypatch)
    process = bot.process(argv=["--data-dir", str(here / "data")], environ={"MOSFET_TRACEBACK": "1"})

    assert _exit(lambda: bot.run(model, process=process)) == 78

    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err
    assert "ValueError: this processor cannot reason" in err


def test_a_data_dir_that_cannot_be_created_exits_78(here: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    blocker = here / "file"
    _ = blocker.write_text("not a directory", encoding="utf-8")
    process = bot.process(argv=["--data-dir", str(blocker / "data")], environ={})

    assert (
        _exit(
            lambda: bot.run(
                bot.define("quiet", bot.cognition(testing.Scripted()), bot.allow("no_devices")), process=process
            )
        )
        == 78
    )
    assert capsys.readouterr().err.startswith("error: cannot create the data dir ")


def test_without_a_process_bot_run_reads_this_ones_flags(
    here: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    del here
    monkeypatch.setattr(sys, "argv", ["greeter", "--bogus"])

    assert _exit(lambda: bot.run(bot.define("quiet", bot.cognition(testing.Scripted()), bot.allow("no_devices")))) == 64
    err = capsys.readouterr().err
    assert err.startswith("error[B0107]: bad process option: unknown option `--bogus`")
    assert "test_run.py:" in err


# The explanations' examples


_RUN_EXPLANATIONS = [explanation for explanation in diagnostics.EXPLANATIONS if explanation.code in _RUN_CODES]


def _namespace() -> dict[str, object]:
    return {"bot": bot, "phone": phone, "mosfet": mosfet, "processor": testing.Scripted()}


@pytest.mark.parametrize("explanation", _RUN_EXPLANATIONS, ids=[explanation.code for explanation in _RUN_EXPLANATIONS])
def test_each_run_explanations_mistake_is_its_code(
    here: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    explanation: diagnostics.Explanation,
) -> None:
    del here
    # The examples read this process's options, as a bot file's __main__ does: run as if with no flags.
    monkeypatch.setattr(sys, "argv", ["greeter"])
    code = compile(explanation.wrong, "<explanation>", "exec")

    # The examples are this repository's own code, documented in docs/bot_diagnostics.md.
    assert _exit(lambda: exec(code, _namespace())) == (64 if explanation.code == "B0107" else 78)
    assert f"error[{explanation.code}]" in capsys.readouterr().err


def test_the_b0107_explanations_fix_reads_cleanly() -> None:
    (explanation,) = [explanation for explanation in _RUN_EXPLANATIONS if explanation.code == "B0107"]
    namespace = _namespace()

    exec(compile(explanation.right, "<explanation>", "exec"), namespace)

    assert isinstance(namespace["process"], bot.Process)


@pytest.mark.parametrize("code", _RUN_CODES)
def test_every_run_code_has_an_explanation(code: str) -> None:
    assert bot.explain(code).startswith(f"error[{code}]: ")


# Child processes


class _Child:
    """One bot file run as a ``python -m`` (or ``python -c``) child process in its own temporary directory."""

    popen: subprocess.Popen[bytes]
    ready: pathlib.Path
    data: pathlib.Path

    def __init__(self, tmp_path: pathlib.Path, *command: str) -> None:
        self.ready = tmp_path / "ready"
        self.data = tmp_path / "data"
        environ = {name: value for name, value in os.environ.items() if not name.startswith(("OTEL_", "BOT_OTEL_"))}
        environ.update(PYTHONPATH=str(_REPO), MOSFET_DATA_DIR=str(self.data), MOSFET_READY_FILE=str(self.ready))
        self.popen = subprocess.Popen(
            [sys.executable, *command],
            cwd=tmp_path,
            env=environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def wait_for(self, condition: collections.abc.Callable[[], bool]) -> None:
        deadline = time.monotonic() + _START_BOUND
        while not condition():
            if self.popen.poll() is not None:
                _, stderr = self.popen.communicate()
                pytest.fail(f"the bot exited {self.popen.returncode} early: {stderr.decode(errors='replace')}")
            if time.monotonic() > deadline:
                self.popen.kill()
                _ = self.popen.communicate()
                pytest.fail("the bot did not reach the expected state")
            time.sleep(0.02)

    def finish(self, bound: float) -> tuple[int, bytes, bytes, float]:
        """Wait at most ``bound`` seconds for the exit; return code, stdout, stderr and the time it took."""

        started = time.monotonic()
        try:
            stdout, stderr = self.popen.communicate(timeout=bound)
        except subprocess.TimeoutExpired:
            self.popen.kill()
            _ = self.popen.communicate()
            pytest.fail(f"the bot did not exit within {bound} s")
        return self.popen.returncode, stdout, stderr, time.monotonic() - started


def _bot_file(tmp_path: pathlib.Path, name: str, *flags: str) -> _Child:
    return _Child(tmp_path, "-m", f"{_FIXTURES}.{name}", *flags)


def _stop_durations(directory: pathlib.Path) -> list[tuple[object, float]]:
    """``(outcome, seconds)`` of every stop the child recorded in its OTEL metric file."""

    points = metric_reader.exported(directory / "otel-metrics.jsonl", "bot.runtime.stop.duration")
    return [(point.attributes.get("outcome"), point.sum) for point in points]


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_a_signal_after_ready_stops_gracefully_with_0_and_empty_stdout(
    tmp_path: pathlib.Path, signum: signal.Signals
) -> None:
    child = _bot_file(tmp_path, "quiet")
    child.wait_for(child.ready.exists)

    child.popen.send_signal(signum)
    code, stdout, stderr, elapsed = child.finish(8 + _EXIT_BOUND + _SLACK)

    assert code == 0, stderr.decode(errors="replace")
    assert stdout == b""
    assert not child.ready.exists()
    assert child.data.stat().st_mode & 0o777 == 0o700
    # The process boundary times the stop from the signal to the runtime's exit.
    ((outcome, seconds),) = _stop_durations(tmp_path)
    assert outcome == "graceful"
    assert 0 < seconds < elapsed


def test_a_stale_ready_file_is_removed_at_boot_and_sigterm_while_activating_exits_0(tmp_path: pathlib.Path) -> None:
    # A SIGKILLed run left its ready file behind.
    (tmp_path / "ready").touch()
    child = _bot_file(tmp_path, "activating")
    child.wait_for(lambda: (child.data / "activating").exists())

    assert not child.ready.exists()
    child.popen.send_signal(signal.SIGTERM)
    code, stdout, stderr, _elapsed = child.finish(8 + _EXIT_BOUND + _SLACK)

    assert code == 0, stderr.decode(errors="replace")
    assert stdout == b""
    assert not child.ready.exists()


def test_a_second_sigterm_while_the_stop_is_under_way_forces_exit_1(tmp_path: pathlib.Path) -> None:
    child = _bot_file(tmp_path, "stubborn")
    child.wait_for(child.ready.exists)

    child.popen.send_signal(signal.SIGTERM)
    # The stubborn actor holds the stop open for the whole 8 s stop_grace.
    time.sleep(0.2)
    child.popen.send_signal(signal.SIGTERM)
    code, stdout, stderr, _elapsed = child.finish(_EXIT_BOUND + _SLACK)

    assert code == 1, stderr.decode(errors="replace")
    assert stdout == b""
    ((outcome, seconds),) = _stop_durations(tmp_path)
    assert outcome == "forced"
    assert seconds >= 0.2


def test_an_actor_that_ignores_cancellation_cannot_hold_the_process_past_its_bound(tmp_path: pathlib.Path) -> None:
    grace = 2.0
    child = _bot_file(
        tmp_path, "stubborn", "--stop-grace", str(grace), "--bot-stop-timeout", "0.5", "--bot-release-timeout", "0.5"
    )
    child.wait_for(child.ready.exists)

    child.popen.send_signal(signal.SIGTERM)
    code, stdout, stderr, elapsed = child.finish(grace + _EXIT_BOUND + _SLACK)

    assert code == 1, stderr.decode(errors="replace")
    assert stdout == b""
    assert elapsed < grace + _EXIT_BOUND + _SLACK


def test_the_observers_run_is_given_watch_each_bot_it_runs(tmp_path: pathlib.Path) -> None:
    """DSL-26: ``bot.run(observe=)`` adds the observers to each bot's body, which only watches."""

    child = _bot_file(tmp_path, "observed")
    child.wait_for(child.ready.exists)
    seen_running = (child.data / "observed").read_text(encoding="utf-8").split()

    child.popen.send_signal(signal.SIGTERM)
    code, stdout, stderr, _elapsed = child.finish(8 + _EXIT_BOUND + _SLACK)

    assert code == 0, stderr.decode(errors="replace")
    assert stdout == b""
    assert seen_running == ["bot.activate"]
    assert (child.data / "observed").read_text(encoding="utf-8").split() == ["bot.activate", "bot.deactivate"]


def test_a_bot_that_exhausts_its_restart_budget_exits_70(tmp_path: pathlib.Path) -> None:
    child = _bot_file(tmp_path, "failing", "--max-restarts", "1", "--backoff-initial", "0.01", "--backoff-max", "0.01")

    code, stdout, stderr, _elapsed = child.finish(_START_BOUND)

    assert code == 70, stderr.decode(errors="replace")
    assert stdout == b""
    assert not child.ready.exists()


_PRELUDE = """\
from mosfet import bot
from mosfet.bot import testing
from mosfet.devices import phone

processor = testing.Scripted()
"""


@pytest.mark.parametrize("code", ["B0104", "B0105", "B0106"])
def test_each_run_explanations_fix_runs_until_it_is_stopped(tmp_path: pathlib.Path, code: str) -> None:
    (explanation,) = [explanation for explanation in _RUN_EXPLANATIONS if explanation.code == code]
    source = _PRELUDE + explanation.right
    _ = ast.parse(source)
    child = _Child(tmp_path, "-c", source)
    child.wait_for(child.ready.exists)

    child.popen.send_signal(signal.SIGTERM)
    exited, stdout, stderr, _elapsed = child.finish(8 + _EXIT_BOUND + _SLACK)

    assert exited == 0, stderr.decode(errors="replace")
    assert stdout == b""

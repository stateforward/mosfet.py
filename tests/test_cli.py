"""``mosfet run``: argument and target validation, exit codes, signals, and the bounded process exit.

Usage and boot failures run ``cli.main`` in this process. Signal behaviour runs the real command in a child
process (``python -m mosfet.cli``) against the test factories in ``tests.runtime.fixtures.factories``;
only the child gets the repository root on ``PYTHONPATH``, and it runs in a temporary directory with
OpenTelemetry export confined there.
"""

import collections.abc
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest
from mosfet import cli

from tests import metric_reader

_REPO = pathlib.Path(__file__).resolve().parents[1]
_FACTORIES = "tests.runtime.fixtures.factories"
# A child imports mosfet and the test doubles before it can be signalled; that start-up is not what is
# under test, so it gets a generous bound.
_START_BOUND = 60.0
# Wall-clock slack for scheduling the child and reaping it, on top of the bound under test.
_SLACK = 1.0
# The process exit after on_exit is bounded by 1.5 s (telemetry flush 0.5 s, tasks and threads 1 s).
_EXIT_BOUND = 1.5


@pytest.fixture
def no_telemetry(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """In-process runs configure telemetry disabled and leave no files outside ``tmp_path``."""

    monkeypatch.setenv("BOT_OTEL_DISABLED", "1")
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([], id="no-command"),
        pytest.param(["serve"], id="unknown-command"),
        pytest.param(["run"], id="no-target"),
        pytest.param(["run", "not-a-target"], id="no-attribute"),
        pytest.param(["run", "pkg.module:"], id="empty-attribute"),
        pytest.param(["run", "pkg-name:build"], id="module-not-identifier"),
        pytest.param(["run", "--stop-grace", "soon", "pkg.module:build"], id="duration-not-a-number"),
        pytest.param(["run", "--stop-grace", "nan", "pkg.module:build"], id="duration-not-finite"),
        pytest.param(["run", "--stop-grace", "0", "pkg.module:build"], id="duration-not-positive"),
        pytest.param(["run", "--max-restarts", "0", "pkg.module:build"], id="restarts-below-one"),
        pytest.param(
            ["run", "--stop-grace", "4", "--bot-stop-timeout", "3", "--bot-release-timeout", "1", "pkg.module:build"],
            id="bot-stop-outside-grace",
        ),
        pytest.param(["run", "--backoff-initial", "5", "--backoff-max", "1", "pkg.module:build"], id="backoff-cap"),
        pytest.param(["run", "--name", "", "pkg.module:build"], id="blank-name"),
    ],
)
def test_bad_arguments_exit_64_with_nothing_on_stdout(
    argv: list[str], capsys: pytest.CaptureFixture[str], no_telemetry: None
) -> None:
    del no_telemetry

    assert cli.main(argv) == 64

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("mosfet: ")


def test_a_target_without_an_attribute_names_the_expected_form(
    capsys: pytest.CaptureFixture[str], no_telemetry: None
) -> None:
    del no_telemetry

    assert cli.main(["run", "not-a-target"]) == 64

    assert "module.path:attribute" in capsys.readouterr().err


def test_help_is_requested_output_and_exits_0(capsys: pytest.CaptureFixture[str], no_telemetry: None) -> None:
    del no_telemetry

    assert cli.main(["run", "--help"]) == 0

    assert "--stop-grace" in capsys.readouterr().out


@pytest.mark.parametrize(
    "attribute",
    [
        pytest.param("absent", id="factory-missing"),
        pytest.param("not_callable", id="factory-not-callable"),
        pytest.param("raising", id="factory-raised"),
        pytest.param("wrong_type", id="invalid-assembly"),
        pytest.param("needs_secret", id="missing-secret"),
    ],
)
def test_a_factory_that_cannot_build_exits_78(
    attribute: str, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], no_telemetry: None
) -> None:
    del no_telemetry
    data = tmp_path / "data"

    code = cli.main(
        ["run", f"{_FACTORIES}:{attribute}", "--data-dir", str(data), "--secrets-dir", str(tmp_path / "secrets")]
    )

    assert code == 78
    assert capsys.readouterr().out == ""


def test_an_unimportable_module_exits_78(capsys: pytest.CaptureFixture[str], no_telemetry: None) -> None:
    del no_telemetry

    assert cli.main(["run", "tests.runtime.fixtures.no_such_module:build", "--data-dir", "data"]) == 78
    assert capsys.readouterr().out == ""


def test_the_data_dir_is_created_private(tmp_path: pathlib.Path, no_telemetry: None) -> None:
    del no_telemetry
    data = tmp_path / "state" / "data"

    _ = cli.main(["run", f"{_FACTORIES}:absent", "--data-dir", str(data)])

    assert data.is_dir()
    assert data.stat().st_mode & 0o777 == 0o700


def test_a_data_dir_that_cannot_be_created_exits_78(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], no_telemetry: None
) -> None:
    del no_telemetry
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")

    assert cli.main(["run", f"{_FACTORIES}:quiet", "--data-dir", str(blocker / "data")]) == 78
    assert capsys.readouterr().out == ""


def test_a_bot_that_exhausts_its_restart_budget_exits_70(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], no_telemetry: None
) -> None:
    del no_telemetry

    code = cli.main(
        [
            "run",
            f"{_FACTORIES}:failing",
            "--data-dir",
            str(tmp_path / "data"),
            "--max-restarts",
            "1",
            "--backoff-initial",
            "0.01",
            "--backoff-max",
            "0.01",
        ]
    )

    assert code == 70
    assert capsys.readouterr().out == ""


class _Child:
    """One ``mosfet run`` child process in its own temporary directory."""

    popen: subprocess.Popen[bytes]
    ready: pathlib.Path
    data: pathlib.Path

    def __init__(self, tmp_path: pathlib.Path, attribute: str, *flags: str) -> None:
        self.ready = tmp_path / "ready"
        self.data = tmp_path / "data"
        environ = {name: value for name, value in os.environ.items() if not name.startswith(("OTEL_", "BOT_OTEL_"))}
        environ["PYTHONPATH"] = str(_REPO)
        self.popen = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mosfet.cli",
                "run",
                f"{_FACTORIES}:{attribute}",
                "--data-dir",
                str(self.data),
                "--ready-file",
                str(self.ready),
                *flags,
            ],
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
                pytest.fail(f"mosfet run exited {self.popen.returncode} early: {stderr.decode(errors='replace')}")
            if time.monotonic() > deadline:
                self.popen.kill()
                _ = self.popen.communicate()
                pytest.fail("mosfet run did not reach the expected state")
            time.sleep(0.02)

    def signal(self, signum: signal.Signals) -> float:
        self.popen.send_signal(signum)
        return time.monotonic()

    def finish(self, bound: float) -> tuple[int, bytes, bytes, float]:
        """Wait at most ``bound`` seconds for the exit; return code, stdout, stderr and the time it took."""

        started = time.monotonic()
        try:
            stdout, stderr = self.popen.communicate(timeout=bound)
        except subprocess.TimeoutExpired:
            self.popen.kill()
            _ = self.popen.communicate()
            pytest.fail(f"mosfet run did not exit within {bound} s")
        return self.popen.returncode, stdout, stderr, time.monotonic() - started


def _stop_durations(directory: pathlib.Path) -> list[tuple[object, float]]:
    """``(outcome, seconds)`` of every stop the child recorded in its OTEL metric file."""

    points = metric_reader.exported(directory / "otel-metrics.jsonl", "bot.runtime.stop.duration")
    return [(point.attributes.get("outcome"), point.sum) for point in points]


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_a_signal_after_ready_stops_gracefully_with_0_and_empty_stdout(
    tmp_path: pathlib.Path, signum: signal.Signals
) -> None:
    child = _Child(tmp_path, "quiet")
    child.wait_for(child.ready.exists)

    _ = child.signal(signum)
    code, stdout, stderr, elapsed = child.finish(8 + _EXIT_BOUND + _SLACK)

    assert code == 0, stderr.decode(errors="replace")
    assert stdout == b""
    assert not child.ready.exists()
    # The process boundary times the stop from the signal to the runtime's exit.
    ((outcome, seconds),) = _stop_durations(tmp_path)
    assert outcome == "graceful"
    assert 0 < seconds < elapsed


def test_sigterm_while_a_bot_is_still_activating_exits_0(tmp_path: pathlib.Path) -> None:
    child = _Child(tmp_path, "activating")
    child.wait_for(lambda: (child.data / "activating").exists())

    _ = child.signal(signal.SIGTERM)
    code, stdout, stderr, _elapsed = child.finish(8 + _EXIT_BOUND + _SLACK)

    assert code == 0, stderr.decode(errors="replace")
    assert stdout == b""
    assert not child.ready.exists()


def test_sigterm_while_the_factory_blocks_exits_0_without_waiting_for_it(tmp_path: pathlib.Path) -> None:
    child = _Child(tmp_path, "blocking")
    child.wait_for(lambda: (child.data / "booting").exists())
    # The factory signals from its worker thread; give the event loop a moment to install the handlers.
    time.sleep(0.2)

    _ = child.signal(signal.SIGTERM)
    code, stdout, stderr, _elapsed = child.finish(_EXIT_BOUND + _SLACK)

    assert code == 0, stderr.decode(errors="replace")
    assert stdout == b""


def test_a_second_sigterm_while_a_bot_will_not_stop_forces_exit_1(tmp_path: pathlib.Path) -> None:
    child = _Child(tmp_path, "mute")
    child.wait_for(child.ready.exists)

    _ = child.signal(signal.SIGTERM)
    # The mute bot holds its supervisor in stopping for the whole 4 s bot_stop_timeout.
    time.sleep(0.2)
    _ = child.signal(signal.SIGTERM)
    code, stdout, stderr, _elapsed = child.finish(_EXIT_BOUND + _SLACK)

    assert code == 1, stderr.decode(errors="replace")
    assert stdout == b""
    ((outcome, seconds),) = _stop_durations(tmp_path)
    assert outcome == "forced"
    assert seconds >= 0.2


def test_an_actor_that_ignores_cancellation_cannot_hold_the_process_past_its_bound(tmp_path: pathlib.Path) -> None:
    grace = 2.0
    child = _Child(
        tmp_path,
        "stubborn",
        "--stop-grace",
        str(grace),
        "--bot-stop-timeout",
        "0.5",
        "--bot-release-timeout",
        "0.5",
    )
    child.wait_for(child.ready.exists)

    _ = child.signal(signal.SIGTERM)
    code, stdout, stderr, elapsed = child.finish(grace + _EXIT_BOUND + _SLACK)

    assert code == 1, stderr.decode(errors="replace")
    assert stdout == b""
    assert elapsed < grace + _EXIT_BOUND + _SLACK

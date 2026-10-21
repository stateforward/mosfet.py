"""``willy_example.evaluate.cli``: ``python -m willy_example.evaluate``, its flags, settings, secret and exit codes.

Every setting is given in the test. Nothing reaches a live service: the one run that gets as far as the speech
engine is pointed at a closed loopback port.
"""

import os
import pathlib
import subprocess
import sys

import pytest
from willy_example.evaluate import cli

_CLOSED = "127.0.0.1:9"
_SETTINGS = {
    "WILLY_REASONING_URL": f"http://{_CLOSED}/v1",
    "WILLY_REASONING_MODEL": "gemma-4-26b-a4b",
    "WILLY_TTS_URL": f"http://{_CLOSED}",
    "WILLY_TRANSCRIBE_URL": f"http://{_CLOSED}/v1",
}


@pytest.fixture
def environ(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pytest.MonkeyPatch:
    for name in (*_SETTINGS, "COMPUTER_TOKEN_SECRET", "COMPUTER_TOKEN_SECRET_FILE", "OPENAI_API_KEY", "WILLY_TRAITS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in _SETTINGS.items():
        monkeypatch.setenv(name, value)
    # Telemetry's files are written under the working directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOT_OTEL_DISABLED", "1")
    return monkeypatch


def test_bad_flags_exit_64_and_nothing_is_written_to_stdout(
    environ: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environ.setenv("COMPUTER_TOKEN_SECRET", "secret-8e3f")
    out = str(tmp_path / "out")

    assert cli.main(["--nonsense", "--out", out]) == 64
    assert cli.main([]) == 64
    assert cli.main(["--out", out, "--url", "http://127.0.0.1:8765"]) == 64
    assert cli.main(["--out", out, "--peer", "Not A Room"]) == 64
    assert cli.main(["--out", out, "--scenario", "no-such-scenario"]) == 64
    assert cli.main(["--out", out, "--reply-timeout", "0"]) == 64
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no-such-scenario" in captured.err
    assert not (tmp_path / "out").exists()


def test_a_missing_secret_or_setting_exits_78_naming_it(
    environ: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = str(tmp_path / "out")

    assert cli.main(["--out", out]) == 78
    assert "computer_token_secret" in capsys.readouterr().err

    environ.setenv("COMPUTER_TOKEN_SECRET", "secret-8e3f")
    for missing in ("WILLY_REASONING_URL", "WILLY_REASONING_MODEL"):
        given = os.environ[missing]
        environ.delenv(missing)
        assert cli.main(["--out", out]) == 78
        assert missing in capsys.readouterr().err
        environ.setenv(missing, given)
    environ.delenv("WILLY_TTS_URL")
    assert cli.main(["--out", out]) == 78
    captured = capsys.readouterr()
    assert "WILLY_TTS_URL" in captured.err
    assert "secret-8e3f" not in captured.err
    assert captured.out == ""


def test_a_speech_engine_that_does_not_answer_exits_1_and_says_so(
    environ: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environ.setenv("COMPUTER_TOKEN_SECRET", "secret-8e3f")
    out = tmp_path / "out"

    assert cli.main(["--out", str(out), "--scenario", "stranger-arrives"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "speech engine" in captured.err
    assert "secret-8e3f" not in captured.err
    # The output directory is the owner's alone.
    assert (out.stat().st_mode & 0o777) == 0o700


def test_the_module_runs_as_python_dash_m_and_refuses_bad_flags_with_64(tmp_path: pathlib.Path) -> None:
    environ = {name: value for name, value in os.environ.items() if not name.startswith(("WILLY_", "COMPUTER_"))}

    child = subprocess.run(
        [sys.executable, "-m", "willy_example.evaluate", "--nonsense"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert child.returncode == 64
    assert child.stdout == ""
    assert child.stderr.startswith("willy-evaluate: ")

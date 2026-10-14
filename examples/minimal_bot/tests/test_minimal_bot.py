"""``minimal_bot_example.build``: what it assembles, what it needs, and that ``mosfet run`` runs it as written.

The command test runs the installed ``mosfet`` console script from this project's environment in a
temporary directory with no ``PYTHONPATH``, so the factory is imported exactly as the README documents.
Its endpoint is a closed local port: the bot gets no input, so it never calls the model.
"""

import os
import pathlib
import signal
import subprocess
import sys
import time

import minimal_bot_example
import pytest
from mosfet import runtime

_ENDPOINT = {"OPENAI_BASE_URL": "http://127.0.0.1:9/v1", "OPENAI_MODEL": "any-model"}
# Starting the child imports mosfet and the provider before it can be ready; that is not under test.
_START_BOUND = 60.0
# Stop grace (8 s default) plus the 1.5 s bounded process exit and scheduling slack.
_STOP_BOUND = 10.5


def _config(tmp_path: pathlib.Path, environ: dict[str, str]) -> runtime.Config:
    return runtime.Config(
        name="minimal_bot_example",
        instance_id="test-1",
        data_dir=tmp_path,
        environ=environ,
        secrets=runtime.Secrets(directory=tmp_path / "secrets", environ=environ),
    )


def test_build_assembles_one_device_free_bot(tmp_path: pathlib.Path) -> None:
    assembly = minimal_bot_example.build(_config(tmp_path, {**_ENDPOINT, "OPENAI_API_KEY": "key"}))

    assert list(assembly.bots) == ["minimal"]
    assert isinstance(assembly.bots["minimal"], minimal_bot_example.MinimalBot)
    assert assembly.actors == ()


@pytest.mark.parametrize("unset", ["OPENAI_BASE_URL", "OPENAI_MODEL"])
def test_build_needs_the_endpoint_and_its_model(tmp_path: pathlib.Path, unset: str) -> None:
    environ = {**_ENDPOINT, "OPENAI_API_KEY": "key"}
    del environ[unset]

    with pytest.raises(ValueError, match="OPENAI_BASE_URL and OPENAI_MODEL"):
        _ = minimal_bot_example.build(_config(tmp_path, environ))


def test_build_needs_the_api_key_secret(tmp_path: pathlib.Path) -> None:
    with pytest.raises(runtime.MissingSecretError, match="openai_api_key"):
        _ = minimal_bot_example.build(_config(tmp_path, dict(_ENDPOINT)))


def test_mosfet_run_serves_the_example_until_sigterm(tmp_path: pathlib.Path) -> None:
    ready = tmp_path / "ready"
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    _ = (secrets / "openai_api_key").write_text("key\n", encoding="utf-8")
    environ = {name: value for name, value in os.environ.items() if not name.startswith(("OTEL_", "BOT_OTEL_"))}
    _ = environ.pop("PYTHONPATH", None)
    environ.update(_ENDPOINT, BOT_OTEL_DISABLED="1")
    command = pathlib.Path(sys.executable).with_name("mosfet")
    child = subprocess.Popen(
        [
            str(command),
            "run",
            "minimal_bot_example:build",
            "--data-dir",
            str(tmp_path / "data"),
            "--secrets-dir",
            str(secrets),
            "--ready-file",
            str(ready),
        ],
        cwd=tmp_path,
        env=environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + _START_BOUND
        while not ready.exists():
            if child.poll() is not None:
                pytest.fail(f"mosfet run exited {child.returncode} early: {child.communicate()[1].decode()}")
            if time.monotonic() > deadline:
                pytest.fail("mosfet run did not become ready")
            time.sleep(0.02)

        child.send_signal(signal.SIGTERM)
        stdout, stderr = child.communicate(timeout=_STOP_BOUND)
    finally:
        if child.poll() is None:
            child.kill()
            _ = child.communicate()

    assert child.returncode == 0, stderr.decode(errors="replace")
    assert stdout == b""
    assert not ready.exists()

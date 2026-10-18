"""``builder_bot_example``: the owner's ``Settings``, and its bot file, ``python -m builder_bot_example``, run as written.

Every setting is given in the test; nothing reads the process environment. The child tests run the bot file
from this project's environment in a temporary directory with no ``PYTHONPATH``, so it is imported exactly as
the README documents. Its endpoint is a closed local port and its bench socket does not exist: nothing reaches
the bot, so it never calls the model, and its bench waits for a host that never serves.
"""

import os
import pathlib
import signal
import socket as sockets
import subprocess
import sys
import time
import typing

import builder_bot_example
import pydantic
import pytest
from mosfet.abilities import persona

_REQUIRED = {
    "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
    "OPENAI_MODEL": "local-model",
    "BUILDER_SANDBOX_SOCKET": "/nonexistent/podman.sock",
}
_SECRETS = {"OPENAI_API_KEY": "key-5d1a", "COMPUTER_TOKEN_SECRET": "secret-8e3f"}
# Starting the child imports mosfet and the providers before the bot file runs; that is not under test.
_RUN_BOUND = 60.0
# Stop grace (8 s default) plus the 1.5 s bounded process exit and scheduling slack.
_STOP_BOUND = 10.5


def test_the_required_settings_alone_take_every_default() -> None:
    settings = builder_bot_example.Settings.model_validate(_REQUIRED)

    assert (settings.base_url, settings.model, settings.socket) == (
        "http://127.0.0.1:9/v1",
        "local-model",
        pathlib.Path("/nonexistent/podman.sock"),
    )
    # Bench image 2 has mosfet installed, so the bot can write and test bots on it (W7-14).
    assert settings.image == builder_bot_example.DEFAULT_IMAGE == "localhost/mosfet-bench:2"
    assert (settings.egress_network, settings.runtime) == (None, "runc")
    assert (settings.host, settings.port, settings.peer, settings.person_id) == (
        "127.0.0.1",
        8765,
        "builder-desk",
        None,
    )
    assert (
        settings.traits == builder_bot_example.DEFAULT_TRAITS == persona.Traits(traits=("careful", "curious", "candid"))
    )


def test_every_owner_setting_has_a_working_value() -> None:
    settings = builder_bot_example.Settings.model_validate(
        {
            **_REQUIRED,
            "BUILDER_SANDBOX_IMAGE": "localhost/mosfet-bench:3",
            "BUILDER_SANDBOX_EGRESS_NETWORK": "builder-egress",
            "BUILDER_SANDBOX_RUNTIME": "runsc",
            "BUILDER_COMPUTER_HOST": "0.0.0.0",
            "BUILDER_COMPUTER_PORT": "9000",
            "BUILDER_COMPUTER_PEER": "desk-2",
            "BUILDER_PERSON_ID": "person-1",
            "BUILDER_TRAITS": "patient, exact",
        }
    )

    assert settings.image == "localhost/mosfet-bench:3"
    assert (settings.egress_network, settings.runtime) == ("builder-egress", "runsc")
    assert (settings.host, settings.port, settings.peer, settings.person_id) == ("0.0.0.0", 9000, "desk-2", "person-1")
    assert settings.traits == persona.Traits(traits=("patient", "exact"))


def test_an_empty_variable_is_unset() -> None:
    settings = builder_bot_example.Settings.model_validate(
        {**_REQUIRED, "BUILDER_SANDBOX_EGRESS_NETWORK": "", "BUILDER_COMPUTER_PORT": "", "BUILDER_TRAITS": ""}
    )

    assert (settings.egress_network, settings.port, settings.traits) == (None, 8765, builder_bot_example.DEFAULT_TRAITS)


def _refused(environ: dict[str, str]) -> str:
    with pytest.raises(pydantic.ValidationError) as raised:
        _ = builder_bot_example.Settings.model_validate(environ)
    return str(raised.value)


@pytest.mark.parametrize("unset", ["OPENAI_BASE_URL", "OPENAI_MODEL", "BUILDER_SANDBOX_SOCKET"])
def test_each_required_setting_is_named_when_it_is_missing(unset: str) -> None:
    for missing in ({}, {unset: ""}):
        environ = {**_REQUIRED, **missing}
        if not missing:
            del environ[unset]

        assert unset in _refused(environ)


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("BUILDER_COMPUTER_PORT", "0"),
        ("BUILDER_COMPUTER_PORT", "65536"),
        ("BUILDER_COMPUTER_PORT", "eighty"),
        ("BUILDER_SANDBOX_RUNTIME", "kata"),
        ("BUILDER_TRAITS", "careful,,candid"),
        ("BUILDER_TRAITS", " "),
        ("BUILDER_SANDBOX_IMAGE", "Not An Image"),
    ],
)
def test_a_bad_setting_is_refused_and_named_without_its_value(variable: str, value: str) -> None:
    message = _refused({**_REQUIRED, variable: value})

    assert variable in message
    assert value.strip() not in message or not value.strip()


def _environ(**given: str) -> dict[str, str]:
    """This process's environment for the child, without its telemetry, endpoint, builder and mosfet variables."""

    environ = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OTEL_", "BOT_OTEL_", "OPENAI_", "BUILDER_", "MOSFET_"))
    }
    _ = environ.pop("PYTHONPATH", None)
    environ.update(given, BOT_OTEL_DISABLED="1")
    return environ


def _free_port() -> int:
    with sockets.socket(sockets.AF_INET, sockets.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return typing.cast(tuple[str, int], probe.getsockname())[1]


def test_python_m_defines_the_builder_and_serves_it_until_sigterm(tmp_path: pathlib.Path) -> None:
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "builder_bot_example",
            "--data-dir",
            str(tmp_path / "data"),
            "--secrets-dir",
            str(tmp_path / "absent"),
            "--ready-file",
            str(ready),
        ],
        cwd=tmp_path,
        env=_environ(**_REQUIRED, **_SECRETS, BUILDER_COMPUTER_PORT=str(_free_port())),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + _RUN_BOUND
        while not ready.exists():
            if child.poll() is not None:
                raise AssertionError(f"the bot exited {child.returncode} early: {child.communicate()[1].decode()}")
            if time.monotonic() > deadline:
                raise AssertionError("the bot did not become ready")
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
    assert (tmp_path / "data" / "memory.sqlite3").exists()
    for secret in _SECRETS.values():
        assert secret.encode() not in stderr


def test_python_m_names_a_missing_setting_and_exits_1_without_a_secret(tmp_path: pathlib.Path) -> None:
    finished = subprocess.run(
        [sys.executable, "-m", "builder_bot_example", "--data-dir", str(tmp_path / "data")],
        cwd=tmp_path,
        env=_environ(
            **{name: value for name, value in {**_REQUIRED, **_SECRETS}.items() if name != "BUILDER_SANDBOX_SOCKET"}
        ),
        capture_output=True,
        timeout=_RUN_BOUND,
        check=False,
    )

    assert finished.returncode == 1, finished.stderr.decode(errors="replace")
    assert finished.stdout == b""
    assert b"BUILDER_SANDBOX_SOCKET" in finished.stderr
    for secret in _SECRETS.values():
        assert secret.encode() not in finished.stderr

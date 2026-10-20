"""``builder_bot_example``: the owner's ``Settings``, its bench check, and its bot file, run as written.

Every setting is given in the test; nothing reads the process environment. The child tests run the bot file
from this project's environment in a temporary directory with no ``PYTHONPATH``, so it is imported exactly as
the README documents. Its endpoint is a closed local port, and its bench socket is a fake daemon that answers
only an image inspect: nothing reaches the bot, so it never calls the model, and its bench never opens.
"""

import collections.abc
import contextlib
import http.server
import json
import os
import pathlib
import signal
import socket as sockets
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import typing

import builder_bot_example
import pydantic
import pytest
from mosfet.abilities import persona
from mosfet.abilities.language import text

_REQUIRED = {
    "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
    "OPENAI_MODEL": "local-model",
    "BUILDER_SANDBOX_SOCKET": "/nonexistent/podman.sock",
    # The decision server intuition decides on: a closed port, since nothing reaches the bot in these tests.
    "DECISION_BASE_URL": "http://127.0.0.1:9",
}
_SECRETS = {"OPENAI_API_KEY": "key-5d1a", "DECISION_API_KEY": "key-9b07", "COMPUTER_TOKEN_SECRET": "secret-8e3f"}
# A local decision server takes no key, so the bot file needs no decision_api_key for it.
_KEYLESS = {name: secret for name, secret in _SECRETS.items() if name != "DECISION_API_KEY"}
# Starting the child imports mosfet and the providers before the bot file runs; that is not under test.
_RUN_BOUND = 60.0
# Stop grace (8 s default) plus the 1.5 s bounded process exit and scheduling slack.
_STOP_BOUND = 10.5
# The commit of this checkout, read independently of the bot file: the bench must be built for it.
_HEAD = subprocess.run(
    ["git", "-C", str(pathlib.Path(__file__).parent), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=True,
    timeout=30,
).stdout.strip()
_STALE = "0" * 40


@contextlib.contextmanager
def _daemon(images: collections.abc.Mapping[str, dict[str, str] | None]) -> collections.abc.Generator[pathlib.Path]:
    """A fake rootless daemon on a unix socket that knows ``images`` (reference -> labels) and answers nothing else.

    An image inspect (``GET /images/<reference>/json``) for a known image is its ``Config.Labels``; every other
    request is a 404, so a bench host that tries to open on it is refused and never serves.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            reference = self.path.removeprefix("/images/").removesuffix("/json")
            known = self.path.startswith("/images/") and self.path.endswith("/json") and reference in images
            body = json.dumps(
                {"Id": "sha256:feed", "Config": {"Labels": images[reference]}}
                if known
                else {"message": f"no such image: {reference}"}
            ).encode()
            self.send_response(200 if known else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            _ = self.wfile.write(body)

        @typing.override
        def address_string(self) -> str:
            # A unix socket peer has no host to log; the base class would index an empty address.
            return "fake-daemon"

    # A unix socket path must stay under 104 bytes on macOS, which pytest's tmp_path does not.
    with tempfile.TemporaryDirectory(prefix="bench-") as directory:
        socket = pathlib.Path(directory) / "d.sock"
        server = socketserver.ThreadingUnixStreamServer(str(socket), Handler)
        server.daemon_threads = True
        serving = threading.Thread(target=server.serve_forever, args=(0.05,), name="fake-daemon", daemon=True)
        serving.start()
        try:
            yield socket
        finally:
            server.shutdown()
            server.server_close()
            serving.join(timeout=5)


_IMAGE = builder_bot_example.DEFAULT_IMAGE
_LABEL = builder_bot_example.REVISION_LABEL


def _refusal(socket: pathlib.Path, commit: str = _HEAD) -> str:
    with pytest.raises(builder_bot_example.BenchError) as raised:
        builder_bot_example.check_bench(socket=socket, image=_IMAGE, commit=commit)
    return str(raised.value)


def test_mosfet_commit_is_the_head_of_the_checkout_mosfet_is_imported_from() -> None:
    assert builder_bot_example.mosfet_commit() == _HEAD


def test_mosfet_commit_outside_a_checkout_is_refused_naming_where_it_looked(tmp_path: pathlib.Path) -> None:
    with pytest.raises(builder_bot_example.BenchError) as raised:
        _ = builder_bot_example.mosfet_commit(tmp_path)

    assert str(tmp_path) in str(raised.value)


def test_a_bench_image_built_for_this_commit_passes_the_check_and_says_what_it_holds() -> None:
    # Live (builder check round 2, E4): nothing the bot was offered said its bench has mosfet. The image's own
    # description label is what the owner says the bench holds.
    holds = "python 3.13, uv, git, pytest; mosfet installed (`from mosfet import bot`)"
    labels = {_LABEL: _HEAD, "org.opencontainers.image.version": "2", builder_bot_example.DESCRIPTION_LABEL: holds}
    with _daemon({_IMAGE: labels}) as socket:
        described = builder_bot_example.check_bench(socket=socket, image=_IMAGE, commit=_HEAD)
    with _daemon({_IMAGE: {_LABEL: _HEAD}}) as socket:
        undescribed = builder_bot_example.check_bench(socket=socket, image=_IMAGE, commit=_HEAD)

    assert builder_bot_example.DESCRIPTION_LABEL == "org.opencontainers.image.description"
    assert (described, undescribed) == (holds, None)


def test_a_stale_bench_image_is_refused_naming_both_commits_and_the_rebuild() -> None:
    with _daemon({_IMAGE: {_LABEL: _STALE}}) as socket:
        message = _refusal(socket)

    assert _IMAGE in message
    assert _STALE in message
    assert _HEAD in message
    assert "--build-arg MOSFET_COMMIT=" in message
    assert "docker/bench/Dockerfile" in message


@pytest.mark.parametrize("labels", [None, {}, {"org.opencontainers.image.version": "2"}])
def test_a_bench_image_without_the_commit_label_is_refused_as_stale(labels: dict[str, str] | None) -> None:
    with _daemon({_IMAGE: labels}) as socket:
        message = _refusal(socket)

    assert _IMAGE in message
    assert _LABEL in message
    assert "--build-arg MOSFET_COMMIT=" in message


def test_a_bench_image_the_daemon_does_not_have_is_refused_with_the_build() -> None:
    with _daemon({}) as socket:
        message = _refusal(socket)

    assert _IMAGE in message
    assert "--build-arg MOSFET_COMMIT=" in message


def test_a_daemon_that_cannot_be_reached_is_refused_naming_its_socket(tmp_path: pathlib.Path) -> None:
    socket = tmp_path / "absent.sock"

    message = _refusal(socket)

    assert str(socket) in message
    assert _IMAGE in message


def test_the_required_settings_alone_take_every_default() -> None:
    settings = builder_bot_example.Settings.model_validate(_REQUIRED)

    assert (settings.base_url, settings.model, settings.socket) == (
        "http://127.0.0.1:9/v1",
        "local-model",
        pathlib.Path("/nonexistent/podman.sock"),
    )
    assert settings.stream is True
    assert settings.tool_choice == text.ToolSelectionPolicy.REQUIRED
    assert (settings.decision_base_url, settings.decision_max_context_tokens) == ("http://127.0.0.1:9", None)
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
            "OPENAI_STREAM": "false",
            "OPENAI_TOOL_CHOICE": "auto",
            "DECISION_MAX_CONTEXT_TOKENS": "2048",
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

    # Live (builder check round 2, E1): the owner's proxy never streams a tool call, so the owner turns it off.
    assert settings.stream is False
    # Live (builder check round 3, H1): under "required" the Spark's model wrote code with no line breaks.
    assert settings.tool_choice == text.ToolSelectionPolicy.AUTO
    # D-R-36: anygame's decide has a 2048-token context.
    assert settings.decision_max_context_tokens == 2048
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


@pytest.mark.parametrize("unset", ["OPENAI_BASE_URL", "OPENAI_MODEL", "BUILDER_SANDBOX_SOCKET", "DECISION_BASE_URL"])
def test_each_required_setting_is_named_when_it_is_missing(unset: str) -> None:
    for missing in ({}, {unset: ""}):
        environ = {**_REQUIRED, **missing}
        if not missing:
            del environ[unset]

        assert unset in _refused(environ)


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("OPENAI_STREAM", "sometimes"),
        ("OPENAI_TOOL_CHOICE", "none"),
        ("OPENAI_TOOL_CHOICE", "dispatch"),
        ("DECISION_MAX_CONTEXT_TOKENS", "-2048"),
        ("DECISION_MAX_CONTEXT_TOKENS", "many"),
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
        if not name.startswith(("OTEL_", "BOT_OTEL_", "OPENAI_", "DECISION_", "BUILDER_", "MOSFET_"))
    }
    _ = environ.pop("PYTHONPATH", None)
    environ.update(given, BOT_OTEL_DISABLED="1")
    return environ


def _free_port() -> int:
    with sockets.socket(sockets.AF_INET, sockets.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return typing.cast(tuple[str, int], probe.getsockname())[1]


@pytest.mark.parametrize(
    ("decider", "secrets"),
    [({"DECISION_MAX_CONTEXT_TOKENS": "2048"}, _SECRETS), ({}, _KEYLESS)],
    ids=["keyed", "keyless"],
)
def test_python_m_defines_the_builder_and_serves_it_until_sigterm(
    tmp_path: pathlib.Path, decider: dict[str, str], secrets: dict[str, str]
) -> None:
    ready = tmp_path / "ready"
    with _daemon({_IMAGE: {_LABEL: _HEAD}}) as socket:
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
            env=_environ(
                **{**_REQUIRED, "BUILDER_SANDBOX_SOCKET": str(socket)},
                **secrets,
                **decider,
                BUILDER_COMPUTER_PORT=str(_free_port()),
            ),
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
    for secret in secrets.values():
        assert secret.encode() not in stderr


def test_python_m_refuses_a_stale_bench_image_and_exits_1_before_anything_starts(tmp_path: pathlib.Path) -> None:
    ready = tmp_path / "ready"
    with _daemon({_IMAGE: {_LABEL: _STALE}}) as socket:
        finished = subprocess.run(
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
            env=_environ(
                **{**_REQUIRED, "BUILDER_SANDBOX_SOCKET": str(socket)},
                **_SECRETS,
                BUILDER_COMPUTER_PORT=str(_free_port()),
            ),
            capture_output=True,
            timeout=_RUN_BOUND,
            check=False,
        )

    assert finished.returncode == 1, finished.stderr.decode(errors="replace")
    assert finished.stdout == b""
    assert not ready.exists()
    assert not (tmp_path / "data" / "memory.sqlite3").exists()
    stderr = finished.stderr.decode(errors="replace")
    assert stderr.startswith("builder_bot_example: ")
    assert _STALE in stderr
    assert _HEAD in stderr
    for secret in _SECRETS.values():
        assert secret not in stderr


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

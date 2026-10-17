"""``builder_bot_example.build`` and ``compose``: what the factory assembles, what it needs, and what it refuses.

Every config is built in the test; nothing reads the process environment. The command test runs the installed
``mosfet`` console script from this project's environment in a temporary directory with no ``PYTHONPATH``, so
the factory is imported exactly as the README documents.
"""

import datetime
import functools
import logging
import os
import pathlib
import subprocess
import sys

import builder_bot_example
import pydantic
import pytest
from mosfet import runtime
from mosfet.abilities import persona, sandbox
from mosfet.providers import docker
from mosfet.providers import openai_compat
from mosfet.providers import websocket

_REQUIRED = {
    "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
    "OPENAI_MODEL": "local-model",
    "BUILDER_SANDBOX_SOCKET": "/nonexistent/podman.sock",
}
_SECRETS = {"OPENAI_API_KEY": "key-5d1a", "COMPUTER_TOKEN_SECRET": "secret-8e3f"}
# Starting the child imports mosfet and the providers before the factory runs; that is not under test.
_RUN_BOUND = 60.0


def _config(tmp_path: pathlib.Path, environ: dict[str, str]) -> runtime.Config:
    return runtime.Config(
        name="builder_bot_example",
        instance_id="test-1",
        data_dir=tmp_path,
        environ=environ,
        secrets=runtime.Secrets(directory=tmp_path / "secrets", environ=environ),
    )


def _full(**overrides: str) -> dict[str, str]:
    return {**_REQUIRED, **_SECRETS, **overrides}


def _compose(tmp_path: pathlib.Path, *, capacity: sandbox.Capacity, spec: sandbox.Spec) -> runtime.Assembly:
    return builder_bot_example.compose(
        processor=openai_compat.Processor(
            client=openai_compat.ChatClient(model="local-model", base_url="http://127.0.0.1:9/v1", api_key="key"),
            provider="builder_bot",
        ),
        server=websocket.Server(
            host="127.0.0.1", port=8765, verifier=websocket.HmacTokens(pydantic.SecretStr("secret"))
        ),
        peer="builder-desk",
        person_id=None,
        host=docker.Host(
            socket=pathlib.Path("/nonexistent/podman.sock"),
            capacity=docker.capacity(max_resources=sandbox.Resources()),
            images=frozenset({spec.image}),
            owner="builder_bot_example/test-1",
        ),
        capacity=capacity,
        spec=spec,
        traits=builder_bot_example.DEFAULT_TRAITS,
        database=str(tmp_path / "memory.sqlite3"),
        clock=functools.partial(datetime.datetime.now, datetime.UTC),
    )


def test_a_full_config_assembles_one_builder_and_its_websocket_server(tmp_path: pathlib.Path) -> None:
    assembly = builder_bot_example.build(_config(tmp_path, _full()))

    assert list(assembly.bots) == ["builder"]
    assert isinstance(assembly.bots["builder"], builder_bot_example.BuilderBot)
    (server,) = assembly.actors
    assert isinstance(server, websocket.Server)
    # Assembling constructs, and constructing opens nothing: the store opens its database on start (DSL-29).
    assert not (tmp_path / "memory.sqlite3").exists()


@pytest.mark.parametrize("unset", ["OPENAI_BASE_URL", "OPENAI_MODEL", "BUILDER_SANDBOX_SOCKET"])
def test_each_required_setting_is_named_when_it_is_missing(tmp_path: pathlib.Path, unset: str) -> None:
    for missing in ({}, {unset: ""}):
        environ = _full(**missing)
        if not missing:
            del environ[unset]

        with pytest.raises(ValueError, match=unset):
            _ = builder_bot_example.build(_config(tmp_path, environ))


@pytest.mark.parametrize(
    ("variable", "secret"), [("OPENAI_API_KEY", "openai_api_key"), ("COMPUTER_TOKEN_SECRET", "computer_token_secret")]
)
def test_each_required_secret_is_named_when_no_source_provides_it(
    tmp_path: pathlib.Path, variable: str, secret: str
) -> None:
    environ = _full()
    del environ[variable]

    with pytest.raises(runtime.MissingSecretError, match=secret):
        _ = builder_bot_example.build(_config(tmp_path, environ))


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
def test_a_bad_setting_is_refused_and_named(tmp_path: pathlib.Path, variable: str, value: str) -> None:
    with pytest.raises(ValueError, match=variable):
        _ = builder_bot_example.build(_config(tmp_path, _full(**{variable: value})))


def test_an_egress_network_that_is_not_the_owners_own_bridge_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="egress_network"):
        _ = builder_bot_example.build(_config(tmp_path, _full(BUILDER_SANDBOX_EGRESS_NETWORK="host")))


def test_every_owner_setting_has_a_working_value(tmp_path: pathlib.Path) -> None:
    environ = _full(
        BUILDER_SANDBOX_IMAGE="localhost/mosfet-bench:2",
        BUILDER_SANDBOX_EGRESS_NETWORK="builder-egress",
        BUILDER_SANDBOX_RUNTIME="runsc",
        BUILDER_COMPUTER_HOST="0.0.0.0",
        BUILDER_COMPUTER_PORT="9000",
        BUILDER_COMPUTER_PEER="desk-2",
        BUILDER_PERSON_ID="person-1",
        BUILDER_TRAITS="patient, exact",
    )

    assembly = builder_bot_example.build(_config(tmp_path, environ))

    assert isinstance(assembly.bots["builder"], builder_bot_example.BuilderBot)


def test_the_bench_is_the_owners_fixed_policy() -> None:
    closed = builder_bot_example.bench(image="localhost/mosfet-bench:1", network=sandbox.Network.NONE)
    open_ = builder_bot_example.bench(image="localhost/mosfet-bench:1", network=sandbox.Network.EGRESS)

    assert closed == sandbox.Spec(
        image="localhost/mosfet-bench:1",
        resources=sandbox.Resources(),
        network=sandbox.Network.NONE,
        workspace="/workspace",
        env={"HOME": "/workspace", "PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_INPUT": "1"},
    )
    assert open_.network is sandbox.Network.EGRESS
    assert builder_bot_example.DEFAULT_IMAGE == "localhost/mosfet-bench:1"
    assert builder_bot_example.DEFAULT_TRAITS == persona.Traits(traits=("careful", "curious", "candid"))


def test_a_bench_beyond_the_hosts_capacity_fails_at_construction(tmp_path: pathlib.Path) -> None:
    spec = builder_bot_example.bench(image="localhost/mosfet-bench:1", network=sandbox.Network.NONE)
    small = sandbox.Capacity(
        isolation=sandbox.Isolation.CONTAINER,
        networks=frozenset({sandbox.Network.NONE}),
        max_resources=sandbox.Resources(memory_bytes=512 * 1024 * 1024),
        files=True,
    )
    no_egress = small.model_copy(update={"max_resources": sandbox.Resources()})

    with pytest.raises(ValueError, match="memory_bytes"):
        _ = _compose(tmp_path, capacity=small, spec=spec)
    with pytest.raises(ValueError, match="egress"):
        _ = _compose(
            tmp_path,
            capacity=no_egress,
            spec=builder_bot_example.bench(image=spec.image, network=sandbox.Network.EGRESS),
        )


def test_the_wiring_is_logged_once_without_a_secret(tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="builder_bot_example"):
        _ = builder_bot_example.build(_config(tmp_path, _full()))

    (record,) = [record for record in caplog.records if record.name == "builder_bot_example"]
    message = record.getMessage()
    for shown in ("local-model", "localhost/mosfet-bench:1", "none", "127.0.0.1:8765"):
        assert shown in message
    for secret in _SECRETS.values():
        assert secret not in message


def test_mosfet_run_exits_78_when_the_sandbox_socket_is_missing(tmp_path: pathlib.Path) -> None:
    environ = {name: value for name, value in os.environ.items() if not name.startswith(("OTEL_", "BOT_OTEL_"))}
    _ = environ.pop("PYTHONPATH", None)
    _ = environ.pop("BUILDER_SANDBOX_SOCKET", None)
    environ.update(
        {name: value for name, value in _full().items() if name != "BUILDER_SANDBOX_SOCKET"}, BOT_OTEL_DISABLED="1"
    )
    command = pathlib.Path(sys.executable).with_name("mosfet")

    finished = subprocess.run(
        [str(command), "run", "builder_bot_example:build", "--data-dir", str(tmp_path / "data")],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        timeout=_RUN_BOUND,
        check=False,
    )

    assert finished.returncode == 78, finished.stderr.decode(errors="replace")
    assert b"BUILDER_SANDBOX_SOCKET" in finished.stderr
    for secret in _SECRETS.values():
        assert secret.encode() not in finished.stderr

"""``builder_bot_example``: the bot ``builder`` defines, the owner's ``Settings``, and ``python -m builder_bot_example``.

Every setting is given in the test; nothing reads the process environment. The child test runs the bot file
from this project's environment in a temporary directory with no ``PYTHONPATH``, so it is imported exactly as
the README documents.
"""

import datetime
import functools
import os
import pathlib
import subprocess
import sys

import builder_bot_example
import pydantic
import pytest
from mosfet import bot
from mosfet.abilities import memory, persona, sandbox
from mosfet.bot import testing
from mosfet.providers import docker
from mosfet.providers import websocket

_REQUIRED = {
    "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
    "OPENAI_MODEL": "local-model",
    "BUILDER_SANDBOX_SOCKET": "/nonexistent/podman.sock",
}
_SECRETS = {"OPENAI_API_KEY": "key-5d1a", "COMPUTER_TOKEN_SECRET": "secret-8e3f"}
# Starting the child imports mosfet and the providers before the bot file runs; that is not under test.
_RUN_BOUND = 60.0


def _builder(
    tmp_path: pathlib.Path, *, capacity: sandbox.Capacity | None = None, spec: sandbox.Spec | None = None
) -> bot.Model:
    spec = spec or builder_bot_example.bench(image=builder_bot_example.DEFAULT_IMAGE, network=sandbox.Network.NONE)
    return builder_bot_example.builder(
        processor=testing.Scripted(),
        server=websocket.Server(verifier=websocket.HmacTokens(pydantic.SecretStr("secret"))),
        host=docker.Host(
            socket=pathlib.Path("/nonexistent/podman.sock"),
            images=frozenset({spec.image}),
            owner="builder_bot_example/test-1",
        ),
        capacity=capacity or docker.capacity(max_resources=sandbox.Resources()),
        spec=spec,
        peer="builder-desk",
        person_id=None,
        traits=builder_bot_example.DEFAULT_TRAITS.traits,
        store=memory.Memory(database=str(tmp_path / "memory.sqlite3")),
        clock=functools.partial(datetime.datetime.now, datetime.UTC),
    )


def test_the_builder_holds_the_desk_and_has_the_bench_and_the_full_cognition(tmp_path: pathlib.Path) -> None:
    model = _builder(tmp_path)

    (outline,) = bot.describe(model).bots
    assert outline.name == "builder"
    assert outline.traits == ("careful", "curious", "candid")
    assert outline.faculties == (
        "intuition",
        "reasoning",
        "reflection",
        "associative",
        "consolidation",
        "autonomy",
        "routines",
    )
    assert [(part.id, part.kind, part.constructor) for part in outline.parts] == [
        ("desk", "device", "Computer"),
        ("sandbox", "ability", "Sandbox"),
    ]
    assert bot.check(model) == ()
    # Defining constructs, and constructing opens nothing: the store opens its database on start (DSL-29).
    assert not (tmp_path / "memory.sqlite3").exists()


def test_a_bench_beyond_the_hosts_capacity_fails_at_construction(tmp_path: pathlib.Path) -> None:
    spec = builder_bot_example.bench(image=builder_bot_example.DEFAULT_IMAGE, network=sandbox.Network.NONE)
    small = sandbox.Capacity(
        isolation=sandbox.Isolation.CONTAINER,
        networks=frozenset({sandbox.Network.NONE}),
        max_resources=sandbox.Resources(memory_bytes=512 * 1024 * 1024),
        files=True,
    )
    no_egress = small.model_copy(update={"max_resources": sandbox.Resources()})

    with pytest.raises(ValueError, match="memory_bytes"):
        _ = _builder(tmp_path, capacity=small, spec=spec)
    with pytest.raises(ValueError, match="egress"):
        _ = _builder(
            tmp_path,
            capacity=no_egress,
            spec=builder_bot_example.bench(image=spec.image, network=sandbox.Network.EGRESS),
        )


def test_the_bench_is_the_owners_fixed_policy() -> None:
    closed = builder_bot_example.bench(image="localhost/mosfet-bench:2", network=sandbox.Network.NONE)
    open_ = builder_bot_example.bench(image="localhost/mosfet-bench:2", network=sandbox.Network.EGRESS)

    assert closed == sandbox.Spec(
        image="localhost/mosfet-bench:2",
        resources=sandbox.Resources(),
        network=sandbox.Network.NONE,
        workspace="/workspace",
        env={"PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_INPUT": "1"},
    )
    assert closed.env["HOME"] == "/workspace"
    assert open_.network is sandbox.Network.EGRESS
    # Bench image 2 has mosfet installed, so the bot can write and test bots on it (W7-14).
    assert builder_bot_example.DEFAULT_IMAGE == "localhost/mosfet-bench:2"
    assert builder_bot_example.DEFAULT_TRAITS == persona.Traits(traits=("careful", "curious", "candid"))


def test_the_required_settings_alone_take_every_default() -> None:
    settings = builder_bot_example.Settings.model_validate(_REQUIRED)

    assert (settings.base_url, settings.model, settings.socket) == (
        "http://127.0.0.1:9/v1",
        "local-model",
        pathlib.Path("/nonexistent/podman.sock"),
    )
    assert settings.image == builder_bot_example.DEFAULT_IMAGE
    assert (settings.egress_network, settings.runtime) == (None, "runc")
    assert (settings.host, settings.port, settings.peer, settings.person_id) == (
        "127.0.0.1",
        8765,
        "builder-desk",
        None,
    )
    assert settings.traits == builder_bot_example.DEFAULT_TRAITS


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


def test_python_m_names_a_missing_setting_and_exits_1_without_a_secret(tmp_path: pathlib.Path) -> None:
    environ = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OTEL_", "BOT_OTEL_", "OPENAI_", "BUILDER_", "MOSFET_"))
    }
    _ = environ.pop("PYTHONPATH", None)
    environ.update(
        {name: value for name, value in {**_REQUIRED, **_SECRETS}.items() if name != "BUILDER_SANDBOX_SOCKET"},
        BOT_OTEL_DISABLED="1",
    )

    finished = subprocess.run(
        [sys.executable, "-m", "builder_bot_example", "--data-dir", str(tmp_path / "data")],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        timeout=_RUN_BOUND,
        check=False,
    )

    assert finished.returncode == 1, finished.stderr.decode(errors="replace")
    assert finished.stdout == b""
    assert b"BUILDER_SANDBOX_SOCKET" in finished.stderr
    for secret in _SECRETS.values():
        assert secret.encode() not in finished.stderr

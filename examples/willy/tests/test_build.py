"""``willy_example``: the owner's ``Settings``, and its bot file, ``python -m willy_example``, run as written.

Every setting is given in the test; nothing reads the process environment. The child tests run the bot file
from this project's environment in a temporary directory with no ``PYTHONPATH``, so it is imported exactly as
the README documents. Every endpoint is a closed local port: nothing reaches the bot, so it never hears anyone,
and its birth (which asks the model for a voice) fails and waits, as it would with the lab down.
"""

import json
import os
import pathlib
import signal
import socket as sockets
import stat
import subprocess
import sys
import time
import typing

import pydantic
import pytest
import sqlalchemy
import willy_example
from mosfet.abilities import memory, persona
from mosfet.environment import space

_CLOSED = "127.0.0.1:9"
_REQUIRED = {
    "WILLY_LLM_URL": f"http://{_CLOSED}/v1",
    "WILLY_MODEL": "local-model",
    "WILLY_EMBED_URL": f"http://{_CLOSED}/v1",
    "WILLY_EMBED_MODEL": "embeddinggemma-2",
    "WILLY_STT_URL": f"ws://{_CLOSED}",
    "WILLY_TRANSCRIBE_URL": f"http://{_CLOSED}/v1",
    "WILLY_TTS_URL": f"http://{_CLOSED}",
    "WILLY_VOICEPRINT_URL": f"http://{_CLOSED}",
    "WILLY_EMOTION_URL": f"http://{_CLOSED}",
    "WILLY_DIAR_URL": f"http://{_CLOSED}",
}
_SECRETS = {"OPENAI_API_KEY": "key-5d1a", "TYPESAFE_API_KEY": "key-9b07", "COMPUTER_TOKEN_SECRET": "secret-8e3f"}
# A local system_one server for intuition to decide on: it takes no key, so the bot file needs none for it.
_LOCAL_DECIDER = {"TYPESAFE_BASE_URL": f"http://{_CLOSED}"}
_KEYLESS = {name: secret for name, secret in _SECRETS.items() if name != "TYPESAFE_API_KEY"}
_HOUSEHOLD = [
    {"id": "gabe", "name": "Gabe", "role": "adult"},
    {"id": "ada", "name": "Ada", "role": "child", "born": 2018},
]
# Starting the child imports mosfet and the providers before the bot file runs; that is not under test.
_RUN_BOUND = 60.0
# Stop grace (8 s default) plus the 1.5 s bounded process exit and scheduling slack.
_STOP_BOUND = 10.5


def _household(tmp_path: pathlib.Path, content: str) -> str:
    path = tmp_path / "household.json"
    _ = path.write_text(content, encoding="utf-8")
    return str(path)


def test_the_required_settings_alone_take_every_default() -> None:
    settings = willy_example.Settings.model_validate(_REQUIRED)

    assert (settings.llm_url, settings.model, settings.embed_url, settings.embed_model) == (
        f"http://{_CLOSED}/v1",
        "local-model",
        f"http://{_CLOSED}/v1",
        "embeddinggemma-2",
    )
    assert (settings.host, settings.port, settings.camera, settings.face_models) == ("127.0.0.1", 8765, False, None)
    # One room, its computer on a shelf; nobody is declared, so Willy learns everyone.
    assert settings.rooms == (
        willy_example.Room(id="living-room", on="shelf", placement=space.Placement(position=space.Position(0.0, 0.0))),
    )
    assert settings.household == ()
    assert (
        settings.traits
        == willy_example.DEFAULT_TRAITS
        == persona.Traits(traits=("warm", "patient", "curious", "honest"))
    )


def test_every_owner_setting_has_a_working_value(tmp_path: pathlib.Path) -> None:
    settings = willy_example.Settings.model_validate(
        {
            **_REQUIRED,
            "WILLY_HOST": "0.0.0.0",
            "WILLY_PORT": "9000",
            "WILLY_ROOMS": "kitchen:counter, den:shelf",
            "WILLY_CAMERA": "1",
            "WILLY_FACE_MODELS": str(tmp_path),
            "WILLY_TRAITS": "patient, exact",
            "WILLY_HOUSEHOLD_FILE": _household(tmp_path, json.dumps(_HOUSEHOLD)),
        }
    )

    assert (settings.host, settings.port, settings.camera, settings.face_models) == ("0.0.0.0", 9000, True, tmp_path)
    # Each room is placed ROOM_SPACING_M further along, in the order the owner lists them.
    assert [(room.id, room.on, room.placement.position.x) for room in settings.rooms] == [
        ("kitchen", "counter", 0.0),
        ("den", "shelf", willy_example.ROOM_SPACING_M),
    ]
    assert all(room.placement.threshold_db is None for room in settings.rooms)
    assert settings.traits == persona.Traits(traits=("patient", "exact"))
    assert settings.household == (
        memory.Member(id="gabe", name="Gabe", role=memory.Role.ADULT),
        memory.Member(id="ada", name="Ada", role=memory.Role.CHILD, born=2018),
    )


def test_an_empty_variable_is_unset() -> None:
    settings = willy_example.Settings.model_validate(
        {**_REQUIRED, "WILLY_ROOMS": "", "WILLY_PORT": "", "WILLY_TRAITS": "", "WILLY_HOUSEHOLD_FILE": ""}
    )

    assert (settings.port, settings.traits, settings.household) == (8765, willy_example.DEFAULT_TRAITS, ())
    assert [room.id for room in settings.rooms] == ["living-room"]


def _refused(environ: dict[str, str]) -> str:
    with pytest.raises(pydantic.ValidationError) as raised:
        _ = willy_example.Settings.model_validate(environ)
    return str(raised.value)


@pytest.mark.parametrize("unset", sorted(_REQUIRED))
def test_each_endpoint_is_required_and_named_when_it_is_missing(unset: str) -> None:
    for missing in ({}, {unset: ""}):
        environ = {**_REQUIRED, **missing}
        if not missing:
            del environ[unset]

        assert unset in _refused(environ)


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("WILLY_PORT", "0"),
        ("WILLY_PORT", "eighty"),
        ("WILLY_STT_URL", "http://127.0.0.1:8001"),
        ("WILLY_ROOMS", "pantry"),
        ("WILLY_ROOMS", "kitchen:counter,kitchen:shelf"),
        ("WILLY_ROOMS", "Living Room:shelf"),
        ("WILLY_ROOMS", "living_room:shelf"),
        ("WILLY_ROOMS", "kitchen:Counter Top"),
        ("WILLY_CAMERA", "maybe"),
        ("WILLY_TRAITS", "warm,,honest"),
    ],
)
def test_a_bad_setting_is_refused_and_named_without_its_value(variable: str, value: str) -> None:
    message = _refused({**_REQUIRED, variable: value})

    assert variable in message
    assert value not in message


def test_seeing_needs_the_face_models() -> None:
    assert "WILLY_FACE_MODELS" in _refused({**_REQUIRED, "WILLY_CAMERA": "1"})


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"id": "ada", "name": "Ada"}),
        json.dumps([{"id": "ada", "name": "Ada", "role": "boss"}]),
        json.dumps([{"id": "self", "name": "Me"}]),
        json.dumps([{"id": "ada", "name": "Ada", "age": 8}]),
    ],
)
def test_a_bad_household_file_is_refused_and_named_without_its_content(tmp_path: pathlib.Path, content: str) -> None:
    message = _refused({**_REQUIRED, "WILLY_HOUSEHOLD_FILE": _household(tmp_path, content)})

    assert "WILLY_HOUSEHOLD_FILE" in message
    assert "boss" not in message and "Me" not in message


def test_a_missing_or_oversized_household_file_is_refused(tmp_path: pathlib.Path) -> None:
    assert "cannot be read" in _refused({**_REQUIRED, "WILLY_HOUSEHOLD_FILE": str(tmp_path / "absent.json")})
    oversized = " " * (willy_example.MAX_HOUSEHOLD_FILE_BYTES + 1)
    assert "larger than" in _refused({**_REQUIRED, "WILLY_HOUSEHOLD_FILE": _household(tmp_path, oversized)})


_PROJECT = pathlib.Path(__file__).resolve().parents[1]


def test_the_env_example_is_the_lab_and_every_setting_in_it_is_valid() -> None:
    lines = (_PROJECT / ".env.example").read_text(encoding="utf-8").splitlines()
    given = dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))

    settings = willy_example.Settings.model_validate(given)

    # Every required address is the lab model server's but speech, which is the speech engine anygame's family
    # shares on CT 107 (D-R-26), and the decision server is reached through a local forward; nothing in it is a
    # secret but the lab's keyless placeholder, and the local decision server takes no key at all.
    assert set(_REQUIRED) <= set(given)
    assert given["WILLY_TTS_URL"] == "http://192.168.2.32:8002"
    assert all("192.168.2.2:" in given[name] for name in _REQUIRED if name.endswith("_URL") and name != "WILLY_TTS_URL")
    assert (settings.model, settings.embed_model) == ("qwen3.8-27b", "embeddinggemma-2")
    assert [room.id for room in settings.rooms] == ["living-room"]
    assert given["OPENAI_API_KEY"] == "local"
    assert settings.typesafe_base_url == "http://127.0.0.1:8102"
    secrets = {"COMPUTER_TOKEN_SECRET", "OPENAI_API_KEY_FILE", "TYPESAFE_API_KEY", "TYPESAFE_API_KEY_FILE"}
    assert not secrets & set(given)


def test_the_household_example_is_a_household() -> None:
    settings = willy_example.Settings.model_validate(
        {**_REQUIRED, "WILLY_HOUSEHOLD_FILE": str(_PROJECT / "household.example.json")}
    )

    assert settings.household == (
        memory.Member(id="gabe", name="Gabe", role=memory.Role.ADULT),
        memory.Member(id="ada", name="Ada", role=memory.Role.CHILD, born=2018),
    )


def _environ(**given: str) -> dict[str, str]:
    """This process's environment for the child, without its telemetry, endpoint, Willy and mosfet variables."""

    environ = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OTEL_", "BOT_OTEL_", "OPENAI_", "TYPESAFE_", "WILLY_", "MOSFET_", "COMPUTER_"))
    }
    _ = environ.pop("PYTHONPATH", None)
    environ.update(given, BOT_OTEL_DISABLED="1")
    return environ


def _free_port() -> int:
    with sockets.socket(sockets.AF_INET, sockets.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return typing.cast(tuple[str, int], probe.getsockname())[1]


@pytest.mark.parametrize(("decider", "secrets"), [({}, _SECRETS), (_LOCAL_DECIDER, _KEYLESS)], ids=["hosted", "local"])
def test_python_m_defines_willy_and_serves_the_house_until_sigterm(
    tmp_path: pathlib.Path, decider: dict[str, str], secrets: dict[str, str]
) -> None:
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "willy_example",
            "--data-dir",
            str(tmp_path / "data"),
            "--secrets-dir",
            str(tmp_path / "absent"),
            "--ready-file",
            str(ready),
        ],
        cwd=tmp_path,
        env=_environ(
            **_REQUIRED,
            **secrets,
            **decider,
            WILLY_PORT=str(_free_port()),
            WILLY_ROOMS="kitchen:counter,den:shelf",
            WILLY_HOUSEHOLD_FILE=_household(tmp_path, json.dumps(_HOUSEHOLD)),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # A host's usual umask: the data dir holds every voiceprint, so it is private however the host is set.
        umask=0o022,
    )
    try:
        deadline = time.monotonic() + _RUN_BOUND
        while not ready.exists():
            if child.poll() is not None:
                raise AssertionError(f"willy exited {child.returncode} early: {child.communicate()[1].decode()}")
            if time.monotonic() > deadline:
                raise AssertionError("willy did not become ready")
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
    # One file database, shared by memory and the voiceprints, in a data dir only its owner can enter.
    assert (tmp_path / "data" / "memory.sqlite3").exists()
    assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700
    for secret in secrets.values():
        assert secret.encode() not in stderr


def test_python_m_opens_its_database_only_once_willy_starts(tmp_path: pathlib.Path) -> None:
    """D-W8-20: the bot file opens nothing; a database that cannot be opened fails as Willy starts, under bot.run.

    The database holds memory tables but no schema revision, so opening it is refused (``MigrationError``).
    That is reported by the running bot's diagnostics, never raised by the bot file before ``bot.run``.
    """

    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    engine = sqlalchemy.create_engine(f"sqlite+pysqlite:///{data / 'memory.sqlite3'}")
    with engine.begin() as connection:
        _ = connection.execute(sqlalchemy.text(f"CREATE TABLE {memory.MEMORY_TABLE} (memory_id TEXT PRIMARY KEY)"))
    engine.dispose()
    child = subprocess.Popen(
        [sys.executable, "-m", "willy_example", "--data-dir", str(data), "--secrets-dir", str(tmp_path / "absent")],
        cwd=tmp_path,
        env=_environ(
            **_REQUIRED,
            **_SECRETS,
            WILLY_PORT=str(_free_port()),
            WILLY_HOUSEHOLD_FILE=_household(tmp_path, json.dumps(_HOUSEHOLD)),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = child.communicate(timeout=_RUN_BOUND)
    finally:
        if child.poll() is None:
            child.kill()
            _ = child.communicate()

    reported = stderr.decode(errors="replace")
    # 70: bot.run's supervisor gave up restarting a Willy whose memory cannot open, and said why.
    assert child.returncode == 70, reported
    assert stdout == b""
    assert "Traceback" not in reported
    assert "Memory could not open its database: MigrationError" in reported
    assert "exhausted its restart budget" in reported
    for secret in _SECRETS.values():
        assert secret not in reported


@pytest.mark.parametrize(
    ("given", "named"),
    [
        ({"WILLY_DIAR_URL": ""}, b"WILLY_DIAR_URL"),
        ({"WILLY_HOUSEHOLD_FILE": "{household}"}, b"WILLY_HOUSEHOLD_FILE"),
    ],
)
def test_python_m_names_a_bad_setting_and_exits_1_without_a_secret(
    tmp_path: pathlib.Path, given: dict[str, str], named: bytes
) -> None:
    household = _household(tmp_path, json.dumps([{"id": "ada", "name": "Ada", "role": "boss"}]))
    environ = {**_REQUIRED, **_SECRETS, **{name: value.format(household=household) for name, value in given.items()}}
    finished = subprocess.run(
        [sys.executable, "-m", "willy_example", "--data-dir", str(tmp_path / "data")],
        cwd=tmp_path,
        env=_environ(**environ),
        capture_output=True,
        timeout=_RUN_BOUND,
        check=False,
    )

    assert finished.returncode == 1, finished.stderr.decode(errors="replace")
    assert finished.stdout == b""
    assert named in finished.stderr
    assert b"boss" not in finished.stderr
    for secret in _SECRETS.values():
        assert secret.encode() not in finished.stderr
    # Nothing started: not even the database was opened.
    assert not (tmp_path / "data" / "memory.sqlite3").exists()

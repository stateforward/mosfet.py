"""``willy_example``: the owner's ``Settings``, and its bot file, ``python -m willy_example``, run as written.

Every setting is given in the test; nothing reads the process environment. The child tests run the bot file
from this project's environment in a temporary directory with no ``PYTHONPATH``, so it is imported exactly as
the README documents. Every endpoint is a closed local port: nothing reaches the bot, so it never hears anyone,
and its birth (which asks the model for a voice) fails and waits, as it would with the lab down.
"""

import asyncio
import collections.abc
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
from mosfet.abilities import memory, persona, processing
from mosfet.environment import space
from mosfet.providers import openai_compat

_CLOSED = "127.0.0.1:9"
_REQUIRED = {
    "WILLY_INTUITION_URL": f"http://{_CLOSED}/v1",
    "WILLY_INTUITION_MODEL": "fast-model",
    "WILLY_REASONING_URL": f"http://{_CLOSED}/v1",
    "WILLY_REASONING_MODEL": "local-model",
    "WILLY_EMBED_URL": f"http://{_CLOSED}/v1",
    "WILLY_EMBED_MODEL": "embeddinggemma-2",
    "WILLY_STT_URL": f"ws://{_CLOSED}",
    "WILLY_TRANSCRIBE_URL": f"http://{_CLOSED}/v1",
    "WILLY_TTS_URL": f"http://{_CLOSED}",
    "WILLY_VOICEPRINT_URL": f"http://{_CLOSED}",
    "WILLY_EMOTION_URL": f"http://{_CLOSED}",
    "WILLY_DIAR_URL": f"http://{_CLOSED}",
    "DECISION_BASE_URL": f"http://{_CLOSED}",
}
_SECRETS = {"OPENAI_API_KEY": "key-5d1a", "DECISION_API_KEY": "key-9b07", "COMPUTER_TOKEN_SECRET": "secret-8e3f"}
# A local decision server takes no key, so the bot file needs no decision_api_key for it.
_KEYLESS = {name: secret for name, secret in _SECRETS.items() if name != "DECISION_API_KEY"}
# Starting the child imports mosfet and the providers before the bot file runs; that is not under test.
_RUN_BOUND = 60.0
# Stop grace (8 s default) plus the 1.5 s bounded process exit and scheduling slack.
_STOP_BOUND = 10.5


def test_the_required_settings_alone_take_every_default() -> None:
    settings = willy_example.Settings.model_validate(_REQUIRED)

    assert (settings.intuition_url, settings.intuition_model, settings.reasoning_url, settings.reasoning_model) == (
        f"http://{_CLOSED}/v1",
        "fast-model",
        f"http://{_CLOSED}/v1",
        "local-model",
    )
    assert (settings.embed_url, settings.embed_model) == (f"http://{_CLOSED}/v1", "embeddinggemma-2")
    assert (settings.host, settings.port, settings.camera, settings.face_models) == ("127.0.0.1", 8765, False, None)
    assert (settings.decision_base_url, settings.decision_max_context_tokens) == (f"http://{_CLOSED}", None)
    # One room, its computer on a shelf.
    assert settings.rooms == (
        willy_example.Room(id="living-room", on="shelf", placement=space.Placement(position=space.Position(0.0, 0.0))),
    )
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
            "DECISION_MAX_CONTEXT_TOKENS": "2048",
        }
    )

    assert (settings.host, settings.port, settings.camera, settings.face_models) == ("0.0.0.0", 9000, True, tmp_path)
    assert settings.decision_max_context_tokens == 2048
    # Each room is placed ROOM_SPACING_M further along, in the order the owner lists them.
    assert [(room.id, room.on, room.placement.position.x) for room in settings.rooms] == [
        ("kitchen", "counter", 0.0),
        ("den", "shelf", willy_example.ROOM_SPACING_M),
    ]
    assert all(room.placement.threshold_db is None for room in settings.rooms)
    assert settings.traits == persona.Traits(traits=("patient", "exact"))


class _Recorder:
    """A chat endpoint that records the body each request carries beyond its messages, and selects nothing."""

    model: str = "local-model"

    def __init__(self) -> None:
        self.bodies: list[dict[str, object]] = []

    async def create_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        del messages, tools, response_format
        self.bodies.append(dict(extra_body or {}))
        return {"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]}

    def stream_chat_completion(
        self,
        *,
        messages: collections.abc.Sequence[dict[str, object]],
        tools: collections.abc.Sequence[object] = (),
        response_format: object | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> collections.abc.AsyncIterator[dict[str, object]]:
        del messages, tools, response_format, extra_body
        raise AssertionError("a turn with no drafts is asked for whole")


@pytest.mark.parametrize(
    ("given", "body"),
    [
        ({}, {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 1024}),
        (
            {"WILLY_THINKING_TOKENS": "256"},
            {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 256},
        ),
        ({"WILLY_THINKING_TOKENS": "0"}, {"chat_template_kwargs": {"enable_thinking": False}}),
    ],
    ids=["default", "owner-budget", "off"],
)
def test_willy_reasons_after_thinking_within_the_owners_budget(given: dict[str, str], body: dict[str, object]) -> None:
    """R-18, D-R-52: reasoning thinks first, within WILLY_THINKING_TOKENS (1024 by default); 0 turns its thinking
    off and sends no budget."""

    settings = willy_example.Settings.model_validate({**_REQUIRED, **given})
    endpoint = _Recorder()
    reasoning = openai_compat.Processor(client=endpoint, provider="willy", extra_body=settings.thinking)

    _ = asyncio.run(reasoning.process(processing.InputData(input="hello", instructions="Select events.")))

    assert endpoint.bodies == [body]


@pytest.mark.parametrize("budget", ["0", "1024"])
def test_willy_intuits_with_thinking_off_whatever_the_reasoning_budget(budget: str) -> None:
    """D-R-52: thinking off is what intuition is, so its body is fixed, never a setting."""

    _ = willy_example.Settings.model_validate({**_REQUIRED, "WILLY_THINKING_TOKENS": budget})
    endpoint = _Recorder()
    intuition = openai_compat.Processor(client=endpoint, provider="willy", extra_body=willy_example.INTUITION_BODY)

    _ = asyncio.run(intuition.process(processing.InputData(input="hello", instructions="Select events.")))

    assert endpoint.bodies == [{"chat_template_kwargs": {"enable_thinking": False}}]


def test_the_bot_file_thinks_in_the_shared_tiers() -> None:
    """D-R-52: the bot file builds its cognition from ``willy_example.tiers``, as the live check does."""

    main = (_PROJECT / "src" / "willy_example" / "__main__.py").read_text(encoding="utf-8")

    assert main.count("Tiers.of(settings, api_key=api_key, decision_key=decision_key)") == 1
    assert "reasoning=tiers.reasoning" in main
    assert "intuition=tiers.intuition" in main
    assert "reflex=tiers.reflex" in main
    assert main.count("Tiers.timeouts()") == 1


@pytest.mark.parametrize("value", ["-1", "lots"])
def test_a_thinking_budget_is_a_count_of_tokens(value: str) -> None:
    message = _refused({**_REQUIRED, "WILLY_THINKING_TOKENS": value})

    assert "WILLY_THINKING_TOKENS" in message
    assert value not in message


def test_the_owner_lists_nobody_willy_learns_who_people_are_itself() -> None:
    """R-11: there is no household to declare; who lives here is something Willy learns and remembers."""

    assert not {name for name in willy_example.Settings.model_fields if "household" in name}
    assert not {name for name in willy_example.__all__ if "HOUSEHOLD" in name}
    assert not (_PROJECT / "household.example.json").exists()
    # An old household variable names nothing Willy reads.
    assert willy_example.Settings.model_validate(
        {**_REQUIRED, "WILLY_HOUSEHOLD_FILE": "./household.json"}
    ) == willy_example.Settings.model_validate(_REQUIRED)


def test_an_empty_variable_is_unset() -> None:
    settings = willy_example.Settings.model_validate(
        {**_REQUIRED, "WILLY_ROOMS": "", "WILLY_PORT": "", "WILLY_TRAITS": ""}
    )

    assert (settings.port, settings.traits) == (8765, willy_example.DEFAULT_TRAITS)
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
        ("DECISION_MAX_CONTEXT_TOKENS", "-2048"),
        ("DECISION_MAX_CONTEXT_TOKENS", "many"),
    ],
)
def test_a_bad_setting_is_refused_and_named_without_its_value(variable: str, value: str) -> None:
    message = _refused({**_REQUIRED, variable: value})

    assert variable in message
    assert value not in message


def test_seeing_needs_the_face_models() -> None:
    assert "WILLY_FACE_MODELS" in _refused({**_REQUIRED, "WILLY_CAMERA": "1"})


_PROJECT = pathlib.Path(__file__).resolve().parents[1]


def test_the_env_example_is_the_lab_and_every_setting_in_it_is_valid() -> None:
    lines = (_PROJECT / ".env.example").read_text(encoding="utf-8").splitlines()
    given = dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))

    settings = willy_example.Settings.model_validate(given)

    # Every required address is Willy's own lab stack, CT 108, speech and the decision server included (R-14);
    # nothing in it is a secret but the lab's keyless placeholder, and the local decision server takes no key.
    assert set(_REQUIRED) <= set(given)
    assert all("192.168.2.2:" in given[name] for name in _REQUIRED if name.endswith("_URL"))
    # D-R-52: intuition and reasoning both run on Gemma for now, intuition with thinking off.
    assert (settings.intuition_url, settings.intuition_model) == ("http://192.168.2.2:8100/v1", "gemma-4-26b-a4b")
    assert (settings.reasoning_url, settings.reasoning_model) == ("http://192.168.2.2:8100/v1", "gemma-4-26b-a4b")
    assert settings.embed_model == "embeddinggemma-2"
    assert [room.id for room in settings.rooms] == ["living-room"]
    assert given["OPENAI_API_KEY"] == "local"
    # D-R-36: decide's context is 2048 tokens, and each decision request is fitted to it.
    assert (settings.decision_base_url, settings.decision_max_context_tokens) == ("http://192.168.2.2:8102", 2048)
    # R-18, D-R-52: the lab's reasoning turns think within 1024 tokens.
    assert settings.thinking_tokens == 1024
    secrets = {"COMPUTER_TOKEN_SECRET", "OPENAI_API_KEY_FILE", "DECISION_API_KEY", "DECISION_API_KEY_FILE"}
    assert not secrets & set(given)


def _environ(**given: str) -> dict[str, str]:
    """This process's environment for the child, without its telemetry, endpoint, Willy and mosfet variables."""

    environ = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OTEL_", "BOT_OTEL_", "OPENAI_", "DECISION_", "WILLY_", "MOSFET_", "COMPUTER_"))
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
        ({"WILLY_PORT": "boss"}, b"WILLY_PORT"),
    ],
)
def test_python_m_names_a_bad_setting_and_exits_1_without_a_secret(
    tmp_path: pathlib.Path, given: dict[str, str], named: bytes
) -> None:
    environ = {**_REQUIRED, **_SECRETS, **given}
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


def test_a_new_voice_or_face_tells_willy_whom_it_resembles() -> None:
    """R-13: below the match threshold but above a floor, a new print says which known people it is like."""

    for policy in (willy_example.VOICE_POLICY, willy_example.FACE_POLICY):
        assert policy.resemble_floor is not None
        assert 0.0 < policy.resemble_floor < policy.threshold

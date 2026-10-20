"""``minimal_bot_example``: a bot of its shape, and its bot file, ``python -m minimal_bot_example``, run as written.

The first test writes its own ``bot.define`` of a bot of the same shape on a scripted model. The child tests run
the bot file from this project's environment in a temporary directory with no ``PYTHONPATH``, so it is imported
exactly as the README documents. Its endpoint is a closed local port: the bot gets no input, so it never calls
the model.
"""

import asyncio
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

from mosfet import bot
from mosfet.abilities import memory
from mosfet.bot import testing

# The decision server intuition decides on is a closed local port too: nothing reaches the bot, so it never asks.
_ENDPOINT = {
    "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
    "OPENAI_MODEL": "any-model",
    "DECISION_BASE_URL": "http://127.0.0.1:9",
}
# Starting the child imports mosfet and the provider before it can be ready; that is not under test.
_START_BOUND = 60.0
# Stop grace (8 s default) plus the 1.5 s bounded process exit and scheduling slack.
_STOP_BOUND = 10.5


def test_a_device_free_bot_on_one_memory_starts_activates_and_stops_with_a_scripted_model() -> None:
    model = testing.Scripted()
    minimal = bot.define(
        "minimal",
        bot.cognition(model, intuition=testing.Decider(), memory=memory.Memory(), associative=True, consolidation=True),
        bot.allow("no_devices"),
    )

    async def run() -> None:
        async with testing.running(minimal) as running:
            assert testing.snapshot(running.bot).State.startswith("/minimal/active")

    (outline,) = bot.describe(minimal).bots
    assert outline.faculties == ("intuition", "reasoning", "reflection", "associative", "consolidation")
    assert outline.parts == ()
    assert bot.check(minimal) == ()
    asyncio.run(run())

    # Nothing reached it, so nothing woke it.
    assert model.calls() == ()


def _child(tmp_path: pathlib.Path, environ: dict[str, str], *flags: str) -> subprocess.Popen[bytes]:
    given = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OTEL_", "BOT_OTEL_", "OPENAI_", "DECISION_", "MOSFET_"))
    }
    _ = given.pop("PYTHONPATH", None)
    given.update(environ, BOT_OTEL_DISABLED="1")
    return subprocess.Popen(
        [sys.executable, "-m", "minimal_bot_example", "--data-dir", str(tmp_path / "data"), *flags],
        cwd=tmp_path,
        env=given,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# Intuition decides on the decision server DECISION_BASE_URL names. The decision_api_key secret is optional: a local
# server takes none, and a hosted one with the same wire format is given its key. DECISION_MAX_CONTEXT_TOKENS is too.
@pytest.mark.parametrize(
    ("decider", "keys"),
    [
        ({}, ("openai_api_key",)),
        ({"DECISION_MAX_CONTEXT_TOKENS": "2048"}, ("openai_api_key", "decision_api_key")),
    ],
    ids=["keyless", "keyed"],
)
def test_python_m_serves_the_bot_until_sigterm(
    tmp_path: pathlib.Path, decider: dict[str, str], keys: tuple[str, ...]
) -> None:
    ready = tmp_path / "ready"
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    for key in keys:
        _ = (secrets / key).write_text("key\n", encoding="utf-8")
    child = _child(tmp_path, {**_ENDPOINT, **decider}, "--secrets-dir", str(secrets), "--ready-file", str(ready))
    try:
        deadline = time.monotonic() + _START_BOUND
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


def test_python_m_without_the_endpoint_or_the_secret_exits_1_naming_it(tmp_path: pathlib.Path) -> None:
    undecided = {name: value for name, value in _ENDPOINT.items() if name != "DECISION_BASE_URL"}
    for environ, named in (
        ({}, b"OPENAI_BASE_URL and OPENAI_MODEL"),
        (_ENDPOINT, b"openai_api_key"),
        ({**undecided, "OPENAI_API_KEY": "key"}, b"DECISION_BASE_URL"),
    ):
        child = _child(tmp_path, environ, "--secrets-dir", str(tmp_path / "absent"))
        _, stderr = child.communicate(timeout=_START_BOUND)

        assert child.returncode == 1
        assert named in stderr


def test_python_m_with_a_bad_flag_exits_64(tmp_path: pathlib.Path) -> None:
    child = _child(tmp_path, _ENDPOINT, "--bogus")
    stdout, stderr = child.communicate(timeout=_START_BOUND)

    assert child.returncode == 64
    assert stdout == b""
    assert stderr.startswith(b"error[B0107]: bad process option: unknown option `--bogus`")

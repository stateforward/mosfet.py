"""``minimal_bot_example``: the bot it defines, and ``python -m minimal_bot_example`` run as written.

The child test runs the bot file from this project's environment in a temporary directory with no
``PYTHONPATH``, so it is imported exactly as the README documents. Its endpoint is a closed local port:
the bot gets no input, so it never calls the model.
"""

import asyncio
import os
import pathlib
import signal
import subprocess
import sys
import time

import minimal_bot_example
from mosfet import bot
from mosfet.abilities import memory
from mosfet.bot import testing

_ENDPOINT = {"OPENAI_BASE_URL": "http://127.0.0.1:9/v1", "OPENAI_MODEL": "any-model"}
# Starting the child imports mosfet and the provider before it can be ready; that is not under test.
_START_BOUND = 60.0
# Stop grace (8 s default) plus the 1.5 s bounded process exit and scheduling slack.
_STOP_BOUND = 10.5


def test_the_minimal_bot_is_one_device_free_bot_whose_faculties_share_one_memory() -> None:
    store = memory.Memory()
    model = minimal_bot_example.minimal(processor=testing.Scripted(), store=store)

    (outline,) = bot.describe(model).bots
    assert outline.name == "minimal"
    assert outline.faculties == ("intuition", "reasoning", "reflection", "associative", "consolidation")
    assert outline.parts == ()
    assert bot.check(model) == ()


def test_it_starts_activates_and_stops_with_a_scripted_model() -> None:
    model = testing.Scripted()

    async def run() -> None:
        async with testing.running(minimal_bot_example.minimal(processor=model, store=memory.Memory())) as running:
            assert testing.snapshot(running.bot).State.startswith("/Bot/active")

    asyncio.run(run())

    # Nothing reached it, so nothing woke it.
    assert model.calls() == ()


def _child(tmp_path: pathlib.Path, environ: dict[str, str], *flags: str) -> subprocess.Popen[bytes]:
    given = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OTEL_", "BOT_OTEL_", "OPENAI_", "MOSFET_"))
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


def test_python_m_serves_the_bot_until_sigterm(tmp_path: pathlib.Path) -> None:
    ready = tmp_path / "ready"
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    _ = (secrets / "openai_api_key").write_text("key\n", encoding="utf-8")
    child = _child(tmp_path, _ENDPOINT, "--secrets-dir", str(secrets), "--ready-file", str(ready))
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
    for environ, named in (({}, b"OPENAI_BASE_URL and OPENAI_MODEL"), (_ENDPOINT, b"openai_api_key")):
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

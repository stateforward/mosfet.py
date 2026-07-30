from __future__ import annotations

import collections.abc
import concurrent.futures
import pathlib
import threading

import bot.telemetry
import pytest
from opentelemetry import _logs

from bot.telemetry.configure import is_enabled, log_file, logger_provider
from bot.telemetry.generator import record_generator_request


def _force_flush() -> None:
    provider = logger_provider()
    if provider is not None:
        _ = provider.force_flush()


@pytest.fixture(autouse=True)
def _reset_telemetry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> collections.abc.Iterator[None]:
    monkeypatch.chdir(tmp_path)
    bot.telemetry.reset()
    monkeypatch.delenv("BOT_OTEL_DISABLED", raising=False)
    monkeypatch.delenv("BOT_OTEL_LOG_FILE", raising=False)
    # Silence set-once warnings; emission uses the module-retained provider.
    def _noop_set_logger_provider(_provider: object) -> None:
        return None

    monkeypatch.setattr(_logs, "set_logger_provider", _noop_set_logger_provider)
    yield
    bot.telemetry.reset()


def test_configure_enables_export_and_exposes_log_file() -> None:
    log_path = pathlib.Path("otel-logs.jsonl")
    assert bot.telemetry.configure(log_file=log_path) is True
    assert is_enabled() is True
    assert log_file() == log_path.resolve()


def test_configure_opt_out_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = pathlib.Path("disabled.jsonl")
    monkeypatch.setenv("BOT_OTEL_DISABLED", "true")
    assert bot.telemetry.configure(log_file=log_path) is False
    assert is_enabled() is False
    record_generator_request(
        provider="openai_compat",
        model="test-model",
        messages=[{"role": "user", "content": "should not persist"}],
        tools=(),
    )
    _force_flush()
    assert not log_path.exists()


def test_configure_enabled_false_writes_nothing() -> None:
    log_path = pathlib.Path("explicit-off.jsonl")
    assert bot.telemetry.configure(enabled=False, log_file=log_path) is False
    record_generator_request(
        provider="openai_compat",
        model=None,
        messages=[{"role": "system", "content": "nope"}],
    )
    _force_flush()
    assert not log_path.exists()


def test_configure_is_idempotent() -> None:
    first = pathlib.Path("first.jsonl")
    second = pathlib.Path("second.jsonl")
    assert bot.telemetry.configure(log_file=first) is True
    assert bot.telemetry.configure(enabled=False, log_file=second) is True
    record_generator_request(
        provider="openai_compat",
        model=None,
        messages=[{"role": "user", "content": "once"}],
    )
    _force_flush()
    assert first.is_file()
    assert not second.exists()


def test_configure_rejects_path_outside_cwd(tmp_path: pathlib.Path) -> None:
    outside = tmp_path.parent / f"outside-{tmp_path.name}.jsonl"
    with pytest.raises(ValueError, match="must resolve under the process working directory"):
        _ = bot.telemetry.configure(log_file=outside)
    assert is_enabled() is False
    # Failed configure must not stick as configured — a good path can still install.
    assert bot.telemetry.configure(log_file="good.jsonl") is True
    assert is_enabled() is True


def test_configure_rejects_symlink_escape(tmp_path: pathlib.Path) -> None:
    outside = tmp_path.parent / f"symlink-target-{tmp_path.name}.jsonl"
    link = pathlib.Path("escape-link.jsonl")
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="must resolve under the process working directory"):
        _ = bot.telemetry.configure(log_file=link)
    assert is_enabled() is False


def test_configure_accepts_relative_path_under_cwd() -> None:
    nested = pathlib.Path("logs") / "nested.jsonl"
    assert bot.telemetry.configure(log_file=nested) is True
    assert log_file() == nested.resolve()
    record_generator_request(
        provider="openai_compat",
        model=None,
        messages=[{"role": "user", "content": "nested"}],
    )
    _force_flush()
    assert nested.is_file()


def test_configure_concurrent_single_outcome() -> None:
    """Hammer configure() from many threads; one consistent enabled outcome and one log file."""

    log_path = pathlib.Path("otel-logs.jsonl")
    workers = 16
    barrier = threading.Barrier(workers)
    results: list[bool] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        enabled = bot.telemetry.configure(log_file=log_path)
        with results_lock:
            results.append(enabled)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker) for _ in range(workers)]
        for future in concurrent.futures.as_completed(futures):
            _ = future.result()

    assert len(results) == workers
    assert set(results) == {True}
    assert is_enabled() is True
    record_generator_request(
        provider="openai_compat",
        model="test-model",
        messages=[{"role": "user", "content": "concurrent"}],
        tools=(),
    )
    _force_flush()
    assert log_path.is_file()
    assert len(list(pathlib.Path(".").glob("*.jsonl"))) == 1


def test_record_is_noop_until_configure() -> None:
    """Library emission must not auto-install; configure() is the app boundary."""

    assert is_enabled() is False
    record_generator_request(
        provider="openai_compat",
        model="auto",
        messages=[{"role": "user", "content": "no-auto-configure"}],
        tools=(),
    )
    _force_flush()
    assert is_enabled() is False
    assert not pathlib.Path("otel-logs.jsonl").exists()


def test_record_opt_out_after_configure(monkeypatch: pytest.MonkeyPatch) -> None:
    """After explicit configure(), env opt-out still disables emission."""

    monkeypatch.setenv("BOT_OTEL_DISABLED", "true")
    log_path = pathlib.Path("otel-logs.jsonl")
    assert bot.telemetry.configure(log_file=log_path) is False
    record_generator_request(
        provider="openai_compat",
        model="auto",
        messages=[{"role": "user", "content": "should-not-persist"}],
        tools=(),
    )
    _force_flush()
    assert is_enabled() is False
    assert not log_path.exists()

from __future__ import annotations

import collections.abc
import json
import os
import pathlib
import stat

import bot.telemetry
import pytest
from opentelemetry import _logs

from bot.telemetry.configure import logger_provider
from bot.telemetry.export import JsonlFileLogRecordExporter
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
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    # Silence set-once warnings; emission uses the module-retained provider.
    def _noop_set_logger_provider(_provider: object) -> None:
        return None

    monkeypatch.setattr(_logs, "set_logger_provider", _noop_set_logger_provider)
    yield
    bot.telemetry.reset()


def test_export_writes_generator_request_with_mode_600() -> None:
    log_path = pathlib.Path("otel-logs.jsonl")
    assert bot.telemetry.configure(log_file=log_path) is True
    record_generator_request(
        provider="openai_compat",
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        tools=(),
    )
    _force_flush()
    assert log_path.is_file()
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["attributes"]["component"] == "text.generator"
    assert payload["attributes"]["provider"] == "openai_compat"
    assert payload["attributes"]["model"] == "test-model"
    assert payload["attributes"]["stage"] == "request"


def test_export_rejects_replaced_symlink_leaf_outside_cwd(tmp_path: pathlib.Path) -> None:
    """After configure, replacing the leaf with an outside symlink must not write out."""

    log_path = pathlib.Path("otel-logs.jsonl")
    assert bot.telemetry.configure(log_file=log_path) is True
    record_generator_request(
        provider="openai_compat",
        model="test-model",
        messages=[{"role": "user", "content": "seed"}],
        tools=(),
    )
    _force_flush()
    assert log_path.is_file()

    outside = tmp_path.parent / f"nofollow-target-{tmp_path.name}.jsonl"
    _ = outside.write_text("sentinel-outside\n", encoding="utf-8")
    log_path.unlink()
    log_path.symlink_to(outside)

    # Processor swallows exporter errors; assert open/export fails and outside is untouched.
    exporter = JsonlFileLogRecordExporter(tmp_path / "otel-logs.jsonl")
    with pytest.raises(OSError):
        _ = exporter.export(())
    exporter.shutdown()

    record_generator_request(
        provider="openai_compat",
        model="test-model",
        messages=[{"role": "user", "content": "must-not-escape"}],
        tools=(),
    )
    _force_flush()
    assert outside.read_text(encoding="utf-8") == "sentinel-outside\n"
    assert "must-not-escape" not in outside.read_text(encoding="utf-8")


def test_export_rejects_replaced_parent_symlink_outside_cwd(tmp_path: pathlib.Path) -> None:
    """After construction, replacing a nested parent dir with a symlink must not escape."""

    log_path = pathlib.Path("logs") / "request.jsonl"
    exporter = JsonlFileLogRecordExporter(log_path)
    assert (tmp_path / "logs").is_dir()
    assert not (tmp_path / "logs").is_symlink()

    outside = tmp_path.parent / f"parent-escape-{tmp_path.name}"
    outside.mkdir()
    (tmp_path / "logs").rmdir()
    (tmp_path / "logs").symlink_to(outside)

    with pytest.raises(OSError):
        _ = exporter.export(())
    exporter.shutdown()

    assert not (outside / "request.jsonl").exists()
    assert list(outside.iterdir()) == []


def test_export_stays_in_configure_cwd_after_chdir(tmp_path: pathlib.Path) -> None:
    """Configure under A, chdir to B, emit — writes stay under A; B stays empty."""

    root_a = tmp_path / "configure-root"
    root_b = tmp_path / "divert"
    root_a.mkdir()
    root_b.mkdir()
    os.chdir(root_a)
    assert bot.telemetry.configure(log_file="otel-logs.jsonl") is True
    os.chdir(root_b)
    record_generator_request(
        provider="openai_compat",
        model="test-model",
        messages=[{"role": "user", "content": "chdir-attack"}],
        tools=(),
    )
    _force_flush()
    written = root_a / "otel-logs.jsonl"
    assert written.is_file()
    assert "chdir-attack" in written.read_text(encoding="utf-8")
    assert list(root_b.iterdir()) == []


def test_export_rejects_hardlinked_leaf(tmp_path: pathlib.Path) -> None:
    """Hard-linking an outside file into the leaf path must fail export and leave sentinel."""

    outside = tmp_path.parent / f"hardlink-src-{tmp_path.name}.jsonl"
    _ = outside.write_text("sentinel-hardlink\n", encoding="utf-8")
    log_path = pathlib.Path("otel-logs.jsonl")
    assert bot.telemetry.configure(log_file=log_path) is True
    os.link(outside, log_path)

    exporter = JsonlFileLogRecordExporter(tmp_path / "otel-logs.jsonl")
    with pytest.raises(OSError, match="hard-linked"):
        _ = exporter.export(())
    exporter.shutdown()

    record_generator_request(
        provider="openai_compat",
        model="test-model",
        messages=[{"role": "user", "content": "must-not-follow-hardlink"}],
        tools=(),
    )
    _force_flush()
    assert outside.read_text(encoding="utf-8") == "sentinel-hardlink\n"
    assert "must-not-follow-hardlink" not in outside.read_text(encoding="utf-8")

from __future__ import annotations

import collections.abc
import json
import math
import os
import pathlib
import stat

import mosfet.telemetry
import pytest
from opentelemetry import _logs
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from mosfet.telemetry.configure import logger_provider
from mosfet.telemetry.export import JsonlFileLogRecordExporter, JsonlFileMetricExporter
from mosfet.telemetry.generator import record_generator_request


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
    mosfet.telemetry.reset()
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
    mosfet.telemetry.reset()


def test_export_writes_generator_request_with_mode_600(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD", "true")
    log_path = pathlib.Path("otel-logs.jsonl")
    assert mosfet.telemetry.configure(log_file=log_path) is True
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
    assert mosfet.telemetry.configure(log_file=log_path) is True
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


def test_export_stays_in_configure_cwd_after_chdir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure under A, chdir to B, emit — writes stay under A; B stays empty."""

    monkeypatch.setenv("BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD", "true")
    root_a = tmp_path / "configure-root"
    root_b = tmp_path / "divert"
    root_a.mkdir()
    root_b.mkdir()
    os.chdir(root_a)
    assert mosfet.telemetry.configure(log_file="otel-logs.jsonl") is True
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
    assert mosfet.telemetry.configure(log_file=log_path) is True
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


def _record_counter(exporter: JsonlFileMetricExporter, *, increments: int) -> None:
    """Drive a private (never global) MeterProvider so the exporter sees one real collection."""

    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=math.inf)
    provider = MeterProvider(
        metric_readers=[reader],
        resource=Resource.create({"service.name": "export-test", "service.instance.id": "host-1"}),
    )
    counter = provider.get_meter("bot.telemetry.export.test").create_counter(
        "bot.telemetry.export.test.count",
        unit="{event}",
        description="Counts test events.",
    )
    counter.add(increments, {"outcome": "delivered"})
    _ = provider.force_flush()
    provider.shutdown()


def test_metric_export_writes_one_line_per_metric_with_mode_600() -> None:
    metric_path = pathlib.Path("otel-metrics.jsonl")
    _record_counter(JsonlFileMetricExporter(metric_path), increments=3)

    assert metric_path.is_file()
    assert stat.S_IMODE(metric_path.stat().st_mode) == 0o600
    lines = [json.loads(line) for line in metric_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = lines[0]
    assert payload["name"] == "bot.telemetry.export.test.count"
    assert payload["unit"] == "{event}"
    assert payload["description"] == "Counts test events."
    assert payload["scope"] == "bot.telemetry.export.test"
    assert payload["kind"] == "Sum"
    assert payload["resource"]["service.name"] == "export-test"
    assert payload["resource"]["service.instance.id"] == "host-1"
    assert isinstance(payload["timestamp"], str)
    points = payload["data"]["data_points"]
    assert [point["value"] for point in points] == [3]
    assert points[0]["attributes"] == {"outcome": "delivered"}
    assert payload["data"]["is_monotonic"] is True


def test_metric_export_rejects_replaced_symlink_leaf_outside_cwd(tmp_path: pathlib.Path) -> None:
    """The metric file shares the confined writer: a swapped leaf symlink is refused."""

    metric_path = pathlib.Path("otel-metrics.jsonl")
    exporter = JsonlFileMetricExporter(metric_path)
    outside = tmp_path.parent / f"metric-escape-{tmp_path.name}.jsonl"
    _ = outside.write_text("sentinel-metric\n", encoding="utf-8")
    metric_path.symlink_to(outside)

    _record_counter(exporter, increments=1)

    assert outside.read_text(encoding="utf-8") == "sentinel-metric\n"

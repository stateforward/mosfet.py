from __future__ import annotations

import collections.abc
import concurrent.futures
import importlib
import json
import pathlib
import threading
import typing

import mosfet.telemetry
import pytest
from opentelemetry import _logs, metrics
from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult, MetricsData, Sum

from mosfet.telemetry import span
from mosfet.telemetry.configure import (
    is_enabled,
    log_file,
    logger_provider,
    meter_provider,
    metric_file,
    span_file,
    tracer_provider,
)
from mosfet.telemetry.generator import record_generator_request

_CONFIGURE_MOD = importlib.import_module("mosfet.telemetry.configure")

# Module-level, like the library meters (``telemetry/hsm.py``): a proxy meter taken at import,
# before any ``configure()`` installs the process MeterProvider.
_PROBE_NAME = "bot.telemetry.configure.probe.count"
_PROBE = metrics.get_meter("bot.telemetry.configure.test").create_counter(
    _PROBE_NAME,
    unit="{probe}",
    description="Counts probe increments made by the configure tests.",
)


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
    monkeypatch.delenv("BOT_OTEL_SPAN_FILE", raising=False)
    monkeypatch.delenv("BOT_OTEL_METRIC_FILE", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    # Silence set-once warnings; emission uses the module-retained provider.
    def _noop_set_logger_provider(_provider: object) -> None:
        return None

    monkeypatch.setattr(_logs, "set_logger_provider", _noop_set_logger_provider)
    yield
    mosfet.telemetry.reset()


def test_configure_enables_export_and_exposes_log_file() -> None:
    log_path = pathlib.Path("otel-logs.jsonl")
    assert mosfet.telemetry.configure(log_file=log_path) is True
    assert is_enabled() is True
    assert log_file() == log_path.resolve()


def test_configure_opt_out_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = pathlib.Path("disabled.jsonl")
    monkeypatch.setenv("BOT_OTEL_DISABLED", "true")
    assert mosfet.telemetry.configure(log_file=log_path) is False
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
    assert mosfet.telemetry.configure(enabled=False, log_file=log_path) is False
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
    assert mosfet.telemetry.configure(log_file=first) is True
    assert mosfet.telemetry.configure(enabled=False, log_file=second) is True
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
        _ = mosfet.telemetry.configure(log_file=outside)
    assert is_enabled() is False
    # Failed configure must not stick as configured — a good path can still install.
    assert mosfet.telemetry.configure(log_file="good.jsonl") is True
    assert is_enabled() is True


def test_configure_rejects_symlink_escape(tmp_path: pathlib.Path) -> None:
    outside = tmp_path.parent / f"symlink-target-{tmp_path.name}.jsonl"
    link = pathlib.Path("escape-link.jsonl")
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="must resolve under the process working directory"):
        _ = mosfet.telemetry.configure(log_file=link)
    assert is_enabled() is False


def test_configure_accepts_relative_path_under_cwd() -> None:
    nested = pathlib.Path("logs") / "nested.jsonl"
    assert mosfet.telemetry.configure(log_file=nested) is True
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
        enabled = mosfet.telemetry.configure(log_file=log_path)
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
    assert mosfet.telemetry.configure(log_file=log_path) is False
    record_generator_request(
        provider="openai_compat",
        model="auto",
        messages=[{"role": "user", "content": "should-not-persist"}],
        tools=(),
    )
    _force_flush()
    assert is_enabled() is False
    assert not log_path.exists()


class _FakeOtlpExporter:
    endpoint: str | None

    def __init__(self, endpoint: str | None = None, **_kwargs: object) -> None:
        self.endpoint = endpoint

    def export(self, spans: object) -> object:
        del spans
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis
        return True


def _capture_otlp_endpoints(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    seen: list[str | None] = []

    class _Recording(_FakeOtlpExporter):
        def __init__(self, endpoint: str | None = None, **kwargs: object) -> None:
            seen.append(endpoint)
            super().__init__(endpoint, **kwargs)

    monkeypatch.setattr(_CONFIGURE_MOD, "OTLPSpanExporter", _Recording)
    _ = _capture_otlp_metrics(monkeypatch)
    return seen


class _RecordingMetricExporter(MetricExporter):
    """Stands in for the OTLP gRPC metric exporter; keeps what it is handed."""

    endpoint: str | None
    batches: list[MetricsData]

    def __init__(self, endpoint: str | None = None) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.batches = []

    @typing.override
    def export(self, metrics_data: MetricsData, timeout_millis: float = 10_000, **kwargs: object) -> MetricExportResult:
        del timeout_millis, kwargs
        self.batches.append(metrics_data)
        return MetricExportResult.SUCCESS

    @typing.override
    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        del timeout_millis
        return True

    @typing.override
    def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
        del timeout_millis, kwargs


def _capture_otlp_metrics(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingMetricExporter]:
    created: list[_RecordingMetricExporter] = []

    def _create(endpoint: str | None = None, **kwargs: object) -> _RecordingMetricExporter:
        del kwargs
        exporter = _RecordingMetricExporter(endpoint)
        created.append(exporter)
        return exporter

    monkeypatch.setattr(_CONFIGURE_MOD, "OTLPMetricExporter", _create)
    return created


def _probe_payloads(path: pathlib.Path) -> list[dict[str, typing.Any]]:
    payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [payload for payload in payloads if payload["name"] == _PROBE_NAME]


def _probe_values(path: pathlib.Path) -> list[float]:
    return [float(point["value"]) for payload in _probe_payloads(path) for point in payload["data"]["data_points"]]


def _batch_probe_values(batches: list[MetricsData]) -> list[float]:
    return [
        float(point.value)
        for batch in batches
        for resource_metrics in batch.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == _PROBE_NAME and isinstance(metric.data, Sum)
        for point in metric.data.data_points
    ]


def test_configure_without_otlp_endpoint_is_jsonl_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_otlp_endpoints(monkeypatch)
    assert mosfet.telemetry.configure() is True
    assert seen == []
    provider = tracer_provider()
    assert provider is not None
    assert provider.resource.attributes.get("bot.otel.otlp_export") is None
    assert span_file() is not None


def test_configure_otlp_traces_endpoint_installs_second_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_otlp_endpoints(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:4317")
    assert mosfet.telemetry.configure() is True
    assert seen == ["http://127.0.0.1:4317"]
    provider = tracer_provider()
    assert provider is not None
    assert provider.resource.attributes.get("bot.otel.otlp_export") == "true"
    assert span_file() is not None


def test_configure_otlp_base_endpoint_does_not_append_traces_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_otlp_endpoints(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    assert mosfet.telemetry.configure() is True
    assert seen == ["http://127.0.0.1:4317"]


def test_configure_otlp_base_endpoint_strips_legacy_http_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_otlp_endpoints(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317/v1/traces")
    assert mosfet.telemetry.configure() is True
    assert seen == ["http://127.0.0.1:4317"]


def test_configure_bot_otlp_endpoint_wins_and_strips_legacy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_otlp_endpoints(monkeypatch)
    monkeypatch.setenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9999/v1/traces")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    assert mosfet.telemetry.configure() is True
    assert seen == ["http://127.0.0.1:9999"]


def test_configure_otlp_keeps_jsonl_span_export(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _capture_otlp_endpoints(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:4317")
    assert mosfet.telemetry.configure() is True
    with span.operation(
        "bot.test.operation",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="probe",
    ):
        pass
    provider = tracer_provider()
    assert provider is not None
    _ = provider.force_flush()
    path = span_file()
    assert path is not None
    assert path.is_file()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1


def test_module_meter_exports_into_each_configuration_across_reset_cycles() -> None:
    """Import-time meters bind to the once-installed provider and follow every reconfiguration."""

    assert mosfet.telemetry.configure() is True
    mosfet.telemetry.reset()

    first = pathlib.Path("a.jsonl")
    assert mosfet.telemetry.configure(metric_file=first) is True
    _PROBE.add(3)
    mosfet.telemetry.reset()
    assert first.is_file()
    assert max(_probe_values(first)) >= 3
    first_contents = first.read_text(encoding="utf-8")

    second = pathlib.Path("b.jsonl")
    assert mosfet.telemetry.configure(metric_file=second) is True
    _PROBE.add(2)
    mosfet.telemetry.reset()
    assert max(_probe_values(second)) >= 5
    # The first configuration's exporter was unbound and shut down by the first reset.
    assert first.read_text(encoding="utf-8") == first_contents


def test_reset_never_shuts_down_the_meter_provider() -> None:
    assert mosfet.telemetry.configure() is True
    installed = mosfet.telemetry.meter_provider()
    assert installed is not None
    assert installed is meter_provider()
    assert metrics.get_meter_provider() is installed

    mosfet.telemetry.reset()
    assert mosfet.telemetry.meter_provider() is installed
    assert mosfet.telemetry.configure(metric_file="again.jsonl") is True
    assert mosfet.telemetry.meter_provider() is installed
    assert mosfet.telemetry.metric_file() == pathlib.Path("again.jsonl").resolve()
    _PROBE.add(1)
    mosfet.telemetry.reset()
    assert max(_probe_values(pathlib.Path("again.jsonl"))) >= 1


def test_configure_stamps_service_identity_on_logs_spans_and_metrics() -> None:
    metric_path = pathlib.Path("identity-metrics.jsonl")
    assert (
        mosfet.telemetry.configure(metric_file=metric_path, service_name="builder", service_instance_id="host-7")
        is True
    )
    for provider in (logger_provider(), tracer_provider()):
        assert provider is not None
        assert provider.resource.attributes.get("service.name") == "builder"
        assert provider.resource.attributes.get("service.instance.id") == "host-7"
    _PROBE.add(1)
    mosfet.telemetry.reset()
    payloads = _probe_payloads(metric_path)
    assert payloads
    identities = {
        (payload["resource"]["service.name"], payload["resource"]["service.instance.id"]) for payload in payloads
    }
    assert identities == {("builder", "host-7")}


def test_configure_service_name_falls_back_to_env_then_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert mosfet.telemetry.configure() is True
    provider = tracer_provider()
    assert provider is not None
    assert provider.resource.attributes.get("service.name") == "stateforward.mosfet"
    assert "service.instance.id" not in provider.resource.attributes
    mosfet.telemetry.reset()

    monkeypatch.setenv("OTEL_SERVICE_NAME", "from-env")
    assert mosfet.telemetry.configure() is True
    provider = tracer_provider()
    assert provider is not None
    assert provider.resource.attributes.get("service.name") == "from-env"
    mosfet.telemetry.reset()

    assert mosfet.telemetry.configure(service_name="from-argument") is True
    provider = tracer_provider()
    assert provider is not None
    assert provider.resource.attributes.get("service.name") == "from-argument"


def test_configure_rejects_blank_service_name() -> None:
    with pytest.raises(ValueError, match="service_name"):
        _ = mosfet.telemetry.configure(service_name="  ")
    assert is_enabled() is False


def test_configure_rejects_blank_service_instance_id() -> None:
    with pytest.raises(ValueError, match="service_instance_id"):
        _ = mosfet.telemetry.configure(service_instance_id="")
    assert is_enabled() is False


def test_metric_file_follows_argument_env_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert mosfet.telemetry.configure() is True
    assert metric_file() == pathlib.Path("otel-metrics.jsonl").resolve()
    mosfet.telemetry.reset()
    assert metric_file() is None

    monkeypatch.setenv("BOT_OTEL_METRIC_FILE", "env-metrics.jsonl")
    assert mosfet.telemetry.configure() is True
    assert metric_file() == pathlib.Path("env-metrics.jsonl").resolve()
    mosfet.telemetry.reset()

    nested = pathlib.Path("metrics") / "nested.jsonl"
    assert mosfet.telemetry.configure(metric_file=nested) is True
    assert metric_file() == nested.resolve()


def test_disabled_configure_binds_no_metric_export() -> None:
    metric_path = pathlib.Path("disabled-metrics.jsonl")
    assert mosfet.telemetry.configure(enabled=False, metric_file=metric_path) is False
    assert metric_file() is None
    _PROBE.add(1)
    mosfet.telemetry.reset()
    assert not metric_path.exists()


def test_configure_rejects_metric_path_outside_cwd(tmp_path: pathlib.Path) -> None:
    outside = tmp_path.parent / f"outside-metrics-{tmp_path.name}.jsonl"
    with pytest.raises(ValueError, match="must resolve under the process working directory"):
        _ = mosfet.telemetry.configure(metric_file=outside)
    assert is_enabled() is False
    assert metric_file() is None


def test_configure_otlp_endpoint_binds_metric_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _capture_otlp_endpoints(monkeypatch)
    created = _capture_otlp_metrics(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317/v1/traces")
    assert mosfet.telemetry.configure(service_name="collector-bound") is True
    assert [exporter.endpoint for exporter in created] == ["http://127.0.0.1:4317"]
    _PROBE.add(4)
    mosfet.telemetry.reset()
    assert max(_batch_probe_values(created[0].batches)) >= 4
    stamped = {
        resource_metrics.resource.attributes.get("service.name")
        for batch in created[0].batches
        for resource_metrics in batch.resource_metrics
    }
    assert stamped == {"collector-bound"}


def test_configure_without_otlp_endpoint_binds_no_metric_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _capture_otlp_metrics(monkeypatch)
    assert mosfet.telemetry.configure() is True
    assert created == []


def test_configure_with_otlp_endpoint_keeps_local_jsonl_files(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _capture_otlp_endpoints(monkeypatch)
    _ = _capture_otlp_metrics(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    assert mosfet.telemetry.configure() is True
    assert is_enabled() is True
    assert logger_provider() is not None
    assert log_file() == pathlib.Path("otel-logs.jsonl").resolve()
    assert span_file() == pathlib.Path("otel-spans.jsonl").resolve()
    assert metric_file() == pathlib.Path("otel-metrics.jsonl").resolve()

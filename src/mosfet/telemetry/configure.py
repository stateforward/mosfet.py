"""Application-boundary OpenTelemetry configuration for logs, spans and metrics.

Configures a ``LoggerProvider`` (log records), a ``TracerProvider`` (spans) and
the process ``MeterProvider`` (metrics). Each signal has a local JSONL file
exporter, so a run leaves greppable ``otel-logs.jsonl``, ``otel-spans.jsonl``
and ``otel-metrics.jsonl`` artifacts. When an OTLP endpoint is configured,
spans and metrics are also exported over OTLP gRPC.

Call ``configure()`` (or ``ensure_configured()``) from an application boundary;
library emission is a no-op until then. When configure runs, default is enabled
(opt-out via ``enabled=False`` / ``BOT_OTEL_DISABLED``).

Rule exceptions (PY-LOG-002, PY-OBJ-002) and the user-approved
``opentelemetry-sdk`` / ``opentelemetry-api`` floors: see ``EXCEPTIONS.md``.

Process-global ``_RUNTIME`` matches OTEL's process-global providers
(``set_logger_provider`` / ``set_tracer_provider`` / ``set_meter_provider``).
Single writer: ``configure()`` / ``ensure_configured()`` / ``reset()`` under
``_RUNTIME_LOCK``. Enablement SoT is ``_RUNTIME.provider`` /
``_RUNTIME.tracer_provider`` only when that provider exports JSONL
(``bot.otel.jsonl_export`` resource marker). Readers never fall back to
``_logs.get_logger_provider()`` / ``trace.get_tracer_provider()``; the global
setters are install-only.

Metrics: OpenTelemetry sets the global ``MeterProvider`` once per process, and
import-time meters (``metrics.get_meter(...)``) are proxies that bind to it when
it is set. So the first enabled ``configure()`` creates and installs one
``MeterProvider`` whose only reader exports through a routing exporter, and
``reset()`` never shuts that provider down. Each ``configure()`` binds the
routing exporter to that configuration's metric exporters and resource;
``reset()`` flushes through them, then unbinds and shuts down those exporters
only. Import-time meters therefore keep exporting across every reconfiguration.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import pathlib
import threading
import typing

from opentelemetry import _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    MetricExportResult,
    MetricsData,
    PeriodicExportingMetricReader,
    ResourceMetrics,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from mosfet.telemetry.export import JsonlFileLogRecordExporter, JsonlFileMetricExporter, JsonlFileSpanExporter

_LOG = logging.getLogger(__name__)
_DISABLED_VALUES = frozenset({"1", "true", "yes", "on"})
_DEFAULT_LOG_FILE = "otel-logs.jsonl"
_DEFAULT_SPAN_FILE = "otel-spans.jsonl"
_DEFAULT_METRIC_FILE = "otel-metrics.jsonl"
_SERVICE_NAME = "stateforward.mosfet"
_EXPORT_ATTR = "bot.otel.jsonl_export"
_OTLP_EXPORT_ATTR = "bot.otel.otlp_export"
_LOG_FILE_ATTR = "bot.otel.log_file"
_SPAN_FILE_ATTR = "bot.otel.span_file"
_TRACES_PATH = "/v1/traces"


@dataclasses.dataclass(frozen=True)
class _MetricRoute:
    """One configuration's metric destinations and the resource its batches carry."""

    resource: Resource
    exporters: tuple[MetricExporter, ...]


class _RoutingMetricExporter(MetricExporter):
    """Hand each collected metric batch to the exporters the current configuration bound.

    The process ``MeterProvider``'s only reader exports here. ``configure()``
    binds a route and ``reset()`` releases it; with nothing bound a batch is
    dropped and the export still succeeds. Each batch is re-stamped with the
    bound configuration's resource, so service identity follows the
    configuration although the provider's own resource is fixed at install. The
    lock is held across export, so an exporter is never handed a batch after
    ``release()`` took it for shutdown.
    """

    _lock: threading.Lock
    _route: _MetricRoute | None

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._route = None

    def bind(self, route: _MetricRoute) -> None:
        """Route later batches to ``route`` (the caller has released any earlier route)."""

        with self._lock:
            self._route = route

    def release(self) -> None:
        """Unbind the current route and shut down its exporters; the router stays usable."""

        with self._lock:
            route, self._route = self._route, None
        if route is None:
            return
        for exporter in route.exporters:
            try:
                exporter.shutdown()
            except Exception as error:
                _LOG.warning("otel metric exporter shutdown failed error=%s", type(error).__name__)

    @typing.override
    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,
        **kwargs: object,
    ) -> MetricExportResult:
        del kwargs
        with self._lock:
            route = self._route
            if route is None:
                return MetricExportResult.SUCCESS
            stamped = MetricsData(
                resource_metrics=[
                    ResourceMetrics(
                        resource=route.resource,
                        scope_metrics=resource_metrics.scope_metrics,
                        schema_url=resource_metrics.schema_url,
                    )
                    for resource_metrics in metrics_data.resource_metrics
                ]
            )
            result = MetricExportResult.SUCCESS
            for exporter in route.exporters:
                try:
                    exported = exporter.export(stamped, timeout_millis=timeout_millis)
                except Exception:
                    # Documented boundary: each exporter logs its own failure
                    # (JsonlFileMetricExporter does); one failing sink must not
                    # starve the others of this batch.
                    exported = MetricExportResult.FAILURE
                if exported is not MetricExportResult.SUCCESS:
                    result = MetricExportResult.FAILURE
            return result

    @typing.override
    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        with self._lock:
            route = self._route
            if route is None:
                return True
            return all(exporter.force_flush(timeout_millis=timeout_millis) for exporter in route.exporters)

    @typing.override
    def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
        """Called once by the ``MeterProvider`` at process exit."""

        del timeout_millis, kwargs
        self.release()


@dataclasses.dataclass(frozen=True)
class _MetricPipeline:
    """The once-installed process ``MeterProvider`` and the router its reader exports through."""

    provider: MeterProvider
    router: _RoutingMetricExporter


@dataclasses.dataclass
class _Runtime:
    """OTEL export runtime boundary (process-global by the OTEL global setters)."""

    configured: bool = False
    provider: LoggerProvider | None = None
    tracer_provider: TracerProvider | None = None
    metric_file: pathlib.Path | None = None
    # Installed by the first enabled configure() and never shut down by reset().
    metric_pipeline: _MetricPipeline | None = None


# Process-global SoT for this module's exporting providers (and idempotency).
# Mutations go through ``configure()`` / ``ensure_configured()`` / ``reset()``
# under ``_RUNTIME_LOCK``. Log enablement reads ``_RUNTIME.provider`` only.
_RUNTIME = _Runtime()
_RUNTIME_LOCK = threading.Lock()


def _env_disabled() -> bool:
    raw = os.environ.get("BOT_OTEL_DISABLED", "").strip().lower()
    return raw in _DISABLED_VALUES


def _resolve_enabled(enabled: bool | None) -> bool:
    if enabled is False:
        return False
    if _env_disabled():
        return False
    if enabled is True:
        return True
    return True


def _resolve_file(
    override: str | pathlib.Path | None,
    *,
    env_var: str,
    default: str,
) -> pathlib.Path:
    if override is not None:
        return pathlib.Path(override)
    env_path = os.environ.get(env_var, "").strip()
    if env_path:
        return pathlib.Path(env_path)
    return pathlib.Path(default)


def _strip_http_traces_path(endpoint: str) -> str:
    """Drop a leftover OTLP/HTTP ``/v1/traces`` suffix; gRPC uses host:port only."""

    trimmed = endpoint.rstrip("/")
    if trimmed.endswith(_TRACES_PATH):
        return trimmed[: -len(_TRACES_PATH)]
    return trimmed


def otlp_endpoint() -> str | None:
    """Return the configured OTLP gRPC endpoint, or ``None`` when unset.

    Reads ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``, then
    ``BOT_OTEL_EXPORTER_OTLP_ENDPOINT``, then ``OTEL_EXPORTER_OTLP_ENDPOINT``.
    Does not append ``/v1/traces``. A leftover HTTP suffix is stripped.
    """

    for name in (
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "BOT_OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ):
        raw = os.environ.get(name, "").strip()
        if raw:
            return _strip_http_traces_path(raw)
    return None


def _confine_file(path: pathlib.Path) -> pathlib.Path:
    """Resolve ``path`` and require it to stay under the process working directory."""

    allowed = pathlib.Path.cwd().resolve()
    try:
        resolved = path.expanduser().resolve(strict=False)
    except OSError as error:
        message = "export file could not be resolved under the process working " + f"directory ({allowed}): {path}"
        raise ValueError(message) from error
    try:
        _ = resolved.relative_to(allowed)
    except ValueError as error:
        message = "export file must resolve under the process working " + f"directory ({allowed}); got {resolved}"
        raise ValueError(message) from error
    return resolved


def _provider_exports(provider: object | None) -> bool:
    if not isinstance(provider, LoggerProvider | TracerProvider):
        return False
    return provider.resource.attributes.get(_EXPORT_ATTR) == "true"


def _active_provider() -> LoggerProvider | None:
    """Return ``_RUNTIME.provider`` when it is an exporting LoggerProvider.

    Enablement SoT is ``_RUNTIME.provider`` only — never
    ``_logs.get_logger_provider()``.
    """

    provider = _RUNTIME.provider
    if isinstance(provider, LoggerProvider) and _provider_exports(provider):
        return provider
    return None


def _active_tracer_provider() -> TracerProvider | None:
    """Return ``_RUNTIME.tracer_provider`` when it is an exporting TracerProvider."""

    provider = _RUNTIME.tracer_provider
    if isinstance(provider, TracerProvider) and _provider_exports(provider):
        return provider
    return None


def _configure_locked(
    *,
    enabled: bool | None = None,
    log_file: str | pathlib.Path | None = None,
    span_file: str | pathlib.Path | None = None,
    metric_file: str | pathlib.Path | None = None,
    service_name: str | None = None,
    service_instance_id: str | None = None,
) -> bool:
    """Install or mark disabled. Caller must hold ``_RUNTIME_LOCK``."""

    if _RUNTIME.configured:
        return is_enabled()

    if service_name is not None and not service_name.strip():
        message = "service_name must not be blank"
        raise ValueError(message)
    if service_instance_id is not None and not service_instance_id.strip():
        message = "service_instance_id must not be blank"
        raise ValueError(message)

    resolved_enabled = _resolve_enabled(enabled)
    if not resolved_enabled:
        _RUNTIME.configured = True
        _RUNTIME.provider = None
        _RUNTIME.tracer_provider = None
        _RUNTIME.metric_file = None
        _LOG.info("OpenTelemetry export disabled")
        return False

    # Validate confinement before committing idempotent state so bad paths can be retried.
    resolved_path = _confine_file(_resolve_file(log_file, env_var="BOT_OTEL_LOG_FILE", default=_DEFAULT_LOG_FILE))
    resolved_span_path = _confine_file(
        _resolve_file(span_file, env_var="BOT_OTEL_SPAN_FILE", default=_DEFAULT_SPAN_FILE)
    )
    resolved_metric_path = _confine_file(
        _resolve_file(metric_file, env_var="BOT_OTEL_METRIC_FILE", default=_DEFAULT_METRIC_FILE)
    )
    otlp_url = otlp_endpoint()
    resource_attributes: dict[str, str] = {
        "service.name": service_name or os.environ.get("OTEL_SERVICE_NAME", "").strip() or _SERVICE_NAME,
        _EXPORT_ATTR: "true",
        _LOG_FILE_ATTR: str(resolved_path),
        _SPAN_FILE_ATTR: str(resolved_span_path),
    }
    if service_instance_id is not None:
        resource_attributes["service.instance.id"] = service_instance_id
    if otlp_url is not None:
        resource_attributes[_OTLP_EXPORT_ATTR] = "true"
    resource = Resource.create(resource_attributes)
    provider = LoggerProvider(resource=resource)
    # Batch export (CORE-OBS-001): keep emit off the hot path; callers that
    # need immediate visibility must force_flush() (tests / live proofs).
    provider.add_log_record_processor(
        BatchLogRecordProcessor(JsonlFileLogRecordExporter(resolved_path)),
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(JsonlFileSpanExporter(resolved_span_path)))
    metric_exporters: list[MetricExporter] = [JsonlFileMetricExporter(resolved_metric_path)]
    if otlp_url is not None:
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_url)))
        metric_exporters.append(OTLPMetricExporter(endpoint=otlp_url))

    pipeline = _RUNTIME.metric_pipeline
    if pipeline is None:
        router = _RoutingMetricExporter()
        meter_provider = MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(router)])
        metrics.set_meter_provider(meter_provider)
        if metrics.get_meter_provider() is not meter_provider:
            _LOG.warning("OpenTelemetry MeterProvider was installed by other code first; import-time meters use it")
        pipeline = _MetricPipeline(provider=meter_provider, router=router)
        _RUNTIME.metric_pipeline = pipeline
    pipeline.router.bind(_MetricRoute(resource=resource, exporters=tuple(metric_exporters)))

    _RUNTIME.provider = provider
    _RUNTIME.tracer_provider = tracer_provider
    _RUNTIME.metric_file = resolved_metric_path
    _RUNTIME.configured = True
    # Best-effort global install; emission uses the retained providers so tests can rebind.
    _logs.set_logger_provider(provider)
    trace.set_tracer_provider(tracer_provider)
    _LOG.info(
        "OpenTelemetry export enabled logs=%s spans=%s metrics=%s otlp=%s",
        resolved_path,
        resolved_span_path,
        resolved_metric_path,
        otlp_url is not None,
    )
    return True


def configure(
    *,
    enabled: bool | None = None,
    log_file: str | pathlib.Path | None = None,
    span_file: str | pathlib.Path | None = None,
    metric_file: str | pathlib.Path | None = None,
    service_name: str | None = None,
    service_instance_id: str | None = None,
) -> bool:
    """Install the OTEL Logger/Tracer providers and bind metric export for this configuration.

    Default is enabled (opt-out). Disabled when ``enabled=False`` or when
    ``BOT_OTEL_DISABLED`` is one of ``1`` / ``true`` / ``yes`` / ``on``
    (case-insensitive).

    **Development security note:** generator request payloads (system/user
    prompts, tools) are captured only when ``BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD``
    or ``BOT_OTEL_CAPTURE=full`` is set; without it the log body carries
    low-cardinality counts, never prompt or tool content. ``BOT_OTEL_CAPTURE=full``
    also logs generator responses, every observed HSM event payload and transition,
    and dispatch outcomes (credentials redacted; see ``mosfet.telemetry.capture``). Enable OTEL is opt-out via ``BOT_OTEL_DISABLED`` /
    ``enabled=False``. Files are created with mode ``0o600``. Paths from
    ``log_file`` / ``BOT_OTEL_LOG_FILE`` (and the span and metric paths) must
    resolve under the process working directory (symlink escapes are rejected).
    See ``EXCEPTIONS.md`` for the PY-LOG-002 exception.

    Process-global ``_RUNTIME`` is the enablement SoT (see module docstring);
    ``set_logger_provider`` is install-only. ``configure()`` is the synchronized
    writer under ``_RUNTIME_LOCK``.

    Log path comes from ``log_file``, else ``BOT_OTEL_LOG_FILE``, else
    ``otel-logs.jsonl``. Span path comes from ``span_file``, else
    ``BOT_OTEL_SPAN_FILE``, else ``otel-spans.jsonl``. Metric path comes from
    ``metric_file``, else ``BOT_OTEL_METRIC_FILE``, else ``otel-metrics.jsonl``.
    When ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``, ``BOT_OTEL_EXPORTER_OTLP_ENDPOINT``,
    or ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, spans and metrics are also
    exported over OTLP gRPC to that collector, in addition to the JSONL files.
    The endpoint is the gRPC origin (``http://127.0.0.1:4317``); ``/v1/traces``
    is not appended and a leftover HTTP suffix is stripped.
    Span and metric attributes stay low-cardinality and payload-free (see
    ``mosfet.telemetry.span``), so no export carries prompts or media.

    Metrics flow through the process ``MeterProvider``, installed once by the
    first enabled call and never shut down by ``reset()``; each call binds this
    configuration's metric exporters to it (see the module docstring).

    Every signal's resource carries ``service.name`` (``service_name``, else
    ``OTEL_SERVICE_NAME``, else ``stateforward.mosfet``) and, when given,
    ``service.instance.id`` (``service_instance_id``). Both are resource
    attributes, never metric dimensions. A blank value raises ``ValueError``.

    Idempotent: subsequent calls return the prior result without reinstalling.
    """

    with _RUNTIME_LOCK:
        return _configure_locked(
            enabled=enabled,
            log_file=log_file,
            span_file=span_file,
            metric_file=metric_file,
            service_name=service_name,
            service_instance_id=service_instance_id,
        )


def ensure_configured() -> bool:
    """Configure once with defaults when nothing has been configured yet.

    Takes ``_RUNTIME_LOCK`` and installs (or marks disabled) if needed so
    concurrent first-use callers cannot race a double install. Returns whether
    JSONL export is enabled afterward.
    """

    with _RUNTIME_LOCK:
        if not _RUNTIME.configured:
            return _configure_locked()
        return is_enabled()


def reset() -> None:
    """Clear process-global OTEL configure state.

    For tests / process re-init only; not for production hot-reload.
    Best-effort shuts down any prior logger and tracer providers retained by
    this module. The process ``MeterProvider`` is never shut down: it is
    flushed through the bound metric exporters, which are then unbound and
    shut down, so the next ``configure()`` can bind its own.
    """

    with _RUNTIME_LOCK:
        prior = _RUNTIME.provider
        prior_tracer = _RUNTIME.tracer_provider
        pipeline = _RUNTIME.metric_pipeline
        _RUNTIME.configured = False
        _RUNTIME.provider = None
        _RUNTIME.tracer_provider = None
        _RUNTIME.metric_file = None
        for retained in (prior, prior_tracer):
            if retained is not None:
                with contextlib.suppress(Exception):
                    retained.shutdown()
        if pipeline is not None:
            try:
                _ = pipeline.provider.force_flush()
            except Exception as error:
                _LOG.warning("otel metric flush on reset failed error=%s", type(error).__name__)
            pipeline.router.release()


def is_enabled() -> bool:
    """Return whether an exporting LoggerProvider with the JSONL marker is active."""

    return _active_provider() is not None


def log_file() -> pathlib.Path | None:
    """Return the JSONL path from the active exporting provider resource, if any."""

    provider = _active_provider()
    if provider is None:
        return None
    raw = provider.resource.attributes.get(_LOG_FILE_ATTR)
    if isinstance(raw, str) and raw:
        return pathlib.Path(raw)
    return None


def span_file() -> pathlib.Path | None:
    """Return the span JSONL path from the active exporting tracer resource, if any."""

    provider = _active_tracer_provider()
    if provider is None:
        return None
    raw = provider.resource.attributes.get(_SPAN_FILE_ATTR)
    if isinstance(raw, str) and raw:
        return pathlib.Path(raw)
    return None


def metric_file() -> pathlib.Path | None:
    """Return the metric JSONL path bound by the active configuration, if any."""

    return _RUNTIME.metric_file


def logger_provider() -> LoggerProvider | None:
    """Return the active exporting LoggerProvider, if JSONL export is enabled."""

    return _active_provider()


def tracer_provider() -> TracerProvider | None:
    """Return the active exporting TracerProvider, if JSONL export is enabled."""

    return _active_tracer_provider()


def meter_provider() -> MeterProvider | None:
    """Return the process MeterProvider once the first enabled ``configure()`` installed it.

    The same object across every ``configure()`` / ``reset()`` cycle.
    """

    pipeline = _RUNTIME.metric_pipeline
    if pipeline is None:
        return None
    return pipeline.provider


__all__ = [
    "configure",
    "ensure_configured",
    "is_enabled",
    "log_file",
    "logger_provider",
    "meter_provider",
    "metric_file",
    "otlp_endpoint",
    "reset",
    "span_file",
    "tracer_provider",
]

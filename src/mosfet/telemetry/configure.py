"""Application-boundary OpenTelemetry configuration for local JSONL export.

Configures both a ``LoggerProvider`` (log records) and a ``TracerProvider``
(spans), each with a local JSONL file exporter, so a run leaves greppable
``otel-logs.jsonl`` and ``otel-spans.jsonl`` artifacts.

Call ``configure()`` (or ``ensure_configured()``) from an application boundary;
library emission is a no-op until then. When configure runs, default is enabled
(opt-out via ``enabled=False`` / ``BOT_OTEL_DISABLED``).

Rule exceptions (PY-LOG-002, PY-OBJ-002) and the user-approved
``opentelemetry-sdk`` / ``opentelemetry-api`` floors: see ``EXCEPTIONS.md``.

Process-global ``_RUNTIME`` matches OTEL's process-global providers
(``set_logger_provider`` / ``set_tracer_provider``). Single writer:
``configure()`` / ``ensure_configured()`` / ``reset()`` under ``_RUNTIME_LOCK``.
Enablement SoT is ``_RUNTIME.provider`` / ``_RUNTIME.tracer_provider`` only when
that provider exports JSONL (``bot.otel.jsonl_export`` resource marker). Readers
never fall back to ``_logs.get_logger_provider()`` / ``trace.get_tracer_provider()``;
the global setters are install-only.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import pathlib
import threading

from opentelemetry import _logs, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from mosfet.telemetry.export import JsonlFileLogRecordExporter, JsonlFileSpanExporter

_LOG = logging.getLogger(__name__)
_DISABLED_VALUES = frozenset({"1", "true", "yes", "on"})
_DEFAULT_LOG_FILE = "otel-logs.jsonl"
_DEFAULT_SPAN_FILE = "otel-spans.jsonl"
_SERVICE_NAME = "stateforward.mosfet"
_EXPORT_ATTR = "bot.otel.jsonl_export"
_OTLP_EXPORT_ATTR = "bot.otel.otlp_export"
_LOG_FILE_ATTR = "bot.otel.log_file"
_SPAN_FILE_ATTR = "bot.otel.span_file"
_TRACES_PATH = "/v1/traces"


@dataclasses.dataclass
class _Runtime:
    """OTEL export runtime boundary (process-global by the OTEL global setters)."""

    configured: bool = False
    provider: LoggerProvider | None = None
    tracer_provider: TracerProvider | None = None


# Process-global SoT for this module's exporting LoggerProvider (and idempotency).
# Mutations go through ``configure()`` / ``ensure_configured()`` / ``reset()``
# under ``_RUNTIME_LOCK``. Enablement reads ``_RUNTIME.provider`` only.
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
) -> bool:
    """Install or mark disabled. Caller must hold ``_RUNTIME_LOCK``."""

    if _RUNTIME.configured:
        return is_enabled()

    resolved_enabled = _resolve_enabled(enabled)
    if not resolved_enabled:
        _RUNTIME.configured = True
        _RUNTIME.provider = None
        _RUNTIME.tracer_provider = None
        _LOG.info("OpenTelemetry export disabled")
        return False

    # Validate confinement before committing idempotent state so bad paths can be retried.
    resolved_path = _confine_file(_resolve_file(log_file, env_var="BOT_OTEL_LOG_FILE", default=_DEFAULT_LOG_FILE))
    resolved_span_path = _confine_file(
        _resolve_file(span_file, env_var="BOT_OTEL_SPAN_FILE", default=_DEFAULT_SPAN_FILE)
    )
    otlp_url = otlp_endpoint()
    resource_attributes: dict[str, str] = {
        "service.name": _SERVICE_NAME,
        _EXPORT_ATTR: "true",
        _LOG_FILE_ATTR: str(resolved_path),
        _SPAN_FILE_ATTR: str(resolved_span_path),
    }
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
    if otlp_url is not None:
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_url)))
    _RUNTIME.provider = provider
    _RUNTIME.tracer_provider = tracer_provider
    _RUNTIME.configured = True
    # Best-effort global install; emission uses the retained providers so tests can rebind.
    _logs.set_logger_provider(provider)
    trace.set_tracer_provider(tracer_provider)
    if otlp_url is None:
        _LOG.info(
            "OpenTelemetry export enabled logs=%s spans=%s",
            resolved_path,
            resolved_span_path,
        )
    else:
        _LOG.info(
            "OpenTelemetry export enabled logs=%s spans=%s otlp=true",
            resolved_path,
            resolved_span_path,
        )
    return True


def configure(
    *,
    enabled: bool | None = None,
    log_file: str | pathlib.Path | None = None,
    span_file: str | pathlib.Path | None = None,
) -> bool:
    """Install the global OTEL Logger/Tracer providers with local JSONL exporters.

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
    ``log_file`` / ``BOT_OTEL_LOG_FILE`` must resolve under the process working
    directory (symlink escapes are rejected). See ``EXCEPTIONS.md`` for the
    PY-LOG-002 exception.

    Process-global ``_RUNTIME`` is the enablement SoT (see module docstring);
    ``set_logger_provider`` is install-only. ``configure()`` is the synchronized
    writer under ``_RUNTIME_LOCK``.

    Log path comes from ``log_file``, else ``BOT_OTEL_LOG_FILE``, else
    ``otel-logs.jsonl``. Span path comes from ``span_file``, else
    ``BOT_OTEL_SPAN_FILE``, else ``otel-spans.jsonl``. When
    ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``, ``BOT_OTEL_EXPORTER_OTLP_ENDPOINT``,
    or ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, a second batch processor exports
    OTLP gRPC to that collector. The endpoint is the gRPC origin
    (``http://127.0.0.1:4317``); ``/v1/traces`` is not appended and a leftover
    HTTP suffix is stripped. Span attributes stay low-cardinality and
    payload-free (see ``mosfet.telemetry.span``), so neither export carries prompts
    or media.

    Idempotent: subsequent calls return the prior result without reinstalling.
    """

    with _RUNTIME_LOCK:
        return _configure_locked(enabled=enabled, log_file=log_file, span_file=span_file)


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
    Best-effort shuts down any prior providers retained by this module.
    """

    with _RUNTIME_LOCK:
        prior = _RUNTIME.provider
        prior_tracer = _RUNTIME.tracer_provider
        _RUNTIME.configured = False
        _RUNTIME.provider = None
        _RUNTIME.tracer_provider = None
        for retained in (prior, prior_tracer):
            if retained is not None:
                with contextlib.suppress(Exception):
                    retained.shutdown()
    from mosfet.telemetry import control

    control.reset()


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


def logger_provider() -> LoggerProvider | None:
    """Return the active exporting LoggerProvider, if JSONL export is enabled."""

    return _active_provider()


def tracer_provider() -> TracerProvider | None:
    """Return the active exporting TracerProvider, if JSONL export is enabled."""

    return _active_tracer_provider()


__all__ = [
    "configure",
    "ensure_configured",
    "is_enabled",
    "log_file",
    "logger_provider",
    "otlp_endpoint",
    "reset",
    "span_file",
    "tracer_provider",
]

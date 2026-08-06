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
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from bot.telemetry.export import JsonlFileLogRecordExporter, JsonlFileSpanExporter

_LOG = logging.getLogger(__name__)
_DISABLED_VALUES = frozenset({"1", "true", "yes", "on"})
_DEFAULT_LOG_FILE = "otel-logs.jsonl"
_DEFAULT_SPAN_FILE = "otel-spans.jsonl"
_SERVICE_NAME = "stateforward.bot"
_EXPORT_ATTR = "bot.otel.jsonl_export"
_LOG_FILE_ATTR = "bot.otel.log_file"
_SPAN_FILE_ATTR = "bot.otel.span_file"


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


def _confine_file(path: pathlib.Path) -> pathlib.Path:
    """Resolve ``path`` and require it to stay under the process working directory."""

    allowed = pathlib.Path.cwd().resolve()
    try:
        resolved = path.expanduser().resolve(strict=False)
    except OSError as error:
        message = (
            "export file could not be resolved under the process working "
            + f"directory ({allowed}): {path}"
        )
        raise ValueError(message) from error
    try:
        _ = resolved.relative_to(allowed)
    except ValueError as error:
        message = (
            "export file must resolve under the process working "
            + f"directory ({allowed}); got {resolved}"
        )
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
    resource = Resource.create(
        {
            "service.name": _SERVICE_NAME,
            _EXPORT_ATTR: "true",
            _LOG_FILE_ATTR: str(resolved_path),
            _SPAN_FILE_ATTR: str(resolved_span_path),
        }
    )
    provider = LoggerProvider(resource=resource)
    # Batch export (CORE-OBS-001): keep emit off the hot path; callers that
    # need immediate visibility must force_flush() (tests / live proofs).
    provider.add_log_record_processor(
        BatchLogRecordProcessor(JsonlFileLogRecordExporter(resolved_path)),
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(JsonlFileSpanExporter(resolved_span_path)))
    _RUNTIME.provider = provider
    _RUNTIME.tracer_provider = tracer_provider
    _RUNTIME.configured = True
    # Best-effort global install; emission uses the retained providers so tests can rebind.
    _logs.set_logger_provider(provider)
    trace.set_tracer_provider(tracer_provider)
    _LOG.info(
        "OpenTelemetry export enabled logs=%s spans=%s",
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

    **Development security note:** when enabled, this writes raw text-generator
    request payloads (system/user prompts, tools) to a local JSONL file for
    debugging. Files are created with mode ``0o600``. Paths from ``log_file`` /
    ``BOT_OTEL_LOG_FILE`` must resolve under the process working directory
    (symlink escapes are rejected). Opt out with ``BOT_OTEL_DISABLED`` or
    ``enabled=False`` when prompts must not hit disk. See ``EXCEPTIONS.md`` for
    the PY-LOG-002 exception.

    Process-global ``_RUNTIME`` is the enablement SoT (see module docstring);
    ``set_logger_provider`` is install-only. ``configure()`` is the synchronized
    writer under ``_RUNTIME_LOCK``.

    Log path comes from ``log_file``, else ``BOT_OTEL_LOG_FILE``, else
    ``otel-logs.jsonl``. Span path comes from ``span_file``, else
    ``BOT_OTEL_SPAN_FILE``, else ``otel-spans.jsonl``. Span attributes stay
    low-cardinality and payload-free (see ``bot.telemetry.span``), so the span
    file carries no prompts or media.

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
    "reset",
    "span_file",
    "tracer_provider",
]

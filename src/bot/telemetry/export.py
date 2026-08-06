"""Local JSONL OpenTelemetry exporters (log records and spans) for development."""

from __future__ import annotations

import collections.abc
import json
import os
import pathlib
import stat
import typing
from datetime import UTC, datetime

from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

_FILE_MODE = 0o600
_DIR_MODE = 0o755


def _severity(record: ReadableLogRecord) -> str | int | None:
    log_record = record.log_record
    if log_record.severity_text:
        return log_record.severity_text
    severity_number = log_record.severity_number
    if severity_number is not None:
        return int(severity_number.value)
    return None


def _iso_nanos(nanos: int | None) -> str | None:
    if nanos is None:
        return None
    seconds, frac = divmod(int(nanos), 1_000_000_000)
    instant = datetime.fromtimestamp(seconds, tz=UTC)
    return instant.strftime("%Y-%m-%dT%H:%M:%S.") + f"{frac:09d}Z"


def _timestamp(record: ReadableLogRecord) -> str | None:
    log_record = record.log_record
    return _iso_nanos(log_record.timestamp if log_record.timestamp is not None else log_record.observed_timestamp)


def _resolve_existing_prefix(path: pathlib.Path) -> pathlib.Path:
    """Resolve symlinks in the longest existing prefix; keep missing suffix lexical.

    Avoids following a leaf symlink while still normalizing macOS ``/var`` vs
    ``/private/var`` (and similar) prefixes for cwd confinement checks.
    """

    absolute = pathlib.Path(os.path.normpath(path.absolute()))
    missing: list[str] = []
    current = absolute
    while not current.exists() and current != current.parent:
        missing.append(current.name)
        current = current.parent
    resolved = current.resolve()
    for name in reversed(missing):
        resolved /= name
    return resolved


def _relative_parts(path: pathlib.Path) -> tuple[str, ...]:
    """Return cwd-relative path components; reject ``..`` and absolute escapes.

    Normalizes the path against cwd without following the leaf symlink, so a
    leaf or parent symlink present at construction is still opened later with
    ``O_NOFOLLOW`` rather than being followed into an escape during setup.
    """

    cwd = pathlib.Path.cwd().resolve()
    expanded = path.expanduser()
    if any(part == ".." for part in expanded.parts):
        message = f"export path must not contain '..' components; got {path}"
        raise ValueError(message)
    absolute = expanded if expanded.is_absolute() else pathlib.Path.cwd() / expanded
    normalized = pathlib.Path(os.path.normpath(absolute))
    # Resolve only existing parents so the leaf name is not symlink-followed.
    candidate = _resolve_existing_prefix(normalized.parent) / normalized.name
    try:
        relative = candidate.relative_to(cwd)
    except ValueError as error:
        message = (
            "export path must resolve under the process working directory "
            + f"({cwd}); got {candidate}"
        )
        raise ValueError(message) from error
    parts = relative.parts
    if not parts or parts[-1] in {"", ".", ".."} or any(part == ".." for part in parts):
        message = f"export path must be a file under the process working directory; got {path}"
        raise ValueError(message)
    return parts


def _open_child_dir(dir_fd: int, name: str, *, create: bool) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except FileNotFoundError:
        if not create:
            raise
        os.mkdir(name, _DIR_MODE, dir_fd=dir_fd)
        return os.open(name, flags, dir_fd=dir_fd)


def _walk_from_root(root_fd: int, parts: tuple[str, ...], *, create: bool) -> int:
    """Open each parent component under ``root_fd`` with ``O_DIRECTORY|O_NOFOLLOW``.

    Returns the parent directory fd for the leaf. When the leaf is directly under
    ``root_fd``, the return value is ``root_fd`` and must not be closed by the
    caller. Otherwise the caller must close the returned fd.
    """

    parent_fd = root_fd
    try:
        for name in parts[:-1]:
            next_fd = _open_child_dir(parent_fd, name, create=create)
            if parent_fd != root_fd:
                os.close(parent_fd)
            parent_fd = next_fd
        return parent_fd
    except BaseException:
        if parent_fd != root_fd:
            os.close(parent_fd)
        raise


def _require_safe_regular_file(fd: int) -> None:
    """Reject non-files, hard-link redirects, and files not owned by this euid."""

    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        message = "export path must be a regular file"
        raise OSError(message)
    if info.st_nlink != 1:
        message = "export path must not be hard-linked (nlink != 1)"
        raise OSError(message)
    if info.st_uid != os.geteuid():
        message = "export path must be owned by the current effective user"
        raise OSError(message)


class _JsonlAppendFile:
    """Append JSON lines to a local file without following symlinks.

    Path components are stored relative to the process cwd at construction.
    Construction retains a directory fd for that cwd; parent walks and leaf
    opens are always relative to that retained root, so a later ``chdir`` cannot
    divert writes. Parent directories and the leaf use ``O_NOFOLLOW``; existing
    leaves must be single-link regular files owned by this euid.
    """

    _parts: tuple[str, ...]
    _root_fd: int | None

    def __init__(self, path: str | pathlib.Path) -> None:
        self._parts = _relative_parts(pathlib.Path(path))
        self._root_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
        try:
            if len(self._parts) > 1:
                parent_fd = _walk_from_root(self._root_fd, self._parts, create=True)
                if parent_fd != self._root_fd:
                    os.close(parent_fd)
        except BaseException:
            os.close(self._root_fd)
            self._root_fd = None
            raise

    def _open_append(self) -> typing.TextIO:
        root_fd = self._root_fd
        if root_fd is None:
            message = "exporterl exporter already shut down"
            raise OSError(message)
        parent_fd = _walk_from_root(root_fd, self._parts, create=True)
        try:
            leaf = self._parts[-1]
            create_flags = os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW
            try:
                fd = os.open(leaf, create_flags, _FILE_MODE, dir_fd=parent_fd)
            except FileExistsError:
                open_flags = os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW
                fd = os.open(leaf, open_flags, dir_fd=parent_fd)
                try:
                    _require_safe_regular_file(fd)
                except BaseException:
                    os.close(fd)
                    raise
            try:
                os.fchmod(fd, _FILE_MODE)
            except OSError:
                os.close(fd)
                raise
            return os.fdopen(fd, "a", encoding="utf-8")
        finally:
            if parent_fd != root_fd:
                os.close(parent_fd)

    def write(self, payloads: collections.abc.Sequence[dict[str, object]]) -> None:
        """Append one JSON line per payload.

        An empty batch still opens the leaf: the open is the check that the
        path has not been swapped for a symlink or hard link since construction.
        """

        lines = [json.dumps(payload, default=str, separators=(",", ":")) for payload in payloads]
        with self._open_append() as handle:
            _ = handle.write("\n".join(lines))
            _ = handle.write("\n")

    def close(self) -> None:
        root_fd = self._root_fd
        if root_fd is None:
            return
        self._root_fd = None
        os.close(root_fd)


class JsonlFileLogRecordExporter(LogRecordExporter):
    """Write one compact JSON object per log record to a local file."""

    _file: _JsonlAppendFile

    def __init__(self, path: str | pathlib.Path) -> None:
        self._file = _JsonlAppendFile(path)

    @typing.override
    def export(self, batch: collections.abc.Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        payloads: list[dict[str, object]] = []
        for record in batch:
            log_record = record.log_record
            payloads.append(
                {
                    "timestamp": _timestamp(record),
                    "body": log_record.body,
                    "attributes": dict(log_record.attributes) if log_record.attributes else {},
                    "severity": _severity(record),
                }
            )
        self._file.write(payloads)
        return LogRecordExportResult.SUCCESS

    @typing.override
    def shutdown(self) -> None:
        self._file.close()


def _span_payload(span: ReadableSpan) -> dict[str, object]:
    context = span.get_span_context()
    parent = span.parent
    start = span.start_time
    end = span.end_time
    scope = span.instrumentation_scope
    return {
        "timestamp": _iso_nanos(end if end is not None else start),
        "name": span.name,
        "trace_id": f"{context.trace_id:032x}" if context is not None else None,
        "span_id": f"{context.span_id:016x}" if context is not None else None,
        "parent_span_id": f"{parent.span_id:016x}" if parent is not None else None,
        "scope": scope.name if scope is not None else None,
        "kind": span.kind.name,
        "start_time": _iso_nanos(start),
        "end_time": _iso_nanos(end),
        "duration_ns": (end - start) if start is not None and end is not None else None,
        "status": span.status.status_code.name,
        "status_message": span.status.description,
        "attributes": dict(span.attributes) if span.attributes else {},
        "events": [
            {
                "name": event.name,
                "timestamp": _iso_nanos(event.timestamp),
                "attributes": dict(event.attributes) if event.attributes else {},
            }
            for event in span.events
        ],
    }


class JsonlFileSpanExporter(SpanExporter):
    """Write one compact JSON object per finished span to a local file.

    Shares the symlink-safe, cwd-confined append path used for log records so a
    run leaves a greppable trace artifact next to ``otel-logs.jsonl``.
    """

    _file: _JsonlAppendFile

    def __init__(self, path: str | pathlib.Path) -> None:
        self._file = _JsonlAppendFile(path)

    @typing.override
    def export(self, spans: collections.abc.Sequence[ReadableSpan]) -> SpanExportResult:
        self._file.write([_span_payload(span) for span in spans])
        return SpanExportResult.SUCCESS

    @typing.override
    def shutdown(self) -> None:
        self._file.close()


__all__ = ["JsonlFileLogRecordExporter", "JsonlFileSpanExporter"]

"""Read the OTEL metrics a test recorded through the one process ``MeterProvider`` telemetry installs.

The OpenTelemetry API accepts exactly one global ``MeterProvider`` per process, and instruments created
at import time bind to it when it is set. ``mosfet.telemetry.configure()`` installs that provider once and
routes it to each configuration's exporters, so a test that asserts metrics configures export into its
own directory and reads the cumulative points back from the JSONL metric file. Points are cumulative for
the whole process: assert on the difference between two reads.
"""

import collections.abc
import contextlib
import dataclasses
import json
import pathlib
import typing

import pytest
from opentelemetry import _logs, trace

from mosfet import telemetry

_METRIC_FILE = "metrics.jsonl"
_ENVIRONMENT = (
    "BOT_OTEL_DISABLED",
    "BOT_OTEL_LOG_FILE",
    "BOT_OTEL_SPAN_FILE",
    "BOT_OTEL_METRIC_FILE",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "BOT_OTEL_EXPORTER_OTLP_ENDPOINT",
)


@dataclasses.dataclass(frozen=True)
class Point:
    """One exported data point: a sum's ``value``, or a histogram's ``count`` and ``sum``."""

    attributes: dict[str, object]
    value: float = 0.0
    count: int = 0
    sum: float = 0.0


def exported(path: pathlib.Path, name: str) -> tuple[Point, ...]:
    """The newest points for the instrument ``name`` in the JSONL metric file at ``path``, if any."""

    latest: list[dict[str, object]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            payload = typing.cast(dict[str, typing.Any], json.loads(line))
            if payload["name"] == name:
                latest = typing.cast(list[dict[str, object]], payload["data"]["data_points"])
    return tuple(
        Point(
            attributes=typing.cast(dict[str, object], point.get("attributes") or {}),
            value=float(typing.cast(float, point.get("value", 0.0))),
            count=int(typing.cast(int, point.get("count", 0))),
            sum=float(typing.cast(float, point.get("sum", 0.0))),
        )
        for point in latest
    )


@dataclasses.dataclass(frozen=True)
class Reader:
    """Reads the latest cumulative points of an instrument from the configured metric file."""

    path: pathlib.Path

    def points(self, name: str) -> tuple[Point, ...]:
        """Flush the process provider, then return the newest exported points for ``name``."""

        provider = telemetry.meter_provider()
        assert provider is not None, "recording() configured telemetry"
        _ = provider.force_flush()
        return exported(self.path, name)


def _install_nothing(provider: object) -> None:
    del provider


@contextlib.contextmanager
def recording(directory: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> collections.abc.Generator[Reader]:
    """Export metrics into ``directory`` for the duration of the block.

    Only metrics are exported: the logger and tracer providers stay module-local, so later tests'
    dispatches are not traced, and nothing is written outside ``directory``.
    """

    monkeypatch.chdir(directory)
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_logs, "set_logger_provider", _install_nothing)
    monkeypatch.setattr(trace, "set_tracer_provider", _install_nothing)
    telemetry.reset()
    assert telemetry.configure(metric_file=_METRIC_FILE) is True
    try:
        yield Reader(directory / _METRIC_FILE)
    finally:
        telemetry.reset()

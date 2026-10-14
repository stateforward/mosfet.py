"""One process-wide in-memory OTEL metric reader for tests that assert recorded metrics.

The OpenTelemetry API accepts exactly one global ``MeterProvider`` per process, and instruments
created at import time bind to it when it is set. Every test that reads metrics therefore shares
this reader instead of installing its own provider.
"""

import functools

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import HistogramDataPoint, InMemoryMetricReader, NumberDataPoint


@functools.cache
def reader() -> InMemoryMetricReader:
    """Install the shared provider once and return its reader."""

    shared = InMemoryMetricReader()
    metrics.set_meter_provider(MeterProvider(metric_readers=[shared]))
    return shared


def points(name: str) -> tuple[NumberDataPoint | HistogramDataPoint, ...]:
    """Cumulative data points recorded so far for the instrument ``name``."""

    data = reader().get_metrics_data()
    if data is None:
        return ()
    return tuple(
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == name
        for point in metric.data.data_points
        if isinstance(point, NumberDataPoint | HistogramDataPoint)
    )

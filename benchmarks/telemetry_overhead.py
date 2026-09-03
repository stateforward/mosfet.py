"""Benchmark the telemetry observation hot path (observer no-op vs configured)."""

from __future__ import annotations

import argparse
import asyncio
import collections.abc
import contextlib
import dataclasses
import datetime
import json
import pathlib
import statistics
import time
import typing

import hsm

import bot.telemetry.hsm as telemetry_hsm
from bot.telemetry.configure import configure, logger_provider, reset, tracer_provider

_BENCH_LOG_FILE = "otel-bench-logs.jsonl"
_BENCH_SPAN_FILE = "otel-bench-spans.jsonl"


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration for one deterministic telemetry benchmark run."""

    iterations: int = 2_000
    repeats: int = 5
    warmup: int = 1

    def validate(self) -> None:
        """Raise ValueError when the benchmark configuration is invalid."""

        if self.iterations < 1:
            raise ValueError("iterations must be positive.")
        if self.repeats < 1:
            raise ValueError("repeats must be positive.")
        if self.warmup < 0:
            raise ValueError("warmup must not be negative.")


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Measured timing result for one telemetry benchmark case."""

    name: str
    iterations: int
    best_ns_per_op: float
    mean_ns_per_op: float
    best_ops_per_second: float
    samples_ns: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _Args:
    iterations: int
    repeats: int
    warmup: int
    cases: list[str] | None
    as_json: bool


Case = collections.abc.Callable[[BenchmarkConfig], collections.abc.Awaitable[int]]


class _BenchInstance(hsm.Instance):
    pass


_CTX = hsm.Context()
_INSTANCE = _BenchInstance()
_OBSERVATION = hsm.Event[dict[str, object]](
    name="hsm/observation",
    source="/Bench/active/observing/to_processing",
    data={
        "event": hsm.Event[None](name="bench.event"),
        "occurrence": "event",
        "time": datetime.datetime.now(datetime.UTC),
    },
)


async def _bench_observer_noop(config: BenchmarkConfig) -> int:
    for _ in range(config.iterations):
        telemetry_hsm.observer(_CTX, _INSTANCE, _OBSERVATION)
    return config.iterations


async def _bench_observer_configured(config: BenchmarkConfig) -> int:
    for _ in range(config.iterations):
        telemetry_hsm.observer(_CTX, _INSTANCE, _OBSERVATION)
    return config.iterations


_CASES: typing.Final[dict[str, Case]] = {
    "observer_noop": _bench_observer_noop,
    "observer_configured": _bench_observer_configured,
}

_CONFIGURED_CASES = frozenset({"observer_configured"})


def _configure_bench_export() -> None:
    configure(
        enabled=True,
        log_file=pathlib.Path(_BENCH_LOG_FILE),
        span_file=pathlib.Path(_BENCH_SPAN_FILE),
    )


def _teardown_bench_export() -> None:
    logger = logger_provider()
    if logger is not None:
        with contextlib.suppress(Exception):
            logger.force_flush()
    tracer = tracer_provider()
    if tracer is not None:
        with contextlib.suppress(Exception):
            tracer.force_flush()
    reset()
    for name in (_BENCH_LOG_FILE, _BENCH_SPAN_FILE):
        with contextlib.suppress(OSError):
            pathlib.Path(name).unlink()


async def run_benchmarks(
    config: BenchmarkConfig,
    *,
    cases: collections.abc.Iterable[str] | None = None,
) -> tuple[BenchmarkResult, ...]:
    """Run selected telemetry benchmarks and return timing results."""

    config.validate()
    selected = tuple(cases) if cases is not None else tuple(_CASES)
    unknown = sorted(set(selected).difference(_CASES))
    if unknown:
        raise ValueError(f"unknown benchmark case(s): {', '.join(unknown)}")

    results: list[BenchmarkResult] = []
    for name in selected:
        case = _CASES[name]
        if name in _CONFIGURED_CASES:
            _configure_bench_export()
        try:
            for _ in range(config.warmup):
                _ = await case(config)

            samples: list[int] = []
            operations = 0
            for _ in range(config.repeats):
                started = time.perf_counter_ns()
                operations = await case(config)
                samples.append(time.perf_counter_ns() - started)
        finally:
            if name in _CONFIGURED_CASES:
                _teardown_bench_export()

        best = min(samples)
        mean = statistics.fmean(samples)
        best_ns_per_op = best / operations
        results.append(
            BenchmarkResult(
                name=name,
                iterations=operations,
                best_ns_per_op=best_ns_per_op,
                mean_ns_per_op=mean / operations,
                best_ops_per_second=1_000_000_000 / best_ns_per_op,
                samples_ns=tuple(samples),
            )
        )
    return tuple(results)


def format_table(results: collections.abc.Sequence[BenchmarkResult]) -> str:
    """Format benchmark results as a compact text table."""

    rows = [
        (
            "case",
            "iterations",
            "best us/op",
            "mean us/op",
            "best ops/s",
        )
    ]
    rows.extend(
        (
            result.name,
            str(result.iterations),
            f"{result.best_ns_per_op / 1_000:.2f}",
            f"{result.mean_ns_per_op / 1_000:.2f}",
            f"{result.best_ops_per_second:,.0f}",
        )
        for result in results
    )
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join("  ".join(cell.rjust(widths[index]) for index, cell in enumerate(row)) for row in rows)


def format_json(results: collections.abc.Sequence[BenchmarkResult]) -> str:
    """Format benchmark results as JSON for archival or comparison."""

    return json.dumps([dataclasses.asdict(result) for result in results], indent=2, sort_keys=True)


def _parse_args() -> _Args:
    defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--iterations", type=int, default=defaults.iterations)
    _ = parser.add_argument("--repeats", type=int, default=defaults.repeats)
    _ = parser.add_argument("--warmup", type=int, default=defaults.warmup)
    _ = parser.add_argument("--case", choices=tuple(_CASES), action="append", dest="cases")
    _ = parser.add_argument("--json", action="store_true", dest="as_json")
    namespace = parser.parse_args()
    return _Args(
        iterations=typing.cast(int, namespace.iterations),
        repeats=typing.cast(int, namespace.repeats),
        warmup=typing.cast(int, namespace.warmup),
        cases=typing.cast(list[str] | None, namespace.cases),
        as_json=typing.cast(bool, namespace.as_json),
    )


async def _main() -> None:
    args = _parse_args()
    config = BenchmarkConfig(
        iterations=args.iterations,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    results = await run_benchmarks(config, cases=args.cases)
    print(format_json(results) if args.as_json else format_table(results))


if __name__ == "__main__":
    asyncio.run(_main())

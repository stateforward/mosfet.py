"""Benchmark the Yamux frame and session HSM hot paths."""

from __future__ import annotations

import argparse
import asyncio
import collections.abc
import dataclasses
import json
import statistics
import time
import typing

import hsm

from mosfet.protocols.yamux import (
    INITIAL_STREAM_WINDOW,
    OpenStreamEvent,
    PingEvent,
    SendDataEvent,
    Client,
    Flag,
    FrameData,
    FrameType,
    OpenStreamData,
    PingData,
    ReadStream,
    SendData,
    Session,
    decode_frame,
)

_MAX_UINT32 = (1 << 32) - 1


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration for one deterministic Yamux benchmark run."""

    iterations: int = 5_000
    repeats: int = 5
    warmup: int = 1
    payload_size: int = 64

    def validate(self) -> None:
        """Raise ValueError when the benchmark configuration is invalid."""

        if self.iterations < 1:
            raise ValueError("iterations must be positive.")
        if self.repeats < 1:
            raise ValueError("repeats must be positive.")
        if self.warmup < 0:
            raise ValueError("warmup must not be negative.")
        if self.payload_size < 1:
            raise ValueError("payload_size must be positive.")
        if self.iterations * self.payload_size > _MAX_UINT32 - INITIAL_STREAM_WINDOW:
            raise ValueError("iterations * payload_size exceeds Yamux's 32-bit window range.")


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Measured timing result for one Yamux benchmark case."""

    name: str
    iterations: int
    payload_size: int
    best_ns_per_op: float
    mean_ns_per_op: float
    best_ops_per_second: float
    samples_ns: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _Args:
    iterations: int
    repeats: int
    warmup: int
    payload_size: int
    cases: list[str] | None
    as_json: bool


Case = collections.abc.Callable[[BenchmarkConfig], collections.abc.Awaitable[int]]


async def run_benchmarks(
    config: BenchmarkConfig,
    *,
    cases: collections.abc.Iterable[str] | None = None,
) -> tuple[BenchmarkResult, ...]:
    """Run selected Yamux benchmarks and return timing results."""

    config.validate()
    selected = tuple(cases) if cases is not None else tuple(_CASES)
    unknown = sorted(set(selected).difference(_CASES))
    if unknown:
        raise ValueError(f"unknown benchmark case(s): {', '.join(unknown)}")

    results: list[BenchmarkResult] = []
    for name in selected:
        case = _CASES[name]
        for _ in range(config.warmup):
            _ = await case(config)

        samples: list[int] = []
        operations = 0
        for _ in range(config.repeats):
            started = time.perf_counter_ns()
            operations = await case(config)
            samples.append(time.perf_counter_ns() - started)

        best = min(samples)
        mean = statistics.fmean(samples)
        best_ns_per_op = best / operations
        results.append(
            BenchmarkResult(
                name=name,
                iterations=operations,
                payload_size=config.payload_size,
                best_ns_per_op=best_ns_per_op,
                mean_ns_per_op=mean / operations,
                best_ops_per_second=1_000_000_000 / best_ns_per_op,
                samples_ns=tuple(samples),
            )
        )
    return tuple(results)


async def _bench_frame_encode_decode(config: BenchmarkConfig) -> int:
    payload = b"x" * config.payload_size
    for _ in range(config.iterations):
        encoded = FrameData.data(stream_id=1, payload=payload).to_bytes()
        frame, rest = decode_frame(encoded)
        if rest or frame.payload != payload or frame.frame_type is not FrameType.DATA:
            raise AssertionError("frame encode/decode round trip failed")
    return config.iterations


async def _bench_session_receive_read(config: BenchmarkConfig) -> int:
    session = await _started_client()
    await session.receive_frame(session.context(), FrameData.data(stream_id=2, flags=Flag.SYN))
    _ = await _read_outbound(session, expected_type=FrameType.WINDOW_UPDATE)
    stream = _require_stream(session, 2)
    payload = b"x" * config.payload_size

    for _ in range(config.iterations):
        await session.receive_frame(session.context(), FrameData.data(stream_id=2, payload=payload))
        data = await stream.readexactly(config.payload_size)
        if data != payload:
            raise AssertionError("ReadStream returned unexpected payload")
        _ = await _read_outbound(session, expected_type=FrameType.WINDOW_UPDATE)
    return config.iterations


async def _bench_session_send_data(config: BenchmarkConfig) -> int:
    session = await _started_client()
    await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))
    _ = await _read_outbound(session, expected_type=FrameType.WINDOW_UPDATE)
    await session.receive_frame(
        session.context(),
        FrameData.window_update(
            stream_id=1,
            delta=config.iterations * config.payload_size,
            flags=Flag.ACK,
        ),
    )
    payload = b"x" * config.payload_size

    for _ in range(config.iterations):
        await session.dispatch(
            session.context(),
            SendDataEvent.with_data(SendData(stream_id=1, payload=payload)),
        )
        frame = await _read_outbound(session, expected_type=FrameType.DATA)
        if frame.payload != payload:
            raise AssertionError("outbound DATA frame carried unexpected payload")
    return config.iterations


async def _bench_session_ping_round_trip(config: BenchmarkConfig) -> int:
    session = await _started_client()
    for index in range(config.iterations):
        opaque = index & _MAX_UINT32
        await session.dispatch(session.context(), PingEvent.with_data(PingData(opaque=opaque)))
        _ = await _read_outbound(session, expected_type=FrameType.PING)
        await session.receive_frame(session.context(), FrameData.ping(opaque=opaque, ack=True))
    return config.iterations


async def _started_client() -> Client:
    session = Client()
    _ = await hsm.started(None, session, session.model)
    return session


async def _read_outbound(session: Session, *, expected_type: FrameType) -> FrameData:
    frame = await session.outbound().read()
    if frame.frame_type is not expected_type:
        raise AssertionError(f"expected {expected_type.name} frame, got {frame.frame_type.name}")
    return frame


def _require_stream(session: Session, stream_id: int) -> ReadStream:
    stream = session.stream(stream_id)
    if stream is None:
        raise AssertionError(f"stream {stream_id} was not created")
    return stream


_CASES: typing.Final[dict[str, Case]] = {
    "frame_encode_decode": _bench_frame_encode_decode,
    "session_receive_read": _bench_session_receive_read,
    "session_send_data": _bench_session_send_data,
    "session_ping_round_trip": _bench_session_ping_round_trip,
}


def format_table(results: collections.abc.Sequence[BenchmarkResult]) -> str:
    """Format benchmark results as a compact text table."""

    rows = [
        (
            "case",
            "iterations",
            "payload",
            "best us/op",
            "mean us/op",
            "best ops/s",
        )
    ]
    rows.extend(
        (
            result.name,
            str(result.iterations),
            str(result.payload_size),
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
    _ = parser.add_argument("--payload-size", type=int, default=defaults.payload_size)
    _ = parser.add_argument("--case", choices=tuple(_CASES), action="append", dest="cases")
    _ = parser.add_argument("--json", action="store_true", dest="as_json")
    namespace = parser.parse_args()
    return _Args(
        iterations=typing.cast(int, namespace.iterations),
        repeats=typing.cast(int, namespace.repeats),
        warmup=typing.cast(int, namespace.warmup),
        payload_size=typing.cast(int, namespace.payload_size),
        cases=typing.cast(list[str] | None, namespace.cases),
        as_json=typing.cast(bool, namespace.as_json),
    )


async def _main() -> None:
    args = _parse_args()
    config = BenchmarkConfig(
        iterations=args.iterations,
        repeats=args.repeats,
        warmup=args.warmup,
        payload_size=args.payload_size,
    )
    results = await run_benchmarks(config, cases=args.cases)
    print(format_json(results) if args.as_json else format_table(results))


if __name__ == "__main__":
    asyncio.run(_main())

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
import pydantic

from mosfet.protocols import transform
from mosfet.protocols.yamux import (
    INITIAL_STREAM_WINDOW,
    Client,
    Flag,
    FrameData,
    FrameType,
    HandOverData,
    HandOverEvent,
    OpenStreamData,
    OpenStreamEvent,
    PingData,
    PingEvent,
    StreamAcceptedData,
    TransmitData,
    decode_frame,
    event_from_frame,
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


class _PayloadData(pydantic.BaseModel):
    """One benchmark message."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    payload: str


_PayloadEvent = hsm.Event[_PayloadData](name="benchmark.yamux.payload", schema=_PayloadData)


class _Sink(hsm.Instance):
    """Stands in for a connection or a stream owner: keeps the last frame, stream and payload it was told."""

    frames: list[FrameData]
    streams: list[hsm.Instance]
    payloads: int
    owned: list[hsm.Instance]

    def __init__(self) -> None:
        super().__init__()
        self.frames = []
        self.streams = []
        self.payloads = 0
        self.owned = []

    @staticmethod
    def _keep(ctx: hsm.Context, instance: "_Sink", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        if isinstance(data, TransmitData):
            instance.frames.append(decode_frame(data.frame)[0])
        elif isinstance(data, StreamAcceptedData):
            instance.streams.append(data.stream)
        elif isinstance(data, transform.OwnedData):
            instance.owned.append(data.stream)
        elif isinstance(data, _PayloadData):
            instance.payloads += 1

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "Sink",
        hsm.initial(hsm.target("keeping")),
        hsm.state("keeping", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_keep))),
    )

    def take_frame(self, expected_type: FrameType) -> FrameData:
        frame = self.frames.pop(0)
        if frame.frame_type is not expected_type:
            raise AssertionError(f"expected {expected_type.name} frame, got {frame.frame_type.name}")
        return frame


def _codec(config: BenchmarkConfig) -> transform.JsonCodec:
    return transform.JsonCodec(_PayloadEvent, max_message_bytes=config.payload_size + 128)


def _framed(codec: transform.JsonCodec, config: BenchmarkConfig) -> bytes:
    message = codec.encode(_PayloadEvent.with_data(_PayloadData(payload="x" * config.payload_size)))
    return len(message).to_bytes(4, "big") + message


async def _bench_session_receive_deliver(config: BenchmarkConfig) -> int:
    connection, client = await _started_client()
    owner = await _started(_Sink())
    codec = _codec(config)
    _ = await client.dispatch(client.context(), event_from_frame(FrameData.data(stream_id=2, flags=Flag.SYN)))
    _ = connection.take_frame(FrameType.WINDOW_UPDATE)
    (stream,) = connection.streams
    hand_over = HandOverEvent.with_data(HandOverData(owner=owner, codec=codec, channel="bench"))
    _ = await stream.dispatch(stream.context(), dataclasses.replace(hand_over, source=hsm.id(connection)))
    framed = _framed(codec, config)

    for index in range(config.iterations):
        _ = await client.dispatch(client.context(), event_from_frame(FrameData.data(stream_id=2, payload=framed)))
        while owner.payloads <= index or not connection.frames:
            await asyncio.sleep(0)
        if connection.take_frame(FrameType.WINDOW_UPDATE).length != len(framed):
            raise AssertionError("window credit did not match the delivered message")
    return config.iterations


async def _bench_session_send(config: BenchmarkConfig) -> int:
    connection, client = await _started_client()
    owner = await _started(_Sink())
    codec = _codec(config)
    _ = await client.dispatch(
        client.context(), OpenStreamEvent.with_data(OpenStreamData(owner=owner, codec=codec, channel="bench"))
    )
    _ = connection.take_frame(FrameType.WINDOW_UPDATE)
    framed = _framed(codec, config)
    grant = FrameData.window_update(stream_id=1, delta=config.iterations * len(framed), flags=Flag.ACK)
    _ = await client.dispatch(client.context(), event_from_frame(grant))
    (stream,) = owner.owned
    sent = _PayloadEvent.with_data(_PayloadData(payload="x" * config.payload_size))

    for _ in range(config.iterations):
        _ = await stream.dispatch(stream.context(), sent)
        if connection.take_frame(FrameType.DATA).payload != framed:
            raise AssertionError("outbound DATA frame carried an unexpected message")
    return config.iterations


async def _bench_session_ping_round_trip(config: BenchmarkConfig) -> int:
    connection, client = await _started_client()
    for index in range(config.iterations):
        opaque = index & _MAX_UINT32
        _ = await client.dispatch(client.context(), PingEvent.with_data(PingData(opaque=opaque)))
        _ = connection.take_frame(FrameType.PING)
        _ = await client.dispatch(client.context(), event_from_frame(FrameData.ping(opaque=opaque, ack=True)))
    return config.iterations


async def _started[T: hsm.Instance](instance: T) -> T:
    return await hsm.started(None, instance, typing.cast(hsm.Model, getattr(type(instance), "model")))


async def _started_client() -> tuple[_Sink, Client]:
    connection = await _started(_Sink())
    return connection, await _started(Client(connection=connection))


_CASES: typing.Final[dict[str, Case]] = {
    "frame_encode_decode": _bench_frame_encode_decode,
    "session_receive_deliver": _bench_session_receive_deliver,
    "session_send": _bench_session_send,
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

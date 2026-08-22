import asyncio
import dataclasses
import inspect

import hsm
import bot
import pytest

from bot.protocols.yamux.frame import INITIAL_STREAM_WINDOW
from bot.protocols.yamux.stream import Stream, StreamSnapshot, StreamState


def explicit_effects(transition: hsm.TransitionElement) -> tuple[str, ...]:
    return tuple(transition.effect)


def test_yamux_stream_is_hsm_model_with_lifecycle_states_and_python_reader() -> None:
    signature = inspect.signature(Stream)

    assert dataclasses.is_dataclass(Stream)
    assert signature.parameters["stream_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert issubclass(Stream, hsm.Instance)
    for name in (
        "lifecycle",
        "initiated_locally",
        "send_window",
        "receive_window",
        "sent_bytes",
        "received_bytes",
        "can_send",
        "can_receive",
        "is_terminal",
    ):
        assert not isinstance(inspect.getattr_static(Stream, name, None), property)
    assert Stream.model.qualified_name == "/Stream"
    assert "/Stream/active/local_opening" in Stream.model.members
    assert "/Stream/active/open" in Stream.model.members
    assert "/Stream/active/local_closed" in Stream.model.members
    assert "/Stream/active/remote_closed" in Stream.model.members
    assert "/Stream/closed" in Stream.model.members
    assert "/Stream/reset" in Stream.model.members
    assert not any(isinstance(member, hsm.ObservationElement) for member in Stream.model.members.values())


def test_yamux_stream_model_does_not_rely_on_observer_effects_for_internal_transitions() -> None:
    targetless = tuple(
        member
        for member in Stream.model.members.values()
        if isinstance(member, hsm.TransitionElement) and not member.target
    )

    assert targetless
    assert all(explicit_effects(transition) for transition in targetless)


def test_yamux_stream_take_snapshot_extends_canonical_hsm_snapshot() -> None:
    async def run() -> None:
        stream = Stream(stream_id=2, locally_initiated=False)
        _ = await bot.started(None, stream, stream.model)

        snapshot = hsm.take_snapshot(None, stream)

        assert isinstance(snapshot, StreamSnapshot)
        assert snapshot.State == "/Stream/active/open"
        assert snapshot.lifecycle is StreamState.OPEN

    asyncio.run(run())


def test_yamux_stream_reads_received_bytes_with_asyncio_stream_api() -> None:
    async def run() -> None:
        stream = Stream(stream_id=2, locally_initiated=False)
        _ = await bot.started(None, stream, stream.model)

        assert stream.take_snapshot().lifecycle is StreamState.OPEN
        stream.receive(b"hello\n")
        assert await stream.readline() == b"hello\n"

        stream.receive(b"tail", end_stream=True)
        assert stream.take_snapshot().lifecycle is StreamState.REMOTE_CLOSED
        assert await stream.read() == b"tail"
        assert stream.at_eof()

    asyncio.run(run())


def test_yamux_stream_reports_application_read_window_consumption() -> None:
    async def run() -> None:
        consumed: list[tuple[int, int]] = []
        stream = Stream(
            stream_id=2,
            locally_initiated=False,
            on_window_consumed=lambda stream_id, count: consumed.append((stream_id, count)),
        )
        _ = await bot.started(None, stream, stream.model)

        stream.receive(b"abcdef")
        assert await stream.read(2) == b"ab"
        assert await stream.readexactly(4) == b"cdef"

        assert consumed == [(2, 2), (2, 4)]

    asyncio.run(run())


def test_yamux_stream_lifecycle_and_flow_control_runtime() -> None:
    async def run() -> None:
        stream = Stream(stream_id=1, locally_initiated=True)
        _ = await bot.started(None, stream, stream.model)

        assert stream.take_snapshot().lifecycle is StreamState.LOCAL_OPENING
        assert stream.take_snapshot().awaiting_ack
        stream.send(b"hello")
        assert stream.take_snapshot().send_window == INITIAL_STREAM_WINDOW - 5
        assert stream.take_snapshot().sent_bytes == 5

        stream.acknowledge()
        assert stream.take_snapshot().lifecycle is StreamState.OPEN
        assert not stream.take_snapshot().awaiting_ack

        stream.update_send_window(10)
        assert stream.take_snapshot().send_window == INITIAL_STREAM_WINDOW + 5

        stream.receive(b"payload")
        assert stream.take_snapshot().receive_window == INITIAL_STREAM_WINDOW - 7
        assert stream.take_snapshot().received_bytes == 7

        stream.remote_fin()
        assert stream.take_snapshot().lifecycle is StreamState.REMOTE_CLOSED

        stream.send(b"", end_stream=True)
        assert stream.take_snapshot().lifecycle is StreamState.CLOSED

    asyncio.run(run())


def test_yamux_stream_accepts_ack_after_local_fin_before_ack() -> None:
    async def run() -> None:
        stream = Stream(stream_id=1, locally_initiated=True)
        _ = await bot.started(None, stream, stream.model)

        stream.send(b"done", end_stream=True)
        assert stream.take_snapshot().lifecycle is StreamState.LOCAL_CLOSED
        assert stream.take_snapshot().awaiting_ack

        stream.acknowledge()

        assert stream.take_snapshot().lifecycle is StreamState.LOCAL_CLOSED
        assert not stream.take_snapshot().awaiting_ack
        with pytest.raises(RuntimeError, match="not awaiting acknowledgement"):
            stream.acknowledge()

        stream = Stream(stream_id=3, locally_initiated=True)
        _ = await bot.started(None, stream, stream.model)
        stream.send(b"done", end_stream=True)
        stream.remote_fin()
        assert stream.take_snapshot().lifecycle is StreamState.CLOSED
        assert stream.take_snapshot().awaiting_ack

        stream.acknowledge()

        assert stream.take_snapshot().lifecycle is StreamState.CLOSED
        assert not stream.take_snapshot().awaiting_ack

    asyncio.run(run())


def test_yamux_stream_reset_is_terminal() -> None:
    async def run() -> None:
        stream = Stream(stream_id=2, locally_initiated=False)
        _ = await bot.started(None, stream, stream.model)

        stream.reset()

        assert stream.take_snapshot().lifecycle is StreamState.RESET
        with pytest.raises(RuntimeError, match="not writable"):
            stream.send(b"late")
        assert stream.take_snapshot().lifecycle is StreamState.RESET
        assert stream.take_snapshot().sent_bytes == 0

    asyncio.run(run())


def test_yamux_stream_rejects_invalid_operations_through_hsm_transitions() -> None:
    async def run() -> None:
        stream = Stream(stream_id=2, locally_initiated=False)
        _ = await bot.started(None, stream, stream.model)

        with pytest.raises(RuntimeError, match="not awaiting acknowledgement"):
            stream.acknowledge()

        with pytest.raises(RuntimeError, match="send window"):
            stream.send(b"x" * (INITIAL_STREAM_WINDOW + 1))

        with pytest.raises(RuntimeError, match="receive window"):
            stream.receive(b"x" * (INITIAL_STREAM_WINDOW + 1))

        with pytest.raises(RuntimeError, match="send window"):
            stream.update_send_window((1 << 32) - INITIAL_STREAM_WINDOW)

        with pytest.raises(RuntimeError, match="receive window"):
            stream.grant_receive_window((1 << 32) - INITIAL_STREAM_WINDOW)

        stream.local_fin()
        with pytest.raises(RuntimeError, match="local writes"):
            stream.local_fin()

        stream = Stream(stream_id=4, locally_initiated=False)
        _ = await bot.started(None, stream, stream.model)
        stream.remote_fin()
        with pytest.raises(RuntimeError, match="remote writes"):
            stream.remote_fin()

        stream = Stream(stream_id=1, locally_initiated=True)
        _ = await bot.started(None, stream, stream.model)
        stream.acknowledge()
        with pytest.raises(RuntimeError, match="not awaiting acknowledgement"):
            stream.acknowledge()

        stream.reset()
        with pytest.raises(RuntimeError, match="not awaiting acknowledgement"):
            stream.acknowledge()
        with pytest.raises(RuntimeError, match="terminal"):
            stream.reset()

    asyncio.run(run())

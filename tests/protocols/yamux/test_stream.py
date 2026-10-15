"""Tests for one yamux stream: it decodes what the peer sends to its owner and encodes what its owner sends."""

import asyncio
import pathlib
import typing

import hsm
import pytest

from mosfet.protocols import transform, yamux
from mosfet.protocols.yamux import Flag, FrameData, StreamState
from tests import metric_reader
from tests.protocols.yamux import wire

_WINDOW = yamux.INITIAL_STREAM_WINDOW


async def _stream(
    *,
    locally_initiated: bool = True,
    codec: transform.Codec | None = None,
    channel: str | None = "notes",
    owner: hsm.Instance | None = None,
    max_pending_bytes: int = _WINDOW,
    initial_send_window: int = _WINDOW,
) -> tuple[yamux.Stream, wire.Recorder, hsm.Instance]:
    session = await wire.start(wire.Recorder(), "session")
    if owner is None:
        owner = await wire.start(wire.Recorder(), "owner")
    stream = yamux.Stream(
        stream_id=1 if locally_initiated else 2,
        session=session,
        owner=owner,
        codec=wire.codec() if codec is None and channel is not None else codec,
        channel=channel,
        locally_initiated=locally_initiated,
        max_pending_bytes=max_pending_bytes,
        initial_send_window=initial_send_window,
    )
    stream = await wire.start(stream, "stream")
    return stream, session, owner


def _data(payload: bytes = b"", flags: Flag | None = None, stream_id: int = 1) -> hsm.Event[typing.Any]:
    return yamux.event_from_frame(FrameData.data(stream_id=stream_id, payload=payload, flags=flags))


def _window(delta: int = 0, flags: Flag | None = None, stream_id: int = 1) -> hsm.Event[typing.Any]:
    return yamux.event_from_frame(FrameData.window_update(stream_id=stream_id, delta=delta, flags=flags))


def test_stream_models_its_lifecycle_and_has_no_read_or_command_methods() -> None:
    members = yamux.Stream.model.members
    for state in (
        "/Stream/owned",
        "/Stream/owned/active",
        "/Stream/owned/active/local_opening",
        "/Stream/owned/active/open",
        "/Stream/owned/active/local_closed",
        "/Stream/owned/active/remote_closed",
        "/Stream/owned/closed",
        "/Stream/owned/reset",
    ):
        assert state in members
    for removed in (
        "read",
        "readexactly",
        "readline",
        "at_eof",
        "send",
        "receive",
        "acknowledge",
        "local_fin",
        "remote_fin",
        "reset",
        "update_send_window",
        "grant_receive_window",
    ):
        assert not hasattr(yamux.Stream, removed)


def test_a_locally_opened_stream_tells_its_owner_it_owns_the_channel() -> None:
    async def run() -> None:
        stream, _, owner = await _stream()
        assert isinstance(owner, wire.Recorder)

        (owned,) = owner.named(transform.OwnedEvent)
        assert isinstance(owned.data, transform.OwnedData)
        assert owned.data.stream is stream
        assert owned.data.channel == "notes"
        assert owned.source == hsm.id(stream)
        assert owned.target == hsm.id(owner)
        assert stream.state() == "/Stream/owned/active/local_opening"
        assert stream.take_snapshot().lifecycle is StreamState.LOCAL_OPENING

    asyncio.run(run())


def test_a_message_decodes_to_the_owner_stamped_by_the_stream_and_is_credited_after() -> None:
    async def run() -> None:
        stream, session, owner = await _stream(locally_initiated=False)
        assert isinstance(owner, wire.Recorder)
        sent = wire.message("hello")

        _ = await stream.dispatch(stream.context(), _data(sent, stream_id=2))
        await wire.settle()

        (received,) = owner.named(wire.NoteEvent)
        assert received.data == wire.NoteData(text="hello")
        assert received.source == hsm.id(stream)
        assert received.target == hsm.id(owner)
        assert received.id
        assert session.frames() == [FrameData.window_update(stream_id=2, delta=len(sent))]
        assert stream.take_snapshot().receive_window == _WINDOW
        assert stream.take_snapshot().received_bytes == len(sent)

    asyncio.run(run())


def test_messages_split_across_frames_or_sharing_one_arrive_whole_and_in_order() -> None:
    async def run() -> None:
        stream, session, owner = await _stream(locally_initiated=False)
        assert isinstance(owner, wire.Recorder)
        first, second, third = wire.message("one"), wire.message("two"), wire.message("three")

        _ = await stream.dispatch(stream.context(), _data(first[:3], stream_id=2))
        _ = await stream.dispatch(stream.context(), _data(first[3:] + second, stream_id=2))
        _ = await stream.dispatch(stream.context(), _data(third, stream_id=2))
        await wire.settle()

        assert owner.notes() == ["one", "two", "three"]
        assert [frame.length for frame in session.frames()] == [len(first), len(second), len(third)]

    asyncio.run(run())


def test_window_credit_waits_until_the_owner_has_the_event_and_the_next_waits_for_it() -> None:
    async def run() -> None:
        gate = wire.Gate()
        stream, session, _ = await _stream(locally_initiated=False, owner=gate)

        _ = await stream.dispatch(stream.context(), _data(wire.message("one") + wire.message("two"), stream_id=2))
        await wire.settle()
        assert gate.notes() == ["one"]
        assert session.frames() == []
        assert stream.take_snapshot().receive_window == _WINDOW - 2 * len(wire.message("one"))

        gate.open()
        await wire.settle()
        assert gate.notes() == ["one", "two"]
        assert session.frames() == [FrameData.window_update(stream_id=2, delta=len(wire.message("one")))]

        gate.open()
        await wire.settle()
        assert len(session.frames()) == 2
        assert stream.take_snapshot().receive_window == _WINDOW

    asyncio.run(run())


def test_a_stream_without_a_codec_holds_its_bytes_unacknowledged_until_its_owner_hands_it_over() -> None:
    async def run() -> None:
        connection = await wire.start(wire.Recorder(), "connection")
        port = await wire.start(wire.Recorder(), "port")
        stream, session, _ = await _stream(locally_initiated=False, owner=connection, channel=None)
        sent = wire.message("held")

        _ = await stream.dispatch(stream.context(), _data(sent, stream_id=2))
        await wire.settle()
        assert connection.events == []
        assert session.frames() == []
        assert stream.take_snapshot().receive_window == _WINDOW - len(sent)

        hand_over = yamux.HandOverEvent.with_data(yamux.HandOverData(owner=port, codec=wire.codec(), channel="notes"))
        _ = await stream.dispatch(stream.context(), wire.from_peer(hand_over, port))
        await wire.settle()
        assert port.events == []

        _ = await stream.dispatch(stream.context(), wire.from_peer(hand_over, connection))
        await wire.settle()
        assert [event.name for event in port.events] == [transform.OwnedEvent.name, wire.NoteEvent.name]
        assert port.notes() == ["held"]
        assert connection.events == []
        assert session.frames() == [FrameData.window_update(stream_id=2, delta=len(sent))]

    asyncio.run(run())


def test_a_stream_grows_its_receive_window_to_fit_a_message_of_its_codec() -> None:
    async def run() -> None:
        stream, session, _ = await _stream(codec=wire.codec(max_message_bytes=2 * _WINDOW))

        assert session.frames() == [FrameData.window_update(stream_id=1, delta=_WINDOW + 4)]
        assert stream.take_snapshot().receive_window == 2 * _WINDOW + 4

    asyncio.run(run())


def test_a_message_larger_than_the_initial_window_reaches_the_owner_it_was_handed_to() -> None:
    async def run() -> None:
        connection = await wire.start(wire.Recorder(), "connection")
        port = await wire.start(wire.Recorder(), "port")
        stream, session, _ = await _stream(locally_initiated=False, owner=connection, channel=None)
        big = wire.codec(max_message_bytes=2 * _WINDOW)
        sent = wire.framed(big.encode(wire.note("x" * (_WINDOW + 1024))))

        hand_over = yamux.HandOverEvent.with_data(yamux.HandOverData(owner=port, codec=big, channel="notes"))
        _ = await stream.dispatch(stream.context(), wire.from_peer(hand_over, connection))
        await wire.settle()
        assert session.frames() == [FrameData.window_update(stream_id=2, delta=_WINDOW + 4)]

        _ = await stream.dispatch(stream.context(), _data(sent[:_WINDOW], stream_id=2))
        _ = await stream.dispatch(stream.context(), _data(sent[_WINDOW:], stream_id=2))
        await wire.settle()

        assert port.notes() == ["x" * (_WINDOW + 1024)]
        assert session.frames()[1:] == [FrameData.window_update(stream_id=2, delta=len(sent))]
        assert stream.take_snapshot().receive_window == 2 * _WINDOW + 4
        assert session.named(yamux.ViolationEvent) == []

    asyncio.run(run())


def test_an_unnamed_hand_over_to_itself_decodes_for_the_connection_without_telling_it_owned() -> None:
    async def run() -> None:
        connection = await wire.start(wire.Recorder(), "connection")
        stream, _, _ = await _stream(locally_initiated=False, owner=connection, channel=None)

        _ = await stream.dispatch(
            stream.context(),
            wire.from_peer(
                yamux.HandOverEvent.with_data(yamux.HandOverData(owner=connection, codec=wire.codec())), connection
            ),
        )
        _ = await stream.dispatch(stream.context(), _data(wire.message("first"), stream_id=2))
        await wire.settle()

        assert [event.name for event in connection.events] == [wire.NoteEvent.name]

    asyncio.run(run())


def test_handing_over_tells_the_old_owner_ended_and_the_new_owner_owned() -> None:
    async def run() -> None:
        stream, _, owner = await _stream(locally_initiated=False)
        assert isinstance(owner, wire.Recorder)
        port = await wire.start(wire.Recorder(), "port")

        hand_over = yamux.HandOverEvent.with_data(yamux.HandOverData(owner=port, codec=wire.codec(), channel="other"))
        _ = await stream.dispatch(stream.context(), wire.from_peer(hand_over, owner))
        _ = await stream.dispatch(stream.context(), _data(wire.message("after"), stream_id=2))
        await wire.settle()

        (ended,) = owner.named(transform.EndedEvent)
        assert isinstance(ended.data, transform.EndedData)
        assert (ended.data.stream, ended.data.channel, ended.data.reason) == (stream, "notes", "handed_over")
        (owned,) = port.named(transform.OwnedEvent)
        assert isinstance(owned.data, transform.OwnedData)
        assert (owned.data.stream, owned.data.channel) == (stream, "other")
        assert owner.notes() == []
        assert port.notes() == ["after"]

    asyncio.run(run())


def test_what_the_owner_sends_is_encoded_framed_and_written_and_other_events_are_ignored() -> None:
    async def run() -> None:
        stream, session, _ = await _stream()

        _ = await stream.dispatch(stream.context(), wire.note("out"))
        _ = await stream.dispatch(stream.context(), wire.OtherEvent.with_data(wire.OtherData(text="no")))
        await wire.settle()

        sent = wire.message("out")
        assert session.frames() == [FrameData.data(stream_id=1, payload=sent)]
        assert stream.take_snapshot().send_window == _WINDOW - len(sent)
        assert stream.take_snapshot().sent_bytes == len(sent)

    asyncio.run(run())


def test_sends_past_the_window_wait_for_credit_and_past_the_pending_bound_are_dropped(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = wire.message("x" * 100)
    key = (("codec", "json"), ("reason", "flow_control"))

    async def run() -> None:
        stream, session, _ = await _stream(max_pending_bytes=2 * len(sent), initial_send_window=50)
        before = len(session.frames())

        for _ in range(3):
            _ = await stream.dispatch(stream.context(), wire.note("x" * 100))
        await wire.settle()
        assert sum(frame.length for frame in session.frames()[before:]) == 50
        assert stream.take_snapshot().send_window == 0

        _ = await stream.dispatch(stream.context(), _window(delta=_WINDOW))
        await wire.settle()
        assert b"".join(frame.payload for frame in session.frames()[before:]) == sent + sent

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        earlier = {tuple(sorted(p.attributes.items())): p.value for p in reader.points("bot.transform.dropped.count")}
        asyncio.run(run())
        later = {tuple(sorted(p.attributes.items())): p.value for p in reader.points("bot.transform.dropped.count")}

    assert later[key] - earlier.get(key, 0.0) == 1.0


def test_an_undecodable_message_resets_the_stream_and_is_counted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[yamux.Stream, wire.Recorder, wire.Recorder]:
        stream, session, owner = await _stream(locally_initiated=False)
        assert isinstance(owner, wire.Recorder)
        _ = await stream.dispatch(
            stream.context(), _data(wire.framed(b'{"name":"test.unknown","data":{}}'), stream_id=2)
        )
        await wire.settle()
        return stream, session, owner

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        stream, session, owner = asyncio.run(run())
        points = reader.points("bot.transform.decode.failed.count")

    assert stream.state() == "/Stream/owned/reset"
    assert session.frames() == [FrameData.data(stream_id=2, flags=Flag.RST)]
    assert owner.ended() == ["reset"]
    assert session.streams_ended() == [2]
    assert any(point.attributes == {"codec": "json", "reason": "unknown_event"} for point in points)


def test_a_length_past_the_codec_bound_resets_before_the_message_is_buffered() -> None:
    async def run() -> None:
        stream, session, _ = await _stream(locally_initiated=False, codec=wire.codec(max_message_bytes=8))

        _ = await stream.dispatch(stream.context(), _data(b"\x00\x00\x01\x00", stream_id=2))
        await wire.settle()

        assert stream.state() == "/Stream/owned/reset"
        assert session.frames() == [FrameData.data(stream_id=2, flags=Flag.RST)]

    asyncio.run(run())


def test_close_writes_fin_after_every_pending_message() -> None:
    async def run() -> None:
        stream, session, _ = await _stream(initial_send_window=0)
        _ = await stream.dispatch(stream.context(), wire.note("tail"))
        _ = await stream.dispatch(stream.context(), yamux.CloseEvent.with_data(yamux.CloseData()))
        await wire.settle()
        assert session.frames() == []
        assert stream.state() == "/Stream/owned/active/local_closed"

        _ = await stream.dispatch(stream.context(), _window(delta=_WINDOW))
        await wire.settle()

        assert session.frames() == [
            FrameData.data(stream_id=1, payload=wire.message("tail")),
            FrameData.data(stream_id=1, flags=Flag.FIN),
        ]

    asyncio.run(run())


def test_both_sides_finished_and_acknowledged_ends_the_stream_for_its_owner_and_session() -> None:
    async def run() -> None:
        stream, session, owner = await _stream()
        assert isinstance(owner, wire.Recorder)

        _ = await stream.dispatch(stream.context(), _data(wire.message("last"), flags=Flag.ACK | Flag.FIN))
        await wire.settle()
        assert stream.state() == "/Stream/owned/active/remote_closed"
        assert owner.notes() == ["last"]

        _ = await stream.dispatch(stream.context(), yamux.CloseEvent.with_data(yamux.CloseData()))
        await wire.settle()

        assert stream.state() == "/Stream/owned/closed"
        assert session.frames()[-1] == FrameData.data(stream_id=1, flags=Flag.FIN)
        assert owner.ended() == ["finished"]
        assert session.streams_ended() == [1]

    asyncio.run(run())


def test_a_closed_stream_still_owed_an_ack_ends_when_the_ack_arrives() -> None:
    async def run() -> None:
        stream, session, _ = await _stream()
        _ = await stream.dispatch(stream.context(), yamux.CloseEvent.with_data(yamux.CloseData()))
        _ = await stream.dispatch(stream.context(), _data(flags=Flag.FIN))
        await wire.settle()
        assert stream.take_snapshot().lifecycle is StreamState.CLOSED
        assert stream.take_snapshot().awaiting_ack
        assert session.named(yamux.StreamEndedEvent) == []

        _ = await stream.dispatch(stream.context(), _window(flags=Flag.ACK))
        await wire.settle()

        assert not stream.take_snapshot().awaiting_ack
        assert len(session.named(yamux.StreamEndedEvent)) == 1
        assert session.named(yamux.ViolationEvent) == []

    asyncio.run(run())


def test_an_ack_after_local_fin_opens_nothing_new_and_keeps_the_half_close() -> None:
    async def run() -> None:
        stream, session, _ = await _stream()
        _ = await stream.dispatch(stream.context(), wire.note("done"))
        _ = await stream.dispatch(stream.context(), yamux.CloseEvent.with_data(yamux.CloseData()))
        _ = await stream.dispatch(stream.context(), _window(flags=Flag.ACK))
        await wire.settle()

        assert stream.state() == "/Stream/owned/active/local_closed"
        assert not stream.take_snapshot().awaiting_ack
        assert session.named(yamux.ViolationEvent) == []

    asyncio.run(run())


def test_the_owner_resets_the_stream() -> None:
    async def run() -> None:
        stream, session, owner = await _stream()
        assert isinstance(owner, wire.Recorder)

        _ = await stream.dispatch(stream.context(), yamux.ResetEvent.with_data(yamux.ResetData()))
        _ = await stream.dispatch(stream.context(), wire.note("too late"))
        await wire.settle()

        assert stream.state() == "/Stream/owned/reset"
        assert session.frames() == [FrameData.data(stream_id=1, flags=Flag.RST)]
        assert owner.ended() == ["reset"]
        assert len(session.named(yamux.StreamEndedEvent)) == 1

    asyncio.run(run())


def test_a_peer_reset_ends_the_stream_without_writing_rst() -> None:
    async def run() -> None:
        stream, session, owner = await _stream(locally_initiated=False)
        assert isinstance(owner, wire.Recorder)

        _ = await stream.dispatch(stream.context(), _data(flags=Flag.RST, stream_id=2))
        await wire.settle()

        assert stream.take_snapshot().lifecycle is StreamState.RESET
        assert session.frames() == []
        assert owner.ended() == ["reset"]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("frames", "kind"),
    [
        ([_data(b"x" * (_WINDOW + 1), stream_id=2)], "receive_window_exhausted"),
        ([_data(flags=Flag.FIN, stream_id=2), _data(b"late", stream_id=2)], "stream_reset"),
        ([_window(flags=Flag.FIN, stream_id=2), _window(flags=Flag.FIN, stream_id=2)], "stream_reset"),
        ([_data(flags=Flag.ACK, stream_id=2)], "protocol_error"),
        ([_window(delta=(1 << 32) - _WINDOW, stream_id=2)], "protocol_error"),
    ],
)
def test_a_peer_breaking_the_stream_rules_resets_it_and_reports_a_violation(
    frames: list[hsm.Event[typing.Any]], kind: str
) -> None:
    async def run() -> None:
        stream, session, _ = await _stream(locally_initiated=False)

        for frame in frames:
            _ = await stream.dispatch(stream.context(), frame)
        await wire.settle()

        assert stream.take_snapshot().lifecycle is StreamState.RESET
        assert session.frames()[-1] == FrameData.data(stream_id=2, flags=Flag.RST)
        (violation,) = session.named(yamux.ViolationEvent)
        assert isinstance(violation.data, yamux.ViolationData)
        assert (violation.data.stream_id, violation.data.kind) == (2, kind)

    asyncio.run(run())


def test_stopping_the_stream_tells_its_owner_closed_once() -> None:
    async def run() -> None:
        stream, _, owner = await _stream()
        assert isinstance(owner, wire.Recorder)

        await hsm.stop(stream)
        await wire.settle()

        assert owner.ended() == ["closed"]

    asyncio.run(run())


def test_stream_snapshot_extends_the_canonical_hsm_snapshot() -> None:
    async def run() -> None:
        stream, _, _ = await _stream()

        snapshot = hsm.take_snapshot(None, stream)

        assert isinstance(snapshot, yamux.StreamSnapshot)
        assert snapshot.stream_id == 1
        assert snapshot.State == "/Stream/owned/active/local_opening"
        assert snapshot.awaiting_ack

    asyncio.run(run())

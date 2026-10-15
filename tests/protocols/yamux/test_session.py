"""Tests for a yamux session: it multiplexes streams over one connection and talks to it, and to them, by message."""

import asyncio
import datetime
import types
import typing

import hsm
import pytest

from mosfet.protocols import transform, yamux
from mosfet.protocols.yamux import Flag, FrameData, GoAwayCode, StreamState
from tests.hsm_model import choice_transitions, transition_map
from tests.protocols.yamux import wire

_WINDOW = yamux.INITIAL_STREAM_WINDOW


async def _session(
    role: typing.Literal["client", "server"] = "client",
    *,
    ping_timeout: datetime.timedelta = yamux.DEFAULT_PING_TIMEOUT,
) -> tuple[yamux.Session, wire.Recorder]:
    connection = await wire.start(wire.Recorder(), f"connection-{role}")
    session = yamux.Session(role=role, connection=connection, ping_timeout=ping_timeout)
    return await wire.start(session, f"session-{role}"), connection


async def _receive(session: yamux.Session, frame: FrameData) -> None:
    _ = await session.dispatch(session.context(), yamux.event_from_frame(frame))
    await wire.settle()


async def _open(session: yamux.Session, owner: hsm.Instance, channel: str = "notes") -> hsm.Event[typing.Any]:
    request = yamux.OpenStreamEvent.with_data_and_id(
        yamux.OpenStreamData(owner=owner, codec=wire.codec(), channel=channel), f"open-{channel}"
    )
    _ = await session.dispatch(session.context(), request)
    await wire.settle()
    return request


def _stream_of(owner: wire.Recorder) -> hsm.Instance:
    (owned,) = owner.named(transform.OwnedEvent)
    assert isinstance(owned.data, transform.OwnedData)
    return owned.data.stream


def test_session_models_protocol_states_and_has_no_frame_queue_or_stream_lookup() -> None:
    model = yamux.Session.model
    for state in (
        "/Session/connected",
        "/Session/connected/open",
        "/Session/connected/pinging",
        "/Session/connected/draining",
        "/Session/connected/routing_disconnect",
        "/Session/errored",
        "/Session/disconnected",
    ):
        assert state in model.members
    for removed in ("outbound", "stream", "receive_frame"):
        assert not hasattr(yamux.Session, removed)
    for removed in ("FrameStream", "ReadStream", "SendDataEvent", "CloseStreamEvent", "ResetStreamEvent"):
        assert not hasattr(yamux, removed)

    transitions = transition_map(model)
    assert yamux.FrameOutEvent.name in transitions["/Session/connected"]
    assert yamux.StreamEndedEvent.name in transitions["/Session/connected"]
    assert yamux.ViolationEvent.name in transitions["/Session/connected"]
    assert yamux.FrameOutEvent.name not in transitions.get("/Session/errored", {})
    routing = choice_transitions(model, "/Session/connected/routing_disconnect")
    assert [transition.target for transition in routing] == ["/Session/connected/draining", "/Session/disconnected"]


def test_a_local_open_writes_syn_with_a_role_specific_id_and_tells_the_owner_owned() -> None:
    async def run() -> None:
        client, client_connection = await _session("client")
        server, server_connection = await _session("server")
        client_owner = await wire.start(wire.Recorder(), "client-owner")
        server_owner = await wire.start(wire.Recorder(), "server-owner")

        _ = await _open(client, client_owner)
        _ = await _open(server, server_owner)

        assert client_connection.frames() == [FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)]
        assert server_connection.frames() == [FrameData.window_update(stream_id=2, delta=0, flags=Flag.SYN)]
        (transmit,) = client_connection.named(yamux.TransmitEvent)
        assert transmit.source == hsm.id(client)
        assert transmit.target == hsm.id(client_connection)
        assert _stream_of(client_owner).take_snapshot().State == "/Stream/owned/active/local_opening"
        assert client.take_snapshot().streams[1].lifecycle is StreamState.LOCAL_OPENING

    asyncio.run(run())


def test_a_larger_receive_window_is_granted_with_the_syn() -> None:
    async def run() -> None:
        session, connection = await _session()
        owner = await wire.start(wire.Recorder(), "owner")

        _ = await session.dispatch(
            session.context(),
            yamux.OpenStreamEvent.with_data(
                yamux.OpenStreamData(owner=owner, codec=wire.codec(), channel="notes", initial_window_size=_WINDOW + 7)
            ),
        )

        assert connection.frames() == [FrameData.window_update(stream_id=1, delta=7, flags=Flag.SYN)]

    asyncio.run(run())


def test_what_the_owner_sends_reaches_the_connection_as_data_frames_and_acks_open_the_stream() -> None:
    async def run() -> None:
        session, connection = await _session()
        owner = await wire.start(wire.Recorder(), "owner")
        _ = await _open(session, owner)
        stream = _stream_of(owner)

        _ = await stream.dispatch(stream.context(), wire.note("hello"))
        await wire.settle()
        assert connection.frames()[-1] == FrameData.data(stream_id=1, payload=wire.message("hello"))

        await _receive(session, FrameData.window_update(stream_id=1, delta=10, flags=Flag.ACK))
        snapshot = session.take_snapshot().streams[1]
        assert snapshot.lifecycle is StreamState.OPEN
        assert snapshot.send_window == _WINDOW - len(wire.message("hello")) + 10

    asyncio.run(run())


def test_a_refused_open_answers_the_owner_with_the_request_id() -> None:
    async def run() -> None:
        session, _ = await _session()
        owner = await wire.start(wire.Recorder(), "owner")
        _ = await session.dispatch(session.context(), yamux.GoAwayEvent.with_data(yamux.GoAwayData(code=0)))

        request = await _open(session, owner)

        (failed,) = owner.named(yamux.OpenStreamFailedEvent)
        assert isinstance(failed.data, yamux.OpenStreamFailedData)
        assert failed.data.kind == "session_draining"
        assert failed.id == request.id
        assert owner.named(transform.OwnedEvent) == []
        assert session.take_snapshot().failed_operations[-1].failure_kind == "session_draining"

    asyncio.run(run())


def test_a_remote_open_is_acked_and_handed_to_the_connection_holding_its_bytes() -> None:
    async def run() -> None:
        session, connection = await _session()
        sent = wire.message("rpc")

        await _receive(session, FrameData.data(stream_id=2, payload=sent, flags=Flag.SYN))

        assert connection.frames() == [FrameData.window_update(stream_id=2, delta=0, flags=Flag.ACK)]
        (accepted,) = connection.named(yamux.StreamAcceptedEvent)
        assert isinstance(accepted.data, yamux.StreamAcceptedData)
        assert accepted.data.stream_id == 2
        assert accepted.source == hsm.id(session)
        snapshot = session.take_snapshot().streams[2]
        assert snapshot.lifecycle is StreamState.OPEN
        assert snapshot.receive_window == _WINDOW - len(sent)

        port = await wire.start(wire.Recorder(), "port")
        stream = accepted.data.stream
        _ = await stream.dispatch(
            stream.context(),
            wire.from_peer(
                yamux.HandOverEvent.with_data(yamux.HandOverData(owner=port, codec=wire.codec(), channel="notes")),
                connection,
            ),
        )
        await wire.settle()

        assert port.notes() == ["rpc"]
        assert connection.frames()[-1] == FrameData.window_update(stream_id=2, delta=len(sent))

    asyncio.run(run())


@pytest.mark.parametrize(
    "frame",
    [
        FrameData.data(stream_id=3, flags=Flag.SYN),
        FrameData.data(stream_id=2, flags=Flag.SYN | Flag.ACK),
        FrameData.data(stream_id=2, payload=b"x"),
        FrameData.window_update(stream_id=3, delta=0, flags=Flag.SYN),
        FrameData.window_update(stream_id=2, delta=(1 << 32) - _WINDOW, flags=Flag.SYN),
    ],
    ids=["bad-parity", "syn-with-ack", "unknown-without-syn", "window-bad-parity", "window-overflow"],
)
def test_a_frame_breaking_the_session_rules_resets_it_goes_away_and_ends_the_session(frame: FrameData) -> None:
    async def run() -> None:
        session, connection = await _session()

        await _receive(session, frame)

        assert session.state() == "/Session/errored"
        assert session.take_snapshot().failed_operations[-1].failure_kind == "protocol_error"
        assert connection.frames() == [
            FrameData.data(stream_id=frame.stream_id, flags=Flag.RST),
            FrameData.go_away(code=GoAwayCode.PROTOCOL_ERROR),
        ]
        assert connection.events[-1].name == yamux.EndedEvent.name
        assert isinstance(connection.events[-1].data, yamux.EndedData)
        assert connection.events[-1].data.reason == "protocol_error"

    asyncio.run(run())


def test_a_duplicate_syn_is_a_protocol_error_and_frames_stop_once_errored() -> None:
    async def run() -> None:
        session, connection = await _session()
        owner = await wire.start(wire.Recorder(), "owner")
        _ = await _open(session, owner)
        await _receive(session, FrameData.data(stream_id=2, flags=Flag.SYN))

        await _receive(session, FrameData.data(stream_id=2, flags=Flag.SYN))
        written = len(connection.frames())
        stream = _stream_of(owner)
        _ = await stream.dispatch(stream.context(), wire.note("after"))
        await wire.settle()

        assert session.state() == "/Session/errored"
        assert connection.frames()[-2:] == [
            FrameData.data(stream_id=2, flags=Flag.RST),
            FrameData.go_away(code=GoAwayCode.PROTOCOL_ERROR),
        ]
        assert len(connection.frames()) == written

    asyncio.run(run())


def test_a_stream_protocol_violation_ends_the_session_and_other_violations_only_its_stream() -> None:
    async def run() -> None:
        session, connection = await _session()
        await _receive(session, FrameData.data(stream_id=2, flags=Flag.SYN))
        await _receive(session, FrameData.data(stream_id=2, payload=b"x" * (_WINDOW + 1)))

        assert session.state() == "/Session/connected/open"
        assert session.take_snapshot().failed_operations[-1].failure_kind == "receive_window_exhausted"
        assert connection.frames()[-1] == FrameData.data(stream_id=2, flags=Flag.RST)

        await _receive(session, FrameData.data(stream_id=4, flags=Flag.SYN))
        await _receive(session, FrameData.data(stream_id=4, flags=Flag.ACK))

        assert session.state() == "/Session/errored"
        assert connection.frames()[-2:] == [
            FrameData.data(stream_id=4, flags=Flag.RST),
            FrameData.go_away(code=GoAwayCode.PROTOCOL_ERROR),
        ]

    asyncio.run(run())


def test_oversized_syn_data_is_refused_before_a_stream_exists() -> None:
    async def run() -> None:
        session, connection = await _session()

        await _receive(session, FrameData.data(stream_id=2, payload=b"x" * (_WINDOW + 1), flags=Flag.SYN))

        assert 2 not in session.take_snapshot().streams
        assert connection.named(yamux.StreamAcceptedEvent) == []
        assert session.take_snapshot().failed_operations[-1].failure_kind == "receive_window_exhausted"
        assert connection.frames() == [FrameData.data(stream_id=2, flags=Flag.RST)]

    asyncio.run(run())


def test_ping_is_echoed_correlated_and_times_out_ending_the_session() -> None:
    async def run() -> None:
        session, connection = await _session(ping_timeout=datetime.timedelta(milliseconds=5))

        _ = await session.dispatch(session.context(), yamux.PingEvent.with_data(yamux.PingData(opaque=42)))
        assert session.take_snapshot().outstanding_ping == 42
        await _receive(session, FrameData.ping(opaque=99, ack=True))
        assert session.state() == "/Session/connected/pinging"
        await _receive(session, FrameData.ping(opaque=42, ack=True))
        assert session.state() == "/Session/connected/open"
        await _receive(session, FrameData.ping(opaque=7))
        assert connection.frames() == [FrameData.ping(opaque=42), FrameData.ping(opaque=7, ack=True)]

        _ = await session.dispatch(session.context(), yamux.PingEvent.with_data(yamux.PingData(opaque=43)))
        await asyncio.sleep(0.03)

        assert session.state() == "/Session/errored"
        assert session.take_snapshot().outstanding_ping is None
        assert session.take_snapshot().failed_operations[-1].failure_kind == "ping_timeout"
        assert connection.session_ended() == ["ping_timeout"]

    asyncio.run(run())


def test_go_away_without_streams_ends_the_session_drained() -> None:
    async def run() -> None:
        session, connection = await _session()

        _ = await session.dispatch(session.context(), yamux.GoAwayEvent.with_data(yamux.GoAwayData(code=0)))
        await wire.settle()

        assert session.state() == "/Session/disconnected"
        assert session.take_snapshot().closed_goaway_code == GoAwayCode.NORMAL
        assert connection.frames() == [FrameData.go_away(code=GoAwayCode.NORMAL)]
        assert connection.session_ended() == ["drained"]

    asyncio.run(run())


def test_go_away_drains_the_streams_left_refusing_new_ones() -> None:
    async def run() -> None:
        session, connection = await _session()
        owner = await wire.start(wire.Recorder(), "owner")
        _ = await _open(session, owner)
        await _receive(session, FrameData.window_update(stream_id=1, delta=0, flags=Flag.ACK))

        await _receive(session, FrameData.go_away(code=GoAwayCode.NORMAL))
        assert session.state() == "/Session/connected/draining"
        await _receive(session, FrameData.data(stream_id=2, flags=Flag.SYN))
        assert connection.frames()[-1] == FrameData.data(stream_id=2, flags=Flag.RST)
        assert session.take_snapshot().failed_operations[-1].failure_kind == "session_draining"

        stream = _stream_of(owner)
        _ = await stream.dispatch(stream.context(), wire.note("finish"))
        _ = await stream.dispatch(stream.context(), yamux.CloseEvent.with_data(yamux.CloseData()))
        await _receive(session, FrameData.data(stream_id=1, flags=Flag.FIN))

        assert session.state() == "/Session/disconnected"
        assert session.take_snapshot().streams == {}
        assert connection.session_ended() == ["drained"]
        assert connection.frames()[-2:] == [
            FrameData.data(stream_id=1, payload=wire.message("finish")),
            FrameData.data(stream_id=1, flags=Flag.FIN),
        ]

    asyncio.run(run())


def test_late_frames_for_a_retired_stream_are_ignored_and_counted() -> None:
    async def run() -> None:
        session, connection = await _session()
        owner = await wire.start(wire.Recorder(), "owner")
        _ = await _open(session, owner)
        stream = _stream_of(owner)
        _ = await stream.dispatch(stream.context(), yamux.ResetEvent.with_data(yamux.ResetData()))
        await wire.settle()
        assert 1 not in session.take_snapshot().streams
        written = len(connection.frames())

        await _receive(session, FrameData.data(stream_id=1, payload=b"late"))
        await _receive(session, FrameData.window_update(stream_id=1, delta=1, flags=Flag.ACK))

        assert session.state() == "/Session/connected/open"
        assert len(connection.frames()) == written
        assert [failure.failure_kind for failure in session.take_snapshot().failed_operations] == [
            "unknown_stream",
            "unknown_stream",
        ]

    asyncio.run(run())


def test_snapshot_lists_live_streams_from_their_own_snapshots() -> None:
    async def run() -> None:
        session, _ = await _session()
        await _receive(session, FrameData.data(stream_id=2, payload=b"", flags=Flag.SYN))

        snapshot = hsm.take_snapshot(None, session)

        assert isinstance(snapshot, yamux.SessionSnapshot)
        assert isinstance(snapshot.streams, types.MappingProxyType)
        assert snapshot.streams[2].State == "/Stream/owned/active/open"
        assert snapshot.has_active_streams

    asyncio.run(run())


def test_stopping_the_session_stops_its_streams_and_their_owners_are_told_closed() -> None:
    async def run() -> None:
        session, _ = await _session()
        owner = await wire.start(wire.Recorder(), "owner")
        _ = await _open(session, owner)

        await hsm.stop(session)
        await wire.settle()

        assert owner.ended() == ["closed"]

    asyncio.run(run())


def test_two_sessions_carry_notes_both_ways_through_a_socket() -> None:
    async def run() -> None:
        server_owner = await wire.start(wire.Recorder(), "server-owner")
        client_pipe = await wire.start(wire.Pipe(), "client-pipe")
        server_pipe = await wire.start(wire.Pipe(owner=server_owner), "server-pipe")
        client = await wire.start(yamux.Client(connection=client_pipe), "client")
        server = await wire.start(yamux.Server(connection=server_pipe), "server")
        client_pipe.peer = server
        server_pipe.peer = client
        client_owner = await wire.start(wire.Recorder(), "client-owner")

        _ = await _open(client, client_owner)
        client_stream = _stream_of(client_owner)
        for text in ("one", "two", "three"):
            _ = await client_stream.dispatch(client_stream.context(), wire.note(text))
        await wire.settle()
        server_stream = _stream_of(server_owner)
        _ = await server_stream.dispatch(server_stream.context(), wire.note("back"))
        await wire.settle()

        assert server_owner.notes() == ["one", "two", "three"]
        assert client_owner.notes() == ["back"]
        assert client.take_snapshot().streams[1].lifecycle is StreamState.OPEN
        assert client.take_snapshot().streams[1].send_window == _WINDOW
        assert server.take_snapshot().streams[1].receive_window == _WINDOW

    asyncio.run(run())


def test_event_from_frame_projects_frames_to_typed_events() -> None:
    event = yamux.event_from_frame(FrameData.data(stream_id=1, payload=b"x", flags=Flag.FIN))

    assert event.name == "protocol.yamux.frame.data.received"
    assert isinstance(event.data, yamux.ReceiveDataFrameData)
    assert (event.data.stream_id, event.data.flags, event.data.payload) == (1, int(Flag.FIN), b"x")

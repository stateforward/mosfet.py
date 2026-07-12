import asyncio
import datetime
import inspect
import types

import hsm

from bot.protocols.yamux.events import (
    CloseStreamEvent,
    GoAwayEvent,
    OpenStreamEvent,
    PingEvent,
    SendDataEvent,
    CloseStreamData,
    GoAwayData,
    OpenStreamData,
    PingData,
    ReceiveDataFrameData,
    SendData,
)
from bot.protocols.yamux.frame import (
    INITIAL_STREAM_WINDOW,
    Flag,
    FrameData,
    FrameType,
    GoAwayCode,
)
from bot.protocols.yamux.session import ReadStream, Session, event_from_frame
from bot.protocols.yamux.session import SessionSnapshot
from bot.protocols.yamux.stream import StreamState
from tests.hsm_model import choice_transitions, transition_map


async def read_outbound(session: Session) -> FrameData:
    return await asyncio.wait_for(session.outbound().read(), timeout=1)


async def read_outbound_frames(session: Session, count: int) -> tuple[FrameData, ...]:
    return tuple([await read_outbound(session) for _ in range(count)])


def test_yamux_session_model_tracks_protocol_states_without_stream_events() -> None:
    model = Session.model

    for name in (
        "role",
        "initial_stream_window",
        "next_stream_id",
        "accepts_new_streams",
        "outbound",
        "failed_operations",
        "outstanding_ping",
        "closed_goaway_code",
        "ping_timeout",
        "streams",
        "has_active_streams",
    ):
        assert not isinstance(inspect.getattr_static(Session, name, None), property)
    assert not hasattr(Session, "peer_stream_id_is_valid")
    assert model.qualified_name == "/Session"
    assert "/Session/connected" in model.members
    assert "/Session/connected/open" in model.members
    assert "/Session/connected/pinging" in model.members
    assert "/Session/connected/draining" in model.members
    assert "/Session/errored" in model.members
    assert "/Session/disconnected" in model.members
    assert "/Session/connected/routing_disconnect" in model.members
    assert "/Session/ready" not in model.members
    assert "/Session/closed" not in model.members
    assert "/Session/failed" not in model.members
    assert "/Session/connected/operational" not in model.members
    assert "/Session/connected/normal" not in model.members
    assert "/Session/sending_data" not in model.members
    assert not any(isinstance(member, hsm.ObservationElement) for member in model.members.values())

    transitions = transition_map(model)
    assert SendDataEvent.name in transitions["/Session/connected"]
    assert CloseStreamEvent.name in transitions["/Session/connected"]
    assert OpenStreamEvent.name in transitions["/Session/connected"]
    assert GoAwayEvent.name in transitions["/Session/connected"]
    assert "protocol.yamux.session.window_consumed" in transitions["/Session/connected"]
    assert PingEvent.name in transitions["/Session/connected/open"]
    assert transitions["/Session/connected/open"][SendDataEvent.name][0].source == "/Session/connected"
    assert transitions["/Session/connected/pinging"][SendDataEvent.name][0].source == "/Session/connected"
    assert len(transitions["/Session/connected"][GoAwayEvent.name]) == 1
    assert transitions["/Session/connected"][GoAwayEvent.name][0].target == "/Session/connected/routing_disconnect"
    routing = choice_transitions(model, "/Session/connected/routing_disconnect")
    assert routing[0].target == "/Session/connected/draining"
    assert routing[1].target == "/Session/disconnected"


def test_yamux_session_take_snapshot_extends_canonical_hsm_snapshot() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.receive_frame(session.context(), FrameData.data(stream_id=2, payload=b"rpc", flags=Flag.SYN))

        snapshot = hsm.take_snapshot(None, session)

        assert isinstance(snapshot, SessionSnapshot)
        assert snapshot.State == "/Session/connected/open"
        assert snapshot.streams[2].State == "/Stream/active/open"
        assert snapshot.streams[2].lifecycle is StreamState.OPEN

    asyncio.run(run())


def test_yamux_session_outbound_frame_stream_wakes_waiting_reader() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)

        pending_frame = asyncio.create_task(read_outbound(session))
        await asyncio.sleep(0)
        assert not pending_frame.done()

        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))

        assert await pending_frame == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)

    asyncio.run(run())


def test_yamux_session_allocates_role_specific_stream_ids_and_opens_with_syn() -> None:
    async def run() -> None:
        client = Session(role="client")
        server = Session(role="server")
        _ = await hsm.started(None, client, client.model)
        _ = await hsm.started(None, server, server.model)

        await client.dispatch(client.context(), OpenStreamEvent.with_data(OpenStreamData()))
        await server.dispatch(server.context(), OpenStreamEvent.with_data(OpenStreamData()))

        assert await read_outbound(client) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)
        assert await read_outbound(server) == FrameData.window_update(stream_id=2, delta=0, flags=Flag.SYN)
        assert client.take_snapshot().streams[1].lifecycle is StreamState.LOCAL_OPENING
        assert server.take_snapshot().streams[2].lifecycle is StreamState.LOCAL_OPENING

    asyncio.run(run())


def test_yamux_session_allows_data_before_ack_and_counts_only_data_bytes() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)

        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))
        await session.dispatch(
            session.context(), SendDataEvent.with_data(SendData(stream_id=1, payload=b"hello"))
        )

        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)
        assert await read_outbound(session) == FrameData.data(stream_id=1, payload=b"hello")
        assert session.take_snapshot().streams[1].lifecycle is StreamState.LOCAL_OPENING
        assert session.take_snapshot().streams[1].send_window == INITIAL_STREAM_WINDOW - 5
        assert session.take_snapshot().streams[1].sent_bytes == 5

        await session.receive_frame(session.context(), FrameData.window_update(stream_id=1, delta=10, flags=Flag.ACK))
        assert session.take_snapshot().streams[1].lifecycle is StreamState.OPEN
        assert session.take_snapshot().streams[1].send_window == INITIAL_STREAM_WINDOW + 5

    asyncio.run(run())


def test_yamux_session_accepts_data_ack_for_locally_opened_stream() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))
        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)

        await session.receive_frame(session.context(), FrameData.data(stream_id=1, payload=b"accepted", flags=Flag.ACK))

        stream = session.stream(1)
        assert isinstance(stream, ReadStream)
        assert "_stream" not in inspect.signature(ReadStream).parameters
        assert not hasattr(stream, "_stream")
        assert not hasattr(stream, "send")
        assert not hasattr(stream, "receive")
        assert not hasattr(stream, "reset")
        assert session.take_snapshot().streams[1].lifecycle is StreamState.OPEN
        assert await stream.readexactly(8) == b"accepted"
        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=8)
        assert session.take_snapshot().streams[1].receive_window == INITIAL_STREAM_WINDOW

    asyncio.run(run())


def test_yamux_session_accepts_remote_syn_and_exposes_python_stream_reader() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)

        await session.receive_frame(session.context(), FrameData.data(stream_id=2, payload=b"rpc", flags=Flag.SYN))

        stream = session.stream(2)
        assert isinstance(stream, ReadStream)
        assert await read_outbound(session) == FrameData.window_update(stream_id=2, delta=0, flags=Flag.ACK)
        assert session.take_snapshot().streams[2].lifecycle is StreamState.OPEN
        assert session.take_snapshot().streams[2].received_bytes == 3
        assert await stream.readexactly(3) == b"rpc"

    asyncio.run(run())


def test_yamux_session_sends_window_update_when_application_reads_stream_bytes() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.receive_frame(session.context(), FrameData.data(stream_id=2, payload=b"window", flags=Flag.SYN))
        assert await read_outbound(session) == FrameData.window_update(stream_id=2, delta=0, flags=Flag.ACK)

        stream = session.stream(2)
        assert isinstance(stream, ReadStream)
        assert isinstance(session.take_snapshot().streams, types.MappingProxyType)
        assert session.take_snapshot().streams[2].receive_window == INITIAL_STREAM_WINDOW - 6

        assert await stream.read(3) == b"win"
        assert await read_outbound(session) == FrameData.window_update(stream_id=2, delta=3)
        assert session.take_snapshot().streams[2].receive_window == INITIAL_STREAM_WINDOW - 3

        assert await stream.readexactly(3) == b"dow"
        assert await read_outbound(session) == FrameData.window_update(stream_id=2, delta=3)
        assert session.take_snapshot().streams[2].receive_window == INITIAL_STREAM_WINDOW

    asyncio.run(run())


def test_yamux_session_rejects_bad_remote_stream_parity() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)

        await session.receive_frame(session.context(), FrameData.data(stream_id=3, flags=Flag.SYN))

        assert session.state() == "/Session/errored"
        assert session.take_snapshot().failed_operations[-1].failure_kind == "protocol_error"
        assert await read_outbound_frames(session, 2) == (
            FrameData.data(stream_id=3, flags=Flag.RST),
            FrameData.go_away(code=GoAwayCode.PROTOCOL_ERROR),
        )

    asyncio.run(run())


def test_yamux_session_rejects_bad_remote_ack_flag() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)

        await session.receive_frame(session.context(), FrameData.data(stream_id=2, flags=Flag.SYN | Flag.ACK))

        assert session.state() == "/Session/errored"
        assert session.take_snapshot().failed_operations[-1].failure_kind == "protocol_error"
        assert await read_outbound_frames(session, 2) == (
            FrameData.data(stream_id=2, flags=Flag.RST),
            FrameData.go_away(code=GoAwayCode.PROTOCOL_ERROR),
        )

    asyncio.run(run())


def test_yamux_session_protocol_error_clears_outstanding_ping() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)

        await session.dispatch(session.context(), PingEvent.with_data(PingData(opaque=77)))
        assert await read_outbound(session) == FrameData.ping(opaque=77)

        await session.receive_frame(session.context(), FrameData.data(stream_id=3, flags=Flag.SYN))

        assert session.state() == "/Session/errored"
        assert session.take_snapshot().outstanding_ping is None
        assert await read_outbound_frames(session, 2) == (
            FrameData.data(stream_id=3, flags=Flag.RST),
            FrameData.go_away(code=GoAwayCode.PROTOCOL_ERROR),
        )

    asyncio.run(run())


def test_yamux_session_fin_half_close_and_rst_runtime() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))
        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)
        await session.receive_frame(session.context(), FrameData.window_update(stream_id=1, delta=0, flags=Flag.ACK))

        await session.receive_frame(session.context(), FrameData.data(stream_id=1, flags=Flag.FIN))
        assert session.take_snapshot().streams[1].lifecycle is StreamState.REMOTE_CLOSED

        await session.dispatch(session.context(), CloseStreamEvent.with_data(CloseStreamData(stream_id=1)))
        assert await read_outbound(session) == FrameData.data(stream_id=1, flags=Flag.FIN)
        assert session.take_snapshot().streams[1].lifecycle is StreamState.CLOSED

        await session.receive_frame(session.context(), FrameData.data(stream_id=2, flags=Flag.SYN))
        await session.receive_frame(session.context(), FrameData.data(stream_id=2, flags=Flag.RST))
        assert session.take_snapshot().streams[2].lifecycle is StreamState.RESET

    asyncio.run(run())


def test_yamux_session_accepts_ack_after_local_fin_before_ack() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))
        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)

        await session.dispatch(
            session.context(),
            SendDataEvent.with_data(SendData(stream_id=1, payload=b"done", end_stream=True)),
        )
        assert await read_outbound(session) == FrameData.data(stream_id=1, payload=b"done", flags=Flag.FIN)
        assert session.take_snapshot().streams[1].lifecycle is StreamState.LOCAL_CLOSED
        assert session.take_snapshot().streams[1].awaiting_ack

        await session.receive_frame(session.context(), FrameData.window_update(stream_id=1, delta=0, flags=Flag.ACK))

        assert session.state() == "/Session/connected/open"
        assert session.take_snapshot().failed_operations == ()
        assert session.take_snapshot().streams[1].lifecycle is StreamState.LOCAL_CLOSED
        assert not session.take_snapshot().streams[1].awaiting_ack
        assert session.outbound().empty()

    asyncio.run(run())


def test_yamux_session_accepts_data_ack_after_remote_fin_before_ack() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))
        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)

        await session.receive_frame(session.context(), FrameData.data(stream_id=1, flags=Flag.FIN))
        assert session.take_snapshot().streams[1].lifecycle is StreamState.REMOTE_CLOSED
        assert session.take_snapshot().streams[1].awaiting_ack

        await session.receive_frame(session.context(), FrameData.data(stream_id=1, flags=Flag.ACK))

        assert session.state() == "/Session/connected/open"
        assert session.take_snapshot().failed_operations == ()
        assert session.take_snapshot().streams[1].lifecycle is StreamState.REMOTE_CLOSED
        assert not session.take_snapshot().streams[1].awaiting_ack
        assert session.outbound().empty()

    asyncio.run(run())


def test_yamux_session_accepts_ack_after_stream_closed_before_ack() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))
        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)

        await session.dispatch(
            session.context(),
            SendDataEvent.with_data(SendData(stream_id=1, payload=b"done", end_stream=True)),
        )
        assert await read_outbound(session) == FrameData.data(stream_id=1, payload=b"done", flags=Flag.FIN)
        await session.receive_frame(session.context(), FrameData.data(stream_id=1, flags=Flag.FIN))
        assert session.take_snapshot().streams[1].lifecycle is StreamState.CLOSED
        assert session.take_snapshot().streams[1].awaiting_ack

        await session.receive_frame(session.context(), FrameData.window_update(stream_id=1, delta=0, flags=Flag.ACK))

        assert session.state() == "/Session/connected/open"
        assert session.take_snapshot().failed_operations == ()
        assert session.take_snapshot().streams[1].lifecycle is StreamState.CLOSED
        assert not session.take_snapshot().streams[1].awaiting_ack
        assert session.outbound().empty()

    asyncio.run(run())


def test_yamux_session_rejects_duplicate_window_fin_without_runtime_error() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)

        await session.receive_frame(session.context(), FrameData.window_update(stream_id=2, delta=0, flags=Flag.SYN))
        assert await read_outbound(session) == FrameData.window_update(stream_id=2, delta=0, flags=Flag.ACK)

        await session.receive_frame(session.context(), FrameData.window_update(stream_id=2, delta=0, flags=Flag.FIN))
        assert session.take_snapshot().streams[2].lifecycle is StreamState.REMOTE_CLOSED

        await session.receive_frame(session.context(), FrameData.window_update(stream_id=2, delta=0, flags=Flag.FIN))

        assert session.state() == "/Session/connected/open"
        assert session.take_snapshot().failed_operations[-1].failure_kind == "stream_reset"
        assert await read_outbound(session) == FrameData.data(stream_id=2, flags=Flag.RST)

    asyncio.run(run())


def test_yamux_session_rejects_send_when_flow_control_is_exhausted() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))
        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)

        await session.dispatch(
            session.context(),
            SendDataEvent.with_data(SendData(stream_id=1, payload=b"x" * (INITIAL_STREAM_WINDOW + 1))),
        )

        assert session.state() == "/Session/connected/open"
        assert session.take_snapshot().failed_operations[-1].failure_kind == "flow_control_exhausted"
        assert session.outbound().empty()

    asyncio.run(run())


def test_yamux_session_rejects_inbound_data_when_receive_window_is_exhausted() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.receive_frame(session.context(), FrameData.data(stream_id=2, flags=Flag.SYN))
        assert await read_outbound(session) == FrameData.window_update(stream_id=2, delta=0, flags=Flag.ACK)

        await session.receive_frame(
            session.context(),
            FrameData.data(stream_id=2, payload=b"x" * (INITIAL_STREAM_WINDOW + 1)),
        )

        assert session.state() == "/Session/connected/open"
        assert session.take_snapshot().failed_operations[-1].failure_kind == "receive_window_exhausted"
        assert await read_outbound(session) == FrameData.data(stream_id=2, flags=Flag.RST)
        assert session.take_snapshot().streams[2].lifecycle is StreamState.RESET

    asyncio.run(run())


def test_yamux_session_rejects_oversized_syn_data_before_opening_remote_stream() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)

        await session.receive_frame(
            session.context(),
            FrameData.data(stream_id=2, payload=b"x" * (INITIAL_STREAM_WINDOW + 1), flags=Flag.SYN),
        )

        assert 2 not in session.take_snapshot().streams
        assert session.take_snapshot().failed_operations[-1].failure_kind == "receive_window_exhausted"
        assert await read_outbound(session) == FrameData.data(stream_id=2, flags=Flag.RST)

    asyncio.run(run())


def test_yamux_session_rejects_stale_terminal_data_and_window_updates() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))
        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)
        await session.receive_frame(session.context(), FrameData.window_update(stream_id=1, delta=0, flags=Flag.ACK))
        await session.receive_frame(session.context(), FrameData.data(stream_id=1, flags=Flag.FIN))
        await session.dispatch(session.context(), CloseStreamEvent.with_data(CloseStreamData(stream_id=1)))
        assert await read_outbound(session) == FrameData.data(stream_id=1, flags=Flag.FIN)
        assert session.take_snapshot().streams[1].lifecycle is StreamState.CLOSED

        await session.receive_frame(session.context(), FrameData.data(stream_id=1, payload=b"late"))
        await session.receive_frame(session.context(), FrameData.window_update(stream_id=1, delta=1))

        assert session.state() == "/Session/connected/open"
        assert session.take_snapshot().failed_operations[-2].failure_kind == "stream_reset"
        assert session.take_snapshot().failed_operations[-1].failure_kind == "stream_reset"
        assert await read_outbound_frames(session, 2) == (
            FrameData.data(stream_id=1, flags=Flag.RST),
            FrameData.data(stream_id=1, flags=Flag.RST),
        )

    asyncio.run(run())


def test_yamux_session_duplicate_syn_is_protocol_error() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.receive_frame(session.context(), FrameData.data(stream_id=2, flags=Flag.SYN))
        assert await read_outbound(session) == FrameData.window_update(stream_id=2, delta=0, flags=Flag.ACK)

        await session.receive_frame(session.context(), FrameData.data(stream_id=2, flags=Flag.SYN))

        assert session.state() == "/Session/errored"
        assert session.take_snapshot().failed_operations[-1].failure_kind == "protocol_error"
        assert await read_outbound_frames(session, 2) == (
            FrameData.data(stream_id=2, flags=Flag.RST),
            FrameData.go_away(code=GoAwayCode.PROTOCOL_ERROR),
        )

    asyncio.run(run())


def test_yamux_session_ping_echo_ack_correlation_and_timeout() -> None:
    async def run() -> None:
        session = Session(role="client", ping_timeout=datetime.timedelta(milliseconds=5))
        _ = await hsm.started(None, session, session.model)

        await session.dispatch(session.context(), PingEvent.with_data(PingData(opaque=42)))
        assert await read_outbound(session) == FrameData.ping(opaque=42)
        assert session.take_snapshot().outstanding_ping == 42

        await session.receive_frame(session.context(), FrameData.ping(opaque=99, ack=True))
        assert session.state() == "/Session/connected/pinging"
        assert session.take_snapshot().outstanding_ping == 42

        await session.receive_frame(session.context(), FrameData.ping(opaque=42, ack=True))
        assert session.state() == "/Session/connected/open"
        assert session.take_snapshot().outstanding_ping is None

        await session.receive_frame(session.context(), FrameData.ping(opaque=7))
        assert await read_outbound(session) == FrameData.ping(opaque=7, ack=True)

    asyncio.run(run())


def test_yamux_session_ping_timeout_clears_outstanding_ping() -> None:
    async def run() -> None:
        session = Session(role="client", ping_timeout=datetime.timedelta(milliseconds=5))
        _ = await hsm.started(None, session, session.model)

        await session.dispatch(session.context(), PingEvent.with_data(PingData(opaque=42)))
        assert await read_outbound(session) == FrameData.ping(opaque=42)

        await asyncio.sleep(0.02)

        assert session.state() == "/Session/errored"
        assert session.take_snapshot().outstanding_ping is None
        assert session.take_snapshot().failed_operations[-1].failure_kind == "ping_timeout"

    asyncio.run(run())


def test_yamux_session_can_open_stream_while_waiting_for_ping_ack() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)

        await session.dispatch(session.context(), PingEvent.with_data(PingData(opaque=42)))
        assert await read_outbound(session) == FrameData.ping(opaque=42)
        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))

        assert session.state() == "/Session/connected/pinging"
        assert session.take_snapshot().outstanding_ping == 42
        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)
        assert session.take_snapshot().streams[1].lifecycle is StreamState.LOCAL_OPENING

    asyncio.run(run())


def test_yamux_session_goaway_stops_new_streams_and_allows_active_stream_drain() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)

        await session.dispatch(session.context(), PingEvent.with_data(PingData(opaque=42)))
        assert await read_outbound(session) == FrameData.ping(opaque=42)

        await session.dispatch(session.context(), GoAwayEvent.with_data(GoAwayData(code=0)))
        assert session.state() == "/Session/disconnected"
        assert session.take_snapshot().closed_goaway_code == GoAwayCode.NORMAL
        assert session.take_snapshot().outstanding_ping is None
        assert await read_outbound(session) == FrameData.go_away(code=GoAwayCode.NORMAL)

        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))
        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)
        await session.receive_frame(session.context(), FrameData.window_update(stream_id=1, delta=0, flags=Flag.ACK))
        await session.dispatch(session.context(), GoAwayEvent.with_data(GoAwayData(code=0)))
        assert await read_outbound(session) == FrameData.go_away(code=GoAwayCode.NORMAL)

        assert session.state() == "/Session/connected/draining"
        await session.receive_frame(session.context(), FrameData.data(stream_id=2, flags=Flag.SYN))
        assert session.state() == "/Session/connected/draining"
        assert await read_outbound(session) == FrameData.data(stream_id=2, flags=Flag.RST)
        assert session.take_snapshot().failed_operations[-1].failure_kind == "session_draining"

        await session.dispatch(
            session.context(),
            SendDataEvent.with_data(SendData(stream_id=1, payload=b"finish", end_stream=False)),
        )
        assert await read_outbound(session) == FrameData.data(stream_id=1, payload=b"finish")

        await session.receive_frame(session.context(), FrameData.data(stream_id=1, flags=Flag.FIN))
        await session.dispatch(session.context(), CloseStreamEvent.with_data(CloseStreamData(stream_id=1)))
        assert session.take_snapshot().streams[1].lifecycle is StreamState.CLOSED
        assert session.state() == "/Session/disconnected"

    asyncio.run(run())


def test_yamux_session_received_goaway_routes_through_disconnect_choice() -> None:
    async def run() -> None:
        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)

        await session.dispatch(session.context(), PingEvent.with_data(PingData(opaque=43)))
        assert await read_outbound(session) == FrameData.ping(opaque=43)

        await session.receive_frame(session.context(), FrameData.go_away(code=GoAwayCode.NORMAL))
        assert session.state() == "/Session/disconnected"
        assert session.take_snapshot().closed_goaway_code == GoAwayCode.NORMAL
        assert session.take_snapshot().outstanding_ping is None

        session = Session(role="client")
        _ = await hsm.started(None, session, session.model)
        await session.dispatch(session.context(), OpenStreamEvent.with_data(OpenStreamData()))
        assert await read_outbound(session) == FrameData.window_update(stream_id=1, delta=0, flags=Flag.SYN)
        await session.receive_frame(session.context(), FrameData.window_update(stream_id=1, delta=0, flags=Flag.ACK))

        await session.receive_frame(session.context(), FrameData.go_away(code=GoAwayCode.NORMAL))
        assert session.state() == "/Session/connected/draining"
        assert session.take_snapshot().closed_goaway_code == GoAwayCode.NORMAL

        await session.receive_frame(session.context(), FrameData.data(stream_id=1, flags=Flag.FIN))
        await session.dispatch(session.context(), CloseStreamEvent.with_data(CloseStreamData(stream_id=1)))
        assert session.state() == "/Session/disconnected"

    asyncio.run(run())


def test_yamux_session_receive_frame_does_not_auto_start() -> None:
    async def run() -> None:
        session = Session(role="client")

        try:
            await session.receive_frame(session.context(), FrameData.data(stream_id=2, flags=Flag.SYN))
        except Exception:
            pass
        else:
            raise AssertionError("receive_frame must not auto-start a Yamux session.")

        assert session.take_snapshot().streams == {}

    asyncio.run(run())


def test_yamux_session_projects_frames_to_typed_events() -> None:
    event = event_from_frame(FrameData.data(stream_id=1, payload=b"x", flags=Flag.FIN))

    assert FrameData.data(stream_id=1, payload=b"x").frame_type is FrameType.DATA
    assert event.name == "protocol.yamux.frame.data.received"
    assert isinstance(event.data, ReceiveDataFrameData)
    assert event.data.stream_id == 1
    assert event.data.flags == int(Flag.FIN)
    assert event.data.payload == b"x"

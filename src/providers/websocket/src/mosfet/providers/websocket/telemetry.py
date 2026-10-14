"""The transport instruments the WebSocket server and its connections record.

The names are the transport-neutral ``bot.transport.*`` instruments every transport records, told apart
by ``transport="websocket"``. Every attribute is a closed set from code: peer ids, tokens, remote
addresses, channel names, stream ids and content are never attributes.
"""

import typing

from opentelemetry import metrics

from mosfet.telemetry import span

SCOPE = "mosfet.providers.websocket"
TRANSPORT = "websocket"

type Outcome = typing.Literal["connected", "failed", "timed_out"]
type Rejection = typing.Literal[
    "unauthenticated", "unknown_peer", "oversize", "protocol_error", "too_many_channels", "refused_channel"
]
type SendDrop = typing.Literal["backlog", "stale_connection"]

_METER = metrics.get_meter(SCOPE)
_PEERS = _METER.create_up_down_counter(
    "bot.transport.peer.active",
    unit="{peer}",
    description="Remote peers a transport holds reachable right now, by transport.",
)
_CONNECTS = _METER.create_counter(
    "bot.transport.connect.count",
    unit="{attempt}",
    description="Transport connection attempts, by transport and outcome: connected, failed, or timed_out.",
)
_REJECTED = _METER.create_counter(
    "bot.transport.rejected.count",
    unit="{message}",
    description="Connections, streams and messages a transport refused at its boundary, by transport and reason.",
)
_TRANSFERRED = _METER.create_counter(
    "bot.transport.transfer.size",
    unit="By",
    description="Bytes a transport carried, by transport and direction.",
)
_SEND_DROPPED = _METER.create_counter(
    "bot.transport.send.dropped.count",
    unit="{message}",
    description="Outbound messages a transport did not send, by transport and reason.",
)


def peer_active(delta: typing.Literal[1, -1]) -> None:
    """One peer became reachable (``1``) or stopped being reachable (``-1``)."""

    _PEERS.add(delta, {"transport": TRANSPORT})


def reject(reason: Rejection) -> None:
    """Count one connection, stream or message refused at the boundary."""

    _REJECTED.add(1, {"transport": TRANSPORT, "reason": reason})


def transferred(size: int, direction: typing.Literal["in", "out"]) -> None:
    """Count socket bytes carried in one direction."""

    _TRANSFERRED.add(size, {"transport": TRANSPORT, "direction": direction})


def send_dropped(reason: SendDrop) -> None:
    """Count one outbound message the transport did not send."""

    _SEND_DROPPED.add(1, {"transport": TRANSPORT, "reason": reason})


def connect(outcome: Outcome, *, component: str) -> None:
    """Count one connection attempt's outcome and leave a span for it; the outcome is the only attribute."""

    _CONNECTS.add(1, {"transport": TRANSPORT, "outcome": outcome})
    with span.operation(
        "transport.connect",
        scope=SCOPE,
        component=component,
        stage="connect",
        attributes={"bot.transport.name": TRANSPORT, "bot.transport.outcome": outcome},
    ) as active:
        if outcome != "connected":
            span.record_failure(active, outcome)

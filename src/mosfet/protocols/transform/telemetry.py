"""The transform instruments, recorded by whichever actor hosts a codec on a stream.

Every attribute is a closed set fixed in code: ``codec`` is the codec's own name constant, ``direction`` is
``in`` or ``out``, and ``reason`` is a literal reason. Peer ids, channel names, event names, stream ids and
message content are never attributes.
"""

import typing

from opentelemetry import metrics

from .codec import Codec, DecodeReason

SCOPE = "mosfet.protocols.transform"

type Direction = typing.Literal["in", "out"]
type DropReason = typing.Literal["unowned", "flow_control", "encode_failed"]

_METER = metrics.get_meter(SCOPE)
_MESSAGES = _METER.create_counter(
    "bot.transform.message.count",
    unit="{message}",
    description="Messages a codec host decoded (in) or encoded (out), by codec and direction.",
)
_TRANSFERRED = _METER.create_counter(
    "bot.transform.transfer.size",
    unit="By",
    description="Message bytes a codec host decoded (in) or encoded (out), by codec and direction.",
)
_DECODE_FAILED = _METER.create_counter(
    "bot.transform.decode.failed.count",
    unit="{message}",
    description="Messages a codec could not decode into an event, by codec and reason.",
)
_DROPPED = _METER.create_counter(
    "bot.transform.dropped.count",
    unit="{event}",
    description=(
        "Events a codec host did not carry, by codec and reason: no stream owned (unowned), no send window or "
        "pending room (flow_control), or the codec refused the event (encode_failed)."
    ),
)


def transferred(codec: Codec, direction: Direction, size: int) -> None:
    """Count one message of ``size`` bytes carried through ``codec``."""

    attributes = {"codec": codec.name, "direction": direction}
    _MESSAGES.add(1, attributes)
    _TRANSFERRED.add(size, attributes)


def decode_failed(codec: Codec, reason: DecodeReason) -> None:
    """Count one message ``codec`` could not decode."""

    _DECODE_FAILED.add(1, {"codec": codec.name, "reason": reason})


def dropped(codec: Codec, reason: DropReason) -> None:
    """Count one event a host did not carry through ``codec``."""

    _DROPPED.add(1, {"codec": codec.name, "reason": reason})


__all__ = ["SCOPE", "Direction", "DropReason", "decode_failed", "dropped", "transferred"]

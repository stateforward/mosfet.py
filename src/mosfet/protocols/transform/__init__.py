"""Transform codecs: a stream's bytes become typed events for the stream's owner, and back.

Whoever owns a stream is the dispatch target of what comes out of it. The actor hosting the stream applies
a plain :class:`Codec` inline: inbound messages decode to one typed event dispatched to the owner, and events
the owner sends to the stream encode to messages. :data:`OwnedEvent` and :data:`EndedEvent` tell a duplex
owner which stream to send to.
"""

from . import telemetry
from .codec import Codec, DecodeError, DecodeReason, EncodeError, EncodeReason, JsonCodec
from .events import ChannelName, EndedData, EndedEvent, OwnedData, OwnedEvent

__all__ = [
    "ChannelName",
    "Codec",
    "DecodeError",
    "DecodeReason",
    "EncodeError",
    "EncodeReason",
    "EndedData",
    "EndedEvent",
    "JsonCodec",
    "OwnedData",
    "OwnedEvent",
    "telemetry",
]

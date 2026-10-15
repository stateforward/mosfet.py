"""Stream ownership events: which stream a duplex owner may send its events to."""

import typing

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

ChannelName = typing.Annotated[
    str,
    pydantic.Field(
        pattern=r"^[a-z][a-z0-9._-]{0,63}$",
        description=(
            "Lower-case wire name of one channel, the stream a peer opens for one port. It names what the stream "
            "carries; it never names a device, a peer, or an actor."
        ),
        examples=["keyboard", "display", "microphone"],
    ),
]

_STREAM_DESCRIPTION = (
    "Runtime-only address of the stream actor. The owner dispatches the events it wants sent to this address; "
    "it is never read and never serialized."
)


class _StreamData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    stream: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]] = pydantic.Field(
        exclude=True,
        repr=False,
        description=_STREAM_DESCRIPTION,
    )
    channel: ChannelName


class OwnedData(_StreamData):
    """The recipient now owns ``stream``: what it decodes comes to the owner, and what the owner sends goes out."""


class EndedData(_StreamData):
    """The recipient no longer owns ``stream``: nothing decoded from it comes here, so stop sending to it."""

    reason: typing.Literal["finished", "reset", "closed", "handed_over"] = pydantic.Field(
        description=(
            "Why ownership ended: the peer finished the stream, the stream was reset, the stream was closed "
            "because its host stopped, or the previous owner handed it to another owner."
        ),
        examples=["finished", "reset", "closed", "handed_over"],
    )


OwnedEvent = hsm.Event[OwnedData](name="protocol.transform.owned", schema=OwnedData)
EndedEvent = hsm.Event[EndedData](name="protocol.transform.ended", schema=EndedData)

__all__ = [
    "ChannelName",
    "EndedData",
    "EndedEvent",
    "OwnedData",
    "OwnedEvent",
]

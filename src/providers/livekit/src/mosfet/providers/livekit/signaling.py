"""Call setup over LiveKit RPC: the wire a phone uses to reach one endpoint in the room.

A LiveKit room is an exchange, not a party line. Joining it is being reachable; *dialing* is
sending call setup addressed to one participant. That distinction is this module: four addressed
methods that carry a call between two phones, modelled on the SIP messages they stand in for.

======================  ==========  ===========================================================
Method                  SIP         Meaning
======================  ==========  ===========================================================
``...phone.setup``      INVITE      Caller asks one endpoint to ring. The ack means *ringing*.
``...phone.accept``     200 OK      Callee answered. The call is up.
``...phone.decline``    603         Callee refused. Somebody was there and said no.
``...phone.bye``        BYE         Either side ended a call that had been set up.
======================  ==========  ===========================================================

The caller's identity is never in a payload: LiveKit hands every handler the authenticated
``caller_identity`` of the participant that invoked it, and a self-declared caller id on the wire
would be a claim rather than a fact.

On this SFU fiction the participant identity **is** the phone number (normalized digits, the same
form ``phone.Number`` / ``DialData`` produce). Setup is addressed to that number by default.
An optional :class:`DialPlan` is only an alias layer for rare remaps and tests — not required for
normal dial-by-number operation.
"""

from __future__ import annotations

import collections.abc
import typing

import pydantic
from livekit import rtc

from mosfet.devices import phone

_NUMBER = pydantic.TypeAdapter(phone.Number)
"""The one rule for what a number is, so a plan cannot register something no handset could dial."""


SetupMethod = "bot.provider.livekit.phone.setup"
"""Caller → callee. Asks the addressed endpoint to ring; the ack means it is ringing, not connected."""

AcceptMethod = "bot.provider.livekit.phone.accept"
"""Callee → caller. The callee answered the call named in the payload."""

DeclineMethod = "bot.provider.livekit.phone.decline"
"""Callee → caller. The callee refused the call named in the payload."""

ByeMethod = "bot.provider.livekit.phone.bye"
"""Either direction. The sender ended the call named in the payload."""

Methods: typing.Final[tuple[str, ...]] = (SetupMethod, AcceptMethod, DeclineMethod, ByeMethod)
"""Every method a phone answers. A phone is both a caller and a callee, so it registers all four."""


class MessageData(pydantic.BaseModel):
    """The payload every call-setup method carries: which call this message is about.

    One field, because the method name is the verb and the call id is the only thing the two
    endpoints must agree on — the SIP Call-ID header, and nothing else. The caller mints it from
    the dial operation it is running; the callee adopts it from setup and echoes it back on
    accept, decline, and bye.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "Call-setup message body exchanged between two LiveKit phone services.",
            "examples": [{"call_id": "livekit:01J8Z2K7Q9"}],
        },
    )

    call_id: str = pydantic.Field(
        min_length=1,
        description=(
            "Provider-neutral handle for the call this message is about. Minted by the caller "
            "from the dial operation it is running, and echoed back unchanged by the callee."
        ),
        examples=["livekit:01J8Z2K7Q9"],
    )


class DialPlan(typing.Protocol):
    """Optional alias map from a dialled number to a LiveKit participant identity.

    Default dialing does not need a plan: identity is the normalized number. Implement this only
    when a number must resolve to a different identity (tests, temporary aliases). Returning
    None means a wrong number for that map — setup is not sent.
    """

    def endpoint(self, number: str) -> str | None:
        """Participant identity for ``number``, or None when this plan does not route it."""
        ...


class MappingDialPlan:
    """Optional ``number -> participant identity`` alias table.

    Default operation uses identity = number and needs no plan. Use this when tests or rare
    provisioning remap a dialled number onto a different participant identity. Numbers are checked
    against the same rule a dialled number is, so a plan cannot promise to route something no
    keypad can produce.
    """

    _endpoints: dict[str, str]

    def __init__(self, entries: collections.abc.Mapping[str, str]) -> None:
        self._endpoints = {}
        for number, identity in entries.items():
            if not identity:
                raise ValueError("A dial plan entry must name the participant identity that answers the number.")
            self._endpoints[_NUMBER.validate_python(number)] = identity

    def endpoint(self, number: str) -> str | None:
        return self._endpoints.get(number)


def failure_kind(error: rtc.RpcError) -> phone.FailureKind:
    """Normalize one LiveKit RPC failure into the phone contract's failure vocabulary.

    ``RECIPIENT_NOT_FOUND`` and ``UNSUPPORTED_METHOD`` both mean the address did not reach a
    phone — either nobody is there, or what is there does not answer calls. Everything else is a
    message that failed in transit, which is a signalling failure and not a statement about the
    far end.
    """

    if error.code in {
        rtc.RpcError.ErrorCode.RECIPIENT_NOT_FOUND,
        rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD,
        rtc.RpcError.ErrorCode.UNSUPPORTED_SERVER,
        rtc.RpcError.ErrorCode.UNSUPPORTED_VERSION,
        rtc.RpcError.ErrorCode.RECIPIENT_DISCONNECTED,
    }:
        return "remote_unavailable"
    return "signaling_failed"


def invocation(data: object) -> tuple[str, MessageData]:
    """Read one RPC invocation as ``(caller identity, message)``.

    Raises :class:`livekit.rtc.RpcError` when the invocation is not a call-setup message, so the
    sender learns its message was rejected instead of the handler failing silently.
    """

    caller_identity = getattr(data, "caller_identity", None)
    payload = getattr(data, "payload", None)
    if not isinstance(caller_identity, str) or not caller_identity or not isinstance(payload, str):
        raise rtc.RpcError(
            rtc.RpcError.ErrorCode.APPLICATION_ERROR,
            "LiveKit phone call setup requires an identified caller and a JSON message payload.",
        )
    try:
        message = MessageData.model_validate_json(payload)
    except pydantic.ValidationError as error:
        raise rtc.RpcError(
            rtc.RpcError.ErrorCode.APPLICATION_ERROR,
            f"LiveKit phone call setup payload is not a valid call message: {error}",
        ) from error
    return caller_identity, message


__all__ = [
    "AcceptMethod",
    "ByeMethod",
    "DeclineMethod",
    "DialPlan",
    "MappingDialPlan",
    "MessageData",
    "Methods",
    "SetupMethod",
    "failure_kind",
    "invocation",
]

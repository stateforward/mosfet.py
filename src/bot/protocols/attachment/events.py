"""Provider-neutral event protocol for actor attachment lifecycles."""

import datetime
import enum
import typing

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema


_ACTOR_DESCRIPTION = (
    "Stable HSM actor requesting an attachment relationship with the event recipient. The recipient owns the "
    "relationship semantics, such as exclusive ability ownership or device membership."
)
_ACTOR_JSON_SCHEMA = {
    "type": "object",
    "description": _ACTOR_DESCRIPTION,
    "properties": {
        "id": {
            "type": "string",
            "description": "Stable identifier for the actor requesting the attachment relationship.",
        }
    },
    "required": ["id"],
    "examples": [{"id": "bot-body"}, {"id": "cognition"}],
}


class _DeserializedActor(hsm.Instance):
    """Typed placeholder for an attachment actor decoded from JSON.

    Owns its ``id`` as a declared field set in its own ``__init__`` (no ``setattr`` workaround).
    Unstarted, so ``hsm.id`` cannot address it; :func:`_actor_to_schema` and peer
    ``_actor_id`` helpers read the declared field through normal attribute access.
    """

    id: str

    def __init__(self, actor_id: str) -> None:
        super().__init__()
        self.id = actor_id


def _actor_from_schema(data: object) -> hsm.Instance:
    if isinstance(data, hsm.Instance):
        return data
    if isinstance(data, dict):
        values = typing.cast(dict[str, object], data)
        identifier = values.get("id")
        if not isinstance(identifier, str):
            raise ValueError("Attachment actor JSON data requires a string id.")
        return _DeserializedActor(identifier)
    raise TypeError("Attachment actor must be an hsm.Instance or an object containing a string id.")


def _actor_to_schema(actor: hsm.Instance) -> dict[str, str]:
    return {"id": hsm.id(actor)}


Actor = typing.Annotated[
    hsm.Instance,
    pydantic.BeforeValidator(_actor_from_schema),
    pydantic.PlainSerializer(_actor_to_schema, return_type=dict[str, str], when_used="json"),
    pydantic.WithJsonSchema(_ACTOR_JSON_SCHEMA),
]


class FailureKind(enum.StrEnum):
    """Stable attachment failure classifications suitable for branching and telemetry."""

    CONFLICT = "conflict"
    INITIALIZATION = "initialization"
    DISPATCH = "dispatch"
    ROLLBACK = "rollback"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class _RequestData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    actor: Actor = pydantic.Field(
        description=_ACTOR_DESCRIPTION,
        examples=[{"id": "bot-body"}, {"id": "cognition"}],
    )
    reply_to: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance] | None] = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
        description=(
            "Optional runtime-only actor receiving the correlated outcome. When omitted, the requesting actor "
            "receives the outcome directly."
        ),
    )


class AttachData(_RequestData):
    """Request that the recipient establish its domain-specific relationship with ``actor``."""

    timeout: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(seconds=30),
        gt=datetime.timedelta(0),
        description="Maximum time the recipient may remain in its attaching state before rolling back.",
        examples=[30],
    )


class DetachData(_RequestData):
    """Request that the recipient remove its domain-specific relationship with ``actor``."""

    timeout: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(seconds=30),
        gt=datetime.timedelta(0),
        description="Maximum time the recipient may remain in its detaching state before recovery begins.",
        examples=[30],
    )


class AttachCompleteData(pydantic.BaseModel):
    """Successful attachment outcome correlated to an ``AttachEvent``."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    actor: Actor = pydantic.Field(description=_ACTOR_DESCRIPTION)
    created: bool = pydantic.Field(
        description="True when this operation created the relationship; false when it was already established.",
        examples=[True, False],
    )
    reply_to: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance] | None] = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
        description="Runtime-only recipient used while the completion event crosses the attaching state boundary.",
    )


class DetachedData(pydantic.BaseModel):
    """Successful detachment outcome correlated to a ``DetachEvent``."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    actor: Actor = pydantic.Field(description=_ACTOR_DESCRIPTION)
    removed: bool = pydantic.Field(
        description="True when this operation removed the relationship; false when it was already absent.",
        examples=[True, False],
    )


class FailedData(pydantic.BaseModel):
    """Correlated attachment or detachment failure."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    actor: Actor = pydantic.Field(description=_ACTOR_DESCRIPTION)
    kind: FailureKind = pydantic.Field(
        description="Normalized failure classification for deterministic handling.",
        examples=[FailureKind.CONFLICT],
    )
    message: str = pydantic.Field(
        min_length=1,
        description="Human-readable explanation of why the lifecycle operation failed.",
        examples=["Ability is already attached to another owner."],
    )


AttachEvent = hsm.Event[AttachData](name="attachment.attach", schema=AttachData)
AttachCompleteEvent = hsm.Event[AttachCompleteData](
    name="attachment.attach.complete",
    kind=hsm.CompletionEventKind,
    schema=AttachCompleteData,
)
AttachFailedEvent = hsm.Event[FailedData](
    name="attachment.attach.failed",
    kind=hsm.ErrorEventKind,
    schema=FailedData,
)
DetachEvent = hsm.Event[DetachData](name="attachment.detach", schema=DetachData)
DetachedEvent = hsm.Event[DetachedData](
    name="attachment.detached",
    kind=hsm.CompletionEventKind,
    schema=DetachedData,
)
DetachFailedEvent = hsm.Event[FailedData](
    name="attachment.detach.failed",
    kind=hsm.ErrorEventKind,
    schema=FailedData,
)


__all__ = [
    "Actor",
    "AttachData",
    "AttachEvent",
    "AttachCompleteData",
    "AttachCompleteEvent",
    "AttachFailedEvent",
    "DetachData",
    "DetachEvent",
    "DetachedData",
    "DetachedEvent",
    "DetachFailedEvent",
    "FailedData",
    "FailureKind",
]

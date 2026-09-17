"""Public attachment protocol, composable group, and lifecycle events."""

from .attachment import Attachment
from .group import Group
from .events import (
    Actor,
    AttachCompleteData,
    AttachCompleteEvent,
    AttachData,
    AttachEvent,
    AttachFailedEvent,
    DetachData,
    DetachEvent,
    DetachedData,
    DetachedEvent,
    DetachFailedEvent,
    FailedData,
    FailureKind,
)

__all__ = [
    "Actor",
    "Attachment",
    "AttachCompleteData",
    "AttachCompleteEvent",
    "AttachData",
    "AttachEvent",
    "AttachFailedEvent",
    "DetachData",
    "DetachEvent",
    "DetachedData",
    "DetachedEvent",
    "DetachFailedEvent",
    "FailedData",
    "FailureKind",
    "Group",
]

"""People: the bot's capacity to know who people are, which it learns itself.

The bot links the refs it perceives (voices, faces, other identities) to persons, names them, unlinks
them and forgets them; memory holds who is who and the registries hold the prints. Nobody lists the
household. Every choice among these tools is the bot's.

    from mosfet.abilities import memory, people

    people.People(memory=store, registries=(voices,))
    people.LinkEvent
"""

from . import people
from .people import (
    ChangedData,
    ChangedEvent,
    FailedEvent,
    FailureData,
    FailureKind,
    ForgetData,
    ForgetEvent,
    LinkData,
    LinkEvent,
    NameData,
    NameEvent,
    Op,
    People,
    UnlinkData,
    UnlinkEvent,
)

__all__ = [
    "ChangedData",
    "ChangedEvent",
    "FailedEvent",
    "FailureData",
    "FailureKind",
    "ForgetData",
    "ForgetEvent",
    "LinkData",
    "LinkEvent",
    "NameData",
    "NameEvent",
    "Op",
    "People",
    "UnlinkData",
    "UnlinkEvent",
    "people",
]

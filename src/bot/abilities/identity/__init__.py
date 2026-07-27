"""Identity: a bot's sense of who it is.

A bot starts with no name. Someone in its environment says one, judgment may select
``identity.AdoptEvent``, and only then can the bot tell it is being addressed.

    from bot.abilities import identity

    identity.Identity(recognizer=..., memory=...)
    identity.AdoptEvent
"""

from . import identity, name, recognition
from .identity import AddressedData, AddressedEvent, AdoptData, AdoptEvent, Identity
from .name import NAME_QUERY, NAME_SUBJECT, Name, adopted_name, name_insert_input, name_select_input, names_from_output
from .recognition import NameRecognizer, SpeechNameRecognizer, hears_name

__all__ = [
    "NAME_QUERY",
    "NAME_SUBJECT",
    "AddressedData",
    "AddressedEvent",
    "AdoptData",
    "AdoptEvent",
    "Identity",
    "Name",
    "NameRecognizer",
    "SpeechNameRecognizer",
    "adopted_name",
    "hears_name",
    "identity",
    "name",
    "name_insert_input",
    "name_select_input",
    "names_from_output",
    "recognition",
]

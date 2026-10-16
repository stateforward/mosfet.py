"""Identity: a bot's sense of who it is.

A bot starts with no name. Someone in its environment says one, cognition may select
``identity.AdoptEvent``, and only then can the bot tell it is being addressed.

    from mosfet.abilities import identity

    identity.Identity(recognizer=..., memory=...)
    identity.AdoptEvent
"""

from . import identity, name, recognition, value
from .identity import AddressedData, AddressedEvent, AdoptData, AdoptEvent, Identity
from .name import NAME_QUERY, Name, adopted_name, name_insert_input, name_select_input, names_from_output
from .recognition import NameRecognizer, SpeechNameRecognizer, hears_name
from .value import (
    IdentitySet,
    IdentityRef,
    IdentityValue,
    identities_json,
    identity_groups,
    normalize_identity,
    normalize_identity_set,
    sorted_identities,
)

__all__ = [
    "NAME_QUERY",
    "AddressedData",
    "AddressedEvent",
    "AdoptData",
    "AdoptEvent",
    "Identity",
    "IdentitySet",
    "IdentityRef",
    "IdentityValue",
    "Name",
    "NameRecognizer",
    "SpeechNameRecognizer",
    "adopted_name",
    "hears_name",
    "identities_json",
    "identity",
    "identity_groups",
    "name",
    "name_insert_input",
    "name_select_input",
    "names_from_output",
    "normalize_identity",
    "normalize_identity_set",
    "recognition",
    "sorted_identities",
    "value",
]

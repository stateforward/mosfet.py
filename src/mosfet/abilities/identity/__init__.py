"""Identity: a bot's sense of who it is.

A bot starts with no name. Someone in its environment says one, cognition may select
``identity.AdoptEvent``, and only then can the bot tell it is being addressed.

    from mosfet.abilities import identity

    identity.Identity(recognizer=..., memory=...)
    identity.AdoptEvent
"""

from . import identity, name, recognition, value
from .identity import AddressedData, AddressedEvent, AdoptData, AdoptEvent, Identity
from .name import NAME_QUERY, NAME_SUBJECT, Name, adopted_name, name_insert_input, name_select_input, names_from_output
from .recognition import NameRecognizer, SpeechNameRecognizer, hears_name
from .value import (
    Embedding,
    IdentitySet,
    IdentityRef,
    IdentityValue,
    cosine_similarity,
    identities_json,
    identity_groups,
    identity_json_value,
    identity_sort_key,
    match_vector,
    normalize_embedding,
    normalize_identity,
    normalize_identity_set,
    sorted_identities,
)

__all__ = [
    "NAME_QUERY",
    "NAME_SUBJECT",
    "AddressedData",
    "AddressedEvent",
    "AdoptData",
    "AdoptEvent",
    "Embedding",
    "Identity",
    "IdentitySet",
    "IdentityRef",
    "IdentityValue",
    "Name",
    "NameRecognizer",
    "SpeechNameRecognizer",
    "adopted_name",
    "cosine_similarity",
    "hears_name",
    "identities_json",
    "identity",
    "identity_groups",
    "identity_json_value",
    "name",
    "name_insert_input",
    "name_select_input",
    "names_from_output",
    "match_vector",
    "identity_sort_key",
    "normalize_identity",
    "normalize_identity_set",
    "normalize_embedding",
    "recognition",
    "sorted_identities",
    "value",
]

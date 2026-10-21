"""Identity: a bot's sense of who it is.

A bot starts with no name. Someone in its environment says one, cognition may select
``identity.AdoptEvent``, and only then can the bot tell it is being addressed.

    from mosfet.abilities import identity

    identity.Identity(recognizer=..., memory=...)
    identity.AdoptEvent

Identity tells the name the bot answers to (``identity.NamedEvent``) to whoever asks
(``identity.AskEvent``) and to whoever adopted it, so cognition knows its own name.

``Identity`` and its events (module ``identity``) are exported lazily: Identity is an input ability whose
product is a cognition input, and Cognition knows the bot's name by these name schemas.
"""

import importlib
import typing

from . import name, recognition, value
from .name import (
    NAME_QUERY,
    AskData,
    AskEvent,
    Name,
    NameData,
    NamedEvent,
    adopted_name,
    name_insert_input,
    name_select_input,
    names_from_output,
)
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

if typing.TYPE_CHECKING:
    from . import identity
    from .identity import AddressedData, AddressedEvent, AdoptData, AdoptEvent, Identity

_LAZY_EXPORTS = frozenset({"AddressedData", "AddressedEvent", "AdoptData", "AdoptEvent", "Identity", "identity"})

__all__ = [
    "NAME_QUERY",
    "AddressedData",
    "AddressedEvent",
    "AdoptData",
    "AdoptEvent",
    "AskData",
    "AskEvent",
    "Identity",
    "IdentitySet",
    "IdentityRef",
    "IdentityValue",
    "Name",
    "NameData",
    "NamedEvent",
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


def __getattr__(attribute: str) -> object:
    if attribute not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {attribute!r}")
    module = importlib.import_module(".identity", __name__)
    found = module if attribute == "identity" else typing.cast(object, getattr(module, attribute))
    globals()[attribute] = found
    return found


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

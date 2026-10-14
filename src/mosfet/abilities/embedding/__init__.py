"""A bot's provider-neutral capacity to embed text and images into one declared vector space.

Usage::

    from mosfet.abilities import embedding
    from mosfet.abilities.language import text

    ability = embedding.Embedding(embedder=provider_embedder)
    request = embedding.InputData(
        role=embedding.Role.QUERY,
        items=(embedding.Item(parts=(text.TextPart(text="a red circle"),)),),
    )

``Embedding`` answers each ``InputEvent`` with one ``OutputEvent`` (one unit vector per item in
the embedder's ``Space``) or one ``FailedEvent`` with a closed ``FailureKind``. Providers implement
``Embedder``; ``Instructions`` holds a model's task prefixes.
"""

from .embedder import EMBEDDING_GEMMA_2, Capacity, Embedder, Error, Instructions
from .embedding import Embedding, FailedEvent, InputEvent, OutputEvent
from .payload import (
    MAX_DIMENSIONS,
    MAX_ITEMS,
    MAX_PARTS,
    UNIT_NORM_TOLERANCE,
    FailureData,
    FailureKind,
    InputData,
    Item,
    OutputData,
    Role,
    Space,
    Vector,
)

__all__ = [
    "EMBEDDING_GEMMA_2",
    "MAX_DIMENSIONS",
    "MAX_ITEMS",
    "MAX_PARTS",
    "UNIT_NORM_TOLERANCE",
    "Capacity",
    "Embedder",
    "Embedding",
    "Error",
    "FailedEvent",
    "FailureData",
    "FailureKind",
    "InputData",
    "InputEvent",
    "Instructions",
    "Item",
    "OutputData",
    "OutputEvent",
    "Role",
    "Space",
    "Vector",
]

"""GGUF weight files for one llama.cpp model."""

from __future__ import annotations

import dataclasses
import pathlib


@dataclasses.dataclass(frozen=True, kw_only=True)
class Weights:
    """The GGUF files one llama.cpp model loads; paths are explicit configuration and nothing is downloaded.

    Example: ``Weights(model=Path("embeddinggemma-2-Q8_0.gguf"), projector=Path("mmproj-Q8_0.gguf"))``.
    """

    model: pathlib.Path
    """The model GGUF."""
    projector: pathlib.Path | None = None
    """The multimodal projector (mmproj) GGUF that lets the model see images; ``None`` means text-only."""

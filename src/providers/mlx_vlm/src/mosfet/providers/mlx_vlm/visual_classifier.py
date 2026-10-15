from __future__ import annotations

import mosfet.media as media
from mosfet.abilities import vision

import dataclasses
import typing


@dataclasses.dataclass(frozen=True, kw_only=True)
class VisualClassifier(vision.VisualClassifier):
    """Classifier that routes reading input by its typed content before MLX VLM image decoding."""

    text_confidence: float | None = 1.0
    image_confidence: float | None = 1.0
    unreadable_confidence: float | None = 1.0

    @typing.override
    async def classify(self, input: vision.InputData) -> vision.OutputData:
        if isinstance(input.content, media.Image):
            return vision.OutputData(kind="image", confidence=self.image_confidence)
        if input.content.strip():
            return vision.OutputData(kind="text", confidence=self.text_confidence)
        return vision.OutputData(kind="unreadable", confidence=self.unreadable_confidence)


__all__ = ["VisualClassifier"]

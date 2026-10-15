from __future__ import annotations

import mosfet.media as media
from mosfet.abilities import vision

import asyncio
import collections.abc

from mosfet.providers.mlx_vlm import VisualClassifier


PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"image bytes", media_type="image/png")


async def await_visual_classification(
    output: collections.abc.Awaitable[vision.OutputData],
) -> vision.OutputData:
    return await output


def test_visual_classifier_routes_text_input() -> None:
    classifier = VisualClassifier(text_confidence=0.99)

    output = asyncio.run(
        await_visual_classification(
            classifier.classify(vision.InputData(content="read this")),
        )
    )

    assert output == vision.OutputData(kind="text", confidence=0.99)
    assert isinstance(classifier, vision.VisualClassifier)


def test_visual_classifier_routes_image_input() -> None:
    classifier = VisualClassifier(image_confidence=0.87)

    output = asyncio.run(
        await_visual_classification(
            classifier.classify(vision.InputData(content=PNG)),
        )
    )

    assert output == vision.OutputData(kind="image", confidence=0.87)


def test_visual_classifier_marks_blank_text_unreadable() -> None:
    classifier = VisualClassifier(unreadable_confidence=0.42)

    output = asyncio.run(await_visual_classification(classifier.classify(vision.InputData(content="   "))))

    assert output == vision.OutputData(kind="unreadable", confidence=0.42)


def test_visual_classifier_is_awaitable() -> None:
    classifier = VisualClassifier()

    output = classifier.classify(vision.InputData(content="read this"))

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output) == vision.OutputData(kind="text", confidence=1.0)

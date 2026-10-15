"""Live check of the generative vision stages over ``openai_compat.TextGenerator`` on a served VLM.

``vision.description.GenerativeDescriber`` and ``vision.detection.GenerativeDetector`` work over any
image-capable text generator. Point this test at an OpenAI-compatible endpoint that serves a
vision-language model (for example Qwen3.6 on vLLM)::

    MOSFET_LIVE_VLM_BASE_URL=http://spark.local:8000/v1 \\
    MOSFET_LIVE_VLM_MODEL=qwen3.6 \\
      uv run --package mosfet-provider-openai-compat --group dev python -m pytest \\
      src/providers/openai_compat/tests/test_vision_live.py -m live -v -s

``MOSFET_LIVE_VLM_API_KEY`` is passed when set. What the model says is printed, never pinned.
"""

from __future__ import annotations

import asyncio
import os
import struct
import zlib

import pytest

import mosfet.media as media
import mosfet.providers.openai_compat as openai_compat
from mosfet.abilities import vision

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("MOSFET_LIVE_VLM_BASE_URL") or not os.environ.get("MOSFET_LIVE_VLM_MODEL"),
        reason="set MOSFET_LIVE_VLM_BASE_URL and MOSFET_LIVE_VLM_MODEL to a served vision-language model",
    ),
]


def _red_circle_png(size: int = 128) -> media.Image:
    """A red circle on white, encoded as a PNG without an imaging dependency."""

    center = (size - 1) / 2
    rows = b"".join(
        b"\x00"
        + b"".join(
            b"\xff\x00\x00" if (x - center) ** 2 + (y - center) ** 2 <= (size / 3) ** 2 else b"\xff\xff\xff"
            for x in range(size)
        )
        for y in range(size)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    content = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    return media.Image(content=content, media_type="image/png", width_px=size, height_px=size)


def _generator() -> openai_compat.TextGenerator:
    return openai_compat.TextGenerator(
        client=openai_compat.ChatClient(
            base_url=os.environ["MOSFET_LIVE_VLM_BASE_URL"],
            model=os.environ["MOSFET_LIVE_VLM_MODEL"],
            api_key=os.environ.get("MOSFET_LIVE_VLM_API_KEY"),
        ),
        image_capacity=media.ImageCapacity(
            max_images=1,
            max_total_bytes=8 * 1024 * 1024,
            media_types=frozenset({"image/png", "image/jpeg", "image/webp"}),
        ),
    )


def test_generative_describer_describes_a_served_image() -> None:
    describer = vision.description.GenerativeDescriber(generator=_generator())

    said = asyncio.run(describer.decode(_red_circle_png()))

    print(f"description: {said!r}")
    assert said.strip()
    assert len(said) <= vision.description.MAX_DESCRIPTION_CHARS


def test_generative_detector_answers_typed_objects() -> None:
    detector = vision.detection.GenerativeDetector(generator=_generator())

    found = asyncio.run(detector.classify(vision.detection.InputData(image=_red_circle_png())))

    print(f"objects: {[(item.label, item.confidence, item.box) for item in found.objects]}")
    assert isinstance(found, vision.detection.OutputData)

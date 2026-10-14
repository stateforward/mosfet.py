"""Live check of ``openai_compat.Embedder`` against a served multimodal ``/v1/embeddings`` endpoint.

Serve EmbeddingGemma-2 with llama-server (at or after llama.cpp ``4fbc76de``) and point the test at it::

    BOT_OPENAI_COMPAT_EMBEDDINGS_BASE_URL=http://spark.local:8080/v1 \\
    BOT_OPENAI_COMPAT_EMBEDDINGS_MODEL=embeddinggemma-2 \\
      uv run --package mosfet-provider-openai-compat python -m pytest \\
      src/providers/openai_compat/tests/test_embeddings_live.py -m live -v
"""

from __future__ import annotations

import asyncio
import os
import struct
import zlib

import hsm
import pytest

import mosfet.providers.openai_compat as openai_compat
import mosfet.media as media
from mosfet.abilities import embedding
from mosfet.abilities.language import text

pytestmark = pytest.mark.live


def _red_circle_png(size: int = 64) -> bytes:
    """Encode a red circle on white as a PNG without an imaging dependency."""

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
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def _dot(left: embedding.Vector, right: embedding.Vector) -> float:
    return sum(a * b for a, b in zip(left.values, right.values, strict=True))


def test_served_model_returns_unit_vectors_and_aligns_image_with_caption() -> None:
    base_url = os.environ.get("BOT_OPENAI_COMPAT_EMBEDDINGS_BASE_URL")
    model = os.environ.get("BOT_OPENAI_COMPAT_EMBEDDINGS_MODEL")
    if not base_url or not model:
        pytest.skip("BOT_OPENAI_COMPAT_EMBEDDINGS_BASE_URL / BOT_OPENAI_COMPAT_EMBEDDINGS_MODEL not set")
    space = embedding.Space(model="google/embeddinggemma-2", dimensions=768)
    embedder = openai_compat.Embedder(
        client=openai_compat.EmbeddingClient(model=model, base_url=base_url),
        capacity=embedding.Capacity(
            space=space,
            max_items=8,
            max_input_tokens=None,
            images=media.ImageCapacity(max_images=1, max_total_bytes=1024 * 1024, media_types=frozenset({"image/png"})),
        ),
        instructions=embedding.EMBEDDING_GEMMA_2,
    )
    request = embedding.InputData(
        role=embedding.Role.DOCUMENT,
        items=(
            embedding.Item(
                parts=(text.ImagePart(image=media.Image(content=_red_circle_png(), media_type="image/png")),)
            ),
            embedding.Item(parts=(text.TextPart(text="a red circle"),)),
            embedding.Item(parts=(text.TextPart(text="a blue square"),)),
        ),
    )

    output = asyncio.run(embedder.embed(request, ctx=hsm.Context()))

    assert output.space == space
    assert [len(vector.values) for vector in output.vectors] == [768, 768, 768]
    image, red, blue = output.vectors
    assert _dot(image, red) > _dot(image, blue)

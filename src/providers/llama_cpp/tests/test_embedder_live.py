"""Live parity of the in-process embedder with the measured spike, on the installed pinned native build.

Needs ``BOT_LLAMA_CPP_EMBEDDING_MODEL`` and ``BOT_LLAMA_CPP_EMBEDDING_PROJECTOR`` (EmbeddingGemma-2 GGUF and its
mmproj). One model load serves every check. The spike (UD-Q4_K_XL + mmproj-Q8_0, 280 image tokens, Metal) measured
image→caption cosines red 0.7612 / blue 0.6149 / dog 0.5136 for the red circle and 0.6193 / 0.7777 / 0.5310 for the
blue square, 289 image tokens per image, and 0.7896 for the interleaved item against the red caption.
"""

from __future__ import annotations

import asyncio
import datetime
import math
import os
import pathlib
import struct
import sys
import typing
import zlib

import hsm
import pytest

import mosfet.media as media
from mosfet.abilities import ability, embedding
from mosfet.abilities.language import text
from mosfet.environment import Environment
from mosfet.protocols import attachment
from mosfet.providers import llama_cpp
from mosfet.start import started

from tests import metric_reader
from tests.llama_cpp_fakes import Recorder

pytestmark = pytest.mark.live

_MODEL = os.environ.get("BOT_LLAMA_CPP_EMBEDDING_MODEL")
_PROJECTOR = os.environ.get("BOT_LLAMA_CPP_EMBEDDING_PROJECTOR")
_SPACE = embedding.Space(model="google/embeddinggemma-2", dimensions=768)
_TIMEOUT = datetime.timedelta(seconds=60)
_TOLERANCE = 0.02
"""Cosine tolerance against the spike: the test draws its shapes without anti-aliasing and parses no special tokens."""


def _png(shape: typing.Literal["circle", "square"], color: tuple[int, int, int]) -> media.Image:
    """A 448×448 RGB PNG of a filled shape on white, drawn like the spike's (stdlib only)."""

    size = 448
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            if shape == "circle":
                inside = (x - 224) ** 2 + (y - 224) ** 2 <= 150**2
            else:
                inside = 84 <= x <= 364 and 84 <= y <= 364
            rows.extend(color if inside else (255, 255, 255))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    content = (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b"")
    )
    return media.Image(content=content, media_type="image/png", width_px=size, height_px=size)


def _cosine(left: embedding.Vector, right: embedding.Vector) -> float:
    return math.fsum(a * b for a, b in zip(left.values, right.values, strict=True))


def _item(*parts: text.ContentPart) -> embedding.Item:
    return embedding.Item(parts=parts)


@pytest.mark.skipif(not (_MODEL and _PROJECTOR), reason="BOT_LLAMA_CPP_EMBEDDING_MODEL/_PROJECTOR are not set")
def test_live_embeddings_match_the_spike(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _MODEL is not None and _PROJECTOR is not None
    weights = llama_cpp.Weights(model=pathlib.Path(_MODEL), projector=pathlib.Path(_PROJECTOR))
    context = llama_cpp.EmbeddingContext()
    red, blue = _png("circle", (220, 20, 20)), _png("square", (20, 40, 220))
    captions = {
        "red": "a red circle on a white background",
        "blue": "a blue square on a white background",
        "dog": "a golden retriever dog running through snow",
    }

    async def run(reader: metric_reader.Reader) -> dict[str, typing.Any]:
        ctx = Environment()
        owner = Recorder()
        _ = await started(ctx, owner, Recorder.model)
        model = llama_cpp.EmbeddingModel(weights=weights, context=context, require_accelerator=sys.platform == "darwin")
        _ = await started(ctx, model, llama_cpp.EmbeddingModel.model)
        embedder = llama_cpp.Embedder(
            model=model,
            capacity=llama_cpp.embedding_capacity(weights, context, space=_SPACE),
            instructions=embedding.EMBEDDING_GEMMA_2,
        )
        owner_ability = embedding.Embedding(embedder=embedder, timeout=_TIMEOUT)
        _ = await owner_ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=owner, timeout=_TIMEOUT), "attach"),
        )
        await owner.until(lambda: bool(owner.named(attachment.AttachCompleteEvent, attachment.AttachFailedEvent)))
        assert owner.events[-1].name == attachment.AttachCompleteEvent.name, owner.events[-1].data

        async def embed(role: embedding.Role, *items: embedding.Item) -> hsm.Event[typing.Any]:
            return await ability.run_terminal_operation(
                ctx,
                child=owner_ability,
                request=embedding.InputEvent.with_data_and_id(
                    embedding.InputData(role=role, items=items), os.urandom(8).hex()
                ),
                terminals=(embedding.OutputEvent, embedding.FailedEvent),
                timeout=_TIMEOUT,
            )

        def vectors(event: hsm.Event[typing.Any]) -> tuple[embedding.Vector, ...]:
            data = event.data
            assert isinstance(data, embedding.OutputData), data
            assert data.space == _SPACE
            return data.vectors

        def image_tokens() -> float:
            points = reader.points("bot.provider.llama_cpp.tokens.count")
            return sum(point.value for point in points if point.attributes.get("bot.llama_cpp.token.kind") == "image")

        try:
            before = image_tokens()
            (red_image,) = vectors(await embed(embedding.Role.DOCUMENT, _item(text.ImagePart(image=red))))
            tokens = image_tokens() - before
            images = vectors(
                await embed(
                    embedding.Role.DOCUMENT,
                    _item(text.ImagePart(image=red)),
                    _item(text.ImagePart(image=blue)),
                )
            )
            texts = dict(
                zip(
                    captions,
                    vectors(
                        await embed(
                            embedding.Role.QUERY, *(_item(text.TextPart(text=value)) for value in captions.values())
                        )
                    ),
                    strict=True,
                )
            )
            (mixed,) = vectors(
                await embed(
                    embedding.Role.DOCUMENT,
                    _item(
                        text.TextPart(text="product photo "), text.ImagePart(image=red), text.TextPart(text=" red logo")
                    ),
                )
            )
            too_long = (await embed(embedding.Role.DOCUMENT, _item(text.TextPart(text="word " * 3000)))).data
            marker = (await embed(embedding.Role.QUERY, _item(text.TextPart(text="look <__media__> here")))).data
            return {
                "repeat": _cosine(red_image, images[0]),
                "tokens": tokens,
                "red": {name: _cosine(images[0], vector) for name, vector in texts.items()},
                "blue": {name: _cosine(images[1], vector) for name, vector in texts.items()},
                "mixed": _cosine(mixed, texts["red"]),
                "norms": [math.sqrt(math.fsum(v * v for v in vector.values)) for vector in (*images, *texts.values())],
                "too_long": too_long,
                "marker": marker,
            }
        finally:
            await owner_ability.stop(owner_ability.context())
            await model.stop(ctx)

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        result = asyncio.run(run(reader))

    print({key: value for key, value in result.items() if key not in ("too_long", "marker")})
    assert result["repeat"] == pytest.approx(1.0, abs=1e-5)
    assert result["tokens"] == 289
    assert all(norm == pytest.approx(1.0, abs=1e-6) for norm in result["norms"])
    assert result["red"] == pytest.approx({"red": 0.7612, "blue": 0.6149, "dog": 0.5136}, abs=_TOLERANCE)
    assert result["blue"] == pytest.approx({"red": 0.6193, "blue": 0.7777, "dog": 0.5310}, abs=_TOLERANCE)
    assert result["red"]["red"] > result["red"]["blue"] > result["red"]["dog"]
    assert result["blue"]["blue"] > result["blue"]["red"] > result["blue"]["dog"]
    assert result["mixed"] == pytest.approx(0.7896, abs=_TOLERANCE)
    too_long = result["too_long"]
    assert isinstance(too_long, embedding.FailureData) and too_long.kind is embedding.FailureKind.TOO_LONG
    assert too_long.item_index == 0
    marker = result["marker"]
    assert isinstance(marker, embedding.FailureData) and marker.kind is embedding.FailureKind.INVALID_INPUT
    if sys.platform == "darwin":
        assert llama_cpp.Library().accelerator() is llama_cpp.Accelerator.METAL

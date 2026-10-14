"""``llama_cpp.Embedder`` and ``embedding_capacity``: a non-actor embedder over a shared, root-started model."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import pathlib
import typing

import hsm
import pytest

import mosfet.media as media
from mosfet.abilities import ability, embedding
from mosfet.abilities.language import text
from mosfet.environment import Environment
from mosfet.start import started
from mosfet.protocols import attachment
from mosfet.providers import llama_cpp

from tests import llama_cpp_fakes as fakes
from tests.llama_cpp_fakes import Recorder

SPACE = embedding.Space(model="google/embeddinggemma-2", dimensions=2)
TEXT_ONLY = llama_cpp.Weights(model=pathlib.Path("embedding.gguf"))
PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"png-body", media_type="image/png")
TIMEOUT = datetime.timedelta(seconds=2)


def test_capacity_with_a_projector_takes_png_and_jpeg_interleaved() -> None:
    capacity = llama_cpp.embedding_capacity(fakes.WEIGHTS, llama_cpp.EmbeddingContext(tokens=1024), space=SPACE)

    assert capacity == embedding.Capacity(
        space=SPACE,
        max_items=32,
        max_input_tokens=1024,
        images=media.ImageCapacity(
            max_images=8, max_total_bytes=32 * 1024 * 1024, media_types=frozenset({"image/png", "image/jpeg"})
        ),
        interleaved=True,
    )


def test_capacity_without_a_projector_is_text_only() -> None:
    capacity = llama_cpp.embedding_capacity(TEXT_ONLY, llama_cpp.EmbeddingContext(), space=SPACE, max_items=4)

    assert capacity.images == media.ImageCapacity()
    assert capacity.interleaved is False
    assert capacity.max_items == 4
    assert capacity.max_input_tokens == 2048


def test_capacity_refuses_webp_which_the_native_decoder_cannot_read() -> None:
    webp = media.ImageCapacity(max_images=1, max_total_bytes=1024, media_types=frozenset({"image/webp"}))

    with pytest.raises(ValueError, match="image/webp"):
        _ = llama_cpp.embedding_capacity(fakes.WEIGHTS, llama_cpp.EmbeddingContext(), space=SPACE, images=webp)


def _embedder(model: llama_cpp.EmbeddingModel, **fields: typing.Any) -> llama_cpp.Embedder:
    capacity = llama_cpp.embedding_capacity(fakes.WEIGHTS, llama_cpp.EmbeddingContext(), space=SPACE)
    return llama_cpp.Embedder(model=model, capacity=capacity, **fields)


def test_embedder_is_a_frozen_value_object_naming_its_model_as_its_attachment() -> None:
    model = llama_cpp.EmbeddingModel(weights=fakes.WEIGHTS, runtime=fakes.Runtime())
    embedder = _embedder(model)

    assert not isinstance(embedder, hsm.Instance)
    assert isinstance(embedder, embedding.Embedder)
    assert embedder.attachments == (model,)
    assert "attachments" not in {field.name for field in dataclasses.fields(embedder) if field.init}
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(embedder, "instructions", None)


async def _ability(
    session: fakes.Session, **fields: typing.Any
) -> tuple[Environment, embedding.Embedding, Recorder, llama_cpp.EmbeddingModel]:
    ctx = Environment()
    owner = Recorder()
    _ = await started(ctx, owner, Recorder.model)
    model = llama_cpp.EmbeddingModel(weights=fakes.WEIGHTS, runtime=fakes.Runtime(session))
    _ = await started(ctx, model, llama_cpp.EmbeddingModel.model)
    ability_ = embedding.Embedding(embedder=_embedder(model, **fields))
    _ = await ability_.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=owner), "a"))
    await owner.until(lambda: bool(owner.named(attachment.AttachCompleteEvent, attachment.AttachFailedEvent)))
    return ctx, ability_, owner, model


async def _embed(ctx: Environment, ability_: embedding.Embedding, data: embedding.InputData) -> hsm.Event[typing.Any]:
    return await ability.run_terminal_operation(
        ctx,
        child=ability_,
        request=embedding.InputEvent.with_data_and_id(data, "request-1"),
        terminals=(embedding.OutputEvent, embedding.FailedEvent),
        timeout=TIMEOUT,
    )


def test_end_to_end_the_ability_attaches_the_model_and_embeds_with_prefixes_and_the_declared_space() -> None:
    async def run() -> tuple[list[str], hsm.Event[typing.Any], str, list[tuple[text.ContentPart, ...]]]:
        session = fakes.Session()
        ctx, ability_, owner, model = await _ability(session, instructions=embedding.EMBEDDING_GEMMA_2)
        attached = [event.name for event in owner.events]
        result = await _embed(
            ctx,
            ability_,
            embedding.InputData(
                role=embedding.Role.QUERY,
                items=(embedding.Item(parts=(text.TextPart(text="a red circle"), text.ImagePart(image=PNG))),),
            ),
        )
        state = model.state()
        await model.stop(ctx)
        return attached, result, state, session.embedded

    attached, result, state, embedded = asyncio.run(run())

    assert attached == [attachment.AttachCompleteEvent.name]
    assert state == "/EmbeddingModel/loaded/idle"
    assert result.name == embedding.OutputEvent.name
    output = result.data
    assert isinstance(output, embedding.OutputData)
    assert output.space == SPACE
    assert output.vectors[0].values == pytest.approx((0.6, 0.8))
    assert embedded == [(text.TextPart(text="task: search result | query: a red circle"), text.ImagePart(image=PNG))]


def test_a_model_failure_reaches_the_ability_as_its_typed_kind_and_item() -> None:
    def vector(index: int) -> tuple[float, ...]:
        raise embedding.Error(embedding.FailureKind.INVALID_INPUT, "text holds the media marker", item_index=index)

    async def run() -> hsm.Event[typing.Any]:
        ctx, ability_, _, model = await _ability(fakes.Session(vector=vector))
        result = await _embed(
            ctx,
            ability_,
            embedding.InputData(
                role=embedding.Role.DOCUMENT, items=(embedding.Item(parts=(text.TextPart(text="x"),)),)
            ),
        )
        await model.stop(ctx)
        return result

    result = asyncio.run(run())

    assert result.name == embedding.FailedEvent.name
    data = result.data
    assert isinstance(data, embedding.FailureData)
    assert data.kind is embedding.FailureKind.INVALID_INPUT
    assert data.item_index == 0

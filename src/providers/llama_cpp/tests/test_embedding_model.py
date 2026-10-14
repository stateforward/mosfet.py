"""``EmbeddingModel``: its request payload, its events, and how one run embeds a request on a fake session."""

from __future__ import annotations

import asyncio
import datetime
import math
import typing

import hsm
import pydantic
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
PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"png-body", media_type="image/png")


def _request(
    *items: embedding.Item,
    space: embedding.Space = SPACE,
    role: embedding.Role = embedding.Role.QUERY,
    instructions: embedding.Instructions | None = None,
) -> llama_cpp.EmbedData:
    return llama_cpp.EmbedData(
        input=embedding.InputData(role=role, items=items or (embedding.Item(parts=(text.TextPart(text="hi"),)),)),
        space=space,
        instructions=instructions,
    )


async def _call(
    session: fakes.Session, data: llama_cpp.EmbedData, *, require_accelerator: bool = False
) -> tuple[hsm.Event[typing.Any], str]:
    """Load ``session`` on a root-started model, send ``data`` through a reply actor, return its terminal."""

    ctx = Environment()
    owner = Recorder()
    _ = await started(ctx, owner, Recorder.model)
    model = llama_cpp.EmbeddingModel(
        weights=fakes.WEIGHTS, runtime=fakes.Runtime(session), require_accelerator=require_accelerator
    )
    _ = await started(ctx, model, llama_cpp.EmbeddingModel.model)
    model_id = hsm.id(model)
    try:
        _ = await model.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=owner), "a"))
        await owner.until(lambda: bool(owner.events))
        if owner.events[0].name != attachment.AttachCompleteEvent.name:
            return owner.events[0], model_id
        terminal = await ability.run_terminal_operation(
            ctx,
            child=model,
            request=llama_cpp.EmbedEvent.with_data_and_id(data, "request-1"),
            terminals=(llama_cpp.EmbeddedEvent, llama_cpp.EmbedFailedEvent),
            timeout=datetime.timedelta(seconds=2),
        )
        return terminal, model_id
    finally:
        await model.stop(ctx)


def _failure(event: hsm.Event[typing.Any]) -> embedding.FailureData:
    assert event.name == llama_cpp.EmbedFailedEvent.name
    data = event.data
    assert isinstance(data, embedding.FailureData)
    return data


def _output(event: hsm.Event[typing.Any]) -> embedding.OutputData:
    assert event.name == llama_cpp.EmbeddedEvent.name
    data = event.data
    assert isinstance(data, embedding.OutputData)
    return data


def test_events_are_typed_and_named_for_the_provider() -> None:
    assert llama_cpp.EmbedEvent.name == "bot.provider.llama_cpp.embed"
    assert llama_cpp.EmbeddedEvent.name == "bot.provider.llama_cpp.embedded"
    assert llama_cpp.EmbedFailedEvent.name == "bot.provider.llama_cpp.embed.failed"
    assert llama_cpp.EmbedEvent.schema is llama_cpp.EmbedData
    assert llama_cpp.EmbeddedEvent.schema is embedding.OutputData
    assert llama_cpp.EmbedFailedEvent.schema is embedding.FailureData


def test_embed_data_is_frozen_strict_and_documented() -> None:
    data = _request()

    with pytest.raises(pydantic.ValidationError):
        _ = llama_cpp.EmbedData.model_validate({**data.model_dump(), "extra": 1})
    with pytest.raises(pydantic.ValidationError):
        _ = llama_cpp.EmbedData.model_validate({"input": data.input.model_dump()})
    with pytest.raises(pydantic.ValidationError):
        setattr(data, "space", SPACE)
    schema = llama_cpp.EmbedData.model_json_schema()
    properties = typing.cast(dict[str, dict[str, object]], schema["properties"])
    assert set(properties) == {"input", "space", "instructions"}
    assert all(field.get("description") for field in properties.values())
    assert schema.get("examples")


def test_terminal_is_correlated_to_the_request_and_addressed_to_its_sender() -> None:
    terminal, model_id = asyncio.run(_call(fakes.Session(), _request()))

    assert terminal.id == "request-1"
    assert terminal.source == model_id
    output = _output(terminal)
    assert output.space == SPACE
    assert output.role is embedding.Role.QUERY
    assert output.vectors[0].values == pytest.approx((0.6, 0.8))


def test_vectors_are_truncated_to_the_space_then_normalized_per_item() -> None:
    session = fakes.Session(vector=lambda index: (1.0 + index, 1.0, 5.0, 5.0))
    items = (embedding.Item(parts=(text.TextPart(text="a"),)), embedding.Item(parts=(text.TextPart(text="b"),)))

    terminal, _ = asyncio.run(_call(session, _request(*items)))

    first, second = _output(terminal).vectors
    assert first.values == pytest.approx((1 / math.sqrt(2), 1 / math.sqrt(2)))
    assert second.values == pytest.approx((2 / math.sqrt(5), 1 / math.sqrt(5)))


def test_instructions_are_applied_once_per_item() -> None:
    session = fakes.Session()
    item = embedding.Item(parts=(text.TextPart(text="red logo"), text.ImagePart(image=PNG)), title="Brand")

    _ = asyncio.run(
        _call(session, _request(item, role=embedding.Role.DOCUMENT, instructions=embedding.EMBEDDING_GEMMA_2))
    )

    assert session.embedded == [(text.TextPart(text="title: Brand | text: red logo"), text.ImagePart(image=PNG))]


def test_a_space_wider_than_the_model_is_unsupported() -> None:
    wide = embedding.Space(model="google/embeddinggemma-2", dimensions=8)

    terminal, _ = asyncio.run(_call(fakes.Session(native_dimensions=4), _request(space=wide)))

    assert _failure(terminal).kind is embedding.FailureKind.UNSUPPORTED_INPUT


def test_an_image_without_a_projector_is_unsupported() -> None:
    item = embedding.Item(parts=(text.ImagePart(image=PNG),))

    terminal, _ = asyncio.run(_call(fakes.Session(vision=False), _request(item)))

    assert _failure(terminal).kind is embedding.FailureKind.UNSUPPORTED_INPUT


def test_a_zero_vector_fails_with_the_item_index() -> None:
    session = fakes.Session(vector=lambda index: (0.0, 0.0, 1.0, 1.0) if index == 1 else (1.0, 0.0, 0.0, 0.0))
    items = (embedding.Item(parts=(text.TextPart(text="a"),)), embedding.Item(parts=(text.TextPart(text="b"),)))

    terminal, _ = asyncio.run(_call(session, _request(*items)))

    failure = _failure(terminal)
    assert failure.kind is embedding.FailureKind.EMBED_FAILED
    assert failure.item_index == 1


def test_an_item_error_propagates_with_its_index() -> None:
    def vector(index: int) -> tuple[float, ...]:
        if index == 1:
            raise embedding.Error(embedding.FailureKind.TOO_LONG, "item 1 is too long", item_index=index)
        return (1.0, 0.0, 0.0, 0.0)

    items = (embedding.Item(parts=(text.TextPart(text="a"),)), embedding.Item(parts=(text.TextPart(text="b"),)))
    terminal, _ = asyncio.run(_call(fakes.Session(vector=vector), _request(*items)))

    failure = _failure(terminal)
    assert failure.kind is embedding.FailureKind.TOO_LONG
    assert failure.item_index == 1


def test_require_accelerator_refuses_a_cpu_session_and_closes_it() -> None:
    session = fakes.Session(accelerator=llama_cpp.Accelerator.CPU)

    reply, _ = asyncio.run(_call(session, _request(), require_accelerator=True))

    assert reply.name == attachment.AttachFailedEvent.name
    data = reply.data
    assert isinstance(data, attachment.FailedData)
    assert data.kind is attachment.FailureKind.INITIALIZATION
    assert session.closes == 1

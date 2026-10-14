import dataclasses

import hsm
import pydantic
import pytest

from mosfet import media
from mosfet.abilities import embedding
from mosfet.abilities.language import text
from mosfet.telemetry import span

from tests.type_helpers import object_dict

_PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"
_SPACE = embedding.Space(model="google/embeddinggemma-2", dimensions=2)
_INSTRUCTIONS = embedding.Instructions(query="task: search result | query: ", document="title: {title} | text: ")


def _image_part() -> text.ImagePart:
    return text.ImagePart(image=media.Image(content=_PNG, media_type="image/png"))


@dataclasses.dataclass(frozen=True, kw_only=True)
class DataclassEmbedder(embedding.Embedder):
    """A provider-shaped embedder: a frozen dataclass whose ``capacity`` is a field."""

    capacity: embedding.Capacity

    async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
        del ctx
        vectors = tuple(embedding.Vector(values=(1.0, 0.0)) for _ in input.items)
        return embedding.OutputData(space=self.capacity.space, role=input.role, vectors=vectors)


@dataclasses.dataclass(frozen=True, kw_only=True)
class AttachedEmbedder(DataclassEmbedder):
    """An embedder that names the provider actors its owner attaches to."""

    attachments: tuple[hsm.Instance, ...] = dataclasses.field(default=())


def test_dataclass_embedder_with_capacity_field_instantiates() -> None:
    """Regression: a ``@property`` abstract ``capacity`` could not be satisfied by a dataclass field."""

    capacity = embedding.Capacity(space=_SPACE, max_items=4, max_input_tokens=2048)
    embedder = DataclassEmbedder(capacity=capacity)

    assert embedder.capacity is capacity
    assert embedder.attachments == ()
    assert not isinstance(embedder, hsm.Instance)


def test_embedder_attachments_are_a_plain_attribute_a_provider_may_set() -> None:
    capacity = embedding.Capacity(space=_SPACE, max_items=1, max_input_tokens=None)
    actor = hsm.Instance()

    assert AttachedEmbedder(capacity=capacity, attachments=(actor,)).attachments == (actor,)
    assert "capacity" not in vars(embedding.Embedder)
    assert vars(embedding.Embedder)["attachments"] == ()


def test_embed_is_the_only_abstract_member() -> None:
    assert embedding.Embedder.__abstractmethods__ == frozenset({"embed"})


def test_query_render_prefixes_text_items() -> None:
    item = embedding.Item(parts=(text.TextPart(text="red circle"),))

    rendered = _INSTRUCTIONS.render(embedding.Role.QUERY, item)

    assert rendered == (text.TextPart(text="task: search result | query: red circle"),)


def test_document_render_substitutes_title_or_untitled() -> None:
    titled = embedding.Item(parts=(text.TextPart(text="body"),), title="Brand")
    untitled = embedding.Item(parts=(text.TextPart(text="body"),))

    assert _INSTRUCTIONS.render(embedding.Role.DOCUMENT, titled) == (text.TextPart(text="title: Brand | text: body"),)
    assert _INSTRUCTIONS.render(embedding.Role.DOCUMENT, untitled) == (text.TextPart(text="title: none | text: body"),)


def test_query_render_ignores_title() -> None:
    item = embedding.Item(parts=(text.TextPart(text="body"),), title="Brand")

    assert _INSTRUCTIONS.render(embedding.Role.QUERY, item) == (
        text.TextPart(text="task: search result | query: body"),
    )


def test_image_only_items_render_unchanged() -> None:
    item = embedding.Item(parts=(_image_part(),), title="Brand")

    assert _INSTRUCTIONS.render(embedding.Role.DOCUMENT, item) == item.parts
    assert _INSTRUCTIONS.render(embedding.Role.QUERY, item) == item.parts


def test_mixed_item_led_by_an_image_gets_a_leading_prefix_part() -> None:
    image = _image_part()
    item = embedding.Item(parts=(image, text.TextPart(text="red logo")))

    rendered = _INSTRUCTIONS.render(embedding.Role.DOCUMENT, item)

    assert rendered == (text.TextPart(text="title: none | text: "), image, text.TextPart(text="red logo"))


def test_leading_text_part_absorbs_the_prefix() -> None:
    image = _image_part()
    item = embedding.Item(parts=(text.TextPart(text="product photo "), image, text.TextPart(text=" red logo")))

    rendered = _INSTRUCTIONS.render(embedding.Role.DOCUMENT, item)

    assert rendered == (
        text.TextPart(text="title: none | text: product photo "),
        image,
        text.TextPart(text=" red logo"),
    )


def test_title_braces_are_substituted_literally() -> None:
    instructions = embedding.Instructions(query="q: ", document="title: {title} | {other} | ")
    item = embedding.Item(parts=(text.TextPart(text="x"),), title="{title}")

    assert instructions.render(embedding.Role.DOCUMENT, item) == (text.TextPart(text="title: {title} | {other} | x"),)


def test_embedding_gemma_2_prefixes_match_the_measured_model() -> None:
    assert embedding.EMBEDDING_GEMMA_2.query == "task: search result | query: "
    assert embedding.EMBEDDING_GEMMA_2.document == "title: {title} | text: "
    assert embedding.EMBEDDING_GEMMA_2.untitled == "none"


def test_error_carries_its_kind_for_failure_events_and_spans() -> None:
    error = embedding.Error(embedding.FailureKind.TOO_LONG, "item 1 exceeds 2048 tokens", item_index=1)

    assert error.kind is embedding.FailureKind.TOO_LONG
    assert error.item_index == 1
    assert error.failure_kind == "too_long"
    assert str(error) == "item 1 exceeds 2048 tokens"
    assert span.failure_kind(error, "embed_failed") == "too_long"
    assert embedding.Error(embedding.FailureKind.TIMEOUT, "slow").item_index is None


def test_capacity_validates_bounds_and_defaults_to_text_only() -> None:
    capacity = embedding.Capacity(space=_SPACE, max_items=embedding.MAX_ITEMS, max_input_tokens=None)

    assert capacity.images == media.ImageCapacity()
    assert capacity.images.max_images == 0
    assert capacity.interleaved is False
    for invalid in (
        {"max_items": 0, "max_input_tokens": None},
        {"max_items": embedding.MAX_ITEMS + 1, "max_input_tokens": None},
        {"max_items": 1, "max_input_tokens": 0},
    ):
        with pytest.raises(pydantic.ValidationError):
            _ = embedding.Capacity.model_validate({"space": _SPACE, **invalid})


@pytest.mark.parametrize("model", [embedding.Capacity, embedding.Instructions])
def test_embedder_value_schemas_describe_every_field(model: type[pydantic.BaseModel]) -> None:
    schema = model.model_json_schema()

    assert schema.get("description")
    assert schema.get("examples")
    for name, value in object_dict(schema["properties"]).items():
        assert object_dict(value).get("description"), f"{model.__name__}.{name} has no description"

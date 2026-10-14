"""Unit tests for ``openai_compat.Embedder``: wire mapping, response validation, and telemetry."""

from __future__ import annotations

import asyncio
import base64
import collections.abc
import dataclasses
import json
import math
import pathlib
import typing

import hsm
import pytest
from opentelemetry import _logs, trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mosfet.providers.openai_compat as openai_compat
import mosfet.media as media
import mosfet.telemetry
from mosfet.abilities import embedding
from mosfet.abilities.language import text
from mosfet.telemetry import span

_PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"
_SPACE = embedding.Space(model="google/embeddinggemma-2", dimensions=2)
_TEXT_ONLY = embedding.Capacity(space=_SPACE, max_items=4, max_input_tokens=None)
_MULTIMODAL = embedding.Capacity(
    space=_SPACE,
    max_items=4,
    max_input_tokens=None,
    images=media.ImageCapacity(max_images=2, max_total_bytes=1024, media_types=frozenset({"image/png"})),
    interleaved=True,
)
_SPAN = "bot.provider.openai_compat.embeddings"
_DURATION = "bot.provider.openai_compat.embeddings.duration"


@dataclasses.dataclass
class Call:
    input: list[object]
    dimensions: int | None
    extra_body: collections.abc.Mapping[str, object] | None


@dataclasses.dataclass
class FakeClient(openai_compat.EmbeddingsClient):
    """Records each request and answers with ``response`` (or raises it)."""

    response: dict[str, object] | Exception
    calls: list[Call] = dataclasses.field(default_factory=list)

    @typing.override
    def create_embeddings(
        self,
        *,
        input: collections.abc.Sequence[object],
        dimensions: int | None = None,
        extra_body: collections.abc.Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self.calls.append(Call(input=list(input), dimensions=dimensions, extra_body=extra_body))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _rows(*vectors: list[object], order: collections.abc.Sequence[int] | None = None) -> dict[str, object]:
    indices = range(len(vectors)) if order is None else order
    return {"data": [{"object": "embedding", "index": index, "embedding": vectors[index]} for index in indices]}


def _image() -> media.Image:
    return media.Image(content=_PNG, media_type="image/png")


def _data_url() -> str:
    return "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii")


def _text_item(value: str, *, title: str | None = None) -> embedding.Item:
    return embedding.Item(parts=(text.TextPart(text=value),), title=title)


def _embed(embedder: openai_compat.Embedder, data: embedding.InputData) -> embedding.OutputData:
    return asyncio.run(embedder.embed(data, ctx=hsm.Context()))


def _failure(embedder: openai_compat.Embedder, data: embedding.InputData) -> embedding.Error:
    with pytest.raises(embedding.Error) as caught:
        _ = _embed(embedder, data)
    return caught.value


def test_embedder_is_a_frozen_core_embedder_value_object() -> None:
    embedder = openai_compat.Embedder(client=FakeClient(response=_rows([1.0, 0.0])), capacity=_TEXT_ONLY)

    assert isinstance(embedder, embedding.Embedder)
    assert not isinstance(embedder, hsm.Instance)
    assert embedder.capacity is _TEXT_ONLY
    assert embedder.attachments == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(embedder, "send_dimensions", True)


def test_text_items_are_sent_as_prefixed_strings() -> None:
    client = FakeClient(response=_rows([1.0, 0.0], [0.0, 1.0]))
    embedder = openai_compat.Embedder(client=client, capacity=_TEXT_ONLY, instructions=embedding.EMBEDDING_GEMMA_2)

    _ = _embed(
        embedder,
        embedding.InputData(
            role=embedding.Role.DOCUMENT,
            items=(
                _text_item("Our logo is a red circle.", title="Brand guide"),
                embedding.Item(parts=(text.TextPart(text="two "), text.TextPart(text="parts"))),
            ),
        ),
    )

    assert client.calls == [
        Call(
            input=["title: Brand guide | text: Our logo is a red circle.", "title: none | text: two parts"],
            dimensions=None,
            extra_body=None,
        )
    ]


def test_text_items_are_sent_unprefixed_without_instructions() -> None:
    client = FakeClient(response=_rows([1.0, 0.0]))
    embedder = openai_compat.Embedder(client=client, capacity=_TEXT_ONLY)

    _ = _embed(embedder, embedding.InputData(role=embedding.Role.QUERY, items=(_text_item("a red circle"),)))

    assert client.calls[0].input == ["a red circle"]


def test_image_and_mixed_items_use_content_arrays_with_data_urls() -> None:
    client = FakeClient(response=_rows([1.0, 0.0], [0.0, 1.0]))
    embedder = openai_compat.Embedder(client=client, capacity=_MULTIMODAL, instructions=embedding.EMBEDDING_GEMMA_2)

    _ = _embed(
        embedder,
        embedding.InputData(
            role=embedding.Role.QUERY,
            items=(
                embedding.Item(parts=(text.ImagePart(image=_image()),)),
                embedding.Item(parts=(text.TextPart(text="a red circle"), text.ImagePart(image=_image()))),
            ),
        ),
    )

    assert client.calls[0].input == [
        {"content": [{"type": "image_url", "image_url": {"url": _data_url()}}]},
        {
            "content": [
                {"type": "text", "text": "task: search result | query: a red circle"},
                {"type": "image_url", "image_url": {"url": _data_url()}},
            ]
        },
    ]


def test_images_outside_the_declared_capacity_never_reach_the_wire() -> None:
    client = FakeClient(response=_rows([1.0, 0.0], [0.0, 1.0]))
    embedder = openai_compat.Embedder(client=client, capacity=_TEXT_ONLY)

    error = _failure(
        embedder,
        embedding.InputData(
            role=embedding.Role.QUERY,
            items=(_text_item("a red circle"), embedding.Item(parts=(text.ImagePart(image=_image()),))),
        ),
    )

    assert error.kind is embedding.FailureKind.UNSUPPORTED_INPUT
    assert error.item_index == 1
    assert client.calls == []


def test_vectors_follow_response_index_and_are_stamped_with_the_capacity_space() -> None:
    client = FakeClient(response=_rows([0.0, 1.0], [1.0, 0.0], order=(1, 0)))
    embedder = openai_compat.Embedder(client=client, capacity=_TEXT_ONLY)

    output = _embed(
        embedder,
        embedding.InputData(role=embedding.Role.DOCUMENT, items=(_text_item("first"), _text_item("second"))),
    )

    assert output == embedding.OutputData(
        space=_SPACE,
        role=embedding.Role.DOCUMENT,
        vectors=(embedding.Vector(values=(0.0, 1.0)), embedding.Vector(values=(1.0, 0.0))),
    )


def test_longer_vectors_are_truncated_to_the_space_then_renormalized() -> None:
    client = FakeClient(response=_rows([3.0, 4.0, 12.0]))
    embedder = openai_compat.Embedder(client=client, capacity=_TEXT_ONLY)

    output = _embed(embedder, embedding.InputData(role=embedding.Role.QUERY, items=(_text_item("a"),)))

    assert all(math.isclose(got, want) for got, want in zip(output.vectors[0].values, (0.6, 0.8), strict=True))


def test_send_dimensions_requests_the_space_dimensions() -> None:
    client = FakeClient(response=_rows([0.6, 0.8]))
    embedder = openai_compat.Embedder(client=client, capacity=_TEXT_ONLY, send_dimensions=True)

    _ = _embed(embedder, embedding.InputData(role=embedding.Role.QUERY, items=(_text_item("a"),)))

    assert client.calls[0].dimensions == 2


@pytest.mark.parametrize(
    "response",
    [
        pytest.param({"data": [{"index": 0, "embedding": [1.0]}]}, id="short-vector"),
        pytest.param({"data": []}, id="wrong-count"),
        pytest.param({"object": "list"}, id="missing-data"),
        pytest.param({"data": [{"index": 0, "embedding": [math.nan, 1.0]}]}, id="non-finite"),
        pytest.param({"data": [{"index": 0, "embedding": [math.inf, 1.0]}]}, id="infinite"),
        pytest.param({"data": [{"index": 0, "embedding": [0.0, 0.0]}]}, id="zero-vector"),
        pytest.param({"data": [{"index": 0, "embedding": ["0.6", 0.8]}]}, id="non-numeric"),
        pytest.param({"data": [{"index": 0, "embedding": [True, 0.0]}]}, id="boolean"),
        pytest.param({"data": [{"index": 1, "embedding": [0.6, 0.8]}]}, id="index-out-of-range"),
        pytest.param({"data": [{"embedding": [0.6, 0.8]}]}, id="missing-index"),
        pytest.param({"data": ["not-an-object"]}, id="non-object-row"),
        pytest.param({"data": [{"index": 0, "embedding": "AACAPwAAAAA="}]}, id="base64-vector"),
    ],
)
def test_malformed_responses_fail_as_invalid_response(response: dict[str, object]) -> None:
    embedder = openai_compat.Embedder(client=FakeClient(response=response), capacity=_TEXT_ONLY)

    error = _failure(embedder, embedding.InputData(role=embedding.Role.QUERY, items=(_text_item("a"),)))

    assert error.kind is embedding.FailureKind.INVALID_RESPONSE


def test_duplicate_indices_fail_as_invalid_response() -> None:
    response: dict[str, object] = {
        "data": [{"index": 0, "embedding": [0.6, 0.8]}, {"index": 0, "embedding": [0.8, 0.6]}]
    }
    embedder = openai_compat.Embedder(client=FakeClient(response=response), capacity=_TEXT_ONLY)

    error = _failure(embedder, embedding.InputData(role=embedding.Role.QUERY, items=(_text_item("a"), _text_item("b"))))

    assert error.kind is embedding.FailureKind.INVALID_RESPONSE


def test_client_errors_keep_their_kind() -> None:
    failure = embedding.Error(embedding.FailureKind.UNAUTHORIZED, "rejected credentials")
    embedder = openai_compat.Embedder(client=FakeClient(response=failure), capacity=_TEXT_ONLY)

    error = _failure(embedder, embedding.InputData(role=embedding.Role.QUERY, items=(_text_item("a"),)))

    assert error is failure


def test_unexpected_client_failures_become_embed_failed() -> None:
    embedder = openai_compat.Embedder(client=FakeClient(response=RuntimeError("socket closed")), capacity=_TEXT_ONLY)

    error = _failure(embedder, embedding.InputData(role=embedding.Role.QUERY, items=(_text_item("a"),)))

    assert error.kind is embedding.FailureKind.EMBED_FAILED
    assert isinstance(error.__cause__, RuntimeError)


@pytest.fixture
def finished_spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)
    return exporter


def _spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [recorded for recorded in exporter.get_finished_spans() if recorded.name == _SPAN]


def test_request_span_carries_bounded_counts_and_no_content(finished_spans: InMemorySpanExporter) -> None:
    embedder = openai_compat.Embedder(
        client=FakeClient(response=_rows([1.0, 0.0], [0.0, 1.0])),
        capacity=_MULTIMODAL,
        instructions=embedding.EMBEDDING_GEMMA_2,
    )

    _ = _embed(
        embedder,
        embedding.InputData(
            role=embedding.Role.QUERY,
            items=(_text_item("secret words"), embedding.Item(parts=(text.ImagePart(image=_image()),))),
        ),
    )

    [recorded] = _spans(finished_spans)
    assert dict(recorded.attributes or {}) == {
        "bot.component.name": "openai_compat.embedding",
        "bot.stage": "request",
        "bot.embedding.items": 2,
        "bot.embedding.images": 1,
        "bot.outcome": "ok",
    }


def test_request_span_records_the_failure_kind(finished_spans: InMemorySpanExporter) -> None:
    embedder = openai_compat.Embedder(client=FakeClient(response={"data": []}), capacity=_TEXT_ONLY)

    _ = _failure(embedder, embedding.InputData(role=embedding.Role.QUERY, items=(_text_item("a"),)))

    [recorded] = _spans(finished_spans)
    attributes = dict(recorded.attributes or {})
    assert attributes["bot.outcome"] == "failed"
    assert attributes["bot.failure.kind"] == "invalid_response"


def _histogram_counts(path: pathlib.Path) -> dict[tuple[tuple[str, object], ...], int]:
    provider = mosfet.telemetry.meter_provider()
    assert provider is not None
    _ = provider.force_flush()
    latest: list[dict[str, typing.Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else ():
        payload = typing.cast(dict[str, typing.Any], json.loads(line))
        if payload["name"] == _DURATION:
            latest = typing.cast(list[dict[str, typing.Any]], payload["data"]["data_points"])
    return {
        tuple(sorted(typing.cast(dict[str, object], point["attributes"]).items())): typing.cast(int, point["count"])
        for point in latest
    }


def _install_nothing(provider: object) -> None:
    del provider


def test_duration_metric_carries_outcome_and_failure_kind_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("BOT_OTEL_DISABLED", "BOT_OTEL_LOG_FILE", "BOT_OTEL_SPAN_FILE", "BOT_OTEL_METRIC_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_logs, "set_logger_provider", _install_nothing)
    monkeypatch.setattr(trace, "set_tracer_provider", _install_nothing)
    mosfet.telemetry.reset()
    assert mosfet.telemetry.configure(metric_file="metrics.jsonl") is True
    try:
        path = tmp_path / "metrics.jsonl"
        before = _histogram_counts(path)
        request = embedding.InputData(role=embedding.Role.QUERY, items=(_text_item("a"),))
        _ = _embed(openai_compat.Embedder(client=FakeClient(response=_rows([1.0, 0.0])), capacity=_TEXT_ONLY), request)
        _ = _failure(openai_compat.Embedder(client=FakeClient(response={"data": []}), capacity=_TEXT_ONLY), request)
        after = _histogram_counts(path)
    finally:
        mosfet.telemetry.reset()

    ok = (("bot.failure.kind", "none"), ("bot.outcome", "ok"))
    failed = (("bot.failure.kind", "invalid_response"), ("bot.outcome", "failed"))
    assert after[ok] - before.get(ok, 0) == 1
    assert after[failed] - before.get(failed, 0) == 1
    assert {key for point in after for key, _ in point} == {"bot.outcome", "bot.failure.kind"}

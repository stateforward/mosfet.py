"""The streaming encoder contract and the generic encoder failure in ``mosfet.abilities.encoding``."""

from mosfet.abilities import encoding
from mosfet.telemetry import span

import asyncio
import collections.abc
import inspect
import typing

import pytest


class _Upper(encoding.StreamEncoder[str, str]):
    """Encodes each piece as it is read, counting the pieces it has read."""

    def __init__(self) -> None:
        self.read: int = 0

    @typing.override
    async def encode(self, input: collections.abc.AsyncIterable[str]) -> collections.abc.AsyncIterator[str]:
        async for piece in input:
            self.read += 1
            yield piece.upper()


async def _pieces(*pieces: str) -> collections.abc.AsyncIterator[str]:
    for piece in pieces:
        yield piece


def test_a_stream_encoder_is_abstract() -> None:
    assert inspect.isabstract(encoding.StreamEncoder)
    assert encoding.StreamEncoder.__abstractmethods__ == frozenset({"encode"})


def test_a_stream_encoder_streams_input_pieces_to_outputs() -> None:
    async def run() -> list[str]:
        return [output async for output in _Upper().encode(_pieces("one ", "two"))]

    assert asyncio.run(run()) == ["ONE ", "TWO"]


def test_closing_the_output_stops_reading_the_input() -> None:
    async def run() -> tuple[str, int]:
        encoder = _Upper()
        stream = encoder.encode(_pieces("one ", "two ", "three"))
        first = await anext(stream)
        await typing.cast(collections.abc.AsyncGenerator[str, None], stream).aclose()
        with pytest.raises(StopAsyncIteration):
            _ = await anext(stream)
        return first, encoder.read

    assert asyncio.run(run()) == ("ONE ", 1)


@pytest.mark.parametrize("kind", ["invalid_input", "unavailable", "timeout", "invalid_response", "encode_failed"])
def test_an_encoding_error_carries_an_existing_failure_kind(kind: encoding.FailureKind) -> None:
    error = encoding.Error(kind, "the encoder failed")

    assert error.kind == kind
    assert error.failure_kind == kind
    assert str(error) == "the encoder failed"
    assert span.failure_kind(error, "encode_failed") == kind


def test_the_encoding_failure_kinds_are_existing_failure_kind_strings() -> None:
    assert typing.get_args(typing.cast(object, encoding.FailureKind.__value__)) == (
        "invalid_input",
        "unavailable",
        "timeout",
        "invalid_response",
        "encode_failed",
    )

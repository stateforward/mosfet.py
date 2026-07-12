from __future__ import annotations

from bot.abilities import reading

import asyncio
import collections.abc

from bot import abilities

from bot.providers.mlx_vlm import ReadingOutputEncoder

async def await_output_encoding(output: collections.abc.Awaitable[reading.OutputData]) -> reading.OutputData:
    return await output

def test_reading_output_encoder_returns_reading_output() -> None:
    encoder = ReadingOutputEncoder()
    input = reading.OutputData(text="Total due: $14.21", source_kind="image", confidence=0.87)

    output = asyncio.run(await_output_encoding(encoder.encode(input)))

    assert output == input
    assert isinstance(encoder, abilities.Encoder)

def test_reading_output_encoder_is_awaitable() -> None:
    encoder = ReadingOutputEncoder()
    input = reading.OutputData(text="Read this note.", source_kind="text", confidence=0.99)

    output = encoder.encode(input)

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output) == input

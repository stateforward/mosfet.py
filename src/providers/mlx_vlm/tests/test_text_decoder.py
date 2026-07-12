from __future__ import annotations

import asyncio
import collections.abc

from bot import abilities
from bot.providers.mlx_vlm import TextDecoder


async def await_text_decoding(output: collections.abc.Awaitable[str]) -> str:
    return await output


def test_text_decoder_passes_text_through() -> None:
    decoder = TextDecoder()

    output = asyncio.run(await_text_decoding(decoder.decode("Read this note.")))

    assert output == "Read this note."
    assert isinstance(decoder, abilities.Decoder)


def test_text_decoder_is_awaitable() -> None:
    decoder = TextDecoder()

    output = decoder.decode("Read this note.")

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output) == "Read this note."

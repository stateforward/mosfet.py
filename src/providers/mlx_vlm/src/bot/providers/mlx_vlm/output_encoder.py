from __future__ import annotations

from bot.abilities import reading

import dataclasses
import typing

import bot.abilities


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReadingOutputEncoder(bot.abilities.Encoder[reading.OutputData, reading.OutputData]):
    """OutputData encoder that preserves normalized reading output."""

    @typing.override
    async def encode(self, input: reading.OutputData) -> reading.OutputData:
        return input


__all__ = ["ReadingOutputEncoder"]

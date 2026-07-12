from __future__ import annotations

from bot.abilities import reading

import dataclasses
import typing

from bot import abilities

@dataclasses.dataclass(frozen=True, kw_only=True)
class ReadingOutputEncoder(abilities.Encoder[reading.OutputData, reading.OutputData]):
    """OutputData encoder that preserves normalized reading output."""

    @typing.override
    async def encode(self, input: reading.OutputData) -> reading.OutputData:
        return input

__all__ = ["ReadingOutputEncoder"]

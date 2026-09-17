from __future__ import annotations

import dataclasses
import typing

import mosfet.abilities


@dataclasses.dataclass(frozen=True, kw_only=True)
class TextDecoder(mosfet.abilities.Decoder[str, str]):
    """Text decoder for reading inputs that are already plain text."""

    @typing.override
    async def decode(self, input: str) -> str:
        return input


__all__ = ["TextDecoder"]

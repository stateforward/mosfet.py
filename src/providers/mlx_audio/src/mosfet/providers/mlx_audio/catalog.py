"""MLX Audio voices as a canonical ``vocal.voice.Catalog``: the voices composition declares the model speaks in.

An on-device model has no listing to ask, so composition declares the voices of the checkpoint it loads, each a
canonical ``voice.Entry`` whose ref key is the model's own voice name (the ``voice`` the speech encoder is built
with). Composition also writes what each voice is: a Kokoro id such as ``af_heart`` names its language (``a``,
American English) and gender (``f``, feminine) in its first two letters, written as the entry's ``languages`` and
``descriptors.gender``. A plain collaborator, not a state machine.
"""

from __future__ import annotations

import collections.abc
import typing

from mosfet.abilities.vocal import speech, voice


class Catalog(voice.Catalog):
    """The declared voices, all of one source, listed in the order they were declared.

    ``source`` is the source every entry's ref names.

    Raises:
        ValueError: no entry is declared, the entries name more than one source, or a voice is declared twice.
    """

    def __init__(self, entries: collections.abc.Iterable[voice.Entry]) -> None:
        declared = tuple(entries)
        sources = {entry.ref.source for entry in declared}
        if len(sources) != 1:
            raise ValueError(f"A MLX Audio catalog declares the voices of one source, not {len(sources)}.")
        if len({entry.ref.key for entry in declared}) != len(declared):
            raise ValueError("A MLX Audio catalog declares each voice once.")
        (source,) = sources
        self.source: str = source
        self._entries: tuple[voice.Entry, ...] = declared

    @typing.override
    async def voices(
        self, *, languages: tuple[speech.Locale, ...] = (), descriptors: voice.Descriptors | None = None
    ) -> tuple[voice.Entry, ...]:
        """Every declared voice. The filters are hints; the declared voices are few, so all are listed."""

        del languages, descriptors
        return self._entries


__all__ = ["Catalog"]

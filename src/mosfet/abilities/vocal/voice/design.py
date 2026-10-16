"""Canonical voice design: a voice made from a description, permanent at its source once made.

A source that can design voices is a :class:`Designer`: given a :class:`DesignRequest` it renders one voice and keeps
it, and answers :class:`Designed` with the :class:`~.catalog.Ref` the voice is held under. Like a catalog it is a plain
collaborator its owner calls directly, raising :class:`~.catalog.Error` (``design_failed`` when no narrower kind
names the failure).
"""

from __future__ import annotations

import abc
import typing

import pydantic

from .. import speech
from . import catalog


class DesignRequest(pydantic.BaseModel):
    """Make one voice from this description, to keep."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    description: str = pydantic.Field(
        min_length=20, max_length=1000, description="What the voice sounds like, in plain language."
    )
    descriptors: catalog.Descriptors | None = pydantic.Field(default=None, description="How it sounds, canonically.")
    languages: tuple[speech.Locale, ...] = pydantic.Field(default=(), description="The languages it will speak.")
    sample_text: str | None = pydantic.Field(
        default=None, min_length=1, max_length=1000, description="Words a source may render it saying, if it needs any."
    )


class Designed(pydantic.BaseModel):
    """A voice the source made and now keeps: permanent at the source from this moment."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    ref: catalog.Ref = pydantic.Field(description="The voice, as the source now holds it.")
    description: str = pydantic.Field(
        min_length=1, max_length=1000, description="The description the voice was made from."
    )
    preview: catalog.Preview | None = pydantic.Field(default=None, description="A sample, when the source made one.")


class Designer(abc.ABC):
    """A source that makes voices from descriptions. ``source`` names it the way its catalog does.

    ``max_description`` is the longest description the source takes, at most 1000 characters. A plain collaborator:
    its owner calls it directly.
    """

    source: str
    max_description: int

    @abc.abstractmethod
    async def create(self, request: DesignRequest) -> Designed:
        """Make one voice from ``request`` and keep it.

        Raises:
            catalog.Error: the source could not make the voice.
        """
        ...


__all__ = ["DesignRequest", "Designed", "Designer"]

"""Canonical voice catalog: the voices a source holds, named and described the same way for every source.

A voice source is whatever makes and keeps voices (a TTS server, an on-device model's set of speakers). Core names a
voice by :class:`Ref`, a pair of the source's composition-assigned name and the source's own opaque key, and never
interprets either. :class:`Entry` describes one voice a source holds, sized from a survey of the major providers: a
name, a description, its languages, :class:`Descriptors` of how it sounds, the styles it knows, an optional
:class:`Preview` and free labels. A provider maps its own listing to entries; no provider name or shape enters here.

:class:`Catalog` is a plain collaborator, not a state machine: its owner calls it directly. A catalog failure is an
:class:`Error` with a closed, low-cardinality kind.
"""

from __future__ import annotations

import abc
import typing

import pydantic

from .. import speech

_Source = typing.Annotated[str, pydantic.StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")]
"""A voice source's name as composition assigned it, such as ``lab``: lower case, at most 64 characters."""

_Word = typing.Annotated[str, pydantic.StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]

FailureKind = typing.Literal[
    "invalid_input",
    "unsupported_input",
    "unauthorized",
    "unavailable",
    "timeout",
    "overloaded",
    "invalid_response",
    "design_failed",
]
"""Why a voice source failed, in ``FAILURE_KINDS.md`` strings."""


class Error(Exception):
    """A voice source failure with a closed, low-cardinality kind. Any catalog or designer may raise it."""

    kind: FailureKind
    failure_kind: str
    """The same string as ``kind``, under the name ``telemetry.span.failure_kind`` reads."""

    def __init__(self, kind: FailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.failure_kind = kind


class Ref(pydantic.BaseModel):
    """One voice, named by the source that holds it and that source's own key for it. Opaque to core."""

    # No examples, deliberately: a ref is model-facing in a shortlisted conception, and an example ref is a voice
    # the model could name without choosing it from the catalog it was shown.
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    source: _Source = pydantic.Field(description="The voice source that holds this voice, as composition named it.")
    key: str = pydantic.Field(min_length=1, max_length=256, description="The source's own key for this voice.")


class Descriptors(pydantic.BaseModel):
    """How a voice sounds, in canonical terms. Every field is optional."""

    # No examples, deliberately: descriptors are model-facing in a described conception, and an example is a voice.
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    gender: typing.Literal["feminine", "masculine", "neutral"] | None = pydantic.Field(
        default=None, description="How the voice's gender is heard, if it is."
    )
    age: typing.Literal["child", "young_adult", "adult", "middle_aged", "senior"] | None = pydantic.Field(
        default=None, description="The apparent age of the speaker."
    )
    accent: _Word | None = pydantic.Field(default=None, description="The accent, in a word or short phrase.")
    pitch: typing.Literal["low", "medium", "high"] | None = pydantic.Field(
        default=None, description="The voice's natural pitch."
    )
    qualities: tuple[_Word, ...] = pydantic.Field(
        default=(), max_length=8, description="Up to eight words for its texture and character, such as 'warm'."
    )


class Preview(pydantic.BaseModel):
    """Synthetic audio of a voice, in place or at a URL. Never enrolled as the bot's own voice."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    pcm: bytes | None = pydantic.Field(default=None, description="The preview as s16le PCM in ``format``.")
    format: speech.Format | None = pydantic.Field(default=None, description="The format of ``pcm``.")
    url: pydantic.AnyHttpUrl | None = pydantic.Field(default=None, description="Where the preview can be fetched.")
    text: str | None = pydantic.Field(
        default=None, min_length=1, max_length=1000, description="The words the preview says, when known."
    )

    @pydantic.model_validator(mode="after")
    def _present(self) -> typing.Self:
        if self.pcm is not None and self.format is None:
            raise ValueError("A preview's pcm needs its format.")
        if self.pcm is None and self.url is None:
            raise ValueError("A preview needs its pcm or its url.")
        return self


class Entry(pydantic.BaseModel):
    """One voice a source holds, described canonically."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "A voice the source holds: its ref, name, description, languages and how it sounds.",
            "examples": [
                {
                    "ref": {"source": "lab", "key": "warm"},
                    "name": "warm",
                    "description": "A warm, unhurried adult voice.",
                    "languages": ["en"],
                }
            ],
        },
    )

    ref: Ref = pydantic.Field(description="The voice.")
    name: str = pydantic.Field(min_length=1, max_length=256, description="The voice's name at its source.")
    description: str | None = pydantic.Field(
        default=None, min_length=1, max_length=1000, description="What the voice sounds like, when the source says."
    )
    languages: tuple[speech.Locale, ...] = pydantic.Field(default=(), description="The languages it speaks.")
    descriptors: Descriptors = pydantic.Field(default=Descriptors(), description="How it sounds.")
    styles: tuple[_Word, ...] = pydantic.Field(default=(), description="The named styles it knows.")
    preview: Preview | None = pydantic.Field(default=None, description="A sample of it, when the source has one.")
    labels: dict[str, str] = pydantic.Field(default_factory=dict, description="Free labels the source attaches to it.")


class Catalog(abc.ABC):
    """The voices one source holds. ``source`` is the composition-assigned name every :class:`Ref` it lists carries.

    A plain collaborator: its owner calls it directly. It raises :class:`Error` when the source cannot answer.
    """

    source: str

    @abc.abstractmethod
    async def voices(
        self, *, languages: tuple[speech.Locale, ...] = (), descriptors: Descriptors | None = None
    ) -> tuple[Entry, ...]:
        """The voices the source holds. The filters are hints a source may apply or ignore.

        Raises:
            Error: the source could not list its voices.
        """
        ...


__all__ = ["Catalog", "Descriptors", "Entry", "Error", "FailureKind", "Preview", "Ref"]

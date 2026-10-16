"""The inception frame: what the inception model is shown, and the one conception of a bot's voice it answers.

The inception model is the bot's makers' tool, not the bot (P-14): it is shown the bot's traits, its languages and its
voice source's capacity (:class:`InceptionInput`), and offered exactly one event, :data:`ConceiveEvent`, on that
request's own schemas and never on a turn. Its :data:`Conception` is a voice :class:`Described` for a source that
designs voices, or one :class:`Shortlisted` from the catalog of a source that does not. No schema here carries an
example: an example conception is a voice the model could hand back without conceiving one.
"""

from __future__ import annotations

import typing

import hsm
import pydantic

from mosfet import event

from ... import persona
from .. import speech
from . import catalog

MAX_ENTRIES = 200
"""The most catalog entries the inception model is shown."""


class Described(pydantic.BaseModel):
    """A voice described in plain language, for a source that designs voices."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "The voice of a newly made bot with these traits: who is speaking (apparent age and character), "
                "pitch, texture, warmth, pace and energy. Never a real person's voice."
            ),
        },
    )

    kind: typing.Literal["described"] = pydantic.Field(description="Always `described`.")
    description: str = pydantic.Field(
        min_length=20,
        max_length=1000,
        description="The voice in plain language, within the source's description_max_chars.",
    )
    descriptors: catalog.Descriptors | None = pydantic.Field(
        default=None, description="How the voice sounds, in canonical terms, when you can say."
    )
    languages: tuple[speech.Locale, ...] = pydantic.Field(
        default=(), description="The languages the voice will speak; the bot's languages when omitted."
    )


class Shortlisted(pydantic.BaseModel):
    """One voice from the catalog of a source that does not design voices."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"description": "The catalog voice that best fits these traits."},
    )

    kind: typing.Literal["shortlisted"] = pydantic.Field(description="Always `shortlisted`.")
    ref: catalog.Ref = pydantic.Field(description="The ref of one of the catalog entries you were shown.")


Conception = typing.Annotated[Described | Shortlisted, pydantic.Field(discriminator="kind")]
"""The one voice conceived for a newly made bot: described, or shortlisted from a catalog."""


class ConceiveData(pydantic.BaseModel):
    """Conceive the one voice a newly made bot speaks in for life.

    No example, deliberately: an example conception is a voice the model could hand back without conceiving one.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Conceive the one voice this newly made bot will speak in for life, from its traits alone. When the "
                "voice source designs voices (description_max_chars is set), describe the voice; otherwise "
                "shortlist the one catalog entry that best fits, by its ref. Select this exactly once."
            ),
        },
    )

    conception: Conception = pydantic.Field(description="The voice: described, or shortlisted from the catalog.")


ConceiveEvent = hsm.Event[ConceiveData](
    name="bot.ability.voice.conceive",
    kind=event.EventKind,
    schema=ConceiveData,
)
"""The inception model's one conception. Offered only on the inception request's own schemas, never on a turn."""


class InceptionInput(pydantic.BaseModel):
    """What the inception model is shown: who the bot is, the languages it speaks, and what its voice source can do."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A bot is being made: the traits its owner gave it, the languages it will speak, and what its voice "
                "source can do. Conceive its voice with bot.ability.voice.conceive."
            ),
        },
    )

    traits: tuple[persona.Trait, ...] = pydantic.Field(
        min_length=1, max_length=24, description="The traits the bot's owner gave it."
    )
    languages: tuple[speech.Locale, ...] = pydantic.Field(default=(), description="The languages the bot speaks.")
    description_max_chars: int | None = pydantic.Field(
        default=None,
        ge=20,
        le=1000,
        description="The longest description the source takes when it designs voices; null when it does not.",
    )
    entries: tuple[catalog.Entry, ...] = pydantic.Field(
        default=(),
        max_length=MAX_ENTRIES,
        description="The catalog's voices, when the source does not design voices; empty when it does.",
    )
    omitted: int = pydantic.Field(default=0, ge=0, description="How many catalog voices were left out of entries.")


__all__ = ["MAX_ENTRIES", "ConceiveData", "ConceiveEvent", "Conception", "Described", "InceptionInput", "Shortlisted"]

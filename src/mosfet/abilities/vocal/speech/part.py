"""A reply as structured parts: what to say, and optionally how to say each part, within the bot's own voice.

A reply is a list of :class:`Part`, each its ``words`` and an optional :class:`Expression`; a part that makes a
sound has no words to have. The words are joined verbatim (:func:`said`) into the text of record; nothing ever parses
them for sounds, tags or cues, because how a part is said is typed data, never text, and the characters of markup
and stage directions are refused in them rather than read. A :class:`Piece` is one append-only delta of such a reply as it is written:
the encoder reads a stream of pieces (``vocal.speech.Text``) and applies each part's expression.

:class:`Expressiveness` is what one speech encoder can do with an expression: the cues it applies, the emotions and
styles it knows, the ranges it clamps to, and the sounds it can make. A cue it does not declare is ignored by the
encoder (Speaking reports it); core never chooses one.

Everything here is canonical and provider-neutral: units are relative to the voice's natural delivery, and a
provider maps them to its own fields or markup.
"""

from __future__ import annotations

import collections.abc
import typing
from xml.sax import saxutils

import pydantic
from pydantic.json_schema import SkipJsonSchema

Locale = typing.Annotated[str, pydantic.StringConstraints(pattern=r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")]
"""A BCP-47 language tag, such as ``en`` or ``en-US``."""

Cue = typing.Literal["instructions", "emotion", "style", "intensity", "pace", "pitch", "volume", "emphasis", "sound"]
"""One kind of direction an expression can carry, named after its :class:`Expression` field."""

Sound = typing.Literal["laugh", "giggle", "chuckle", "sigh", "gasp", "breath", "cough", "clear_throat"]
"""A non-verbal vocal sound, made with a part's words or as a part of its own: a closed vocabulary that each encoder
renders or ignores."""

_SPOKEN = r"^[^*<>\[\]{}]+$"
"""What words may hold: anything but the characters of stage directions and markup (``* < > [ ] { }``), which a
voice would read aloud. A refusal at decoding (a grammar honours the pattern) and at validation, never a parse."""

_Words = typing.Annotated[str, pydantic.StringConstraints(min_length=1, max_length=2000, pattern=_SPOKEN)]

_Instructions = typing.Annotated[str, pydantic.StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
_Word = typing.Annotated[str, pydantic.StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class Range(pydantic.BaseModel):
    """A closed range of values an encoder applies; a value outside it is clamped to its nearer end."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{"min": 0.5, "max": 2.0}]}
    )

    min: float = pydantic.Field(description="The lowest value applied.", examples=[0.5])
    max: float = pydantic.Field(description="The highest value applied; at least ``min``.", examples=[2.0])

    @pydantic.model_validator(mode="after")
    def _ordered(self) -> typing.Self:
        if self.min > self.max:
            raise ValueError("A range's min must not exceed its max.")
        return self


class Expression(pydantic.BaseModel):
    """How to say one part of a reply, within the bot's own voice: every field optional, never all absent.

    Units are relative to the voice's natural delivery. An encoder applies only the cues it declares
    (:class:`Expressiveness`) and clamps values to its ranges; anything else is ignored and reported, never an error.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "How to say this part, within your own voice. Every field is optional; set only what this part "
                "needs. A voice applies what it can and ignores the rest."
            )
        },
    )

    instructions: _Instructions | None = pydantic.Field(
        default=None, description="Natural-language acting direction for this part."
    )
    emotion: _Word | None = pydantic.Field(
        default=None,
        description=(
            "A feeling, as free text. A voice with a closed vocabulary of emotions applies it only on an exact match."
        ),
    )
    style: _Word | None = pydantic.Field(
        default=None, description="A named style the voice declares; 'whisper' is the canonical name for whispering."
    )
    intensity: float | None = pydantic.Field(
        default=None, ge=0.0, le=1.0, description="How strongly to carry the emotion or style, from 0 to 1."
    )
    pace: float | None = pydantic.Field(
        default=None, ge=0.5, le=2.0, description="Speaking rate relative to natural: 1 is natural, 2 twice as fast."
    )
    pitch: float | None = pydantic.Field(
        default=None, ge=-12.0, le=12.0, description="Pitch in semitones relative to natural: 0 is natural."
    )
    volume: float | None = pydantic.Field(
        default=None, ge=0.25, le=2.0, description="Loudness relative to natural, linear: 1 is natural."
    )
    emphasis: bool = pydantic.Field(default=False, description="Stress the words of this part.")
    sound: Sound | None = pydantic.Field(
        default=None,
        description=(
            "A non-verbal sound (a laugh, sigh, gasp and the like), one per part: made beside this part's words, "
            "placed by sound_at, or alone as a part of its own with no words. This is the only way to make one: a "
            "written 'sigh' would be read aloud as speech. A voice that cannot make the sound ignores it."
        ),
    )
    sound_at: typing.Literal["before", "after"] = pydantic.Field(
        default="before",
        description="Whether the sound comes before or after this part's words; meaningful only with a sound.",
    )

    @pydantic.model_validator(mode="after")
    def _meaningful(self) -> typing.Self:
        if "sound_at" in self.model_fields_set and self.sound is None:
            raise ValueError("An expression places a sound only when it has one.")
        if not self.cues():
            raise ValueError("An expression must direct at least one cue.")
        return self

    def cues(self) -> frozenset[Cue]:
        """The cues this expression directs: each field that is set (``emphasis`` only when true)."""

        directed: dict[Cue, bool] = {
            "instructions": self.instructions is not None,
            "emotion": self.emotion is not None,
            "style": self.style is not None,
            "intensity": self.intensity is not None,
            "pace": self.pace is not None,
            "pitch": self.pitch is not None,
            "volume": self.volume is not None,
            "emphasis": self.emphasis,
            "sound": self.sound is not None,
        }
        return frozenset(cue for cue, set_ in directed.items() if set_)


class Part(pydantic.BaseModel):
    """One stretch of a reply: its words verbatim, optionally how to say them, or a sound of its own.

    A part has words, or an expression that makes a sound, or both; never neither.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "One stretch of your reply and, optionally, how to say it. Split the reply into parts where the way "
                "you say it changes, or to make a sound between words. The words are only what is spoken; a laugh, "
                "sigh or other non-verbal sound is expression.sound, never written into the words, and may be a part "
                "of its own with no words. Every part has words or a sound."
            )
        },
    )

    # Declared before the words: a model writes a part's fields in schema order, and how to say the part rides its
    # first draft only when it is written before the words begin (D-R-38).
    expression: Expression | None = pydantic.Field(
        default=None,
        description="How to say this part, written before its words; omit it to say the words naturally.",
    )
    words: _Words | SkipJsonSchema[None] = pydantic.Field(
        default=None,
        description=(
            "The words of this part; omit them for a part that is only a sound. The voice speaks every character "
            "exactly as written, so this is speech only: no stage directions, actions, emotes, markup or tags, and "
            "none of the characters * < > [ ] { }. For a sound use expression.sound; for how to say the words use "
            "expression. Continue the reply exactly, including any space or punctuation that joins this part to the "
            "one before it."
        ),
    )

    @pydantic.field_validator("words")
    @classmethod
    def _spoken(cls, words: str | None) -> str | None:
        if words is not None and not words.strip():
            raise ValueError("A part's words must not be blank.")
        return words

    @pydantic.model_validator(mode="after")
    def _something(self) -> typing.Self:
        if self.words is None and (self.expression is None or self.expression.sound is None):
            raise ValueError("A part must have words or a sound.")
        return self


class Piece(pydantic.BaseModel):
    """One append-only delta of a reply as it is written: words newly written into one part.

    ``part`` starts at 0 and only ever stays or grows by one; ``expression`` rides only the first piece of a part,
    as known when the part began, and a part with no words is that first piece alone. A part ends when the next one begins or the stream ends, and the words of every
    piece, joined in order, are the reply's text of record.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    part: int = pydantic.Field(ge=0, description="Index of the part these words belong to.")
    words: str = pydantic.Field(
        default="",
        description="Characters newly written into that part's words; empty only on a part's opening piece.",
    )
    expression: Expression | None = pydantic.Field(
        default=None, description="How the part is said; only on the first piece of a part."
    )


class Expressiveness(pydantic.BaseModel):
    """What one speech encoder can do with an :class:`Expression`, fixed with its voice.

    ``emotions`` is the closed vocabulary of emotions it applies, or ``None`` when it takes free text. ``styles``
    are the named styles it applies, ``sounds`` the sounds it can make, and ``pace``, ``pitch`` and ``volume`` the
    ranges it clamps those values to. Declaring a style, a sound or a range implies its cue.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    cues: frozenset[Cue] = pydantic.Field(default=frozenset[Cue](), description="The cues this encoder applies.")
    emotions: tuple[_Word, ...] | None = pydantic.Field(
        default=None, description="The emotions it applies, matched exactly; None means any free text."
    )
    styles: tuple[_Word, ...] = pydantic.Field(default=(), description="The named styles it applies.")
    pace: Range | None = pydantic.Field(default=None, description="The pace range it clamps to.")
    pitch: Range | None = pydantic.Field(default=None, description="The pitch range it clamps to.")
    volume: Range | None = pydantic.Field(default=None, description="The volume range it clamps to.")
    sounds: frozenset[Sound] = pydantic.Field(default=frozenset[Sound](), description="The sounds it can make.")

    @pydantic.model_validator(mode="after")
    def _declared(self) -> typing.Self:
        implied: dict[Cue, bool] = {
            "style": bool(self.styles),
            "sound": bool(self.sounds),
            "pace": self.pace is not None,
            "pitch": self.pitch is not None,
            "volume": self.volume is not None,
        }
        missing = sorted(cue for cue, declared in implied.items() if declared and cue not in self.cues)
        if missing:
            raise ValueError(f"Expressiveness declares values for cues it does not apply: {', '.join(missing)}.")
        return self

    def __model_repr__(self) -> str:
        """What this voice can do, for the bot to read: cues, vocabularies, ranges and sounds, in a stable order."""

        parts = [f"<cues>{' '.join(sorted(self.cues))}</cues>"]
        if self.emotions is not None:
            parts.append(f"<emotions>{saxutils.escape(' | '.join(self.emotions))}</emotions>")
        if self.styles:
            parts.append(f"<styles>{saxutils.escape(' | '.join(self.styles))}</styles>")
        for name, bounds in (("pace", self.pace), ("pitch", self.pitch), ("volume", self.volume)):
            if bounds is not None:
                parts.append(f'<{name} min="{bounds.min:g}" max="{bounds.max:g}"/>')
        if self.sounds:
            parts.append(f"<sounds>{' '.join(sorted(self.sounds))}</sounds>")
        return "".join(parts)


def said(parts: collections.abc.Iterable[Part]) -> str:
    """The text of record of a reply: its parts' words joined verbatim, with nothing added between them; a part
    with no words adds nothing."""

    return "".join(part.words or "" for part in parts)


__all__ = ["Cue", "Expression", "Expressiveness", "Locale", "Part", "Piece", "Range", "Sound", "said"]

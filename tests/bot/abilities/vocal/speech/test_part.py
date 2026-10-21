"""A reply as structured parts: Part, Piece, Expression, Expressiveness, Range, Locale, Cue, Sound and said."""

from __future__ import annotations

import json
import typing

import pydantic
import pytest

from mosfet.abilities.vocal import speech


def test_a_part_is_words_verbatim_and_an_optional_expression_and_its_schema_has_no_examples() -> None:
    part = speech.Part(words=" you got me.")

    assert part.words == " you got me."
    assert part.expression is None
    for blank in ("", "   ", "\n"):
        with pytest.raises(pydantic.ValidationError):
            _ = speech.Part(words=blank)
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Part(words="x" * 2001)
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Part.model_validate({"words": "Hi", "sound": "laugh"})
    schema = speech.Part.model_json_schema()
    assert "examples" not in json.dumps(schema)
    assert "Split the reply into parts where the way you say it changes" in schema["description"]
    # The model writes a part's fields in schema order: how to say it comes before the words, so it rides the first draft.
    assert tuple(schema["properties"]) == ("expression", "words")


def test_a_part_without_words_is_a_sound_of_its_own_and_a_part_is_never_empty() -> None:
    chuckle = speech.Part.model_validate({"expression": {"sound": "chuckle"}})

    assert chuckle.words is None
    assert chuckle.expression == speech.Expression(sound="chuckle")
    for empty in ({}, {"expression": {"emotion": "amused"}}, {"words": None}):
        with pytest.raises(pydantic.ValidationError, match="words or a sound"):
            _ = speech.Part.model_validate(empty)
    schema = speech.Part.model_json_schema()
    assert "words" not in schema.get("required", [])
    assert "a part of its own" in schema["description"]


def test_words_refuse_the_characters_of_stage_directions_and_markup_in_the_schema_too() -> None:
    for marked in ("*chuckles* Hi.", "Hi <laugh>", "[pause] Hi.", "Hi {sigh}", "</div>", "}}"):
        with pytest.raises(pydantic.ValidationError, match="pattern"):
            _ = speech.Part(words=marked)
    assert speech.Part(words="Well — “quoted”, (aside) & 50% — ok?!").words is not None
    words = speech.Part.model_json_schema()["properties"]["words"]
    assert words["pattern"] == r"^[^*<>\[\]{}]+$"
    # Omitted, never null: the model sees a string or nothing.
    assert (words["type"], "anyOf" in words) == ("string", False)


def test_an_expression_directs_at_least_one_cue_within_its_bounds() -> None:
    with pytest.raises(pydantic.ValidationError, match="at least one cue"):
        _ = speech.Expression()
    with pytest.raises(pydantic.ValidationError, match="at least one cue"):
        _ = speech.Expression(emphasis=False)
    with pytest.raises(pydantic.ValidationError, match="only when it has one"):
        _ = speech.Expression(sound_at="after", emotion="amused")
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Expression.model_validate({"emotion": "amused", "delivery": "fast"})
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Expression.model_validate({"sound": "yawn"})
    for field, value in (
        ("instructions", "x" * 301),
        ("emotion", "x" * 65),
        ("style", " "),
        ("intensity", 1.1),
        ("pace", 0.4),
        ("pace", 2.1),
        ("pitch", 12.5),
        ("volume", 0.2),
    ):
        with pytest.raises(pydantic.ValidationError):
            _ = speech.Expression.model_validate({field: value})

    every = speech.Expression(
        instructions="Like sharing a secret.",
        emotion="amused",
        style="whisper",
        intensity=0.5,
        pace=1.2,
        pitch=-2.0,
        volume=0.8,
        emphasis=True,
        sound="laugh",
        sound_at="after",
    )
    assert every.cues() == frozenset(typing.get_args(speech.Cue))
    assert speech.Expression(sound="giggle").sound_at == "before"
    assert speech.Expression(sound="giggle").cues() == {"sound"}
    assert speech.Expression(emphasis=True).cues() == {"emphasis"}


def test_the_canonical_cues_and_sounds_are_closed_vocabularies() -> None:
    assert set(typing.get_args(speech.Cue)) == {
        "instructions",
        "emotion",
        "style",
        "intensity",
        "pace",
        "pitch",
        "volume",
        "emphasis",
        "sound",
    }
    assert set(typing.get_args(speech.Sound)) == {
        "laugh",
        "giggle",
        "chuckle",
        "sigh",
        "gasp",
        "breath",
        "cough",
        "clear_throat",
    }


def test_a_piece_is_one_append_only_delta_of_a_part() -> None:
    opening = speech.Piece(part=0, expression=speech.Expression(sound="laugh"))

    assert opening.words == ""
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Piece(part=-1, words="x")
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Piece.model_validate({"part": 0, "words": "x", "text": "x"})


def test_said_joins_the_words_of_every_part_verbatim() -> None:
    parts = (
        speech.Part(words="Oh, ", expression=speech.Expression(sound="laugh")),
        speech.Part(words="you got me."),
        speech.Part(words="好的"),
    )

    assert speech.said(parts) == "Oh, you got me.好的"
    assert speech.said(()) == ""


def test_a_range_is_ordered_and_a_locale_is_a_bcp47_tag() -> None:
    assert speech.Range(min=0.5, max=0.5).max == 0.5
    with pytest.raises(pydantic.ValidationError, match="min"):
        _ = speech.Range(min=2.0, max=1.0)
    locale: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(speech.Locale)
    assert locale.validate_python("en-US") == "en-US"
    assert locale.validate_python("yue") == "yue"
    for invalid in ("EN", "english", "en_US", ""):
        with pytest.raises(pydantic.ValidationError):
            _ = locale.validate_python(invalid)


def test_expressiveness_declares_what_an_encoder_applies_and_implies_each_declared_cue() -> None:
    assert speech.Expressiveness().cues == frozenset()
    for declared in (
        {"styles": ("whisper",)},
        {"sounds": {"laugh"}},
        {"pace": speech.Range(min=0.5, max=2.0)},
        {"pitch": speech.Range(min=-2.0, max=2.0)},
        {"volume": speech.Range(min=0.5, max=1.5)},
    ):
        with pytest.raises(pydantic.ValidationError, match="does not apply"):
            _ = speech.Expressiveness.model_validate(declared)
    with pytest.raises(pydantic.ValidationError):
        _ = speech.Expressiveness.model_validate({"cues": {"timbre"}})


def test_expressiveness_renders_for_the_bot_in_a_stable_order() -> None:
    lab = speech.Expressiveness(
        cues=frozenset({"instructions", "emotion", "intensity", "emphasis", "style", "sound"}),
        styles=("whisper",),
        sounds=frozenset({"laugh", "giggle", "chuckle", "gasp", "sigh"}),
    )
    closed = speech.Expressiveness(
        cues=frozenset({"emotion", "pace"}), emotions=("calm", "angry & loud"), pace=speech.Range(min=0.7, max=1.2)
    )

    assert lab.__model_repr__() == (
        "<cues>emotion emphasis instructions intensity sound style</cues>"
        "<styles>whisper</styles>"
        "<sounds>chuckle gasp giggle laugh sigh</sounds>"
    )
    assert closed.__model_repr__() == (
        '<cues>emotion pace</cues><emotions>calm | angry &amp; loud</emotions><pace min="0.7" max="1.2"/>'
    )
    assert speech.Expressiveness().__model_repr__() == "<cues></cues>"

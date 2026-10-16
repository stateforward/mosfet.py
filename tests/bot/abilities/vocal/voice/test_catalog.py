"""Canonical voice catalog schemas: Ref, Descriptors, Preview, Entry, Error, and the Catalog collaborator."""

from __future__ import annotations

import inspect
import json

import pydantic
import pytest

from mosfet.abilities.vocal import speech
from mosfet.abilities.vocal import voice


@pytest.mark.parametrize("source", ["lab", "lab-2", "on.device_1", "0x"])
def test_a_ref_names_a_source_by_its_composition_name_and_a_key_opaque_to_core(source: str) -> None:
    ref = voice.Ref(source=source, key="v-3f2a9c41d07e")

    assert (ref.source, ref.key) == (source, "v-3f2a9c41d07e")


@pytest.mark.parametrize(
    ("source", "key"),
    [
        ("", "warm"),
        ("Lab", "warm"),
        ("-lab", "warm"),
        ("lab src", "warm"),
        ("x" * 65, "warm"),
        ("lab", ""),
        ("lab", "k" * 257),
    ],
)
def test_a_ref_is_bounded(source: str, key: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = voice.Ref(source=source, key=key)


def test_descriptors_are_canonical_and_bounded() -> None:
    described = voice.Descriptors(
        gender="neutral", age="young_adult", accent="Irish", pitch="high", qualities=("bright",)
    )

    assert described.qualities == ("bright",)
    assert voice.Descriptors() == voice.Descriptors(gender=None, age=None, accent=None, pitch=None, qualities=())
    for invalid in (
        {"gender": "female"},
        {"age": "teen"},
        {"pitch": "very high"},
        {"accent": ""},
        {"qualities": ("q",) * 9},
        {"timbre": "warm"},
    ):
        with pytest.raises(pydantic.ValidationError):
            _ = voice.Descriptors.model_validate(invalid)


def test_a_preview_is_audio_in_place_with_its_format_or_at_a_url() -> None:
    pcm_format = speech.Format(sample_rate_hz=24000, channels=1)

    assert voice.Preview(pcm=b"\x00\x00", format=pcm_format).url is None
    assert str(voice.Preview(url=pydantic.AnyHttpUrl("http://lab/v1/voices/warm/preview")).url).endswith("preview")
    with pytest.raises(pydantic.ValidationError, match="format"):
        _ = voice.Preview(pcm=b"\x00\x00")
    with pytest.raises(pydantic.ValidationError, match="pcm or its url"):
        _ = voice.Preview(text="Hello there.")


def test_an_entry_describes_one_voice_canonically() -> None:
    entry = voice.Entry(ref=voice.Ref(source="lab", key="warm"), name="warm", languages=("en", "en-US"))

    assert entry.description is None
    assert entry.descriptors == voice.Descriptors()
    assert (entry.styles, entry.preview, entry.labels) == ((), None, {})
    with pytest.raises(pydantic.ValidationError):
        _ = voice.Entry(ref=entry.ref, name="warm", languages=("English",))
    with pytest.raises(pydantic.ValidationError):
        _ = voice.Entry(ref=entry.ref, name="")
    schema = json.dumps(voice.Entry.model_json_schema())
    for provider in ("elevenlabs", "gemini", "openai", "breeze", "kokoro"):
        assert provider not in schema.lower()


def test_a_voice_source_error_carries_its_closed_kind_for_telemetry() -> None:
    error = voice.Error("design_failed", "the source could not render it")

    assert (error.kind, error.failure_kind, str(error)) == (
        "design_failed",
        "design_failed",
        "the source could not render it",
    )


def test_a_catalog_is_an_abstract_collaborator() -> None:
    class Listing(voice.Catalog):
        def __init__(self) -> None:
            self.source = "lab"

        async def voices(
            self, *, languages: tuple[speech.Locale, ...] = (), descriptors: voice.Descriptors | None = None
        ) -> tuple[voice.Entry, ...]:
            del languages, descriptors
            return ()

    assert inspect.isabstract(voice.Catalog)
    assert Listing().source == "lab"

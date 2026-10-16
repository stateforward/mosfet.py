"""Unit tests for ``mlx_audio.Catalog``: the voices composition declares the on-device model speaks in."""

from __future__ import annotations

import asyncio

import hsm
import pytest

from mosfet.abilities.vocal import voice
from mosfet.providers import mlx_audio

_HEART = voice.Entry(
    ref=voice.Ref(source="kokoro", key="af_heart"),
    name="Heart",
    languages=("en-US",),
    descriptors=voice.Descriptors(gender="feminine"),
)
_GEORGE = voice.Entry(
    ref=voice.Ref(source="kokoro", key="bm_george"),
    name="George",
    languages=("en-GB",),
    descriptors=voice.Descriptors(gender="masculine"),
)


def test_a_catalog_is_a_core_voice_catalog_of_its_entries_source() -> None:
    catalog = mlx_audio.Catalog((_HEART, _GEORGE))

    assert isinstance(catalog, voice.Catalog)
    assert not isinstance(catalog, hsm.Instance)
    assert catalog.source == "kokoro"


def test_the_voices_are_the_declared_entries_in_order_whatever_the_hints() -> None:
    catalog = mlx_audio.Catalog([_HEART, _GEORGE])

    hinted = asyncio.run(catalog.voices(languages=("fr",), descriptors=voice.Descriptors(gender="neutral")))

    assert hinted == (_HEART, _GEORGE)
    assert asyncio.run(catalog.voices()) == (_HEART, _GEORGE)


@pytest.mark.parametrize(
    "entries",
    [
        (),
        (_HEART, _GEORGE.model_copy(update={"ref": voice.Ref(source="other", key="bm_george")})),
        (_HEART, _HEART.model_copy(update={"name": "Heart again"})),
    ],
    ids=("empty", "two-sources", "one-key-twice"),
)
def test_a_catalog_is_one_source_with_each_voice_once(entries: tuple[voice.Entry, ...]) -> None:
    with pytest.raises(ValueError):
        _ = mlx_audio.Catalog(entries)

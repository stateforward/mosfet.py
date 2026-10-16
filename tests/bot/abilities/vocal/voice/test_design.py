"""Canonical voice design schemas: DesignRequest, Designed, and the Designer collaborator."""

from __future__ import annotations

import inspect

import pydantic
import pytest

from mosfet.abilities.vocal import voice

_DESCRIPTION = "A bright, quick young voice with a smile in it."


def test_a_design_request_is_one_described_voice_to_keep() -> None:
    request = voice.DesignRequest(description=_DESCRIPTION, languages=("en",))

    assert (request.descriptors, request.sample_text) == (None, None)
    for invalid in (
        {"description": "Too short."},
        {"description": "x" * 1001},
        {"description": _DESCRIPTION, "languages": ("English",)},
        {"description": _DESCRIPTION, "sample_text": ""},
        {"description": _DESCRIPTION, "count": 4},
    ):
        with pytest.raises(pydantic.ValidationError):
            _ = voice.DesignRequest.model_validate(invalid)


def test_a_designed_voice_is_permanent_at_its_source_under_its_ref() -> None:
    designed = voice.Designed(ref=voice.Ref(source="lab", key="v-3f2a9c41d07e"), description=_DESCRIPTION)

    assert designed.preview is None
    with pytest.raises(pydantic.ValidationError):
        _ = voice.Designed(ref=designed.ref, description="")


def test_a_designer_is_an_abstract_collaborator_with_its_source_and_description_bound() -> None:
    assert inspect.isabstract(voice.Designer)
    assert {"source", "max_description"} <= set(voice.Designer.__annotations__)

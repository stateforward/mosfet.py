"""Direct contract tests for cognition-owned model-facing event projections."""

from mosfet.abilities import cognition
from mosfet.environment import SoundData, SoundEvent

import typing

import pydantic
import pytest


def test_model_facing_xml_stamps_envelope_identity() -> None:
    rendered = cognition.event.model_facing_xml(
        SoundEvent.with_data_and_id(
            SoundData(audio=b"ring", media_type="audio/pcm", sample_rate_hz=16_000, channels=1, kind="phone.ringing"),
            "turn-1",
        )
    )

    assert SoundEvent.name in rendered
    assert "turn-1" in rendered
    assert "phone.ringing" in rendered


def test_model_facing_xml_never_renders_raw_media() -> None:
    with pytest.raises(TypeError):
        _ = cognition.event.model_facing_xml(b"raw-bytes")


def test_model_facing_xml_enforces_scalar_budget() -> None:
    rendered = cognition.event.model_facing_xml("short stimulus")
    assert "short stimulus" in rendered

    class _LoudCarrier(pydantic.BaseModel):
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

        text: str

    with pytest.raises(ValueError, match="budget"):
        _ = cognition.event.model_facing_xml(_LoudCarrier(text="x" * 70_000))

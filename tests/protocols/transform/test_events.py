"""Tests for the stream ownership events a duplex owner is told."""

import typing

import hsm
import pydantic
import pytest

import mosfet.event
from mosfet.protocols import transform
from tests.type_helpers import object_dict


@pytest.mark.parametrize(
    ("public", "name"),
    [
        (transform.OwnedEvent, "protocol.transform.owned"),
        (transform.EndedEvent, "protocol.transform.ended"),
    ],
)
def test_ownership_events_are_named_in_their_domain_and_are_not_tools(public: hsm.Event[object], name: str) -> None:
    assert public.name == name
    assert public.kind is not mosfet.event.EventKind


def test_owned_names_the_channel_and_carries_the_stream_at_runtime_only() -> None:
    stream = hsm.Instance()

    data = transform.OwnedData(stream=stream, channel="display")

    assert data.stream is stream
    assert data.model_dump() == {"channel": "display"}
    assert set(object_dict(typing.cast(object, transform.OwnedData.model_json_schema()["properties"]))) == {"channel"}


@pytest.mark.parametrize("reason", ["finished", "reset", "closed", "handed_over"])
def test_ended_says_why_the_owner_no_longer_owns_the_stream(reason: str) -> None:
    stream = hsm.Instance()

    data = transform.EndedData.model_validate({"stream": stream, "channel": "speaker", "reason": reason})

    assert data.stream is stream
    assert data.model_dump() == {"channel": "speaker", "reason": reason}
    assert set(object_dict(typing.cast(object, transform.EndedData.model_json_schema()["properties"]))) == {
        "channel",
        "reason",
    }


def test_ended_refuses_a_reason_outside_its_closed_set() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = transform.EndedData.model_validate({"stream": hsm.Instance(), "channel": "speaker", "reason": "lost"})


def test_ended_is_not_an_owned_payload() -> None:
    ended = transform.EndedData(stream=hsm.Instance(), channel="display", reason="finished")

    assert not isinstance(typing.cast(object, ended), transform.OwnedData)


@pytest.mark.parametrize("channel", ["keyboard", "display", "a", "camera.v2", "mic_1-left", "a" * 64])
def test_channel_names_are_short_lower_case_wire_labels(channel: str) -> None:
    assert transform.OwnedData(stream=hsm.Instance(), channel=channel).channel == channel


@pytest.mark.parametrize("channel", ["", "Keyboard", "1keyboard", "-keyboard", "key board", "a" * 65, "kéy"])
def test_channel_names_outside_the_pattern_are_refused(channel: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = transform.OwnedData(stream=hsm.Instance(), channel=channel)

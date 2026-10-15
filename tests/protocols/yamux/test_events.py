import typing

import hsm
import pydantic
import pytest

import mosfet.event
from mosfet.protocols import yamux
from tests.protocols.yamux import wire
from tests.type_helpers import object_dict


@pytest.mark.parametrize(
    ("event", "name", "schema"),
    [
        (yamux.OpenStreamEvent, "protocol.yamux.open_stream", yamux.OpenStreamData),
        (yamux.OpenStreamFailedEvent, "protocol.yamux.open_stream.failed", yamux.OpenStreamFailedData),
        (yamux.ReceiveDataFrameEvent, "protocol.yamux.frame.data.received", yamux.ReceiveDataFrameData),
        (yamux.TransmitEvent, "protocol.yamux.transmit", yamux.TransmitData),
        (yamux.StreamAcceptedEvent, "protocol.yamux.stream.accepted", yamux.StreamAcceptedData),
        (yamux.EndedEvent, "protocol.yamux.session.ended", yamux.EndedData),
        (yamux.CloseEvent, "protocol.yamux.stream.close", yamux.CloseData),
        (yamux.ResetEvent, "protocol.yamux.stream.reset", yamux.ResetData),
        (yamux.HandOverEvent, "protocol.yamux.stream.hand_over", yamux.HandOverData),
        (yamux.FrameOutEvent, "protocol.yamux.frame.out", yamux.FrameOutData),
        (yamux.StreamEndedEvent, "protocol.yamux.stream.ended", yamux.StreamIdData),
        (yamux.ViolationEvent, "protocol.yamux.stream.violation", yamux.ViolationData),
    ],
)
def test_yamux_events_are_named_in_their_domain_typed_and_not_tools(
    event: hsm.Event[typing.Any], name: str, schema: type[pydantic.BaseModel]
) -> None:
    assert event.name == name
    assert object_dict(event.schema) == schema.model_json_schema()
    assert event.kind is not mosfet.event.EventKind


def test_session_drained_is_a_completion() -> None:
    assert yamux.SessionDrainedEvent.kind == hsm.CompletionEventKind
    assert object_dict(yamux.SessionDrainedEvent.schema) == yamux.SessionDrainedData.model_json_schema()


def test_runtime_addresses_and_codecs_never_reach_the_wire_schema() -> None:
    owner = hsm.Instance()
    codec = wire.codec()

    opened = yamux.OpenStreamData(owner=owner, codec=codec, channel="notes")
    handed = yamux.HandOverData(owner=owner, codec=codec)
    accepted = yamux.StreamAcceptedData(stream_id=2, stream=owner)

    assert (opened.owner, opened.codec) == (owner, codec)
    assert opened.model_dump() == {"initial_window_size": yamux.INITIAL_STREAM_WINDOW, "channel": "notes"}
    assert handed.channel is None
    assert handed.model_dump() == {"channel": None}
    assert accepted.model_dump() == {"stream_id": 2}
    for schema in (yamux.OpenStreamData, yamux.HandOverData, yamux.StreamAcceptedData):
        properties = object_dict(typing.cast(object, schema.model_json_schema()["properties"]))
        assert not {"owner", "codec", "stream"} & set(properties)


def test_open_names_a_channel_inside_its_pattern() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = yamux.OpenStreamData(owner=hsm.Instance(), codec=wire.codec(), channel="Not A Channel")
    with pytest.raises(pydantic.ValidationError):
        _ = yamux.OpenStreamData.model_validate({"owner": hsm.Instance(), "codec": wire.codec()})


def test_closed_reasons_and_kinds() -> None:
    assert yamux.EndedData(reason="drained").reason == "drained"
    with pytest.raises(pydantic.ValidationError):
        _ = yamux.EndedData.model_validate({"reason": "went_away"})
    with pytest.raises(pydantic.ValidationError):
        _ = yamux.OpenStreamFailedData.model_validate({"kind": "busy"})
    with pytest.raises(pydantic.ValidationError):
        _ = yamux.CloseData.model_validate({"stream_id": 1})

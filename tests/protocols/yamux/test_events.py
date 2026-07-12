import hsm

from bot.protocols.yamux.events import (
    OpenStreamEvent,
    ReceiveDataFrameEvent,
    SendDataEvent,
    SessionDrainedEvent,
    OpenStreamData,
    ReceiveDataFrameData,
    SendData,
    SessionDrainedData,
)
from tests.type_helpers import object_dict


def test_yamux_events_use_pydantic_schemas() -> None:
    assert OpenStreamEvent.name == "protocol.yamux.open_stream"
    assert object_dict(OpenStreamEvent.schema) == OpenStreamData.model_json_schema()
    assert object_dict(SendDataEvent.schema) == SendData.model_json_schema()
    assert object_dict(ReceiveDataFrameEvent.schema) == ReceiveDataFrameData.model_json_schema()

    assert SessionDrainedEvent.kind == hsm.CompletionEventKind
    assert object_dict(SessionDrainedEvent.schema) == SessionDrainedData.model_json_schema()

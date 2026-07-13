"""Tests for the public attachment event contract."""

import hsm

from bot.protocols import attachment
from tests.type_helpers import object_dict


def test_attachment_protocol_defines_correlated_lifecycle_outcomes() -> None:
    actor = hsm.Instance()
    reply_to = hsm.Instance()

    attach = attachment.AttachEvent.with_data_and_id(
        attachment.AttachData(actor=actor, reply_to=reply_to),
        "attach-operation",
    )
    detach = attachment.DetachEvent.with_data_and_id(
        attachment.DetachData(actor=actor, reply_to=reply_to),
        "detach-operation",
    )

    assert attachment.AttachEvent.name == "attachment.attach"
    assert attachment.AttachCompleteEvent.name == "attachment.attach.complete"
    assert attachment.AttachFailedEvent.name == "attachment.attach.failed"
    assert attachment.DetachEvent.name == "attachment.detach"
    assert attachment.DetachedEvent.name == "attachment.detached"
    assert attachment.DetachFailedEvent.name == "attachment.detach.failed"
    assert attach.id == "attach-operation"
    assert detach.id == "detach-operation"
    assert attach.data.reply_to is reply_to
    assert detach.data.reply_to is reply_to


def test_attachment_protocol_schemas_expose_actor_ids_but_not_runtime_reply_targets() -> None:
    schema = attachment.AttachData.model_json_schema()
    properties = object_dict(schema["properties"])

    assert properties["actor"]
    assert "reply_to" not in properties
    assert schema["required"] == ["actor"]

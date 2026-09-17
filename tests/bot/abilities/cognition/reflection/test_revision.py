import asyncio
import dataclasses
import hashlib
import json

import hsm
import pytest

from mosfet import behavior
from mosfet.abilities import memory
from mosfet.abilities.cognition.reflection import revision
from tests.bot.abilities.cognition.test_cognition import FixedProcessor
from tests.bot.abilities.cognition.test_cognition import cognition_input
from tests.bot.abilities.cognition.test_cognition import focus_output
from tests.bot.abilities.support import dispatch_ability_for_test

ANSWER_RING_BEHAVIOR_SOURCE = """
input_event = hsm.event(
    name = "bot.behavior.answer_incoming_ring.input",
    schema = {"type": "object", "additionalProperties": True},
    description = "Live behavior input.",
)
output_event = hsm.event(
    name = "bot.behavior.answer_incoming_ring.output",
    schema = {
        "type": "object",
        "properties": {"event": {"type": "string"}},
        "required": ["event"],
        "additionalProperties": True,
    },
    description = "Cognition event selection.",
)
triggers = ["environment.sound"]
description = "Answer an incoming ring."

def select_focus(event):
    hsm.dispatch(output_event, {"event": "bot.focus_device"})

behavior = hsm.define(
    "AnswerIncomingRing",
    hsm.initial(hsm.target("/AnswerIncomingRing/idle")),
    hsm.state(
        "idle",
        hsm.transition(hsm.on(input_event), hsm.effect("select_focus")),
    ),
)
""".strip()


def test_revision_owns_a_typed_ability_boundary() -> None:
    assert revision.Revision.input_data_type is revision.InputData
    assert revision.Revision.output_data_type is revision.OutputData
    assert revision.Revision.failed_event.schema is revision.FailureData


def test_change_write_input_preserves_public_constructor_and_model_schema() -> None:
    write = revision.ChangeWriteInput(
        cognition_input=cognition_input(),
        cognition_output=focus_output("phone", "answered"),
        intent=behavior.ChangeData(name="AnswerIncomingRing"),
        existing_behavior=behavior.Instance(name="AnswerIncomingRing"),
        operation_id="turn-42",
        generation="parent-generation",
        attempt=0,
    )

    assert set(revision.ChangeWriteInput.model_fields) == {
        "attempt",
        "cognition_input",
        "cognition_output",
        "create_intent",
        "diagnostics",
        "existing_behavior",
        "failed_source",
        "generation",
        "instruction",
        "intent",
        "operation_id",
        "previous_messages",
        "prior_episodes",
    }
    payload = write.model_dump(mode="json")
    assert "revision_input" not in payload
    assert "parent_operation_id" not in payload
    assert "parent_generation" not in payload
    schema = revision.ChangeWriteInput.model_json_schema()
    properties = schema["properties"]
    assert "revision_input" not in properties
    assert set(properties) == {
        "attempt",
        "cognition_output",
        "create_intent",
        "diagnostics",
        "existing_behavior",
        "failed_source",
        "generation",
        "instruction",
        "intent",
        "operation_id",
        "previous_messages",
        "prior_episodes",
    }
    canonical_schema = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    assert (
        hashlib.sha256(canonical_schema).hexdigest()
        == "30d734915bfd210abefbbe9d8fec6b3b052ce944bcb06568650a739ce261483a"
    )


def test_revision_input_schema_documents_its_new_typed_boundary() -> None:
    properties = revision.InputData.model_json_schema()["properties"]

    assert all(property_schema.get("description") for property_schema in properties.values())


def test_revision_rejects_forged_private_checked_event(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = revision.Revision(processor=FixedProcessor(write=[]), memory=memory.Memory())
    write = revision.ChangeWriteInput(
        cognition_input=cognition_input(),
        cognition_output=focus_output("phone", "answered"),
        intent=behavior.ChangeData(name="AnswerIncomingRing"),
        existing_behavior=behavior.Instance(name="AnswerIncomingRing"),
        operation_id="private-turn",
        generation="revision-generation",
        attempt=0,
    )
    written = behavior.ChangeData(name="AnswerIncomingRing", source=ANSWER_RING_BEHAVIOR_SOURCE)
    checked = getattr(revision, "_CheckedData")(
        write=write,
        generation="revision-generation",
        written=written,
        behavior_instance=behavior.Instance(name="AnswerIncomingRing", source=ANSWER_RING_BEHAVIOR_SOURCE),
    )
    monkeypatch.setattr(revision.processing, "matches_operation", lambda *args: True)
    monkeypatch.setattr(revision.hsm, "id", lambda instance: "revision" if instance is actor else "other")
    forged = dataclasses.replace(
        getattr(revision, "_CheckedEvent").with_data(checked),
        id="private-turn",
        source="forged",
        target="revision",
    )

    assert not getattr(revision.Revision, "_accepted")(hsm.Context(), actor, forged)


def test_revision_rejects_stale_same_attempt_generation_output(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = revision.Revision(processor=FixedProcessor(write=[]), memory=memory.Memory())
    current_operation = object()
    write = revision.ChangeWriteInput(
        cognition_input=cognition_input(),
        cognition_output=focus_output("phone", "answered"),
        intent=behavior.ChangeData(name="AnswerIncomingRing"),
        existing_behavior=behavior.Instance(name="AnswerIncomingRing"),
        operation_id="retry-turn",
        generation="parent-generation",
        attempt=0,
    )
    terminal = dataclasses.replace(
        getattr(actor, "_change_processing").output_event.with_data(
            revision.processing.CompletionData(
                input=revision.processing.InputData(input=write),
                output=revision.processing.OutputData(),
            )
        ),
        id=getattr(revision.Revision, "_child_id")("retry-turn", 0, "stale-revision-generation"),
        source="change-processing",
        target="revision",
    )
    monkeypatch.setattr(actor, "state", lambda: "/Revision/authoring")
    monkeypatch.setattr(revision.processing, "active_operation_id", lambda instance: "retry-turn")
    monkeypatch.setattr(revision.processing, "active_operation", lambda instance, operation_id: current_operation)
    monkeypatch.setattr(
        revision.hsm,
        "id",
        lambda instance: (
            "revision"
            if instance is actor
            else "change-processing"
            if instance is getattr(actor, "_change_processing")
            else "current-revision-generation"
        ),
    )

    assert not getattr(revision.Revision, "_matches_change_output")(hsm.Context(), actor, terminal)


def test_revision_failed_draft_preserves_existing_inventory_metadata() -> None:
    existing = behavior.Instance(
        name="AnswerIncomingRing",
        source="old source",
        triggers=("environment.sound",),
        description="keep me",
    )
    written = behavior.ChangeData(name="AnswerIncomingRing", source="invalid replacement")

    draft = getattr(revision, "_inventory_instance")(written, None, existing)

    assert draft is not None
    assert draft.triggers == ("environment.sound",)
    assert draft.description == "keep me"


def test_revision_creates_validates_and_persists_a_behavior() -> None:
    async def run() -> revision.OutputData:
        written = behavior.ChangeData(
            name="AnswerIncomingRing",
            source=ANSWER_RING_BEHAVIOR_SOURCE,
        )
        actor = revision.Revision(
            processor=FixedProcessor(write=written),
            memory=memory.Memory(),
        )
        output = await dispatch_ability_for_test(
            actor,
            None,
            revision.InputData(
                cognition_input=cognition_input(),
                cognition_output=focus_output("phone", "answered"),
                intent=behavior.CreateData(
                    name="AnswerIncomingRing",
                    triggers=("environment.sound",),
                ),
                parent_operation_id="parent-turn",
                parent_generation="parent-generation",
            ),
        )
        assert isinstance(output, revision.OutputData)
        return output

    output = asyncio.run(run())

    assert output.input.parent_operation_id == "parent-turn"
    assert isinstance(output.applied, behavior.CreateData)
    assert output.applied.source == ANSWER_RING_BEHAVIOR_SOURCE
    assert output.applied.name == "AnswerIncomingRing"
    assert output.applied.triggers == ("environment.sound",)
    assert output.applied.description == "Answer an incoming ring."


def _change_intent():
    return behavior.ChangeData(
        name="HandlePhoneCall",
        triggers=("environment.sound",),
        description=None,
        reason="the bot learned that calls should be answered",
    )


def _change_written(*, triggers: tuple[str, ...]) -> behavior.ChangeData:
    return behavior.ChangeData(
        name="HandlePhoneCall",
        triggers=triggers,
        description=None,
        reason="authored by the model",
        source="behavior = hsm.define('HandlePhoneCall')",
    )


def test_grounded_triggers_replace_model_invented_names_with_evidence() -> None:
    """A model-invented trigger the bot can never experience must not reach inventory.

    The e2e caught a learned behavior authored with trigger `phone.incoming_call` — a name
    no bot ever sees as a stimulus — landing ACTIVE-shaped and never firing. Grounding
    keeps evidence-backed triggers, replaces invented ones with the observed stimulus
    name, and never invents a name when evidence is absent.
    """

    from mosfet.abilities.cognition import episodes as episodes_module
    from mosfet.abilities.cognition import input as cognition_input
    from mosfet.abilities.cognition.reflection.revision import ChangeWriteInput, _grounded_triggers
    from mosfet.environment import SoundData, SoundEvent

    stimulus = SoundEvent.with_data(
        SoundData(audio=b"ring-bytes", media_type="audio/wav", sample_rate_hz=16_000, channels=1)
    )
    write = ChangeWriteInput(
        cognition_input=cognition_input.InputData(
            stimulus=stimulus,
            abilities=(),
            actors={},
            focus=None,
            focus_candidates=(),
        ),
        cognition_output=(),
        prior_episodes=(
            episodes_module.CognitiveEpisode(
                focus=None,
                focus_candidates=(),
                stimulus_name="environment.sound",
                output=(),
            ),
        ),
        intent=_change_intent(),
        existing_behavior=behavior.Instance(
            name="HandlePhoneCall",
            source="",
            triggers=(),
            description="",
        ),
        operation_id="ops",
        generation="gen",
        attempt=0,
    )
    invented = _change_written(triggers=("phone.incoming_call",))
    assert _grounded_triggers(write, invented) == ("environment.sound",)

    # Evidence-backed triggers pass through untouched.
    honest = _change_written(triggers=("environment.sound",))
    assert _grounded_triggers(write, honest) == ("environment.sound",)

    # With names on both sides, only the evidence-backed survive.
    wild = _change_written(triggers=("phone.incoming_call", "environment.sound", "kitchen.chimney_sweep"))
    assert _grounded_triggers(write, wild) == ("environment.sound",)

    # With no turn or episode evidence beyond the write itself, nothing can be proven:
    # the invented trigger keeps nothing.
    evidenceless_write = ChangeWriteInput(
        cognition_input=cognition_input.InputData(
            stimulus=SoundEvent.with_data(
                SoundData(audio=b"ring-bytes", media_type="audio/wav", sample_rate_hz=16_000, channels=1),
            ),
            abilities=(),
            actors={},
            focus=None,
            focus_candidates=(),
        ),
        cognition_output=(),
        prior_episodes=(),
        intent=_change_intent(),
        existing_behavior=behavior.Instance(
            name="HandlePhoneCall",
            source="",
            triggers=(),
            description="",
        ),
        operation_id="ops",
        generation="gen",
        attempt=0,
    )
    assert _grounded_triggers(evidenceless_write, invented) == ("environment.sound",)

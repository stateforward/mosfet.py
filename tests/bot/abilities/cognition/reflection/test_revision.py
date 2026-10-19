import asyncio
import dataclasses
import datetime
import hashlib
import json
import typing

import hsm
import pytest

from mosfet.abilities.cognition import inventory

from mosfet import behavior
from mosfet.abilities import memory, processing
from mosfet.abilities.cognition.reflection import revision
from tests.bot.abilities.cognition.test_cognition import FixedProcessor
from tests.bot.abilities.cognition.test_cognition import cognition_input
from tests.bot.abilities.cognition.test_cognition import focus_output
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test

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


# Checking a written behavior dry-runs it synchronously; that alone can take seconds on a loaded host.
_REVISION_TIMEOUT_SECONDS = 60.0


async def _attached_memory() -> tuple[hsm.Context, memory.Memory]:
    """The bot's memory, attached by its own owner; Revision is only its peer (no direct calls)."""

    store = memory.Memory()
    ctx = shared_hsm_context()
    await start_abilities_for_test(ctx, store)
    return ctx, store


def _revision_input(intent: behavior.CreateData | behavior.ChangeData) -> revision.InputData:
    return revision.InputData(
        cognition_input=cognition_input(),
        cognition_output=focus_output("phone", "answered"),
        intent=intent,
        parent_operation_id="parent-turn",
        parent_generation="parent-generation",
    )


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
        == "5718d6f466e94758473c14ea94af83885463aee72ba3eeb4b3c5b0641f729a0b"
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
        _, store = await _attached_memory()
        actor = revision.Revision(
            processor=FixedProcessor(write=written),
            memory=store,
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
            timeout=_REVISION_TIMEOUT_SECONDS,
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


def test_revision_keeps_the_existing_behaviors_own_grounded_triggers() -> None:
    """A program-only revision from an unrelated turn keeps the installed trigger.

    The installed behavior already fires on `environment.sound`; revising it from a turn
    whose stimulus is a different name must not strip that trigger and retarget the
    behavior onto the current stimulus. An invented name with no evidence still drops.
    """

    from mosfet.behavior import storage as behavior_storage

    revised_source = ANSWER_RING_BEHAVIOR_SOURCE.replace(
        'description = "Answer an incoming ring."',
        'description = "Answer an incoming ring, revised."',
    )
    written = behavior.ChangeData(
        name="AnswerIncomingRing",
        triggers=("environment.sound", "phone.ringing"),
        source=revised_source,
    )

    async def run() -> revision.OutputData:
        ctx, store = await _attached_memory()
        installed = behavior.start(
            ANSWER_RING_BEHAVIOR_SOURCE,
            name="AnswerIncomingRing",
            triggers=("environment.sound",),
        )
        _ = await memory.transact(
            ctx,
            store,
            memory.InputData(
                statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(installed))
            ),
        )
        actor = revision.Revision(processor=FixedProcessor(write=written), memory=store)
        output = await dispatch_ability_for_test(
            actor,
            None,
            revision.InputData(
                cognition_input=cognition_input(),
                cognition_output=focus_output("phone", "answered"),
                intent=behavior.ChangeData(name="AnswerIncomingRing", reason="revise the program only"),
                parent_operation_id="parent-turn",
                parent_generation="parent-generation",
            ),
            timeout=_REVISION_TIMEOUT_SECONDS,
        )
        assert isinstance(output, revision.OutputData)
        return output

    output = asyncio.run(run())

    assert isinstance(output.applied, behavior.ChangeData)
    assert output.applied.name == "AnswerIncomingRing"
    assert output.applied.triggers == ("environment.sound",)
    assert output.applied.source == revised_source


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
            memory.Episode(
                node_id="5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f",
                stimulus="environment.sound",
                text="",
                outcome=memory.Outcome.COMPLETED,
                turn_id="turn-1",
                created_at=datetime.datetime(2026, 10, 14, 12, 0, tzinfo=datetime.UTC),
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


def test_revision_installs_a_persistent_routine_checked_by_one_tick() -> None:
    """A routine is a legitimate create: it is checked by a dry-run tick, not the live turn."""

    from tests.bot.behavior.support import routine_source

    program = routine_source(
        schedule='hsm.at(time = "08:00", days = ["mon", "tue", "wed", "thu", "fri"], tz = "America/New_York")'
    )

    async def run() -> tuple[revision.OutputData, behavior.Instance | None]:
        ctx, store = await _attached_memory()
        actor = revision.Revision(
            processor=FixedProcessor(write=behavior.ChangeData(name="MorningBriefing", source=program)),
            memory=store,
        )
        output = await dispatch_ability_for_test(
            actor,
            None,
            revision.InputData(
                cognition_input=cognition_input(),
                cognition_output=focus_output("phone", "answered"),
                intent=behavior.CreateData(
                    name="MorningBriefing",
                    reason="The user asked for a briefing every weekday morning.",
                ),
                parent_operation_id="parent-turn",
                parent_generation="parent-generation",
            ),
            timeout=_REVISION_TIMEOUT_SECONDS,
        )
        assert isinstance(output, revision.OutputData)
        return output, await inventory.load_behavior(ctx, store, "MorningBriefing")

    output, installed = asyncio.run(run())

    assert isinstance(output.applied, behavior.CreateData)
    assert installed is not None
    assert installed.status == "ACTIVE"
    assert installed.lifetime == "persistent"


def test_revision_instructions_describe_persistent_routines() -> None:
    for fragment in ('lifetime = "persistent"', "hsm.every(seconds=...)", "dry-run tick"):
        assert fragment in revision.CHANGE_INSTRUCTIONS


def test_revision_instructions_teach_dispatch_and_reply_waiting() -> None:
    for fragment in ("Callbacks are synchronous one-shot work", "transition on its reply event", "hsm.after"):
        assert fragment in revision.CHANGE_INSTRUCTIONS


def test_revision_persists_a_draft_by_message_before_retrying_a_failed_check() -> None:
    """A change that fails its check is written back as a draft, then the fix is accepted."""

    broken = behavior.ChangeData(name="AnswerIncomingRing", source="def broken(:")
    fixed = behavior.ChangeData(name="AnswerIncomingRing", source=ANSWER_RING_BEHAVIOR_SOURCE)

    async def run() -> tuple[revision.OutputData, behavior.Instance | None]:
        ctx, store = await _attached_memory()
        actor = revision.Revision(processor=FixedProcessor(write=[broken, fixed]), memory=store)
        output = await dispatch_ability_for_test(
            actor,
            None,
            _revision_input(behavior.CreateData(name="AnswerIncomingRing", triggers=("environment.sound",))),
            timeout=_REVISION_TIMEOUT_SECONDS,
        )
        assert isinstance(output, revision.OutputData)
        return output, await inventory.load_behavior(ctx, store, "AnswerIncomingRing")

    output, stored = asyncio.run(run())

    assert output.applied.source == ANSWER_RING_BEHAVIOR_SOURCE
    assert stored is not None
    assert stored.status == "ACTIVE"
    assert stored.source == ANSWER_RING_BEHAVIOR_SOURCE


def test_revision_abandons_a_repeated_failure_after_persisting_the_draft() -> None:
    broken = behavior.ChangeData(name="AnswerIncomingRing", source="def broken(:")

    async def run() -> tuple[str, behavior.Instance | None]:
        ctx, store = await _attached_memory()
        actor = revision.Revision(processor=FixedProcessor(write=[broken, broken]), memory=store)
        with pytest.raises(RuntimeError) as failure:
            _ = await dispatch_ability_for_test(
                actor,
                None,
                _revision_input(behavior.CreateData(name="AnswerIncomingRing", triggers=("environment.sound",))),
                timeout=_REVISION_TIMEOUT_SECONDS,
            )
        return str(failure.value), await inventory.load_behavior(ctx, store, "AnswerIncomingRing")

    message, stored = asyncio.run(run())

    assert "abandoned" in message
    assert stored is not None
    assert stored.status == "DRAFT"


def test_revision_reports_a_typed_failure_when_memory_cannot_be_reached() -> None:
    """A memory that is not attached anywhere answers nothing; Revision fails the request, typed."""

    async def run() -> None:
        actor = revision.Revision(
            processor=FixedProcessor(write=behavior.ChangeData(name="AnswerIncomingRing")),
            memory=memory.Memory(),
        )
        _ = await dispatch_ability_for_test(
            actor,
            None,
            _revision_input(behavior.CreateData(name="AnswerIncomingRing", triggers=("environment.sound",))),
            timeout=_REVISION_TIMEOUT_SECONDS,
        )

    with pytest.raises(RuntimeError, match="cannot be reached"):
        asyncio.run(run())


class _SlowProcessor(processing.Processor):
    """Authors as ``inner`` does, after ``delay``: a slow model's authoring call."""

    inner: processing.Processor
    delay: datetime.timedelta

    def __init__(self, inner: processing.Processor, delay: datetime.timedelta) -> None:
        self.inner = inner
        self.delay = delay

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        await asyncio.sleep(self.delay.total_seconds())
        return await self.inner.process(input, drafts=drafts)


def test_revision_bounds_each_authoring_call_by_the_step_timeout_it_is_given() -> None:
    """D-R-17 (live round 3, H4): a slow model's authoring call is bounded by the step timeout, not a fixed 30 s."""

    async def run() -> revision.OutputData:
        _, store = await _attached_memory()
        actor = revision.Revision(
            processor=_SlowProcessor(
                FixedProcessor(
                    write=behavior.ChangeData(name="AnswerIncomingRing", source=ANSWER_RING_BEHAVIOR_SOURCE)
                ),
                datetime.timedelta(milliseconds=400),
            ),
            memory=store,
            child_operation_timeout=datetime.timedelta(seconds=5),
        )
        output = await dispatch_ability_for_test(
            actor,
            None,
            _revision_input(behavior.CreateData(name="AnswerIncomingRing", triggers=("environment.sound",))),
            timeout=_REVISION_TIMEOUT_SECONDS,
        )
        assert isinstance(output, revision.OutputData)
        return output

    assert isinstance(asyncio.run(run()).applied, behavior.CreateData)


def test_revision_refuses_a_step_timeout_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="child_operation_timeout must be positive"):
        _ = revision.Revision(
            processor=FixedProcessor(), memory=memory.Memory(), child_operation_timeout=datetime.timedelta()
        )

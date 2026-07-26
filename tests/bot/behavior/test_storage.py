"""Behavior inventory storage: status lifecycle and usage telemetry."""

from bot.behavior.instance import (
    STATUS_ACTIVE,
    STATUS_BROKEN,
    STATUS_DRAFT,
    STATUS_REASON_VALIDATION,
    Instance,
)
from bot.behavior import storage


def _base(**overrides: object) -> Instance:
    values: dict[str, object] = {
        "name": "AnswerGreeting",
        "source": "behavior = hsm.define('AnswerGreeting')",
        "triggers": ("environment.sound",),
        "description": "demo",
    }
    values.update(overrides)
    return Instance.model_validate(values)


def test_instance_defaults_status_active_and_zero_usage() -> None:
    behavior = _base()
    assert behavior.status == STATUS_ACTIVE
    assert behavior.status_reason is None
    assert behavior.status_updated_at is None
    assert behavior.used_count == 0
    assert behavior.last_used_at is None
    assert behavior.failed_count == 0
    assert behavior.last_failed_at is None


def test_mark_used_increments_canonical_used_counters() -> None:
    behavior = _base()
    used = storage.mark_used(behavior, at="2026-07-11T12:00:00Z")
    assert used.used_count == 1
    assert used.last_used_at == "2026-07-11T12:00:00Z"
    assert used.failed_count == 0
    again = storage.mark_used(used, at="2026-07-11T13:00:00Z")
    assert again.used_count == 2
    assert again.last_used_at == "2026-07-11T13:00:00Z"


def test_mark_failed_increments_failure_counters() -> None:
    behavior = _base(used_count=3, last_used_at="2026-07-11T10:00:00Z")
    failed = storage.mark_failed(behavior, at="2026-07-11T14:00:00Z")
    assert failed.failed_count == 1
    assert failed.last_failed_at == "2026-07-11T14:00:00Z"
    assert failed.used_count == 3
    assert failed.last_used_at == "2026-07-11T10:00:00Z"


def test_status_transitions_preserve_usage() -> None:
    behavior = _base(used_count=2, last_used_at="2026-07-11T09:00:00Z", failed_count=1)
    draft = storage.mark_draft(behavior, reason=STATUS_REASON_VALIDATION, at="2026-07-11T15:00:00Z")
    assert draft.status == STATUS_DRAFT
    assert draft.status_reason == STATUS_REASON_VALIDATION
    assert draft.status_updated_at == "2026-07-11T15:00:00Z"
    assert draft.used_count == 2
    broken = storage.mark_broken(draft, reason="harmful side effect", at="2026-07-11T16:00:00Z")
    assert broken.status == STATUS_BROKEN
    assert broken.status_reason == "harmful side effect"
    assert broken.status_updated_at == "2026-07-11T16:00:00Z"
    assert broken.failed_count == 1
    active = storage.mark_active(broken, at="2026-07-11T17:00:00Z")
    assert active.status == STATUS_ACTIVE
    assert active.status_reason is None
    assert active.status_updated_at == "2026-07-11T17:00:00Z"
    assert active.used_count == 2


def test_preserve_usage_copies_practice_telemetry() -> None:
    prior = _base(used_count=4, last_used_at="t1", failed_count=2, last_failed_at="t2")
    rewritten = _base(source="new source", description="revised")
    merged = storage.preserve_usage(rewritten, prior)
    assert merged.source == "new source"
    assert merged.used_count == 4
    assert merged.last_used_at == "t1"
    assert merged.failed_count == 2
    assert merged.last_failed_at == "t2"


def test_instances_from_behavior_results_round_trips_status_and_usage() -> None:
    behavior = _base(
        status=STATUS_DRAFT,
        status_reason=STATUS_REASON_VALIDATION,
        status_updated_at="2026-07-11T08:00:00Z",
        used_count=5,
        last_used_at="2026-07-11T07:00:00Z",
        failed_count=1,
        last_failed_at="2026-07-11T06:00:00Z",
        examples=("when ring",),
    )
    _ = storage.insert_behavior_clauses(behavior)
    rows = [
        {
            "name": behavior.name,
            "source": behavior.source,
            "description": behavior.description,
            "examples_json": '["when ring"]',
            "status": STATUS_BROKEN,
            "status_reason": "retired",
            "status_updated_at": "2026-07-11T08:00:00Z",
            "used_count": "5",
            "last_used_at": "2026-07-11T07:00:00Z",
            "failed_count": "1",
            "last_failed_at": "2026-07-11T06:00:00Z",
        }
    ]
    triggers = [{"behavior_name": behavior.name, "trigger": "environment.sound"}]
    loaded = storage.instances_from_behavior_results(rows, triggers)
    assert len(loaded) == 1
    got = loaded[0]
    assert got.status == STATUS_BROKEN
    assert got.status_reason == "retired"
    assert got.status_updated_at == "2026-07-11T08:00:00Z"
    assert got.used_count == 5
    assert got.last_used_at == "2026-07-11T07:00:00Z"
    assert got.failed_count == 1
    assert got.last_failed_at == "2026-07-11T06:00:00Z"
    assert got.triggers == ("environment.sound",)
    assert got.examples == ("when ring",)


def test_legacy_rows_map_safely() -> None:
    # No status column: active defaults.
    rows = [
        {
            "name": "Legacy",
            "source": "x",
            "description": "",
            "examples_json": "[]",
        }
    ]
    loaded = storage.instances_from_behavior_results(rows, ())
    assert loaded[0].status == STATUS_ACTIVE
    assert loaded[0].used_count == 0

    # Legacy broken boolean column.
    broken_rows = [
        {
            "name": "OldBroken",
            "source": "x",
            "description": "",
            "examples_json": "[]",
            "broken": "true",
            "broken_reason": "draft",
            "broken_at": "2026-01-01T00:00:00Z",
        }
    ]
    legacy = storage.instances_from_behavior_results(broken_rows, ())
    assert legacy[0].status == STATUS_BROKEN
    assert legacy[0].status_reason == "draft"
    assert legacy[0].status_updated_at == "2026-01-01T00:00:00Z"

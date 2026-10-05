from mosfet.behavior import verify


def _transfer_selection_behavior_source() -> str:
    """A behavior that selects an event whose ``call_id`` differs from the live event id.

    Models a legitimate case (e.g. transferring to a second call leg) where the
    selection's domain payload intentionally names a call other than the one
    carrying the live turn.
    """

    return """
input_event = hsm.event(
    name = "bot.behavior.test_verify.transfer.input",
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    description = "Turn text recognized by cognition.",
    examples = [{"text": "transfer me"}],
)

output_event = hsm.event(
    name = "bot.behavior.test_verify.transfer.output",
    schema = {"type": "object"},
    description = "Selected event to dispatch without deliberation.",
    examples = [{"event": "phone.transfer_call", "data": {"call_id": "other-leg-999"}}],
)

triggers = ["conversation.transfer.recognized"]
description = "Select a transfer to a different call leg than the live event."

def emit_selection(event):
    hsm.dispatch(output_event, {
        "event": "phone.transfer_call",
        "data": {"call_id": "other-leg-999"},
    })

behavior = hsm.define(
    "TransferSelection",
    hsm.initial(hsm.target("/TransferSelection/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.effect("emit_selection"),
        ),
    ),
)
""".strip()


def test_verify_apply_allows_selection_naming_a_different_call_id_than_the_live_event() -> None:
    """A selection's ``call_id`` is domain payload, not a modeled id/source/target.

    It must only be checked against the live event when ``call_id`` itself was
    present in the live ``input_data`` — never implicitly bound from the event id.
    Before the fix, ``verify_apply`` hardcoded ``call_id`` to the live event id,
    so any selection naming a *different* call (a transfer target, a second call
    leg, etc.) was rejected with a spurious binding-mismatch error.
    """

    checked = verify.verify_apply(
        _transfer_selection_behavior_source(),
        input_data={"text": "transfer me"},
        event_id="live-call-123",
    )

    assert checked.ok, checked.report.render()


def test_racing_warm_and_callback_fork_survives_forkserver_bootstrap() -> None:
    """Forkserver bootstrapping never races a concurrent process start in this module.

    The warm worker boots the shared forkserver; a callback worker forking while that
    bootstrap runs dies with the interpreter's "bootstrapping phase" failure, which
    surfaced as flaky E0008 apply timeouts during Revision dry-runs (observed live
    inside the composed bot where several verifies ran back to back).
    """

    import concurrent.futures

    from mosfet.behavior import runtime as behavior_runtime

    source = """
input_event = hsm.event(
    name = "bot.behavior.test_verify.race.input",
    schema = {"type": "object", "additionalProperties": True},
    description = "Live behavior input.",
)
output_event = hsm.event(
    name = "bot.behavior.test_verify.race.output",
    schema = {"type": "object"},
    description = "Live behavior output.",
)

def emit_selection(event):
    hsm.dispatch(output_event, {"event": "phone.answer_call"})

behavior = hsm.define(
    "RaceProbeBehavior",
    hsm.initial(hsm.target("/RaceProbeBehavior/idle")),
    hsm.state(
        "idle",
        hsm.transition(hsm.on(input_event), hsm.effect("emit_selection")),
    ),
)
""".strip()

    failures: list[BaseException] = []
    callback_runtime = behavior_runtime.CallbackRuntime(source="", declared_events={})
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        warm = pool.submit(callback_runtime.warm)
        checks = [
            pool.submit(
                verify.verify_apply,
                source,
                input_data={"kind": "phone.ringing"},
            )
            for _ in range(2)
        ]
        try:
            _ = warm.result(timeout=60)
        except BaseException as error:  # noqa: BLE001 — the test asserts on any failure text
            failures.append(error)
        for check in checks:
            try:
                _ = check.result(timeout=60)
            except BaseException as error:  # noqa: BLE001
                failures.append(error)

    for failure in failures:
        message = str(failure)
        assert "bootstrapping phase" not in message, message
        assert "warmup exceeded" not in message, message


_WEEKDAY_AT = 'hsm.at(time = "08:00", days = ["mon", "tue", "wed", "thu", "fri"], tz = "America/New_York")'


def test_verify_apply_dry_runs_one_tick_of_a_persistent_routine() -> None:
    import datetime

    from tests.bot.behavior.support import routine_source

    calls: list[datetime.datetime] = []

    def clock() -> datetime.datetime:
        now = datetime.datetime(2026, 10, 5, 12, 0, 0, 5000, tzinfo=datetime.UTC)
        calls.append(now)
        return now

    checked = verify.verify_apply(routine_source(schedule=_WEEKDAY_AT), input_data={"ignored": True}, clock=clock)

    assert checked.ok, checked.report.render()
    assert checked.value is not None
    assert checked.value.lifetime == "persistent"
    # The forced tick read the injected wall clock (start check, tick stamp, next delay).
    assert calls


def test_verify_apply_rejects_a_routine_tick_without_selections() -> None:
    from tests.bot.behavior.support import routine_source

    program = routine_source(schedule="hsm.every(seconds = 300)").replace(
        "def emit(event):\n    hsm.dispatch(output_event, {",
        'def emit(event):\n    hsm.dispatch(output_event, {"event": ""})\n\ndef unused(event):\n    hsm.dispatch(output_event, {',
    )

    checked = verify.verify_apply(program, input_data={})

    assert not checked.ok
    assert checked.report.errors[0].code == "E0008"


def test_verify_apply_rejects_a_routine_interval_below_the_minimum() -> None:
    import datetime

    from tests.bot.behavior.support import routine_source

    program = routine_source(schedule="hsm.every(seconds = 5)")

    rejected = verify.verify_apply(program, input_data={})
    assert [error.code for error in rejected.report.errors] == ["E0009"]
    assert verify.verify_apply(program, input_data={}, min_every=datetime.timedelta(seconds=5)).ok

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

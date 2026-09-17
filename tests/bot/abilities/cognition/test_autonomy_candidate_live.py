import asyncio
from mosfet.abilities import cognition
from mosfet.abilities import memory
from mosfet import behavior
from mosfet.behavior import storage as behavior_storage
from mosfet.environment import SoundData, SoundEvent

_MINIMAL_ANSWER = """input_event = hsm.event(
    name = "bot.behavior.autonomy_probe.input",
    schema = {"type": "object", "additionalProperties": True},
    description = "d",
)
output_event = hsm.event(
    name = "bot.behavior.autonomy_probe.output",
    schema = {"type": "object",
              "properties": {"event": {"type": "string"}},
              "required": ["event"],
              "additionalProperties": True},
    description = "o",
)
def select_answer(event):
    hsm.dispatch(output_event, {"event": "phone.answer_call"})
behavior = hsm.define(
    "AutonomyProbeAnswer",
    hsm.initial(hsm.target("/AutonomyProbeAnswer/idle")),
    hsm.state("idle", hsm.transition(hsm.on(input_event), hsm.effect("select_answer"))),
)"""


def test_autonomy_runs_a_late_authored_behavior_on_a_live_turn() -> None:
    """A behavior authored after Autonomy's attach dispatches its selection on a real turn.

    The compiled Starlark candidate path — the bot's reflex arc for what it learned — was
    never live-exercised end to end before the e2e exposed the per-turn inventory gap. This
    pins it: late-authored rows materialize, run, and emit their terminal selections."""

    async def run() -> tuple[str, int]:
        store = memory.Memory()
        installed = behavior.Instance(
            name="AutonomyProbeAnswer", source=_MINIMAL_ANSWER, triggers=("environment.sound",), description="probe"
        )
        installed = behavior_storage.mark_active(installed)
        _ = store.execute(
            memory.InputData(statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(installed)))
        )
        autonomy = cognition.Autonomy(memory=store)
        from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context

        ctx = shared_hsm_context()
        turn = cognition.types.TurnData(
            input=cognition.input.InputData(
                stimulus=SoundEvent.with_data(
                    SoundData(audio=b"ring", media_type="audio/wav", sample_rate_hz=16_000, channels=1)
                ),
                abilities=(),
                actors={},
                focus=None,
                focus_candidates=(),
            ),
            operation_id="lateauthored",
            generation="lateauthored:gen",
        )
        out = await dispatch_ability_for_test(autonomy, ctx, turn, timeout=25.0)
        output = out.output
        selections = [item.event for item in output] if isinstance(output, tuple) else []
        return ",".join(selections), 0

    names, _ = asyncio.run(run())
    assert "phone.answer_call" in names, names

from bot.abilities import cognition
from bot.abilities import memory
from bot.abilities import processing
from bot.abilities.cognition import autonomy as autonomy_module
from bot.protocols import attachment
from bot.behavior import storage as behavior_storage

import asyncio
import collections.abc
import dataclasses
import typing

import hsm
import pytest

from bot import behavior
from bot.devices import phone as phone_device
from bot.devices.phone.events import PhoneSoundData
from bot.environment import SoundData, SoundEvent, Environment
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test
from tests.bot.abilities.cognition.metadata_contract import assert_metadata_is_not_coordination
from tests.hsm_instance_state import phone_firmware


def test_autonomy_never_uses_metadata_for_coordination() -> None:
    assert_metadata_is_not_coordination(autonomy_module)


_DELAYED_BEHAVIOR_SOURCE = """
input_event = hsm.event(
    name = "bot.behavior.delayed_ring.input",
    schema = {"type": "object", "additionalProperties": True},
    description = "Input for a delayed behavior.",
)
output_event = hsm.event(
    name = "bot.behavior.delayed_ring.output",
    schema = {
        "type": "object",
        "properties": {"event": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["event"],
        "additionalProperties": True,
    },
    description = "Delayed behavior selection.",
)
triggers = ["environment.sound"]

def emit(event):
    hsm.dispatch(output_event, {"event": "bot.clear_focus", "reason": "delayed behavior"})

behavior = hsm.define(
    "DelayedRing",
    hsm.initial(hsm.target("/DelayedRing/idle")),
    hsm.state(
        "idle",
        hsm.transition(hsm.on(input_event), hsm.target("/DelayedRing/waiting")),
    ),
    hsm.state(
        "waiting",
        hsm.transition(hsm.after(seconds=0.01), hsm.effect("emit")),
    ),
)
""".strip()

_WAITING_BEHAVIOR_SOURCE = _DELAYED_BEHAVIOR_SOURCE.replace("seconds=0.01", "seconds=10")


class _Owner(hsm.Instance):
    model: typing.ClassVar[hsm.Model]
    lifecycle: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle = []


def _record(ctx: hsm.Context, instance: _Owner, event: hsm.Event[typing.Any]) -> None:
    del ctx
    instance.lifecycle.append(event)


_Owner.model = hsm.define(
    "AutonomyTestOwner",
    hsm.initial(hsm.target("/AutonomyTestOwner/ready")),
    hsm.state(
        "ready",
        hsm.transition(hsm.on(processing.CancelledEvent), hsm.effect(_record)),
    ),
)


async def _wait_until(predicate: typing.Callable[[], bool], *, timeout: float = 0.3) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def _turn(operation_id: str = "autonomy-turn") -> cognition.types.TurnData:
    return cognition.types.TurnData(
        input=cognition.InputData(
            stimulus=SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing")),
            abilities=(),
            actors={},
            focus=None,
            focus_candidates=(),
        ),
        operation_id=operation_id,
        generation="operation-token",
    )


def test_autonomy_waits_for_asynchronous_behavior_terminal() -> None:
    async def run() -> tuple[object, str, int]:
        store = memory.Memory()
        installed = behavior.start(
            _DELAYED_BEHAVIOR_SOURCE,
            name="DelayedRing",
            triggers=(SoundEvent.name,),
        )
        _ = store.execute(
            memory.InputData(statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(installed)))
        )
        autonomy = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)

        output = await dispatch_ability_for_test(
            autonomy,
            ctx,
            _turn(),
            timeout=0.2,
        )
        instances = autonomy.context().value(hsm.Keys.Instances)
        candidate_count = (
            sum(isinstance(actor, autonomy_module._CandidateRun) for actor in instances.values())
            if isinstance(instances, collections.abc.Mapping)
            else 0
        )
        return output, autonomy.state(), candidate_count

    output, state, candidate_count = asyncio.run(run())

    assert output == cognition.types.CompletionData(
        turn=_turn(),
        output=(
            cognition.types.EventData(
                event="bot.clear_focus",
                reason="delayed behavior",
            ),
        ),
    )
    assert state == "/AutonomyLifecycle/attached/behavior/idle"
    assert candidate_count == 0


def test_autonomy_acknowledges_cancel_only_after_candidate_detaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        store = memory.Memory()
        installed = behavior.start(
            _WAITING_BEHAVIOR_SOURCE,
            name="DelayedRing",
            triggers=(SoundEvent.name,),
        )
        _ = store.execute(
            memory.InputData(statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(installed)))
        )
        autonomy = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        owner = _Owner()
        _ = await hsm.started(ctx, owner, owner.model)
        await autonomy.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await _wait_until(lambda: autonomy.state().endswith("/idle"))

        operation_id = "cancelled-autonomy-turn"
        _ = await hsm.dispatch(
            ctx,
            autonomy,
            autonomy.input_event.with_data_and_id(
                _turn(operation_id),
                operation_id,
            ),
        )
        await _wait_until(lambda: autonomy.state().endswith("/running"))

        detached = asyncio.Event()
        original_detach = behavior.Behavior.detach

        def delayed_detach(
            behavior: behavior.Behavior,
            detach_ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> asyncio.Task[object]:
            async def detach_after_release() -> object:
                await detached.wait()
                return await original_detach(behavior, detach_ctx, event)

            return asyncio.create_task(detach_after_release())

        monkeypatch.setattr(behavior.Behavior, "detach", delayed_detach)
        cancel = dataclasses.replace(
            processing.CancelEvent.with_data(
                processing.CancelData(operation_id=operation_id, token="exact-cancel-token")
            ),
            id=f"{operation_id}:autonomy",
            source=hsm.id(owner),
            target=hsm.id(autonomy),
        )
        _ = await hsm.dispatch(ctx, autonomy, cancel)
        await _wait_until(lambda: autonomy.state().endswith("/cancelling"))
        assert owner.lifecycle == []

        detached.set()
        await _wait_until(lambda: bool(owner.lifecycle))
        return owner.lifecycle, autonomy.state()

    lifecycle, state = asyncio.run(run())

    assert len(lifecycle) == 1
    assert lifecycle[0].id == "cancelled-autonomy-turn:autonomy"
    assert lifecycle[0].data == processing.CancelledData(
        operation_id="cancelled-autonomy-turn",
        token="exact-cancel-token",
    )
    assert state.endswith("/idle")


_ANSWER_INCOMING_RING_BEHAVIOR_SOURCE = """
input_event = hsm.event(
    name = "bot.behavior.answer_incoming_ring.input",
    schema = {"type": "object", "additionalProperties": True},
    description = "Live behavior input for the observed turn stimulus.",
)
output_event = hsm.event(
    name = "bot.behavior.answer_incoming_ring.output",
    schema = {
        "type": "object",
        "properties": {
            "event": {"type": "string"},
            "target": {"type": "string"},
            "data": {"type": "object"},
            "reason": {"type": "string"},
        },
        "required": ["event"],
        "additionalProperties": True,
    },
    description = "Cognition event selection.",
)
triggers = ["environment.sound"]
description = "Answer an incoming phone ring as practiced automatic behavior."

def always(event):
    return True

def answer_ring(event):
    # Answering names no call: the phone answers whatever it has ringing.
    hsm.dispatch(output_event, {
        "event": "phone.answer_call",
        "target": "phone",
        "reason": "practiced ring answer",
    })

behavior = hsm.define(
    "AnswerIncomingRing",
    hsm.initial(hsm.target("/AnswerIncomingRing/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.guard("always"),
            hsm.effect("answer_ring"),
        ),
    ),
)
""".strip()


def test_autonomy_installed_behavior_answers_phone_ring() -> None:
    """Pre-installed ACTIVE behavior matches environment.sound ring and dispatches phone.answer_call."""

    call_id = "call-answer-behavior"

    async def run() -> tuple[object, tuple[str, ...], str]:
        store = memory.Memory()
        installed = behavior.start(
            _ANSWER_INCOMING_RING_BEHAVIOR_SOURCE,
            name="AnswerIncomingRing",
            triggers=(SoundEvent.name,),
            description="Answer an incoming phone ring as practiced automatic behavior.",
        )
        _ = store.execute(
            memory.InputData(statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(installed)))
        )

        environment = Environment()
        phone = phone_device.Phone()
        autonomy = cognition.Autonomy(memory=store)
        shared = shared_hsm_context(environment)
        await start_abilities_for_test(shared, autonomy)
        _ = await hsm.started(shared, phone, typing.cast(hsm.Model, phone.model))

        firmware = phone_firmware(phone)
        assert firmware is not None
        await firmware.event_recorder().receive(
            phone.context(),
            phone_device.IncomingCallEvent.with_data(
                phone_device.IncomingCallData(call_id=call_id, caller="caller")
            ),
        )
        await _wait_until(lambda: "ringing" in (firmware.state() or ""), timeout=2.0)

        ring = dataclasses.replace(
            SoundEvent.with_data(
                PhoneSoundData(
                    audio=phone_device.RING_SOUND_WAV,
                    media_type="audio/wav",
                    sample_rate_hz=16_000,
                    channels=1,
                    kind="phone.ringing",
                    caller="caller",
                )
            ),
            id=call_id,
            source=hsm.id(phone),
        )
        # Public contract: full event is readable by behaviors (not data-only).
        event_id, event_source, _event_target = autonomy_module.behavior_event_fields(
            cognition.InputData(
                stimulus=ring,
                abilities=(),
                actors={"phone": phone},
                focus="phone",
                focus_candidates=("phone",),
            )
        )
        assert event_id == call_id
        assert event_source == hsm.id(phone)

        turn = cognition.types.TurnData(
            input=cognition.InputData(
                stimulus=ring,
                abilities=(),
                actors={"phone": phone},
                focus="phone",
                focus_candidates=("phone",),
            ),
            operation_id="answer-ring-turn",
            generation="operation-token",
        )
        output = await dispatch_ability_for_test(autonomy, shared, turn, timeout=2.0)
        await asyncio.sleep(0.2)
        published = tuple(event.name for event in firmware.event_recorder().events)
        return output, published, firmware.state() or ""

    output, published, phone_state = asyncio.run(run())

    assert isinstance(output, cognition.types.CompletionData)
    assert output.output is not None
    answer_selections = [item for item in output.output if item.event == phone_device.AnswerCallEvent.name]
    assert answer_selections, f"expected phone.answer_call, got {output.output!r}"
    assert answer_selections[0].target == "phone"
    # Nothing to carry: the command has no identity to get wrong.
    assert answer_selections[0].data is None
    assert phone_device.ServiceAnswerRequestedEvent.name in published, (
        f"phone never requested answer; published={published!r} state={phone_state!r}"
    )
    assert any(token in phone_state for token in ("answering", "answered", "media")), (
        f"phone should leave ringing after behavior answer, state={phone_state!r}"
    )


def test_autonomy_without_installed_behavior_leaves_ring_unhandled() -> None:
    """Empty behavior inventory does not invent an answer — ring turn stays unhandled for deliberation."""

    async def run() -> object:
        store = memory.Memory()
        autonomy = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        return await dispatch_ability_for_test(autonomy, ctx, _turn(), timeout=0.5)

    output = asyncio.run(run())
    assert isinstance(output, cognition.types.CompletionData)
    # Unhandled: no selections (or empty). Deliberative stages own answer policy when no behavior matches.
    assert output.output is None or output.output == ()

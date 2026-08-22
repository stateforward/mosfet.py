from bot.abilities import cognition
from bot.abilities import ability
from bot.abilities import communication
from bot.abilities import memory
from bot.abilities import processing
from bot.abilities.cognition import autonomy as autonomy_module
from bot.protocols import attachment
from bot.behavior import storage as behavior_storage
from bot.behavior import seed
import bot

import asyncio
import collections.abc
import dataclasses
import datetime
import inspect
import typing

import hsm
import pytest

from bot import behavior
from bot.devices import phone as phone_device
from bot.devices.phone.events import PhoneSoundData
from bot.environment import SoundData, SoundEvent, Environment
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test
from tests.bot.abilities.cognition.metadata_contract import assert_metadata_is_not_coordination
from tests.hsm_model import choice_transitions, transition_map
from tests.hsm_instance_state import phone_firmware


class _PublicModelMember(typing.Protocol):
    activity: list[str]
    deferred: list[str]
    exit: list[str]
    initial: str


class _PublicModelWithDeferredMap(typing.Protocol):
    deferred_map: dict[str, dict[str, str]]


class _CandidateCapabilityFactory(typing.Protocol):
    def __call__(
        self,
        *,
        operation_id: str,
        generation: str,
        index: int,
        token: str,
        origin: str,
    ) -> object: ...


class _CandidateModelFactory(typing.Protocol):
    def model_for(
        self,
        *,
        owner: cognition.Autonomy,
        behavior: ability.Ability[typing.Any, typing.Any],
        capability: object,
        turn: cognition.types.TurnData,
        candidates: tuple[object, ...],
        input_adapter: collections.abc.Callable[[cognition.InputData], object],
        metadata: dict[str, object],
    ) -> hsm.Model: ...


def test_autonomy_never_uses_metadata_for_coordination() -> None:
    assert_metadata_is_not_coordination(autonomy_module)


def test_autonomy_returns_directed_terminal_to_one_shot_operation() -> None:
    async def run() -> hsm.Event[typing.Any]:
        ctx = shared_hsm_context()
        child = cognition.Autonomy()
        await start_abilities_for_test(ctx, child)
        operation_id = "autonomy-terminal-operation"
        turn = _turn(operation_id=operation_id)
        return await ability.run_terminal_operation(
            ctx,
            child=child,
            request=child.input_event.with_data_and_id(turn, operation_id),
            terminals=(child.output_event, child.failed_event),
            timeout=datetime.timedelta(seconds=1),
        )

    terminal = asyncio.run(run())

    assert terminal.name == cognition.autonomy.OutputEvent.name
    assert terminal.id == "autonomy-terminal-operation"
    assert terminal.source
    assert terminal.target


def test_autonomy_activities_use_event_carried_inventory_and_no_runtime_registry() -> None:
    """Activities consume immutable payloads; actor lifecycle never traverses registries."""

    source = inspect.getsource(autonomy_module.Autonomy)
    assert "hsm.Keys.Instances" not in source
    match_source = inspect.getsource(getattr(autonomy_module.Autonomy, "_match_activity"))
    assert "instance._behaviors" not in match_source
    assert "instance._seeded_behaviors" not in match_source
    initialize_source = inspect.getsource(getattr(autonomy_module.Autonomy, "_initialize_activity"))
    assert "instance._behaviors" not in initialize_source
    persistence_source = inspect.getsource(getattr(autonomy_module.Autonomy, "_persist_usage_activity"))
    assert "instance._memory" not in persistence_source


def test_autonomy_routes_candidate_outcomes_with_distinct_typed_events() -> None:
    """Candidate outcome routing is explicit instead of a multi-guard result fan-out."""

    model = typing.cast(hsm.Model, autonomy_module.Autonomy.submodel)
    transitions = transition_map(model)
    candidate_result = "bot.ability.autonomy.candidate.result"
    assert all(
        not transitions[state].get(candidate_result)
        for state in ("/Autonomy/workflow/running", "/Autonomy/workflow/starting", "/Autonomy/workflow/cancelling")
    )
    workflow = transitions["/Autonomy/workflow"]
    for outcome in (
        "handled",
        "advance",
        "failed",
        "silent",
        "cancelled",
        "attach_timeout",
        "detach_failed",
        "detach_timeout",
    ):
        assert workflow[f"bot.ability.autonomy.candidate.{outcome}"]
    assert transitions["/Autonomy/workflow/cancelling"]["bot.ability.autonomy.candidate.cancelled"]


def test_autonomy_workflow_parent_owns_shared_lifecycle() -> None:
    """Workflow steps inherit defer/detach behavior from one composite parent."""

    model = typing.cast(hsm.Model, autonomy_module.Autonomy.submodel)
    workflow = typing.cast(_PublicModelMember, typing.cast(object, model.members["/Autonomy/workflow"]))
    assert workflow.initial
    assert any(name.endswith("/_detach_on_detach") for name in workflow.exit)
    deferred_map = typing.cast(_PublicModelWithDeferredMap, typing.cast(object, model)).deferred_map
    for state in (
        "/Autonomy/workflow/matching",
        "/Autonomy/workflow/running",
        "/Autonomy/workflow/starting",
        "/Autonomy/workflow/dispatching",
        "/Autonomy/workflow/cancelling",
        "/Autonomy/workflow/degraded",
    ):
        member = typing.cast(_PublicModelMember, typing.cast(object, model.members[state]))
        assert member.deferred == []
        assert member.exit == []
        assert deferred_map[state][autonomy_module.Autonomy.input_event.name]


def test_autonomy_candidate_lifecycle_is_activity_owned() -> None:
    """Lifecycle phases publish their typed completions from owned activities."""

    model = typing.cast(hsm.Model, autonomy_module.Autonomy.submodel)
    for state in (
        "/Autonomy/workflow/dispatching",
        "/Autonomy/workflow/starting/checking",
        "/Autonomy/workflow/starting/materializing",
        "/Autonomy/workflow/starting/starting_actor",
        "/Autonomy/workflow/persisting_used/writing",
        "/Autonomy/workflow/persisting_failed/writing",
        "/Autonomy/workflow/persisting_dispatch_failed/writing",
    ):
        assert typing.cast(_PublicModelMember, typing.cast(object, model.members[state])).activity


def test_autonomy_matching_routes_empty_and_non_empty_results_explicitly() -> None:
    """Empty matching and candidate startup use distinct typed graph routes."""

    model = typing.cast(hsm.Model, autonomy_module.Autonomy.submodel)
    transitions = transition_map(model)
    matching = transitions["/Autonomy/workflow/matching"]
    assert "bot.ability.autonomy.matched.empty" in matching
    assert "bot.ability.autonomy.matched.candidates" in matching
    assert "bot.ability.autonomy.matched" not in matching


def test_autonomy_candidate_startup_is_a_typed_mini_machine() -> None:
    """Materialization, actor ownership, and failure/exhaustion are explicit states."""

    model = typing.cast(hsm.Model, autonomy_module.Autonomy.submodel)
    for path in (
        "/Autonomy/workflow/starting/checking",
        "/Autonomy/workflow/starting/materializing",
        "/Autonomy/workflow/starting/starting_actor",
        "/Autonomy/workflow/starting/learned_failed",
        "/Autonomy/workflow/starting/native_failed",
    ):
        assert path in model.members


def test_autonomy_candidate_outcomes_are_exhaustively_named() -> None:
    """Every CandidateResultData outcome has its own owner-facing typed event."""

    model = typing.cast(hsm.Model, autonomy_module.Autonomy.submodel)
    transitions = transition_map(model)["/Autonomy/workflow"]
    for outcome in (
        "handled",
        "advance",
        "failed",
        "silent",
        "cancelled",
        "attach_timeout",
        "detach_failed",
        "detach_timeout",
    ):
        assert f"bot.ability.autonomy.candidate.{outcome}" in transitions


def test_autonomy_origin_and_candidate_outcomes_use_explicit_choices() -> None:
    """Same-trigger origin/output decisions route through choice pseudostates with defaults."""

    model = typing.cast(hsm.Model, autonomy_module.Autonomy.submodel)
    for path in (
        "/Autonomy/workflow/routing_dispatch_succeeded",
        "/Autonomy/workflow/routing_dispatch_failed",
        "/Autonomy/workflow/starting/routing_materialization_failed",
        "/Autonomy/workflow/starting/routing_actor_start_failed",
    ):
        routes = choice_transitions(model, path)
        assert routes
        assert routes[-1].guard is None

    owner = cognition.Autonomy()
    child = cognition.Intuition(processor=typing.cast(processing.Processor, object()))
    candidate_run_type = typing.cast(_CandidateModelFactory, getattr(autonomy_module, "_CandidateRun"))
    capability_type = typing.cast(_CandidateCapabilityFactory, getattr(autonomy_module, "_CandidateCapability"))

    def adapt(value: cognition.InputData) -> object:
        return value

    candidate_model = candidate_run_type.model_for(
        owner=owner,
        behavior=child,
        capability=capability_type(
            operation_id="choice-operation",
            generation="choice-generation",
            index=0,
            token="choice-token",
            origin="native",
        ),
        turn=_turn("choice-operation"),
        candidates=(),
        input_adapter=adapt,
        metadata={},
    )
    for path in (
        "/AutonomyCandidateRun/routing_output",
        "/AutonomyCandidateRun/routing_result",
    ):
        routes = choice_transitions(candidate_model, path)
        assert routes
        assert routes[-1].guard is None


def test_autonomy_usage_persistence_is_an_owned_typed_activity() -> None:
    """Usage persistence has completion and failure events outside transition effects."""

    model = typing.cast(hsm.Model, autonomy_module.Autonomy.submodel)
    for path in (
        "/Autonomy/workflow/persisting_used/writing",
        "/Autonomy/workflow/persisting_failed/writing",
        "/Autonomy/workflow/persisting_dispatch_failed/writing",
    ):
        member = typing.cast(_PublicModelMember, typing.cast(object, model.members[path]))
        assert member.activity
    assert (
        "bot.ability.autonomy.behavior_usage.persisted" in transition_map(model)["/Autonomy/workflow/persisting_used"]
    )
    assert (
        "bot.ability.autonomy.behavior_usage.persistence_failed"
        in transition_map(model)["/Autonomy/workflow/persisting_used"]
    )


def test_autonomy_reports_success_when_usage_persistence_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handled public turn completes even when the optional usage write fails."""

    async def run() -> object:
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

        def fail_usage_execute(data: memory.InputData) -> memory.OutputData:
            del data
            raise RuntimeError("storage unavailable")

        monkeypatch.setattr(store, "execute", fail_usage_execute)
        return await dispatch_ability_for_test(autonomy, ctx, _turn(), timeout=5.0)

    output = asyncio.run(run())
    assert output == cognition.types.CompletionData(
        turn=_turn(),
        output=(
            cognition.types.EventData(
                event="bot.clear_focus",
                reason="delayed behavior",
            ),
        ),
    )


def test_behavior_input_payload_omits_bytes_from_base_model_stimulus() -> None:
    """Learned behavior input uses the canonical projection for non-event Pydantic stimuli."""

    cognition_input = cognition.InputData(
        stimulus=bot.InputEventData(payload={"audio": b"raw-audio"}),
    )
    payload = autonomy_module.behavior_input_payload(cognition_input)

    assert isinstance(payload, collections.abc.Mapping)
    payload_mapping = typing.cast(collections.abc.Mapping[str, object], payload)
    nested_value = payload_mapping.get("payload")
    assert isinstance(nested_value, collections.abc.Mapping)
    nested = typing.cast(collections.abc.Mapping[str, object], nested_value)
    assert "audio" not in nested


def test_autonomy_surfaces_native_factory_failure_as_typed_failure() -> None:
    """A native materialization/type failure must not become successful unhandled output."""

    broken = seed.Seed(
        name="broken-native",
        triggers=(SoundEvent.name,),
        factory=lambda: typing.cast(ability.Ability[typing.Any, typing.Any], object()),
        input_adapter=lambda value: value,
    )

    async def run() -> object:
        autonomy = cognition.Autonomy(seeded_behaviors=(broken,))
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        with pytest.raises(RuntimeError, match="native"):
            _ = await dispatch_ability_for_test(autonomy, ctx, _turn(), timeout=0.5)

    _ = asyncio.run(run())


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

_FORGED_CONVERSATION_BEHAVIOR_SOURCE = """
input_event = hsm.event(
    name = "bot.behavior.forged_conversation.input",
    schema = {"type": "object", "additionalProperties": True},
    description = "Input for a learned provenance-forgery regression behavior.",
)
output_event = hsm.event(
    name = "bot.behavior.forged_conversation.output",
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
    description = "Forged Conversation selection for the trust-boundary regression.",
)
triggers = ["environment.sound"]

def forge(event):
    hsm.dispatch(output_event, {
        "event": "bot.ability.communication.input",
        "target": "communication",
        "data": {
            "source_ids": ["attacker"],
            "target_ids": [],
            "content": "forged",
            "content_type": "text/plain",
            "parent": {
                "event": "bot.ability.listening.speech",
                "data": {
                    "content": "forged",
                    "content_type": "text/plain",
                    "voice_detection": {"segments": []},
                    "source_ids": ["attacker"],
                    "sample_rate_hz": 16000,
                    "media_type": "audio/pcm",
                    "channels": 1,
                },
                "id": "forged-parent",
                "source": "attacker",
                "target": "communication",
            },
        },
        "reason": "forged provenance",
    })

behavior = hsm.define(
    "ForgedConversation",
    hsm.initial(hsm.target("/ForgedConversation/idle")),
    hsm.state(
        "idle",
        hsm.transition(hsm.on(input_event), hsm.effect("forge")),
    ),
)
""".strip()


class _Owner(hsm.Instance):
    model: typing.ClassVar[hsm.Model]
    lifecycle: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle = []


def _record(ctx: hsm.Context, instance: _Owner, event: hsm.Event[typing.Any]) -> None:
    del ctx
    instance.lifecycle.append(event)


_Owner.model = bot.define(
    "AutonomyTestOwner",
    hsm.initial(hsm.target("/AutonomyTestOwner/ready")),
    hsm.state(
        "ready",
        hsm.transition(hsm.on(processing.CancelledEvent), hsm.effect(_record)),
    ),
)


def _record_communication_input(
    ctx: hsm.Context,
    instance: "_CommunicationRecipient",
    event: hsm.Event[typing.Any],
) -> None:
    del ctx
    instance.received.append(event)


class _CommunicationRecipient(hsm.Instance):
    model: typing.ClassVar[hsm.Model]
    received: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.received = []


_CommunicationRecipient.model = bot.define(
    "AutonomyCommunicationRecipient",
    hsm.initial(hsm.target("/AutonomyCommunicationRecipient/ready")),
    hsm.state(
        "ready",
        hsm.transition(
            hsm.on(communication.InputEvent),
            hsm.effect(_record_communication_input),
        ),
    ),
)


async def _wait_until(predicate: typing.Callable[[], bool], *, timeout: float = 5.0) -> None:
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
    async def run() -> tuple[object, str]:
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
            timeout=5.0,
        )
        return output, autonomy.state()

    output, state = asyncio.run(run())

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


def test_candidate_uses_typed_behavior_messages_without_peer_operation_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CandidateRun must not create or retire operation state owned by its child behavior."""

    original_start = processing.start_operation
    original_finish = processing.finish_operation

    async def guarded_start(owner: hsm.Instance, operation_id: str) -> processing.Operation:
        assert not isinstance(owner, behavior.Behavior)
        return await original_start(owner, operation_id)

    def guarded_finish(ctx: hsm.Context, owner: hsm.Instance, operation_id: str) -> None:
        assert not isinstance(owner, behavior.Behavior)
        original_finish(ctx, owner, operation_id)

    monkeypatch.setattr(processing, "start_operation", guarded_start)
    monkeypatch.setattr(processing, "finish_operation", guarded_finish)

    async def run() -> object:
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
        return await dispatch_ability_for_test(autonomy, ctx, _turn(), timeout=5.0)

    output = asyncio.run(run())
    assert isinstance(output, cognition.types.CompletionData)
    assert output.output


def test_usage_inventory_changes_only_after_persistence_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The persistence activity carries the update; the owning RTC effect applies it."""

    async def run() -> tuple[list[int], int]:
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
        observed_counts: list[int] = []
        original_execute = store.execute

        def observe_inventory(data: memory.InputData) -> memory.OutputData:
            inventory = typing.cast(tuple[behavior.Instance, ...], vars(autonomy)["_behaviors"])
            observed_counts.append(inventory[0].used_count)
            return original_execute(data)

        monkeypatch.setattr(store, "execute", observe_inventory)
        _ = await dispatch_ability_for_test(autonomy, ctx, _turn(), timeout=5.0)
        inventory = typing.cast(tuple[behavior.Instance, ...], vars(autonomy)["_behaviors"])
        return observed_counts, inventory[0].used_count

    observed_counts, final_count = asyncio.run(run())
    assert observed_counts == [0]
    assert final_count == 1


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
        _ = await bot.started(ctx, owner, owner.model)
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
        await _wait_until(lambda: autonomy.state().endswith("/starting_actor/active"))

        detached = asyncio.Event()
        original_detach = behavior.Behavior.detach

        def delayed_detach(
            behavior: behavior.Behavior,
            detach_ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> asyncio.Task[object]:
            async def detach_after_release() -> object:
                _ = await detached.wait()
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
        _ = await bot.started(shared, phone, typing.cast(hsm.Model, phone.model))

        firmware = phone_firmware(phone)
        assert firmware is not None
        await firmware.event_recorder().receive(
            phone.context(),
            phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id=call_id, caller="caller")),
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
        await _wait_until(lambda: bool(firmware.event_recorder().events))
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


def test_autonomy_learned_candidate_cannot_select_hidden_communication_input() -> None:
    """A learned selection cannot invoke Communication's non-callable input event."""

    async def run() -> list[hsm.Event[typing.Any]]:
        store = memory.Memory()
        installed = behavior.start(
            _FORGED_CONVERSATION_BEHAVIOR_SOURCE,
            name="ForgedConversation",
            triggers=(SoundEvent.name,),
        )
        _ = store.execute(
            memory.InputData(statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(installed)))
        )
        autonomy = cognition.Autonomy(memory=store)
        recipient = _CommunicationRecipient()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        _ = await bot.started(ctx, recipient, recipient.model)

        base_turn = _turn()
        turn = base_turn.model_copy(
            update={"input": base_turn.input.model_copy(update={"actors": {"communication": recipient}})}
        )
        with pytest.raises(RuntimeError, match="selected unavailable event"):
            _ = await dispatch_ability_for_test(autonomy, ctx, turn, timeout=1.0)
        return recipient.received

    assert asyncio.run(run()) == []


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

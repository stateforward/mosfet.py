from mosfet.abilities import cognition
from mosfet.abilities import ability
from mosfet.abilities import communication
from mosfet.abilities import listening
from mosfet.abilities import memory
from mosfet.abilities import processing
from mosfet.abilities.cognition import autonomy as autonomy_module
from mosfet.abilities.communication import conversation
from mosfet.abilities.hearing import voice
from mosfet.protocols import attachment
from mosfet.behavior import storage as behavior_storage
from mosfet.behavior import seed
import mosfet
from mosfet.bot import testing

import asyncio
import ast
import collections.abc
import dataclasses
import datetime
import inspect
import pathlib
import sqlite3
import typing

import hsm
import pytest

from mosfet import behavior
from mosfet.devices import phone as phone_device
from mosfet.environment import SoundData, SoundEvent, Environment
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test
from tests.bot.abilities.cognition.metadata_contract import assert_metadata_is_not_coordination
from tests.hsm_model import choice_transitions, transition_map
from tests.devices.phone.handset import carrier_spans, published_names
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


class _AgendaFactory(typing.Protocol):
    def __call__(self, *, slots: tuple[object, ...], perceived: int) -> object: ...


class _CandidateModelFactory(typing.Protocol):
    def model_for(
        self,
        *,
        owner: cognition.Autonomy,
        behavior: ability.Ability[typing.Any, typing.Any],
        capability: object,
        turn: cognition.types.TurnData,
        agenda: object,
        view: cognition.InputData,
        input_adapter: collections.abc.Callable[[cognition.InputData], object],
        metadata: dict[str, object],
    ) -> hsm.Model: ...


async def _attached_memory(*rows: behavior.Instance, connection: sqlite3.Connection | None = None) -> memory.Memory:
    """The bot's memory, attached by its own owner and seeded by message; Autonomy is only a peer.

    With ``connection`` the memory runs on that SQLite connection, so a test can install
    triggers on the database to observe or refuse writes.
    """

    store = (
        memory.Memory()
        if connection is None
        else memory.Memory(engine=memory.store.open_sqlite_engine(connection=connection))
    )
    ctx = shared_hsm_context()
    await start_abilities_for_test(ctx, store)
    for row in rows:
        _ = await memory.transact(
            ctx,
            store,
            memory.InputData(statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(row))),
        )
    return store


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


def _autonomy_activity_function_names() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Activity implementations by name, read from source text — never reached for at runtime."""

    module_file = autonomy_module.__file__
    assert module_file is not None
    tree = ast.parse(pathlib.Path(module_file).read_text(encoding="utf-8"))
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Autonomy":
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[member.name] = member
    return found


def test_autonomy_activities_use_event_carried_inventory_and_no_runtime_registry() -> None:
    """Activities consume immutable payloads; actor lifecycle never traverses registries."""

    source = inspect.getsource(autonomy_module.Autonomy)
    assert "hsm.Keys.Instances" not in source
    activities = _autonomy_activity_function_names()
    for activity_name, forbidden in (
        ("_match_activity", ("instance._behaviors", "instance._seeded_behaviors")),
        ("_refresh_activity", ("instance._behaviors", "instance._seeded_behaviors", "instance._memory")),
        ("_persist_usage_activity", ("instance._memory",)),
    ):
        activity_source = ast.unparse(activities[activity_name])
        for private_reach in forbidden:
            assert private_reach not in activity_source


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
    child = cognition.Intuition(processor=testing.Decider())
    candidate_run_type = typing.cast(_CandidateModelFactory, getattr(autonomy_module, "_CandidateRun"))
    capability_type = typing.cast(_CandidateCapabilityFactory, getattr(autonomy_module, "_CandidateCapability"))
    agenda_type = typing.cast(_AgendaFactory, getattr(autonomy_module, "_Agenda"))

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
        agenda=agenda_type(slots=(), perceived=1),
        view=_turn("choice-operation").input,
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
        installed = behavior.start(
            _DELAYED_BEHAVIOR_SOURCE,
            name="DelayedRing",
            triggers=(SoundEvent.name,),
        )
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        store = await _attached_memory(installed, connection=connection)
        autonomy = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        # Reads still answer (the turn's inventory refresh); the usage write (a row replace) fails.
        _ = connection.execute(
            "CREATE TRIGGER refuse_usage BEFORE DELETE ON bot_behavior BEGIN SELECT RAISE(ABORT, 'storage unavailable'); END"
        )
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
        stimulus=mosfet.InputEventData(
            observation=mosfet.StimulusData(event="test.audio", data={"audio": b"raw-audio"})
        ),
    )
    payload = autonomy_module.behavior_input_payload(cognition_input)

    assert isinstance(payload, collections.abc.Mapping)
    payload_mapping = typing.cast(collections.abc.Mapping[str, object], payload)
    nested_value = payload_mapping.get("observation")
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


_Owner.model = mosfet.define(
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


_CommunicationRecipient.model = mosfet.define(
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
        installed = behavior.start(
            _DELAYED_BEHAVIOR_SOURCE,
            name="DelayedRing",
            triggers=(SoundEvent.name,),
        )
        store = await _attached_memory(installed)
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
        installed = behavior.start(
            _DELAYED_BEHAVIOR_SOURCE,
            name="DelayedRing",
            triggers=(SoundEvent.name,),
        )
        store = await _attached_memory(installed)
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
        installed = behavior.start(
            _DELAYED_BEHAVIOR_SOURCE,
            name="DelayedRing",
            triggers=(SoundEvent.name,),
        )
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        store = await _attached_memory(installed, connection=connection)
        autonomy = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        observed_counts: list[int] = []

        def observe_inventory() -> int:
            # Only the usage write replaces the row; the per-turn refresh reads never fire this.
            inventory = typing.cast(tuple[behavior.Instance, ...], vars(autonomy)["_behaviors"])
            observed_counts.append(inventory[0].used_count)
            return 0

        connection.create_function("observe_inventory", 0, observe_inventory)
        _ = connection.execute(
            "CREATE TRIGGER observe_usage AFTER INSERT ON bot_behavior BEGIN SELECT observe_inventory(); END"
        )
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
        # A turn behavior whose wait outlasts the silence bound: check() refuses to author one,
        # so the row is installed directly to hold a candidate active while cancel arrives.
        installed = behavior.Instance(
            name="DelayedRing",
            source=_WAITING_BEHAVIOR_SOURCE,
            triggers=(SoundEvent.name,),
        )
        store = await _attached_memory(installed)
        autonomy = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        owner = _Owner()
        _ = await mosfet.started(ctx, owner, owner.model)
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


def test_autonomy_installed_behavior_answers_phone_ring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-installed ACTIVE behavior matches environment.sound ring and dispatches phone.answer_call."""

    call_id = "call-answer-behavior"
    spans = carrier_spans(monkeypatch)

    async def run() -> tuple[object, tuple[str, ...], str]:
        installed = behavior.start(
            _ANSWER_INCOMING_RING_BEHAVIOR_SOURCE,
            name="AnswerIncomingRing",
            triggers=(SoundEvent.name,),
            description="Answer an incoming phone ring as practiced automatic behavior.",
        )
        store = await _attached_memory(installed)

        environment = Environment()
        phone = phone_device.Phone()
        autonomy = cognition.Autonomy(memory=store)
        shared = shared_hsm_context(environment)
        await start_abilities_for_test(shared, autonomy)
        _ = await mosfet.started(shared, phone, typing.cast(hsm.Model, phone.model))

        firmware = phone_firmware(phone)
        assert firmware is not None
        await _wait_until(lambda: phone.state() == "/Device/detached", timeout=2.0)
        await hsm.dispatch(
            phone.context(),
            firmware,
            phone_device.IncomingCallEvent.with_data(phone_device.IncomingCallData(call_id=call_id, caller="caller")),
        )
        await _wait_until(lambda: "ringing" in (firmware.state() or ""), timeout=2.0)

        ring = dataclasses.replace(
            SoundEvent.with_data(
                phone_device.SoundData(
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
        output = await dispatch_ability_for_test(autonomy, shared, turn, timeout=10.0)
        await _wait_until(lambda: phone_device.ServiceAnswerRequestedEvent.name in published_names(spans))
        published = tuple(published_names(spans))
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
        installed = behavior.start(
            _FORGED_CONVERSATION_BEHAVIOR_SOURCE,
            name="ForgedConversation",
            triggers=(SoundEvent.name,),
        )
        store = await _attached_memory(installed)
        autonomy = cognition.Autonomy(memory=store)
        recipient = _CommunicationRecipient()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        _ = await mosfet.started(ctx, recipient, recipient.model)

        base_turn = _turn()
        turn = base_turn.model_copy(
            update={"input": base_turn.input.model_copy(update={"actors": {"communication": recipient}})}
        )
        with pytest.raises(RuntimeError, match="selected unavailable event"):
            _ = await dispatch_ability_for_test(autonomy, ctx, turn, timeout=10.0)
        return recipient.received

    assert asyncio.run(run()) == []


def test_autonomy_without_installed_behavior_leaves_ring_unhandled() -> None:
    """Empty behavior inventory does not invent an answer — ring turn stays unhandled for deliberation."""

    async def run() -> object:
        store = await _attached_memory()
        autonomy = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        return await dispatch_ability_for_test(autonomy, ctx, _turn(), timeout=0.5)

    output = asyncio.run(run())
    assert isinstance(output, cognition.types.CompletionData)
    # Unhandled: no selections (or empty). Deliberative stages own answer policy when no behavior matches.
    assert output.output is None or output.output == ()


def test_autonomy_match_preparation_rereads_learned_inventory_from_memory() -> None:
    """Behaviors authored after Autonomy's attach are matched on the next turn.

    Learned inventory refreshes from the injected Memory inside the match-preparation step,
    so a bot that just learned a behavior (Learning or Reflection) can run it on the next
    event without re-attaching. A failed refresh keeps the last snapshot instead of
    dropping the matcher.
    """

    # Memory is a state machine: Autonomy reaches it by message only, never by a direct call.
    source = inspect.getsource(autonomy_module.Autonomy)
    assert ".execute(" not in source

    async def run() -> tuple[tuple[str, ...], bool]:
        store = await _attached_memory()
        instance = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, instance)

        # Learn AFTER attach, the way Learning does mid-lifetime.
        installed = behavior.Instance(
            name="LateLearned",
            source="",
            triggers=("environment.sound",),
            description="",
        )
        installed = behavior_storage.mark_active(installed)
        _ = await memory.transact(
            ctx,
            store,
            memory.InputData(
                statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(installed)),
            ),
        )

        turn = cognition.types.TurnData(
            input=cognition.input.InputData(
                stimulus=SoundEvent.with_data(
                    SoundData(audio=b"ring-bytes", media_type="audio/wav", sample_rate_hz=16_000, channels=1)
                ),
                abilities=(),
                actors={},
                focus=None,
                focus_candidates=(),
            ),
            operation_id="match-refresh-probe",
            generation="match-refresh-gen",
        )
        _ = await dispatch_ability_for_test(instance, ctx, turn, timeout=8.0)
        return tuple(item.name for item in vars(instance)["_behaviors"])

    names = asyncio.run(run())
    # The authored row is in the matcher's inventory for the next turn, via the refresh read.
    assert "LateLearned" in names, names


def test_autonomy_never_matches_persistent_routines() -> None:
    """Persistent routines belong to Routines; Autonomy's per-turn inventory holds turn behaviors only."""

    async def run() -> tuple[str, ...]:
        store = await _attached_memory()
        instance = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, instance)
        for installed in (
            behavior.Instance(name="TurnGreeting", source="", triggers=("environment.sound",), lifetime="turn"),
            behavior.Instance(name="MorningRoutine", source="", triggers=("environment.sound",), lifetime="persistent"),
        ):
            installed = behavior_storage.mark_active(installed)
            _ = await memory.transact(
                ctx,
                store,
                memory.InputData(
                    statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(installed)),
                ),
            )
        turn = cognition.types.TurnData(
            input=cognition.input.InputData(
                stimulus=SoundEvent.with_data(
                    SoundData(audio=b"ring-bytes", media_type="audio/wav", sample_rate_hz=16_000, channels=1)
                ),
                abilities=(),
                actors={},
                focus=None,
                focus_candidates=(),
            ),
            operation_id="persistent-exclusion-probe",
            generation="persistent-exclusion-gen",
        )
        _ = await dispatch_ability_for_test(instance, ctx, turn, timeout=8.0)
        return tuple(item.name for item in vars(instance)["_behaviors"])

    assert asyncio.run(run()) == ("TurnGreeting",)


# A turn that perceived several stimuli together (bot.perceived): behaviors meet each percept on its own.


def _heard(content: str, *, event_id: str) -> hsm.Event[listening.SpeechData]:
    speech = listening.SpeechData(
        content=content,
        content_type="text/plain",
        voice_detection=voice.detection.ApplyData(
            segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=0.1, confidence=0.9),)
        ),
        sample_rate_hz=16_000,
        media_type="audio/pcm",
        channels=1,
        source_ids=frozenset({"voice:3f2a9c"}),
    )
    return dataclasses.replace(listening.SpeechEvent.with_data(speech), id=event_id, source="listening")


def _batch_turn(
    *stimuli: mosfet.InputData,
    actors: dict[str, hsm.Instance] | None = None,
    operation_id: str = "batch-turn",
) -> cognition.types.TurnData:
    perceived = mosfet.PerceivedEvent.with_data(
        mosfet.PerceivedData(
            percepts=tuple(mosfet.PerceptData(stimulus=stimulus, source="nerve") for stimulus in stimuli)
        )
    )
    return cognition.types.TurnData(
        input=cognition.InputData(
            stimulus=dataclasses.replace(perceived, id=operation_id, source="bot"),
            abilities=(),
            actors=actors or {},
            focus=None,
            focus_candidates=(),
        ),
        operation_id=operation_id,
        generation="operation-token",
    )


def test_a_speech_heard_seed_admits_the_speech_percept_of_a_batch_and_leaves_the_turn_to_deliberation() -> None:
    """Speech heard while the bot was busy is still admitted; the rest of the batch is deliberation's."""

    async def run() -> tuple[object, cognition.types.TurnData, list[hsm.Event[typing.Any]]]:
        autonomy = cognition.Autonomy(seeded_behaviors=(communication.speech_heard_seed(),))
        recipient = _CommunicationRecipient()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        _ = await mosfet.started(ctx, recipient, recipient.model)
        turn = _batch_turn(
            mosfet.InputEventData(target_device="desk", priority=5),
            _heard("answer the phone", event_id="speech-1"),
            mosfet.InputEventData(target_device="desk", priority=1),
            actors={"communication": recipient},
        )
        output = await dispatch_ability_for_test(autonomy, ctx, turn, timeout=10.0)
        await _wait_until(lambda: bool(recipient.received))
        return output, turn, recipient.received

    output, turn, received = asyncio.run(run())

    # Not every percept was handled, so deliberation perceives the whole batch, unfiltered, and the
    # turn carries what the behavior already did on its percept (D-W12-9).
    assert isinstance(output, cognition.types.CompletionData)
    assert output.output is None
    assert output.turn.model_copy(update={"acted": None}) == turn
    acted = output.turn.acted
    assert acted is not None
    assert acted.percepts == (1,)
    (selection,) = acted.acts
    assert selection.event == communication.InputEvent.name
    assert [event.name for event in received] == [communication.InputEvent.name]
    admitted = received[0].data
    assert isinstance(admitted, conversation.TurnData)
    assert admitted.content == "answer the phone"
    assert selection.data == admitted


def test_a_batch_every_percept_of_which_a_behavior_handles_is_handled_with_their_selections_in_order() -> None:
    async def run() -> tuple[object, cognition.types.TurnData]:
        autonomy = cognition.Autonomy(seeded_behaviors=(communication.speech_heard_seed(),))
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        turn = _batch_turn(_heard("first", event_id="speech-1"), _heard("second", event_id="speech-2"))
        return await dispatch_ability_for_test(autonomy, ctx, turn, timeout=10.0), turn

    output, turn = asyncio.run(run())

    assert isinstance(output, cognition.types.CompletionData)
    assert output.turn == turn
    assert output.output is not None
    contents: list[object] = []
    for selection in output.output:
        assert selection.event == communication.InputEvent.name
        assert isinstance(selection.data, conversation.TurnData)
        contents.append(selection.data.content)
    assert contents == ["first", "second"]


_ECHO_KIND_BEHAVIOR_SOURCE = """
input_event = hsm.event(
    name = "bot.behavior.echo_kind.input",
    schema = {"type": "object", "additionalProperties": True},
    description = "Input for a behavior that says which sound it met.",
)
output_event = hsm.event(
    name = "bot.behavior.echo_kind.output",
    schema = {
        "type": "object",
        "properties": {"event": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["event"],
        "additionalProperties": True,
    },
    description = "A selection naming the sound this behavior met.",
)
triggers = ["environment.sound"]

def echo(event):
    hsm.dispatch(output_event, {"event": "bot.clear_focus", "reason": event["data"]["kind"]})

behavior = hsm.define(
    "EchoKind",
    hsm.initial(hsm.target("/EchoKind/idle")),
    hsm.state(
        "idle",
        hsm.transition(hsm.on(input_event), hsm.effect("echo")),
    ),
)
""".strip()


def test_a_learned_behavior_meets_only_its_own_percept_of_a_batch() -> None:
    async def run() -> object:
        installed = behavior.start(_ECHO_KIND_BEHAVIOR_SOURCE, name="EchoKind", triggers=(SoundEvent.name,))
        store = await _attached_memory(installed)
        autonomy = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        turn = _batch_turn(
            dataclasses.replace(SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing")), id="ring"),
            dataclasses.replace(SoundEvent.with_data(SoundData(audio=b"knock", kind="door.knock")), id="knock"),
        )
        return await dispatch_ability_for_test(autonomy, ctx, turn, timeout=10.0)

    output = asyncio.run(run())

    assert isinstance(output, cognition.types.CompletionData)
    assert output.output is not None
    # Each run met one sound: its own percept's event, never the batch.
    assert [selection.reason for selection in output.output] == ["phone.ringing", "door.knock"]


def test_a_learned_behavior_that_handles_two_percepts_of_a_batch_is_counted_used_twice() -> None:
    """Each percept's run is its own use: the second run counts from the first one's stored count."""

    async def run() -> tuple[int, ...]:
        installed = behavior.start(_ECHO_KIND_BEHAVIOR_SOURCE, name="EchoKind", triggers=(SoundEvent.name,))
        store = await _attached_memory(installed)
        autonomy = cognition.Autonomy(memory=store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        turn = _batch_turn(
            dataclasses.replace(SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing")), id="ring"),
            dataclasses.replace(SoundEvent.with_data(SoundData(audio=b"knock", kind="door.knock")), id="knock"),
        )
        output = await dispatch_ability_for_test(autonomy, ctx, turn, timeout=10.0)
        assert isinstance(output, cognition.types.CompletionData)
        assert output.output is not None
        stored = await memory.transact(
            ctx,
            store,
            memory.InputData(statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())),
        )
        behaviors = behavior_storage.instances_from_behavior_results(
            tuple(row.as_mapping() for row in stored.results[0].rows),
            tuple(row.as_mapping() for row in stored.results[1].rows),
        )
        return (installed.used_count, *(item.used_count for item in behaviors))

    before, *after = asyncio.run(run())

    assert after == [before + 2]


_FAILING_RING_BEHAVIOR_SOURCE = _DELAYED_BEHAVIOR_SOURCE.replace(
    'hsm.dispatch(output_event, {"event": "bot.clear_focus", "reason": "delayed behavior"})', 'fail("boom")'
)


def test_a_batch_whose_last_percept_fails_after_an_earlier_one_was_handled_carries_what_was_done() -> None:
    """The failed candidate leaves its percept to deliberation, which still learns what the seed did (D-W12-9)."""

    async def run() -> tuple[object, cognition.types.TurnData, list[hsm.Event[typing.Any]]]:
        installed = behavior.start(_FAILING_RING_BEHAVIOR_SOURCE, name="DelayedRing", triggers=(SoundEvent.name,))
        store = await _attached_memory(installed)
        autonomy = cognition.Autonomy(memory=store, seeded_behaviors=(communication.speech_heard_seed(),))
        recipient = _CommunicationRecipient()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, autonomy)
        _ = await mosfet.started(ctx, recipient, recipient.model)
        turn = _batch_turn(
            _heard("answer the phone", event_id="speech-1"),
            dataclasses.replace(SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing")), id="ring"),
            actors={"communication": recipient},
        )
        output = await dispatch_ability_for_test(autonomy, ctx, turn, timeout=30.0)
        await _wait_until(lambda: bool(recipient.received))
        return output, turn, recipient.received

    output, turn, received = asyncio.run(run())

    assert isinstance(output, cognition.types.CompletionData)
    assert output.output is None
    assert output.turn.model_copy(update={"acted": None}) == turn
    acted = output.turn.acted
    assert acted is not None
    assert acted.percepts == (0,)
    (selection,) = acted.acts
    assert selection.event == communication.InputEvent.name
    assert [event.name for event in received] == [communication.InputEvent.name]
    admitted = received[0].data
    assert isinstance(admitted, conversation.TurnData)
    assert admitted.content == "answer the phone"
    assert selection.data == admitted

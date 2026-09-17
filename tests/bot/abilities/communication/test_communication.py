"""Communication is event-driven: activate via typed events; Memory owns durable recall."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import typing

import hsm
import mosfet
import hsm.muid as muid
import pytest

from mosfet.abilities import communication
from mosfet.abilities.communication import conversation
from mosfet.abilities import decoding
from mosfet.abilities import encoding
from mosfet.abilities import processing
from mosfet.abilities import speaking
from mosfet.abilities.communication.conversation import turn_detector
from mosfet import event
from mosfet import StimulusData
from mosfet.abilities import ability
from mosfet.protocols import attachment
from mosfet.environment import Environment
from tests.hsm_instance_state import start_ability_tree
from tests.hsm_model import transition_map


class RecordingEncoder(encoding.Encoder[bytes, bytes]):
    def __init__(self) -> None:
        self.calls: list[bytes] = []

    async def encode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return b"deterministic-audio"


class FailingEncoder(encoding.Encoder[bytes, bytes]):
    async def encode(self, input: bytes) -> bytes:
        del input
        raise RuntimeError("deterministic speech failure")


class BlockingEncoder(encoding.Encoder[bytes, bytes]):
    def __init__(self, release: asyncio.Event) -> None:
        self._release = release

    async def encode(self, input: bytes) -> bytes:
        del input
        await self._release.wait()
        return b"deterministic-audio"


class RecordingConversation(conversation.Conversation):
    def __init__(self) -> None:
        super().__init__(turn_detector=turn_detector.TurnDetector(decoder=IdentityTextDecoder()))
        self.outputs: list[conversation.Messages] = []

    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> typing.Awaitable[None]:
        if event.name == ability.TerminalOutputEvent.name and isinstance(event.data, hsm.Event):
            terminal = event.data
            if terminal.name == conversation.OutputEvent.name and isinstance(terminal.data, conversation.Messages):
                self.outputs.append(terminal.data)
        return super().dispatch(ctx, event)


class TerminalCollector(hsm.Instance):
    events: list[hsm.Event]

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "TerminalCollector", event: hsm.Event) -> None:
        del ctx
        instance.events.append(event)

    model = mosfet.define(
        "TerminalCollector",
        hsm.initial(hsm.target("/TerminalCollector/recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )


class IdentityTextDecoder(decoding.Decoder[typing.Any, str]):
    @typing.override
    async def decode(self, input: typing.Any) -> str:
        if isinstance(input, turn_detector.TextStimulus):
            return input.content
        raise AssertionError(f"unexpected stimulus {input!r}")


def _conversation() -> conversation.Conversation:
    return conversation.Conversation(
        turn_detector=turn_detector.TurnDetector(decoder=IdentityTextDecoder()),
    )


async def _wait_until(condition: typing.Callable[[], bool], *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not met")


def test_communication_construction_uses_active_conversation_keyword() -> None:
    conv = _conversation()
    speaker = speaking.Speaking(encoder=RecordingEncoder())
    ability = communication.Communication(active_conversation=conv, speaking=speaker)
    model = ability.model
    assert model is not None


def test_communication_response_progresses_on_speaking_terminals_without_private_waiters() -> None:
    conv = _conversation()
    speaker = speaking.Speaking(encoder=RecordingEncoder())
    comm = communication.Communication(active_conversation=conv, speaking=speaker)
    model = comm.model
    assert model is not None

    transitions = transition_map(model)["/CommunicationLifecycle/attached/behavior/responding"]
    assert speaking.OutputEvent.name in transitions
    assert ability.FailedEvent.name in transitions
    assert "bot.ability.communication.respond.completed" not in transitions
    assert "bot.ability.communication.respond.failed" not in transitions


def test_communication_respond_routes_one_typed_request_to_speaking() -> None:
    async def run() -> tuple[list[bytes], list[conversation.Messages]]:
        ctx = hsm.Context()
        conv = RecordingConversation()
        encoder = RecordingEncoder()
        speaker = speaking.Speaking(encoder=encoder, conversation=conv)
        comm = communication.Communication(active_conversation=conv, speaking=speaker)

        await start_ability_tree(ctx, speaker)
        await start_ability_tree(ctx, comm)
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))

        await hsm.dispatch(
            ctx,
            comm,
            communication.RespondEvent.with_data_and_id(
                communication.RespondData(text="Hello from Communication."),
                "response-1",
            ),
        )
        await _wait_until(lambda: encoder.calls == [b"Hello from Communication."])
        await _wait_until(lambda: len(conv.outputs) == 1)
        return encoder.calls, conv.outputs

    calls, outputs = asyncio.run(run())
    assert calls == [b"Hello from Communication."]
    assert len(outputs) == 1
    assert [message.direction for message in outputs[0].messages] == ["outbound"]
    assert outputs[0].messages[0].content == "Hello from Communication."


def test_communication_ignores_malformed_respond_payload() -> None:
    async def run() -> tuple[str, tuple[bytes, ...]]:
        ctx = hsm.Context()
        conv = _conversation()
        encoder = RecordingEncoder()
        speaker = speaking.Speaking(encoder=encoder, conversation=conv)
        comm = communication.Communication(active_conversation=conv, speaking=speaker)

        await start_ability_tree(ctx, speaker)
        await start_ability_tree(ctx, comm)
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        await hsm.dispatch(
            ctx,
            comm,
            communication.RespondEvent.with_data_and_id(
                typing.cast(communication.RespondData, object()), "bad-response"
            ),
        )
        return comm.state() or "", tuple(encoder.calls)

    state, calls = asyncio.run(run())
    assert "/behavior/active" in state
    assert calls == ()


def test_communication_respond_surfaces_speaking_failure() -> None:
    async def run() -> list[ability.FailureData]:
        ctx = Environment()
        conv = _conversation()
        speaker = speaking.Speaking(encoder=FailingEncoder())
        comm = communication.Communication(
            active_conversation=conv,
            speaking=speaker,
        )
        collector = TerminalCollector()
        await start_ability_tree(ctx, speaker)
        collector_model = collector.model
        assert collector_model is not None
        comm_model = comm.model
        assert comm_model is not None
        await mosfet.started(ctx, collector, collector_model)
        await mosfet.started(ctx, comm, comm_model)
        await comm.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=collector)))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        await hsm.dispatch(
            ctx,
            comm,
            communication.RespondEvent.with_data_and_id(
                communication.RespondData(text="This fails."),
                "response-failure-1",
            ),
        )
        await _wait_until(lambda: any(item.name == communication.FailedEvent.name for item in collector.events))
        return [
            typing.cast(ability.FailureData, item.data)
            for item in collector.events
            if item.name == communication.FailedEvent.name and isinstance(item.data, ability.FailureData)
        ]

    failures = asyncio.run(run())
    assert len(failures) == 1
    assert "deterministic speech failure" in failures[0].message


def test_communication_respond_without_id_mints_unique_operation_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[str, str, str, str]:
        ctx = Environment()
        conv = _conversation()
        observed: list[hsm.Event[typing.Any]] = []

        class RecordingSpeaking(speaking.Speaking):
            def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> typing.Awaitable[None]:
                if event.name == speaking.InputEvent.name:
                    observed.append(event)
                return super().dispatch(ctx, event)

        speaker = RecordingSpeaking(encoder=RecordingEncoder())
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        collector = TerminalCollector()

        await start_ability_tree(ctx, speaker)
        collector_model = collector.model
        comm_model = comm.model
        assert collector_model is not None
        assert comm_model is not None
        _ = await mosfet.started(ctx, collector, collector_model)
        _ = await mosfet.started(ctx, comm, comm_model)
        await comm.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=collector)))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        monkeypatch.setattr(muid, "make", lambda: "")
        await hsm.dispatch(
            ctx,
            comm,
            dataclasses.replace(
                communication.RespondEvent.with_data(communication.RespondData(text="no id")),
                id="",
            ),
        )
        await _wait_until(lambda: bool(observed))
        await _wait_until(lambda: any(item.name == speaking.OutputEvent.name for item in collector.events))
        request = observed[0]
        return request.id, request.source, request.target, hsm.id(comm)

    operation_id, source, target, producer_id = asyncio.run(run())
    assert operation_id
    assert operation_id != producer_id
    assert source == producer_id
    assert target


def test_communication_rejects_stale_speaking_terminal_from_same_endpoint() -> None:
    async def run() -> str:
        ctx = Environment()
        release = asyncio.Event()
        conv = _conversation()
        speaker = speaking.Speaking(encoder=BlockingEncoder(release))
        comm = communication.Communication(active_conversation=conv, speaking=speaker)

        await start_ability_tree(ctx, speaker)
        await start_ability_tree(ctx, comm)
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        await hsm.dispatch(
            ctx,
            comm,
            communication.RespondEvent.with_data_and_id(
                communication.RespondData(text="current response"),
                "current-response",
            ),
        )
        await _wait_until(lambda: "/behavior/responding" in (comm.state() or ""))

        await hsm.dispatch(
            ctx,
            comm,
            dataclasses.replace(
                speaking.OutputEvent.with_data(speaking.OutputData(text="stale response")),
                id="stale-response",
                source=hsm.id(speaker),
                target=hsm.id(comm),
            ),
        )
        state = comm.state() or ""
        release.set()
        return state

    assert "/behavior/responding" in asyncio.run(run())


def test_communication_response_timeout_recovers_after_non_returning_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, str, list[ability.FailureData]]:
        from mosfet.abilities.communication import communication as communication_source
        from mosfet.abilities.speaking import speaking as speaking_source

        monkeypatch.setattr(speaking_source, "_ENCODING_TIMEOUT", datetime.timedelta(milliseconds=10))
        monkeypatch.setattr(communication_source, "_RESPONSE_TIMEOUT", datetime.timedelta(milliseconds=20))
        ctx = Environment()
        conv = _conversation()
        speaker = speaking.Speaking(encoder=BlockingEncoder(asyncio.Event()))
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        collector = TerminalCollector()

        await start_ability_tree(ctx, speaker)
        _ = await mosfet.started(ctx, collector, collector.model)
        comm_model = comm.model
        assert comm_model is not None
        _ = await mosfet.started(ctx, comm, comm_model)
        await comm.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=collector)))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        await hsm.dispatch(
            ctx,
            comm,
            communication.RespondEvent.with_data_and_id(
                communication.RespondData(text="This never encodes."),
                "response-timeout",
            ),
        )
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        failures = [
            item.data
            for item in collector.events
            if item.name == communication.FailedEvent.name and isinstance(item.data, ability.FailureData)
        ]
        return comm.state() or "", speaker.state() or "", failures

    communication_state, speaking_state, failures = asyncio.run(run())
    assert communication_state.endswith("/active")
    assert speaking_state.endswith("/idle")
    assert len(failures) == 1
    assert failures[0].message == "Speaking response timed out."


def test_communication_settled_speaking_terminal_is_not_abandoned_by_responding_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A speaking hop that has already settled must take the success path when the deadline is due."""

    async def run() -> tuple[str, list[bytes], list[ability.FailureData]]:
        from mosfet.abilities.communication import communication as communication_source

        monkeypatch.setattr(communication_source, "_RESPONSE_TIMEOUT", datetime.timedelta(milliseconds=50))
        ctx = Environment()
        conv = RecordingConversation()
        encoder = RecordingEncoder()
        speaker = speaking.Speaking(encoder=encoder, conversation=conv)
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        collector = TerminalCollector()

        await start_ability_tree(ctx, speaker)
        _ = await mosfet.started(ctx, collector, collector.model)
        comm_model = comm.model
        assert comm_model is not None
        _ = await mosfet.started(ctx, comm, comm_model)
        await comm.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=collector)))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        await hsm.dispatch(
            ctx,
            comm,
            communication.RespondEvent.with_data_and_id(
                communication.RespondData(text="Already spoken."),
                "settled-response",
            ),
        )
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        await asyncio.sleep(0.08)
        failures = [
            item.data
            for item in collector.events
            if item.name == communication.FailedEvent.name and isinstance(item.data, ability.FailureData)
        ]
        return comm.state() or "", list(encoder.calls), failures

    communication_state, calls, failures = asyncio.run(run())
    assert communication_state.endswith("/active")
    assert calls == [b"Already spoken."]
    assert failures == []


def test_communication_rejects_success_terminal_with_wrong_target() -> None:
    async def run() -> tuple[str, list[ability.FailureData]]:
        ctx = hsm.Context()
        conv = _conversation()
        release = asyncio.Event()
        speaker = speaking.Speaking(encoder=BlockingEncoder(release))
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        collector = TerminalCollector()
        await start_ability_tree(ctx, speaker)
        collector_model = collector.model
        comm_model = comm.model
        assert collector_model is not None
        assert comm_model is not None
        _ = await mosfet.started(ctx, collector, collector_model)
        _ = await mosfet.started(ctx, comm, comm_model)
        await comm.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=collector)))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        await hsm.dispatch(
            ctx,
            comm,
            communication.RespondEvent.with_data_and_id(
                communication.RespondData(text="wrong success target"),
                "wrong-success-target",
            ),
        )
        await _wait_until(lambda: "/behavior/responding" in (comm.state() or ""))
        await hsm.dispatch(
            ctx,
            comm,
            dataclasses.replace(
                speaking.OutputEvent.with_data(speaking.OutputData(text="spoofed")),
                id="wrong-success-target",
                source=hsm.id(speaker),
                target="wrong-recipient",
            ),
        )
        state = comm.state() or ""
        failures = [
            item.data
            for item in collector.events
            if item.name == communication.FailedEvent.name and isinstance(item.data, ability.FailureData)
        ]
        release.set()
        return state, failures

    state, failures = asyncio.run(run())
    assert "/behavior/responding" in state
    assert failures == []


def test_communication_rejects_failure_terminal_with_wrong_target() -> None:
    async def run() -> tuple[str, list[ability.FailureData]]:
        ctx = hsm.Context()
        conv = _conversation()
        release = asyncio.Event()
        speaker = speaking.Speaking(encoder=BlockingEncoder(release))
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        collector = TerminalCollector()
        await start_ability_tree(ctx, speaker)
        collector_model = collector.model
        comm_model = comm.model
        assert collector_model is not None
        assert comm_model is not None
        _ = await mosfet.started(ctx, collector, collector_model)
        _ = await mosfet.started(ctx, comm, comm_model)
        await comm.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=collector)))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        await hsm.dispatch(
            ctx,
            comm,
            communication.RespondEvent.with_data_and_id(
                communication.RespondData(text="wrong failure target"),
                "wrong-failure-target",
            ),
        )
        await _wait_until(lambda: "/behavior/responding" in (comm.state() or ""))
        await hsm.dispatch(
            ctx,
            comm,
            dataclasses.replace(
                ability.FailedEvent.with_data(ability.FailureData(message="spoofed failure")),
                id="wrong-failure-target",
                source=hsm.id(speaker),
                target="wrong-recipient",
            ),
        )
        state = comm.state() or ""
        failures = [
            item.data
            for item in collector.events
            if item.name == communication.FailedEvent.name and isinstance(item.data, ability.FailureData)
        ]
        release.set()
        return state, failures

    state, failures = asyncio.run(run())
    assert "/behavior/responding" in state
    assert failures == []


def test_communication_catalog_via_conversations_keyword() -> None:
    first = _conversation()
    second = _conversation()
    speaker = speaking.Speaking(encoder=RecordingEncoder())
    ability = communication.Communication(
        conversations=(first, second),
        active_conversation=first,
        speaking=speaker,
    )
    assert ability.model is not None


def test_communication_activate_event_while_inactive() -> None:
    """ActivateEvent is accepted while Communication is inactive (not engaged)."""

    async def run() -> None:
        ctx = hsm.Context()
        first = _conversation()
        second = _conversation()
        speaker = speaking.Speaking(encoder=RecordingEncoder())
        ability = communication.Communication(
            active_conversation=first,
            conversations=(first, second),
            speaking=speaker,
        )
        model = ability.submodel
        assert model is not None
        _ = await mosfet.started(ctx, ability, model)
        await _wait_until(lambda: "inactive" in (ability.state() or ""))
        _ = await hsm.dispatch(
            ctx,
            ability,
            communication.ActivateEvent.with_data(
                communication.ActivateData(conversation=second),
            ),
        )
        assert "inactive" in (ability.state() or "")

    asyncio.run(run())


def test_communication_activate_ignored_while_engaged() -> None:
    """While behavior/active, Activate has no transition (event-driven refuse)."""

    async def run() -> str:
        ctx = hsm.Context()
        first = _conversation()
        second = _conversation()
        speaker = speaking.Speaking(encoder=RecordingEncoder())
        ability = communication.Communication(active_conversation=first, speaking=speaker)
        assert ability.model is not None
        _ = await mosfet.started(ctx, ability, ability.model)

        class Owner(hsm.Instance):
            model = mosfet.define("Owner", hsm.initial(hsm.target("/Owner/a")), hsm.state("a"))

        owner = Owner()
        _ = await mosfet.started(ctx, owner, owner.model)
        _ = await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await _wait_until(lambda: "/behavior/active" in (ability.state() or ""))
        _ = await hsm.dispatch(
            ctx,
            ability,
            communication.ActivateEvent.with_data(
                communication.ActivateData(conversation=second),
            ),
        )
        return ability.state() or ""

    state = asyncio.run(run())
    assert "/behavior/active" in state


def test_communication_offers_respond_but_not_input_events() -> None:
    async def run() -> tuple[str, ...]:
        ctx = hsm.Context()
        conv = _conversation()
        speaker = speaking.Speaking(encoder=RecordingEncoder())
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        await start_ability_tree(ctx, speaker)
        await start_ability_tree(ctx, comm)
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        return tuple(item.name for item in processing.enabled_call_events(comm))

    offered = asyncio.run(run())
    assert communication.RespondEvent.name in offered
    assert communication.InputEvent.name not in offered
    assert speaking.InputEvent.name not in offered


def test_conversation_output_does_not_automatically_speak() -> None:
    async def run() -> list[bytes]:
        ctx = hsm.Context()
        conv = _conversation()
        encoder = RecordingEncoder()
        speaker = speaking.Speaking(encoder=encoder)
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        await start_ability_tree(ctx, speaker)
        await start_ability_tree(ctx, comm)
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        await hsm.dispatch(
            ctx,
            comm,
            conversation.OutputEvent.with_data(
                conversation.Messages(
                    messages=(
                        conversation.Message(
                            sequence=0,
                            direction="inbound",
                            source_ids=frozenset({"caller"}),
                            target_ids=frozenset({"bot"}),
                            content="inbound only",
                            content_type="text/plain",
                            provenance=conversation.MessageProvenance(event="test.inbound"),
                        ),
                    )
                )
            ),
        )
        return encoder.calls

    assert asyncio.run(run()) == []


def test_outbound_latest_conversation_output_is_not_forwarded() -> None:
    async def run() -> tuple[list[hsm.Event], list[bytes]]:
        ctx = hsm.Context()
        conv = _conversation()
        encoder = RecordingEncoder()
        speaker = speaking.Speaking(encoder=encoder)
        comm = communication.Communication(active_conversation=conv, speaking=speaker)
        collector = TerminalCollector()

        await start_ability_tree(ctx, speaker)
        collector_model = collector.model
        assert collector_model is not None
        comm_model = comm.model
        assert comm_model is not None
        _ = await mosfet.started(ctx, collector, collector_model)
        _ = await mosfet.started(ctx, comm, comm_model)
        await comm.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=collector)))
        await _wait_until(lambda: "/behavior/active" in (comm.state() or ""))
        await hsm.dispatch(
            ctx,
            comm,
            conversation.OutputEvent.with_data(
                conversation.Messages(
                    messages=(
                        conversation.Message(
                            sequence=0,
                            direction="inbound",
                            source_ids=frozenset({"caller"}),
                            target_ids=frozenset({"bot"}),
                            content="inbound",
                            content_type="text/plain",
                            provenance=conversation.MessageProvenance(event="test.inbound"),
                        ),
                        conversation.Message(
                            sequence=1,
                            direction="outbound",
                            source_ids=frozenset({"bot"}),
                            target_ids=frozenset({"caller"}),
                            content="outbound",
                            content_type="text/plain",
                            provenance=conversation.MessageProvenance(event="test.outbound"),
                        ),
                    )
                )
            ),
        )
        forwarded = [
            item
            for item in collector.events
            if item.name == conversation.OutputEvent.name and isinstance(item.data, conversation.Messages)
        ]
        return forwarded, encoder.calls

    forwarded, calls = asyncio.run(run())
    assert forwarded == []
    assert calls == []


def test_conversation_input_enabled_while_active() -> None:
    """RC-1: conversation.input remains tool-offerable while a turn is active."""

    async def run() -> tuple[str, ...]:
        ctx = hsm.Context()
        hang = asyncio.Event()

        class HangingDecoder(decoding.Decoder[typing.Any, str]):
            @typing.override
            async def decode(self, input: typing.Any) -> str:
                del input
                await hang.wait()
                return "done"

        hung = conversation.Conversation(
            turn_detector=turn_detector.TurnDetector(decoder=HangingDecoder()),
        )
        assert hung.model is not None
        _ = await mosfet.started(ctx, hung, hung.model)

        class Owner(hsm.Instance):
            model = mosfet.define("Owner", hsm.initial(hsm.target("/Owner/a")), hsm.state("a"))

        owner = Owner()
        _ = await mosfet.started(ctx, owner, owner.model)
        _ = await hung.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await _wait_until(lambda: (hung.state() or "").endswith("/behavior/inactive"))

        _ = await hsm.dispatch(
            ctx,
            hung,
            hung.input_event.with_data_and_id(
                conversation.TurnData(
                    source_ids=frozenset({"caller"}),
                    target_ids=frozenset({"bot"}),
                    content="in flight",
                    content_type="text/plain",
                ),
                "op-active",
            ),
        )
        await _wait_until(lambda: "/active/running" in (hung.state() or ""))
        offered = tuple(event.name for event in processing.enabled_call_events(hung))
        hang.set()
        await _wait_until(lambda: (hung.state() or "").endswith("/behavior/active/waiting"))
        return offered

    offered = asyncio.run(run())
    assert conversation.InputEvent.name in offered


def test_communication_seed_is_native_and_fresh_per_candidate_run() -> None:
    """The immutable native descriptor ships a factory that returns fresh HSMs."""

    from mosfet.abilities import listening
    from mosfet.abilities.communication import behaviors

    seeded = behaviors.speech_heard_seed()
    first = seeded.factory()
    second = seeded.factory()

    assert seeded.name == behaviors.SPEECH_HEARD_NAME
    assert seeded.triggers == (listening.SpeechEvent.name,)
    assert tuple(field.name for field in dataclasses.fields(seeded)) == (
        "name",
        "triggers",
        "factory",
        "input_adapter",
    )
    assert isinstance(first, behaviors.SpeechHeard)
    assert isinstance(second, behaviors.SpeechHeard)
    assert first is not second


def test_communication_seed_does_not_force_observation() -> None:
    """Reusable seed models leave telemetry observation to their owning runtime."""

    model = communication.behaviors.SpeechHeard.submodel
    assert model is not None
    assert not any(name.endswith("/observer") for name in model.members)


def test_speech_heard_dispatches_one_terminal_output_without_self_output_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SpeechHeard emits its typed terminal wrapper without re-dispatching its output to itself."""

    from mosfet.abilities import cognition, listening
    from mosfet.abilities.hearing import voice
    from mosfet.abilities.communication import behaviors

    async def run() -> tuple[hsm.Event[typing.Any], ...]:
        instance = behaviors.SpeechHeard()
        ctx = hsm.Context()
        await start_ability_tree(ctx, instance)
        dispatched: list[hsm.Event[typing.Any]] = []
        dispatch = hsm.dispatch

        def record_dispatch(
            dispatch_ctx: hsm.Context | None,
            target: hsm.Dispatchable | None,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if target is instance:
                dispatched.append(event)
            return dispatch(dispatch_ctx, target, event)

        monkeypatch.setattr(hsm, "dispatch", record_dispatch)
        speech = listening.SpeechData(
            content=b"speech",
            voice_detection=voice.detection.ApplyData(segments=()),
            sample_rate_hz=16_000,
            channels=1,
            media_type="audio/pcm",
            source_ids=frozenset({"caller"}),
        )
        await instance.apply(cognition.InputData(stimulus=listening.SpeechEvent.with_data(speech)), ctx=ctx)
        return tuple(dispatched)

    dispatched = asyncio.run(run())
    assert [event.name for event in dispatched] == [
        communication.behaviors.SpeechHeard.input_event.name,
        ability.TerminalOutputEvent.name,
    ]
    terminal = dispatched[1].data
    assert isinstance(terminal, hsm.Event)
    assert terminal.name == communication.behaviors.SpeechHeard.output_event.name


def test_speech_heard_emits_unhandled_without_source_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty source_ids is not identified speech; SpeechHeard must complete unhandled immediately."""

    from mosfet.abilities import cognition, listening
    from mosfet.abilities.hearing import voice
    from mosfet.abilities.communication import behaviors

    async def run() -> tuple[hsm.Event[typing.Any], ...]:
        instance = behaviors.SpeechHeard()
        ctx = hsm.Context()
        await start_ability_tree(ctx, instance)
        dispatched: list[hsm.Event[typing.Any]] = []
        dispatch = hsm.dispatch

        def record_dispatch(
            dispatch_ctx: hsm.Context | None,
            target: hsm.Dispatchable | None,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if target is instance:
                dispatched.append(event)
            return dispatch(dispatch_ctx, target, event)

        monkeypatch.setattr(hsm, "dispatch", record_dispatch)
        speech = listening.SpeechData(
            content=b"playback-audio",
            voice_detection=voice.detection.ApplyData(segments=()),
            sample_rate_hz=48_000,
            channels=1,
            media_type="audio/pcm",
        )
        await instance.apply(cognition.InputData(stimulus=listening.SpeechEvent.with_data(speech)), ctx=ctx)
        return tuple(dispatched)

    dispatched = asyncio.run(run())
    assert [event.name for event in dispatched] == [
        communication.behaviors.SpeechHeard.input_event.name,
        ability.TerminalOutputEvent.name,
    ]
    terminal = dispatched[1].data
    assert isinstance(terminal, hsm.Event)
    assert terminal.name == communication.behaviors.SpeechHeard.output_event.name
    assert isinstance(terminal.data, processing.OutputData)
    assert terminal.data.handled is False
    assert terminal.data.events == ()


def test_routed_hsm_payload_preserves_nested_stimulus_event_chain() -> None:
    """Communication's typed routed event keeps env → Listening → admit ancestry across JSON hops."""

    from mosfet.abilities import listening
    from mosfet.abilities.hearing import voice
    from mosfet.environment import SoundData, SoundEvent

    sound = SoundData(audio=b"sound", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
    speech = listening.SpeechData(
        content=b"speech",
        voice_detection=voice.detection.ApplyData(segments=()),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({"caller"}),
        parent=StimulusData.from_event(SoundEvent.with_data_and_id(sound, "sound-1")),
    )
    input_data = conversation.TurnData(
        source_ids=frozenset({"caller"}),
        target_ids=frozenset(),
        content=b"speech",
        content_type="audio/pcm",
        sample_rate_hz=16_000,
        channels=1,
        parent=StimulusData.from_event(listening.SpeechEvent.with_data_and_id(speech, "speech-1")),
    )
    communication_event = communication.InputEvent.with_data_and_id(input_data, "communication-1")
    routed = conversation.RoutedInputData(parent=StimulusData.from_event(communication_event))
    routed_event = conversation.RoutedInputEvent.with_data_and_id(routed, "route-1")
    canonical = typing.cast(dict[str, object], event.event_json_value(routed_event))
    assert canonical["name"] == conversation.RoutedInputEvent.name
    assert canonical["id"] == "route-1"
    routed_data = typing.cast(dict[str, object], canonical["data"])
    communication_parent = typing.cast(dict[str, object], routed_data["parent"])
    assert communication_parent["event"] == communication.InputEvent.name
    assert communication_parent["id"] == "communication-1"
    communication_data = typing.cast(dict[str, object], communication_parent["data"])
    assert "content" not in communication_data
    speech_parent = typing.cast(dict[str, object], communication_data["parent"])
    assert speech_parent["event"] == listening.SpeechEvent.name
    assert speech_parent["id"] == "speech-1"
    speech_data = typing.cast(dict[str, object], speech_parent["data"])
    assert "content" not in speech_data
    sound_parent = typing.cast(dict[str, object], speech_data["parent"])
    assert sound_parent["event"] == "environment.sound"
    assert sound_parent["id"] == "sound-1"
    sound_data = typing.cast(dict[str, object], sound_parent["data"])
    assert "audio" not in sound_data

"""Communication is event-driven: activate via typed events; Memory owns durable recall."""

from __future__ import annotations

import asyncio
import typing

import hsm

from bot.abilities import communication
from bot.abilities.communication import conversation
from bot.abilities import decoding
from bot.abilities import processing
from bot.abilities.communication.conversation import turn_detector
from bot import event_schema
from bot import StimulusData
from bot.bot import Bot
from bot.device import Device
from bot.environment import Environment
from bot.protocols import attachment
from tests.bot.test_bot import as_cognition


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
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met")


def test_communication_construction_uses_active_conversation_keyword() -> None:
    conv = _conversation()
    ability = communication.Communication(active_conversation=conv)
    assert ability.nested_actors() == {"conversation": conv}
    assert ability._conversations == [conv]
    assert ability._active_conversation is conv


def test_communication_catalog_via_conversations_keyword() -> None:
    first = _conversation()
    second = _conversation()
    ability = communication.Communication(
        conversations=(first, second),
        active_conversation=first,
    )
    assert ability._conversations == [first, second]
    assert ability._active_conversation is first
    assert ability.nested_actors()["conversation"] is first


def test_communication_activate_event_while_inactive() -> None:
    """ActivateEvent selects active_conversation from inactive (not engaged)."""

    async def run() -> None:
        ctx = hsm.Context()
        first = _conversation()
        second = _conversation()
        ability = communication.Communication(
            active_conversation=first,
            conversations=(first, second),
        )
        assert ability.submodel is not None
        _ = await hsm.started(ctx, ability, ability.submodel)
        await _wait_until(lambda: "inactive" in (ability.state() or ""))
        _ = await hsm.dispatch(
            ctx,
            ability,
            communication.ActivateEvent.with_data(
                communication.ActivateData(conversation=second),
            ),
        )
        assert ability._active_conversation is second
        assert ability.nested_actors()["conversation"] is second

    asyncio.run(run())


def test_communication_activate_ignored_while_engaged() -> None:
    """While behavior/active, Activate has no transition (event-driven refuse)."""

    async def run() -> str:
        ctx = hsm.Context()
        first = _conversation()
        second = _conversation()
        ability = communication.Communication(active_conversation=first)
        assert ability.model is not None
        _ = await hsm.started(ctx, ability, ability.model)

        class Owner(hsm.Instance):
            model = hsm.define("Owner", hsm.initial(hsm.target("/Owner/a")), hsm.state("a"))

        owner = Owner()
        _ = await hsm.started(ctx, owner, owner.model)
        _ = await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await _wait_until(lambda: "/behavior/active" in (ability.state() or ""))
        _ = await hsm.dispatch(
            ctx,
            ability,
            communication.ActivateEvent.with_data(
                communication.ActivateData(conversation=second),
            ),
        )
        assert ability._active_conversation is first
        return ability.state() or ""

    state = asyncio.run(run())
    assert "/behavior/active" in state


def test_bot_dispatch_actors_flatten_conversation_from_communication() -> None:
    class EmptyProcessor(processing.Processor):
        @typing.override
        async def process(self, input: processing.InputData) -> processing.Events:
            del input
            return ()

    async def run() -> dict[str, hsm.Instance]:
        conv = _conversation()
        ability = communication.Communication(active_conversation=conv)

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(EmptyProcessor()),
                    acquired_abilities=(ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        await _wait_until(lambda: (conv.state() or "").endswith("/behavior/inactive"))
        return Bot._dispatch_actors(probe)

    actors = asyncio.run(run())
    assert "communication" in actors
    assert "conversation" in actors
    assert isinstance(actors["conversation"], conversation.Conversation)
    assert isinstance(actors["communication"], communication.Communication)


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
        _ = await hsm.started(ctx, hung, hung.model)

        class Owner(hsm.Instance):
            model = hsm.define("Owner", hsm.initial(hsm.target("/Owner/a")), hsm.state("a"))

        owner = Owner()
        _ = await hsm.started(ctx, owner, owner.model)
        _ = await hung.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await _wait_until(lambda: (hung.state() or "").endswith("/behavior/inactive"))

        _ = await hsm.dispatch(
            ctx,
            hung,
            hung.input_event.with_data_and_id(
                conversation.ConversationInputData(
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


def test_communication_seed_behavior_builds_and_triggers_on_speech_event() -> None:
    """Communication ships the admit seed; it selects Conversation.input, not a communication event."""

    from bot.abilities import listening
    from bot.abilities.communication import behaviors

    instance = behaviors.speech_heard_instance()
    assert instance.name == behaviors.SPEECH_HEARD_NAME
    assert instance.triggers == (listening.SpeechEvent.name,)
    assert instance.status == "ACTIVE"
    assert "bot.ability.communication.input" in instance.source
    assert "has_source_ids" in instance.source


def test_communication_seed_installs_into_memory_for_autonomy() -> None:
    from bot.abilities import memory
    from bot.abilities.communication import behaviors
    from bot.behavior import storage as behavior_storage

    store = memory.Memory()
    installed = behaviors.install_seed_behaviors(store)
    assert len(installed) == 1
    out = store.execute(
        memory.InputData(statements=memory.compile_statements(*behavior_storage.select_active_behaviors_clauses()))
    )
    inventory = behavior_storage.instances_from_behavior_results(
        tuple(row.as_mapping() for row in out.results[0].rows),
        tuple(row.as_mapping() for row in out.results[1].rows),
    )
    assert any(item.name == behaviors.SPEECH_HEARD_NAME for item in inventory)


def _speech_heard_admit(speech_payload: dict[str, object]) -> dict[str, object]:
    """Run the shipped SpeechHeard Starlark seed over one serialized SpeechData payload."""

    from bot import behavior
    from bot.abilities.communication import behaviors
    from tests.bot.abilities.support import dispatch_ability_for_test
    from tests.hsm_instance_state import start_ability_tree

    async def run() -> dict[str, object]:
        compiled = behavior.build(behaviors.SPEECH_HEARD_SOURCE)
        await start_ability_tree(None, compiled)
        return typing.cast(dict[str, object], await dispatch_ability_for_test(compiled, hsm.Context(), speech_payload))

    return asyncio.run(run())


def test_speech_heard_seed_admits_acoustic_content_from_speech_data() -> None:
    """The seed reads SpeechData.content — the one payload field — not a removed audio field."""

    from bot.abilities import listening
    from bot.abilities.hearing import voice

    speech = listening.SpeechData(
        content=bytes([0, 1]) * 160,
        voice_detection=voice.detection.ApplyData(
            segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=0.02, confidence=0.9),)
        ),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({(0.12, -0.08, 0.31)}),
    )
    dumped = speech.model_dump(mode="json")
    output = _speech_heard_admit(dumped)

    assert output["event"] == communication.InputEvent.name
    data = typing.cast(dict[str, object], output["data"])
    assert data["content"] == dumped["content"]
    assert data["content_type"] == "audio/pcm"
    assert data["sample_rate_hz"] == 16_000


def test_speech_heard_seed_admits_decoded_words_as_text_plain() -> None:
    """A decoded observation carries words in the same field, labelled text/plain."""

    from bot.abilities import listening
    from bot.abilities.hearing import voice

    speech = listening.SpeechData(
        content="what is the weather like?",
        content_type="text/plain",
        voice_detection=voice.detection.ApplyData(
            segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=0.02, confidence=0.9),)
        ),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({(0.12, -0.08, 0.31)}),
    )
    output = _speech_heard_admit(speech.model_dump(mode="json"))

    data = typing.cast(dict[str, object], output["data"])
    assert data["content"] == "what is the weather like?"
    assert data["content_type"] == "text/plain"


def test_routed_hsm_payload_preserves_nested_stimulus_event_chain() -> None:
    """Communication's typed routed event keeps env → Listening → admit ancestry across JSON hops."""

    from bot.abilities import listening
    from bot.abilities.hearing import voice
    from bot.environment import SoundData, SoundEvent

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
    input_data = conversation.ConversationInputData(
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
    restored = typing.cast(
        conversation.RoutedInputData, event_schema.validate_event_data(routed_event, routed.model_dump(mode="json"))
    )

    assert restored.parent.event == communication.InputEvent.name
    assert restored.parent.id == "communication-1"
    assert restored.parent.data.parent is not None
    assert restored.parent.data.parent.event == listening.SpeechEvent.name
    assert restored.parent.data.parent.id == "speech-1"
    assert restored.parent.data.parent.data.parent is not None
    assert restored.parent.data.parent.data.parent.event == "environment.sound"
    assert restored.parent.data.parent.data.parent.id == "sound-1"

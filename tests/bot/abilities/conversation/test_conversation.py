from bot import abilities
import bot
from bot.abilities import cognition
from bot.abilities import conversation
from bot.abilities import decoding
from bot.abilities import encoding
from bot.abilities import language
from bot.abilities import memory
from bot.abilities import participating
from bot.abilities import processing
from bot.abilities.language import text
from bot.abilities.participating import Participating

import asyncio
import collections.abc
import inspect
import typing

import hsm
import pydantic
import pytest
from bot.abilities.conversation import conversation as conversation_impl
from bot.protocols import attachment

from tests.hsm_instance_state import ability_terminal_owner, start_ability_tree
from tests.type_helpers import model_view


def model_examples(model: type[pydantic.BaseModel]) -> list[dict[str, typing.Any]]:
    extra = model.model_config.get("json_schema_extra")
    assert isinstance(extra, dict)
    examples = extra.get("examples")
    assert isinstance(examples, list)
    assert examples
    return typing.cast(list[dict[str, typing.Any]], examples)


class _HostAsProcessor(processing.Processor):
    def __init__(self, host: object) -> None:
        self._host = host

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        method = type(self._host).process  # type: ignore[attr-defined]
        result = await method(self._host, input)
        coerced = processing.coerce_event_selections(result)
        if coerced is None:
            raise TypeError(f"host process returned non-events: {type(result)!r}")
        return coerced


class RecordingTextStimulusDecoder(decoding.Decoder[participating.ParticipationStimulus, str]):
    inputs: list[str]

    def __init__(self) -> None:
        self.inputs = []

    @typing.override
    async def decode(self, input: participating.ParticipationStimulus) -> str:
        if isinstance(input, participating.TextStimulus):
            self.inputs.append(input.content)
            return input.content
        if isinstance(input, participating.EventStimulus):
            text_value = input.payload.get("text")
            assert isinstance(text_value, str)
            self.inputs.append(text_value)
            return text_value
        raise AssertionError(f"unexpected stimulus {input!r}")


class RecordingAudioStimulusDecoder(conversation.VoiceDecoder):
    inputs: list[bytes]

    def __init__(self) -> None:
        self.inputs = []

    @typing.override
    async def decode(self, input: participating.AudioStimulus) -> str:
        self.inputs.append(input.content)
        return "hello"


class StaticTextGenerator(language.TextGenerator):
    inputs: list[text.generation.InputData]
    response: str

    def __init__(self, response: str = "Conversation fixture response.") -> None:
        self.inputs = []
        self.response = response

    @typing.override
    async def generate(self, input: text.generation.InputData) -> text.generation.OutputData:
        self.inputs.append(input)
        return text.generation.OutputData(content=self.response)


class RecordingConversationMemory(memory.ShortTermMemory):
    inputs: list[abilities.memory.InputData]

    def __init__(self) -> None:
        super().__init__()
        self.inputs = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.input_event.name:
            data = event.data
            assert isinstance(data, abilities.memory.InputData)
            self.inputs.append(data)
        return super().dispatch(ctx, event)


class PassthroughConversationEncoder(encoding.Encoder[str, str | bytes]):
    inputs: list[str]

    def __init__(self) -> None:
        self.inputs = []

    @typing.override
    async def encode(self, input: str) -> str | bytes:
        self.inputs.append(input)
        return input


class RecordingVoiceEncoder(conversation.VoiceEncoder):
    inputs: list[conversation.EncodeData]

    def __init__(self) -> None:
        self.inputs = []

    @typing.override
    async def encode(self, input: conversation.EncodeData) -> str | bytes:
        self.inputs.append(input)
        return f"Voice fixture response for {input.decoded_text}.".encode()


class RecordingIntuitionProcessor(processing.Processor):
    calls: list[processing.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        return ()


class RecordingReasoningProcessor(processing.Processor):
    calls: list[processing.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        return ()


class RecordingParticipating(participating.Participating):
    calls: list[participating.participating.InputData]

    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.input_event.name:
            input = event.data
            assert isinstance(input, participating.participating.InputData)
            self.calls.append(input)
        return super().dispatch(ctx, event)


def text_decoding_ability(
    decoder: decoding.Decoder[participating.ParticipationStimulus, str] | None = None,
) -> decoding.Decoding[participating.ParticipationStimulus, str]:
    return decoding.Decoding(decoder=RecordingTextStimulusDecoder() if decoder is None else decoder)


def text_encoding_ability(
    encoder: encoding.Encoder[str, str | bytes] | None = None,
) -> encoding.Encoding[str, str | bytes]:
    return encoding.Encoding(encoder=PassthroughConversationEncoder() if encoder is None else encoder)


def typing_ability(generator: language.TextGenerator | None = None) -> language.TextGeneration:
    return language.TextGeneration(generator=StaticTextGenerator() if generator is None else generator)


def memory_ability() -> memory.Memory:
    return RecordingConversationMemory()


def brain_for_test(
    intuition_processor: processing.Processor | None = None,
) -> cognition.Cognition:
    return cognition.Cognition(
        intuition=cognition.Intuition(
            processor=intuition_processor if intuition_processor is not None else RecordingIntuitionProcessor()
        ),
        reasoning=cognition.Reasoning(processor=RecordingReasoningProcessor()),
        reflection=cognition.Reflection(
            processor=RecordingReasoningProcessor(),
            memory=memory.Memory(),
        ),
    )


def text_message(conversation_ref: str = "support-call", content: str = "hello") -> conversation.TextMessage:
    return conversation.TextMessage(
        conversation_ref=conversation_ref,
        self_participant_ref="bot",
        participants=(
            participating.ParticipantSnapshot(
                ref="bot",
                kind="bot",
                state=participating.ParticipantStateSnapshot(
                    presence="present", attention="available", turn="listening"
                ),
            ),
            participating.ParticipantSnapshot(
                ref="caller",
                kind="human",
                state=participating.ParticipantStateSnapshot(presence="present", attention="available", turn="holding"),
            ),
        ),
        content=participating.TextStimulus(source_participant_ref="caller", content=content),
    )


def voice_message(conversation_ref: str = "support-call") -> conversation.VoiceMessage:
    return conversation.VoiceMessage(
        conversation_ref=conversation_ref,
        self_participant_ref="bot",
        participants=text_message(conversation_ref).participants,
        content=participating.AudioStimulus(source_participant_ref="caller", content=b"\x01\x00"),
    )


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout_seconds: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise RuntimeError("Timed out waiting for conversation condition.")


async def start_conversation(conversation: conversation.Conversation[typing.Any, typing.Any]) -> None:
    await start_ability_tree(None, conversation)


class RecordingConversation(conversation.TextConversation):
    outputs: list[conversation.Response]
    failures: list[conversation.FailureData]
    snapshots: list[conversation.Snapshot]

    def __init__(
        self,
        *,
        decoding: decoding.Decoding[participating.ParticipationStimulus, str] | None = None,
        participating: participating.Participating | None = None,
        typing: language.TextGeneration | None = None,
        encoding: encoding.Encoding[str, str | bytes] | None = None,
    ) -> None:
        super().__init__(
            decoding=text_decoding_ability() if decoding is None else decoding,
            participating=Participating() if participating is None else participating,
            typing=typing_ability() if typing is None else typing,
            encoding=text_encoding_ability() if encoding is None else encoding,
        )
        self.outputs = []
        self.failures = []
        self.snapshots = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, conversation.Response)
            self.outputs.append(output)
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, conversation.FailureData)
            self.failures.append(failure)
        if event.name == self.snapshot_output_event.name:
            snapshot = event.data
            assert isinstance(snapshot, conversation.Snapshot)
            self.snapshots.append(snapshot)
        return super().dispatch(ctx, event)


class RecordingVoiceConversation(conversation.VoiceConversation):
    outputs: list[conversation.Response]
    failures: list[conversation.FailureData]

    def __init__(
        self,
        *,
        decoder: conversation.VoiceDecoder | None = None,
        encoder: conversation.VoiceEncoder | None = None,
        participating: participating.Participating | None = None,
    ) -> None:
        super().__init__(
            decoder=RecordingAudioStimulusDecoder() if decoder is None else decoder,
            encoder=RecordingVoiceEncoder() if encoder is None else encoder,
            participating=Participating() if participating is None else participating,
        )
        self.outputs = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            output = event.data
            assert isinstance(output, conversation.Response)
            self.outputs.append(output)
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, conversation.FailureData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)


TEXT_CONVERSATION_BEHAVIOR = "/TextConversationLifecycle/attached/behavior"
VOICE_CONVERSATION_BEHAVIOR = "/VoiceConversationLifecycle/attached/behavior"
RECORDING_CONVERSATION_BEHAVIOR = "/RecordingConversationLifecycle/attached/behavior"


def test_conversation_exports_and_stage_contract() -> None:
    assert conversation.Stage.__args__ == ("decoding", "participating")  # type: ignore[attr-defined]
    assert set(conversation_impl.__all__) >= {
        "Conversation",
        "ParticipatedTurn",
        "Response",
        "define_conversation_model",
    }


def test_conversation_module_has_no_decide_memory_product_pipeline() -> None:
    source = inspect.getsource(conversation_impl)
    assert "deciding" not in source
    assert "active/memory" not in source
    assert "BotInputData" not in source
    assert "InputEventData" not in source
    assert "cognition.processing" not in source
    assert "ProcessingInput" not in source
    assert "_cognition" not in source
    assert "self._memory" not in source
    assert "active/participating" in source
    assert "active/decoding" in source


def test_text_and_voice_constructors_are_thin() -> None:
    text_value = conversation.TextConversation(
        decoding=text_decoding_ability(),
        participating=participating.Participating(),
        typing=typing_ability(),
        encoding=text_encoding_ability(),
    )
    assert text_value.child_for_kind("decoding") is not None
    assert text_value.child_for_kind("participating") is not None
    assert text_value.typing is not None
    assert text_value.encoding is not None

    voice = conversation.VoiceConversation(
        decoder=RecordingAudioStimulusDecoder(),
        encoder=RecordingVoiceEncoder(),
        participating=participating.Participating(),
    )
    assert voice.child_for_kind("decoding") is not None
    assert voice.encoding is not None


def test_conversation_requires_decoding_and_participating() -> None:
    with pytest.raises(ValueError, match="Conversation requires decoding."):
        _ = conversation.TextConversation(
            decoding=None,
            participating=participating.Participating(),
        )
    with pytest.raises(ValueError, match="Conversation requires participating."):
        _ = conversation.TextConversation(
            decoding=text_decoding_ability(),
            participating=typing.cast(participating.Participating, typing.cast(object, None)),
        )
    with pytest.raises(ValueError, match="VoiceConversation requires decoder."):
        _ = conversation.VoiceConversation(
            decoder=None,
            encoder=RecordingVoiceEncoder(),
            participating=participating.Participating(),
        )
    with pytest.raises(ValueError, match="VoiceConversation requires encoder."):
        _ = conversation.VoiceConversation(
            decoder=RecordingAudioStimulusDecoder(),
            encoder=None,
            participating=participating.Participating(),
        )


def test_thin_conversation_ends_after_participation() -> None:
    async def run() -> tuple[
        list[participating.InputData],
        list[conversation.Response],
        conversation.ParticipatedTurn,
        str,
    ]:
        participating_ability = RecordingParticipating()
        conversation_ability = RecordingConversation(participating=participating_ability)
        await start_conversation(conversation_ability)
        participated = await conversation.contribute_conversation_turn(conversation_ability, text_message())
        await wait_until(lambda: bool(conversation_ability.outputs) or bool(conversation_ability.failures))
        return (
            participating_ability.calls,
            conversation_ability.outputs,
            participated,
            conversation_ability.state(),
        )

    participating_inputs, outputs, participated, state = asyncio.run(run())

    assert len(participating_inputs) == 1
    assert len(outputs) == 1
    assert outputs[0].content is None
    assert outputs[0].conversation_ref == "support-call"
    assert participated.decoded_text == "hello"
    assert state.endswith("/silent")
    assert "/active/deciding" not in state
    assert "/active/memory" not in state
    assert "/active/encoding" not in state


def test_voice_conversation_contribution_is_thin() -> None:
    async def run() -> tuple[list[bytes], list[conversation.Response], conversation.ParticipatedTurn]:
        decoder = RecordingAudioStimulusDecoder()
        conversation_ability = RecordingVoiceConversation(decoder=decoder)
        await start_conversation(conversation_ability)
        participated = await conversation.contribute_conversation_turn(conversation_ability, voice_message())
        await wait_until(lambda: bool(conversation_ability.outputs) or bool(conversation_ability.failures))
        return decoder.inputs, conversation_ability.outputs, participated

    decoding_inputs, outputs, participated = asyncio.run(run())
    assert decoding_inputs == [b"\x01\x00"]
    assert len(outputs) == 1
    assert outputs[0].content is None
    assert participated.decoded_text == "hello"


def test_host_contribution_does_not_observe_machine_state(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> conversation.ParticipatedTurn:
        conversation_ability = RecordingConversation()
        await start_conversation(conversation_ability)

        def fail_state_read() -> str:
            raise AssertionError("host composition must not inspect machine state")

        monkeypatch.setattr(conversation_ability, "state", fail_state_read)
        return await conversation.contribute_conversation_turn(conversation_ability, text_message())

    participated = asyncio.run(run())

    assert participated.decoded_text == "hello"


def test_host_contribution_does_not_replace_machine_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> conversation.ParticipatedTurn:
        conversation_ability = RecordingConversation()
        await start_conversation(conversation_ability)
        original_setattr = RecordingConversation.__setattr__

        def reject_dispatch_assignment(instance: object, name: str, value: object) -> None:
            if name == "dispatch":
                raise AssertionError("host composition must not replace machine dispatch")
            original_setattr(instance, name, value)

        monkeypatch.setattr(RecordingConversation, "__setattr__", reject_dispatch_assignment)
        return await conversation.contribute_conversation_turn(conversation_ability, text_message())

    participated = asyncio.run(run())

    assert participated.decoded_text == "hello"


def test_concurrent_host_contributions_return_their_correlated_turns() -> None:
    async def run() -> tuple[conversation.ParticipatedTurn, conversation.ParticipatedTurn, bool]:
        conversation_ability = RecordingConversation()
        await start_conversation(conversation_ability)
        original_dispatch = conversation_ability.dispatch
        first, second = await asyncio.gather(
            conversation.contribute_conversation_turn(
                conversation_ability,
                text_message("first-call", "first"),
            ),
            conversation.contribute_conversation_turn(
                conversation_ability,
                text_message("second-call", "second"),
            ),
        )
        return first, second, conversation_ability.dispatch == original_dispatch

    first, second, dispatch_restored = asyncio.run(run())

    assert (first.input.conversation_ref, first.decoded_text) == ("first-call", "first")
    assert (second.input.conversation_ref, second.decoded_text) == ("second-call", "second")
    assert dispatch_restored


def test_host_contribution_reports_terminal_contract_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        conversation_ability = RecordingConversation()
        await start_conversation(conversation_ability)
        monkeypatch.setattr(
            conversation_impl.Conversation,
            "_build_contribution_response",
            staticmethod(lambda instance, participated: object()),
        )
        with pytest.raises(RuntimeError, match="output type does not match"):
            await conversation.contribute_conversation_turn(conversation_ability, text_message())

    asyncio.run(run())


def test_host_correlation_metadata_is_not_owner_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> list[dict[str, object]]:
        conversation_ability = RecordingConversation()
        await start_conversation(conversation_ability)
        owner = ability_terminal_owner(conversation_ability)
        assert owner is not None
        metadata: list[dict[str, object]] = []
        original_record = owner.record

        def record(event: hsm.Event[typing.Any]) -> None:
            metadata.append(dict(event.metadata))
            original_record(event)

        monkeypatch.setattr(owner, "record", record)
        _ = await conversation.contribute_conversation_turn(conversation_ability, text_message())
        return metadata

    metadata = asyncio.run(run())

    # Host waiters and stage context must not leak into owner-visible event.metadata.
    forbidden = {
        "bot.conversation.result",
        "bot.ability.terminal.result",
        "bot.conversation.message",
        "bot.conversation.decoded",
    }
    assert all(forbidden.isdisjoint(item) for item in metadata)


def test_start_ability_tree_waits_for_attachment_events(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> conversation.ParticipatedTurn:
        conversation_ability = RecordingConversation()

        def fail_state_read() -> str:
            raise AssertionError("test lifecycle setup must await typed attachment events")

        monkeypatch.setattr(conversation_ability, "state", fail_state_read)
        await start_conversation(conversation_ability)
        return await conversation.contribute_conversation_turn(conversation_ability, text_message())

    participated = asyncio.run(run())

    assert participated.decoded_text == "hello"


def test_start_ability_tree_surfaces_typed_attachment_failure() -> None:
    async def run() -> None:
        decoding_ability = text_decoding_ability()
        await start_ability_tree(None, decoding_ability)
        conversation_ability = conversation.TextConversation(
            decoding=decoding_ability,
            participating=participating.Participating(),
        )
        with pytest.raises(RuntimeError, match="cannot accept this attachment"):
            await start_ability_tree(None, conversation_ability)

    asyncio.run(run())


def test_conversation_detach_clears_prior_turn_state() -> None:
    async def run() -> conversation.Snapshot:
        conversation_ability = RecordingConversation()
        await start_conversation(conversation_ability)
        _ = await conversation.contribute_conversation_turn(conversation_ability, text_message())
        await wait_until(lambda: bool(conversation_ability.outputs))
        owner = ability_terminal_owner(conversation_ability)
        assert owner is not None
        _ = await conversation_ability.detach(
            conversation_ability.context(),
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)),
        )
        await wait_until(lambda: (conversation_ability.state() or "").endswith("/detached"))
        await start_conversation(conversation_ability)
        _ = await conversation_ability.dispatch(
            conversation_ability.context(),
            conversation.SnapshotRequestEvent.with_data(conversation.SnapshotRequest(request_ref="after-detach")),
        )
        await wait_until(lambda: bool(conversation_ability.snapshots))
        return conversation_ability.snapshots[-1]

    snapshot = asyncio.run(run())

    assert snapshot.conversation_ref is None
    assert snapshot.participants == ()


def test_host_text_respond_turn_yields_encoded_response() -> None:
    async def run() -> tuple[
        conversation.Response,
        list[processing.InputData],
        list[abilities.memory.InputData],
        list[str],
    ]:
        decoder = RecordingTextStimulusDecoder()
        encoder = PassthroughConversationEncoder()
        intuition = RecordingIntuitionProcessor()
        store = RecordingConversationMemory()
        typing = typing_ability()
        encoding_ability = text_encoding_ability(encoder)
        conversation_ability = conversation.TextConversation(
            decoding=text_decoding_ability(decoder),
            participating=participating.Participating(),
            typing=typing,
            encoding=encoding_ability,
        )
        cognition_ability = brain_for_test(intuition_processor=intuition)
        await start_conversation(conversation_ability)
        for stage in (cognition_ability, store, typing, encoding_ability):
            await start_ability_tree(None, stage)
        response = await conversation.run_host_text_respond_turn(
            conversation=conversation_ability,
            cognition=cognition_ability,
            memory=store,
            text_generation=typing,
            encoding=encoding_ability,
            message=text_message(),
        )
        return response, intuition.calls, store.inputs, encoder.inputs

    response, cognition_inputs, memory_inputs, encoding_inputs = asyncio.run(run())
    assert response.content == "Conversation fixture response."
    assert len(cognition_inputs) == 1
    assert isinstance(cognition_inputs[0].input, bot.InputEventData)
    assert len(memory_inputs) == 1
    assert encoding_inputs == ["Conversation fixture response."]


def test_host_voice_respond_turn_yields_bytes_response() -> None:
    async def run() -> tuple[conversation.Response, list[conversation.EncodeData]]:
        encoder = RecordingVoiceEncoder()
        conversation_ability = conversation.VoiceConversation(
            decoder=RecordingAudioStimulusDecoder(),
            encoder=encoder,
            participating=participating.Participating(),
        )
        cognition_ability = brain_for_test()
        store = memory_ability()
        await start_conversation(conversation_ability)
        assert conversation_ability.encoding is not None
        for stage in (cognition_ability, store, conversation_ability.encoding):
            await start_ability_tree(None, stage)
        response = await conversation.run_host_voice_respond_turn(
            conversation=conversation_ability,
            cognition=cognition_ability,
            memory=store,
            message=voice_message(),
        )
        return response, encoder.inputs

    response, encoding_inputs = asyncio.run(run())
    assert isinstance(response.content, bytes)
    assert response.content.startswith(b"Voice fixture response")
    assert len(encoding_inputs) == 1
    assert encoding_inputs[0].decoded_text == "hello"


def test_bot_conversation_decision_input_builds_host_policy_frame() -> None:
    stimulus = participating.EventStimulus(
        source_participant_ref="caller",
        event="conversation.decoding",
        payload={"source_kind": "text", "text": "hello"},
    )
    participated = conversation.ParticipatedTurn(
        input=text_message(),
        stimulus=stimulus,
        decoded_text="hello",
        participation=participating.participating.OutputData(
            participant_ref="bot",
            contribution=participating.ParticipantContribution(
                conversation_ref="support-call",
                participant_ref="caller",
                perception=participating.Perception(
                    source_participant_ref="caller",
                    modality="event",
                    structured={"event": "conversation.decoding", "payload": stimulus.payload},
                ),
            ),
            reason="DecodedData stimulus was accepted.",
        ),
    )
    built = conversation.agent_conversation_decision_input(
        participated,
        target_device="caller",
    )
    assert isinstance(built.input, bot.InputEventData)
    assert built.input.source_event == "bot.ability.conversation.participating"
    assert built.actors == {}


def test_conversation_snapshot_request() -> None:
    async def run() -> list[conversation.Snapshot]:
        conversation_ability = RecordingConversation()
        await start_conversation(conversation_ability)
        _ = await conversation_ability.apply(text_message())
        await wait_until(lambda: bool(conversation_ability.outputs))
        _ = await conversation_ability.dispatch(
            conversation_ability.context(),
            conversation.SnapshotRequestEvent.with_data(conversation.SnapshotRequest(request_ref="panel")),
        )
        await wait_until(lambda: bool(conversation_ability.snapshots))
        return conversation_ability.snapshots

    snapshots = asyncio.run(run())
    assert snapshots[-1].request_ref == "panel"
    assert snapshots[-1].conversation_ref == "support-call"


def test_text_conversation_model_topology_is_thin() -> None:
    model = conversation.TextConversation.model
    assert model is not None
    view = model_view(model)
    assert view.qualified_name == "/TextConversationLifecycle"
    joined = "\n".join(f"{key}" for key in view.transition_map)
    assert "deciding" not in joined
    assert "/memory" not in joined
    assert "decoding" in joined or "participating" in joined


def test_text_message_requires_self_participant_state_and_unique_participants() -> None:
    with pytest.raises(ValueError, match="self_participant_ref must match one participant snapshot."):
        _ = conversation.AnyMessage(
            conversation_ref="support-call",
            self_participant_ref="missing",
            participants=(
                participating.ParticipantSnapshot(
                    ref="bot",
                    kind="bot",
                    state=participating.ParticipantStateSnapshot(
                        presence="present", attention="available", turn="listening"
                    ),
                ),
            ),
            content=participating.TextStimulus(source_participant_ref="bot", content="hello"),
        )

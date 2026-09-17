"""Speaking output ability: text → encoder → ``environment.sound`` elevation."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import typing

import hsm
import mosfet
import pytest
from mosfet import lifecycle
from mosfet.abilities import processing

from mosfet.abilities import ability, encoding
from mosfet.abilities import speaking
from mosfet.abilities.communication import conversation
from mosfet.devices import audio
from mosfet.abilities.speaking import EfferenceData, EfferenceEvent
from mosfet.environment import SoundData, SoundEvent, Environment
from tests.hsm_instance_state import device_bots, start_ability_tree
from tests.bot.abilities.support import require_model


class RecordingEncoder(encoding.Encoder[bytes, bytes]):
    def __init__(self, audio: bytes = b"pcm-audio") -> None:
        self.calls: list[bytes] = []
        self._audio = audio

    @typing.override
    async def encode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return self._audio


class RecordingConversation(conversation.Conversation):
    def __init__(self) -> None:
        super().__init__()
        self.outputs: list[conversation.Messages] = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == ability.TerminalOutputEvent.name and isinstance(event.data, hsm.Event):
            terminal = event.data
            if terminal.name == conversation.OutputEvent.name and isinstance(terminal.data, conversation.Messages):
                self.outputs.append(terminal.data)
        return super().dispatch(ctx, event)


class FailingEncoder(encoding.Encoder[bytes, bytes]):
    @typing.override
    async def encode(self, input: bytes) -> bytes:
        del input
        raise RuntimeError("tts failed")


class NeverReturningEncoder(encoding.Encoder[bytes, bytes]):
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    @typing.override
    async def encode(self, input: bytes) -> bytes:
        del input
        try:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        finally:
            self.cancelled.set()


async def _wait_until(condition: typing.Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not met")


def test_speaking_rejects_non_ability_conversation_targets() -> None:
    with pytest.raises(TypeError):
        speaking.Speaking(encoder=RecordingEncoder(), conversation=typing.cast(typing.Any, hsm.Instance()))

    speaker = speaking.Speaking(encoder=RecordingEncoder())
    with pytest.raises(TypeError):
        speaker.link_conversation(typing.cast(typing.Any, hsm.Instance()))


def test_speaking_records_one_trusted_outbound_message() -> None:
    async def run() -> tuple[list[conversation.Messages], str, str]:
        environment = Environment()
        target = RecordingConversation()
        speaker = speaking.Speaking(encoder=RecordingEncoder(), conversation=target)
        await start_ability_tree(environment, target)
        await start_ability_tree(environment, speaker)

        _ = await speaker.apply(speaking.InputData(text="Hello there."), ctx=environment)
        await _wait_until(lambda: bool(target.outputs))
        return target.outputs, hsm.id(speaker), hsm.id(target)

    histories, speaker_id, conversation_id = asyncio.run(run())

    assert histories
    assert all(len(history.messages) == 1 and history.messages[0].direction == "outbound" for history in histories)
    provenance_ids = {history.messages[0].provenance.id for history in histories}
    assert len(provenance_ids) == 1
    message = histories[0].messages[0]
    assert message.content == "Hello there."
    assert message.provenance.event == conversation.AppendEvent.name
    assert message.provenance.id in provenance_ids
    assert message.provenance.source == speaker_id
    assert message.provenance.target == conversation_id


def test_speaking_input_is_not_a_cognition_call_event() -> None:
    assert speaking.InputEvent.name == "bot.ability.speaking.input"
    assert speaking.InputEvent.kind == hsm.EventKind


def _examples_in(schema: object, path: str = "") -> list[str]:
    """Every place the generated schema carries an example, named by where it sits."""

    found: list[str] = []
    if isinstance(schema, dict):
        for key, value in typing.cast(dict[str, object], schema).items():
            here = f"{path}.{key}" if path else key
            if key in ("examples", "example", "default"):
                found.append(f"{here} = {value!r}")
            else:
                found.extend(_examples_in(value, here))
    elif isinstance(schema, list):
        for index, item in enumerate(typing.cast(list[object], schema)):
            found.extend(_examples_in(item, f"{path}[{index}]"))
    return found


def test_speak_input_schema_offers_no_example_utterance() -> None:
    """The schema a model reads in order to speak must not hand it something to say.

    Speaking's payload is free text, so an example there is never documentation of a format —
    it is a complete, valid answer sitting in the one field the model fills in, and a model
    that is unsure what to say emits it verbatim and the bot says it out loud. Asserted on the
    projected JSON schema rather than the Python model because that projection is what the
    model actually reads.
    """

    schema = processing.model_facing_event_json_schema(speaking.InputEvent)

    assert _examples_in(schema) == []


class SoundListener(hsm.Instance):
    """Environment participant that records the ``environment.sound`` a speaker transduces."""

    def __init__(self, sounds: list[SoundData]) -> None:
        super().__init__()
        self._sounds = sounds

    @staticmethod
    def _record(ctx: hsm.Context, instance: "SoundListener", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        if isinstance(data, SoundData):
            instance._sounds.append(data)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "SoundListener",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(SoundEvent), hsm.effect(_record))),
    )


def test_speaking_encodes_text_and_elevates_to_environment_sound() -> None:
    """Speaking elevates playout as ``environment.sound`` (ability emits; no Speaker required)."""

    async def run() -> tuple[list[bytes], list[SoundData], speaking.OutputData | None]:
        encoder = RecordingEncoder(audio=b"\x00\x01")
        speaking_ability = speaking.Speaking(
            encoder=encoder,
            sample_rate_hz=24_000,
            channels=1,
            media_type="audio/pcm",
        )
        environment = Environment()
        sounds: list[SoundData] = []
        listener = SoundListener(sounds)

        await start_ability_tree(environment, speaking_ability)
        _ = await mosfet.started(environment, listener, listener.model, hsm.Config(id="environment-ear"))
        environment.join(listener)

        outputs: list[speaking.OutputData] = []
        original = speaking_ability.dispatch

        def capture_terminal(ctx: hsm.Context, event: hsm.Event) -> typing.Awaitable[None]:
            from mosfet.abilities import ability

            if event.name == ability.TerminalOutputEvent.name and isinstance(event.data, hsm.Event):
                data = event.data.data
                if isinstance(data, speaking.OutputData):
                    outputs.append(data)
            return original(ctx, event)

        speaking_ability.dispatch = capture_terminal  # type: ignore[method-assign]
        try:
            _ = await speaking_ability.apply(speaking.InputData(text="  Hello there.  "), ctx=environment)
            await _wait_until(lambda: bool(outputs))
        finally:
            speaking_ability.dispatch = original  # type: ignore[method-assign]

        return encoder.calls, sounds, outputs[0] if outputs else None

    calls, sounds, product = asyncio.run(run())
    assert calls == [b"Hello there."]
    assert product is not None
    assert product.text == "Hello there."
    assert product.sample_rate_hz == 24_000
    assert product.media_type == "audio/pcm"
    assert len(sounds) == 1
    assert sounds[0].audio == b"\x00\x01"


def test_speaking_failure_surfaces_on_failed_event() -> None:
    async def run() -> str:
        speaking_ability = speaking.Speaking(encoder=FailingEncoder(), speaker=audio.Speaker())
        environment = Environment()
        await start_ability_tree(environment, speaking_ability)
        failures: list[str] = []
        original = speaking_ability.dispatch

        def capture(ctx: hsm.Context, event: hsm.Event) -> typing.Awaitable[None]:
            from mosfet.abilities import ability

            if event.name == ability.TerminalErrorEvent.name and isinstance(event.data, hsm.Event):
                data = event.data.data
                if isinstance(data, ability.FailureData):
                    failures.append(data.message)
            return original(ctx, event)

        speaking_ability.dispatch = capture  # type: ignore[method-assign]
        try:
            _ = await speaking_ability.apply(speaking.InputData(text="hi"), ctx=environment)
            await _wait_until(lambda: bool(failures))
        finally:
            speaking_ability.dispatch = original  # type: ignore[method-assign]
        return failures[0]

    message = asyncio.run(run())
    assert "tts failed" in message


def test_speaking_returns_correlated_terminal_directly_to_request_source() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        environment = Environment()
        terminals: list[hsm.Event[typing.Any]] = []

        class Requester(hsm.Instance):
            @staticmethod
            def record(ctx: hsm.Context, instance: "Requester", event: hsm.Event[typing.Any]) -> None:
                del ctx, instance
                terminals.append(event)

            model = mosfet.define(
                "SpeakingRequester",
                hsm.initial(hsm.target("/SpeakingRequester/waiting")),
                hsm.state(
                    "waiting",
                    hsm.transition(hsm.on(speaking.OutputEvent), hsm.effect(record)),
                ),
            )

        requester = Requester()
        speaker = speaking.Speaking(encoder=RecordingEncoder())
        _ = await mosfet.started(environment, requester, requester.model)
        await start_ability_tree(environment, speaker)

        await hsm.dispatch(
            environment,
            speaker,
            dataclasses.replace(
                speaking.InputEvent.with_data_and_id(speaking.InputData(text="Hello."), "speak-1"),
                source=hsm.id(requester),
                target=hsm.id(speaker),
            ),
        )
        await _wait_until(lambda: bool(terminals))
        return terminals

    terminals = asyncio.run(run())
    assert len(terminals) == 1
    assert terminals[0].id == "speak-1"


def test_speaking_encoding_timeout_cancels_non_returning_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[str, bool]:
        from mosfet.abilities.speaking import speaking as speaking_source

        monkeypatch.setattr(speaking_source, "_ENCODING_TIMEOUT", datetime.timedelta(milliseconds=10))
        encoder = NeverReturningEncoder()
        speaker = speaking.Speaking(encoder=encoder)
        environment = Environment()
        await start_ability_tree(environment, speaker)
        await hsm.dispatch(
            environment,
            speaker,
            speaking.InputEvent.with_data_and_id(speaking.InputData(text="Never returns."), "timeout-1"),
        )
        await _wait_until(lambda: (speaker.state() or "").endswith("/idle"))
        await encoder.cancelled.wait()
        return speaker.state() or "", encoder.cancelled.is_set()

    state, cancelled = asyncio.run(run())
    assert state.endswith("/idle")
    assert cancelled


def test_speaking_is_not_cognition_callable() -> None:
    """Speaking is an effector reached through behavior/topology wiring."""

    speaking_ability = speaking.Speaking(encoder=RecordingEncoder())
    assert speaking.InputEvent.kind == hsm.EventKind
    assert speaking_ability.input_event is speaking.InputEvent
    assert speaking_ability.input_event.kind == hsm.EventKind


def test_cognition_does_not_directly_select_speaking_output() -> None:
    """Body output wiring is absent from cognition's actor/tool inventory."""

    import mosfet
    from mosfet.bot import Bot
    from mosfet.device import Device
    from mosfet.abilities import processing
    from mosfet.environment import Environment
    from tests.bot.test_bot import as_cognition

    class SpeakProcessor(processing.Processor):
        inputs: list[processing.InputData]

        def __init__(self) -> None:
            self.inputs = []

        @typing.override
        async def process(self, input: processing.InputData) -> processing.Events:
            self.inputs.append(input)
            names = {event.name for event in input.schemas}
            assert "bot.ability.speaking.input" not in names
            assert "speaking" not in input.actors
            return ()

    async def run() -> tuple[list[bytes], list[processing.InputData]]:
        encoder = RecordingEncoder(audio=b"\x11\x22")
        speaking_ability = speaking.Speaking(encoder=encoder, speaker=None)
        processor = SpeakProcessor()

        class Probe(Bot):
            def __init__(self) -> None:
                super().__init__(
                    devices={"phone": Device()},
                    cognition=as_cognition(processor),
                    output=(speaking_ability,),
                )

        probe = Probe()
        environment = Environment()
        await probe.attach(environment)
        await _wait_until(lambda: (probe.state() or "").endswith("/unfocused"))
        assert (speaking_ability.state() or "").endswith("/idle")

        await probe.dispatch(
            environment,
            mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", priority=0)),
        )
        return encoder.calls, processor.inputs

    calls, inputs = asyncio.run(run())
    assert calls == []
    assert len(inputs) == 1
    assert "bot.ability.speaking.input" not in {event.name for event in inputs[0].schemas}
    assert "speaking" not in inputs[0].actors


class ListeningPeer(hsm.Instance):
    """Stands where Listening stands: receives the Speaking→Listening motor-command copy.

    The efference copy is delivered to registered peers, not the body. This records arrival order
    for the nerve under test.
    """

    def __init__(self, seen: list[str]) -> None:
        super().__init__()
        self._seen = seen

    @staticmethod
    def _record(ctx: hsm.Context, instance: "ListeningPeer", event: hsm.Event[typing.Any]) -> None:
        del ctx
        if isinstance(event.data, EfferenceData):
            instance._seen.append("efference")

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "ListeningPeer",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(EfferenceEvent), hsm.effect(_record))),
    )


class OrderingSoundListener(hsm.Instance):
    """Environment participant that notes when the sound actually shows up in the room."""

    def __init__(self, seen: list[str]) -> None:
        super().__init__()
        self._seen = seen

    @staticmethod
    def _record(ctx: hsm.Context, instance: "OrderingSoundListener", event: hsm.Event[typing.Any]) -> None:
        del ctx
        if isinstance(event.data, SoundData):
            instance._seen.append("sound")

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "OrderingSoundListener",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(SoundEvent), hsm.effect(_record))),
    )


class AbilityOwner(hsm.Instance):
    """Attachment owner so Speaking can leave detached lifecycle and run behavior."""

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "AbilityOwner",
        hsm.initial(hsm.target("owning")),
        hsm.state("owning"),
    )


async def _speak_with_listening_peer(
    *,
    media_type: str = "audio/pcm",
    audio_bytes: bytes = b"\x00" * 32_000,
    speaker: audio.Speaker | None = None,
) -> tuple[list[str], list[EfferenceData]]:
    """Say one thing with a real mouth; return order and copies seen by the linked Listening peer."""

    from mosfet.protocols import attachment

    mouth = audio.Speaker() if speaker is None else speaker
    environment = Environment()
    order: list[str] = []
    copies: list[EfferenceData] = []

    peer = ListeningPeer(order)
    owner = AbilityOwner()
    _ = await mosfet.started(environment, peer, peer.model)
    _ = await mosfet.started(environment, owner, owner.model)
    speaking_ability = speaking.Speaking(
        encoder=RecordingEncoder(audio=audio_bytes),
        speaker=mouth,
        listening=peer,
        sample_rate_hz=16_000,
        channels=1,
        media_type=media_type,
    )
    _ = await mosfet.started(environment, mouth, typing.cast(hsm.Model, mouth.model))
    listener = OrderingSoundListener(order)
    _ = await mosfet.started(environment, listener, listener.model, hsm.Config(id="environment-ear"))
    environment.join(listener)

    _ = await mosfet.started(environment, speaking_ability, typing.cast(hsm.Model, speaking_ability.model))
    # Attach is ability lifecycle only; the motor-command copy goes to ``listening=peer``, not owner.
    _ = await speaking_ability.attach(
        environment,
        attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
    )

    original = peer.dispatch

    def capture(ctx: hsm.Context, event: hsm.Event[typing.Any]) -> typing.Awaitable[None]:
        if isinstance(event.data, EfferenceData):
            copies.append(event.data)
        return original(ctx, event)

    peer.dispatch = capture  # type: ignore[method-assign]
    try:
        await speaking_ability.dispatch(
            environment, speaking.InputEvent.with_data(speaking.InputData(text="Hello there."))
        )
        await _wait_until(lambda: "sound" in order)
    finally:
        peer.dispatch = original  # type: ignore[method-assign]
    return order, copies


def test_the_copy_of_a_command_leaves_before_the_sound_does() -> None:
    """An efference copy that arrived after the sound would be a report, not a prediction.

    Issued on entry to playout: after the audio exists, strictly before it reaches the mouth.
    Not at the request — synthesis takes seconds during which nothing is being produced — and
    not at completion, which means the act was committed, not that the sound stopped.
    """

    order, copies = asyncio.run(_speak_with_listening_peer())

    assert order[:2] == ["efference", "sound"]
    assert len(copies) == 1


def test_the_copy_says_which_mouth_how_long_and_in_what_form_and_never_the_words() -> None:
    """Words on the copy would make this self-recognition, which is the thing being avoided.

    Duration is measured, not guessed: 32000 bytes of 16-bit mono at 16 kHz is one second of
    sound, and that is how long the consequences of this command are expected to last.
    """

    async def run() -> tuple[audio.Speaker, list[EfferenceData]]:
        mouth = audio.Speaker()
        _, copies = await _speak_with_listening_peer(speaker=mouth)
        return mouth, copies

    mouth, copies = asyncio.run(run())

    assert len(copies) == 1
    copy = copies[0]
    assert copy.mouth == hsm.id(mouth)
    assert copy.duration == 1.0
    assert copy.media_type == "audio/pcm"
    assert copy.sample_rate_hz == 16_000
    assert copy.channels == 1
    assert "Hello" not in copy.model_dump_json()


def test_no_copy_is_issued_for_a_form_whose_byte_count_is_not_a_duration() -> None:
    """A compressed buffer says nothing about time, and a made-up window is worse than none."""

    order, copies = asyncio.run(_speak_with_listening_peer(media_type="audio/opus"))

    assert copies == []
    assert "sound" in order


def test_the_efference_event_is_never_offerable_to_a_model() -> None:
    """A nerve, not a tool. Nothing decides to send one, so nothing may select one."""

    assert EfferenceEvent.kind != processing.EventKind
    assert EfferenceEvent.name == "bot.ability.speaking.efference"
    # Speaking is also an internal effector boundary; behaviors/topology own the route.
    assert speaking.InputEvent.kind == hsm.EventKind


def test_speaking_powers_an_unstarted_mouth_it_was_given() -> None:
    """The mouth is part of the bot: the ability brings it up, the way firmware brings up an earpiece.

    An injected speaker that nothing else started is started by this ability when the ability
    starts — parented under the ability's own lifetime, so it outlives any single utterance —
    and the very first word already plays out.
    """

    async def run() -> tuple[list[SoundData], bool]:
        speaker = audio.Speaker()
        speaking_ability = speaking.Speaking(encoder=RecordingEncoder(audio=b"\x00\x01"), speaker=speaker)
        environment = Environment()
        sounds: list[SoundData] = []
        listener = SoundListener(sounds)

        await start_ability_tree(environment, speaking_ability)
        _ = await mosfet.started(environment, listener, listener.model, hsm.Config(id="environment-ear"))
        environment.join(listener)

        _ = await speaking_ability.apply(speaking.InputData(text="Hello."), ctx=environment)
        await _wait_until(lambda: bool(sounds))
        return sounds, lifecycle.is_started(speaker)

    sounds, started = asyncio.run(run())

    assert started
    assert [sound.audio for sound in sounds] == [b"\x00\x01"]


def test_speaking_stops_the_mouth_it_started() -> None:
    """Stop powers down what start powered up — the same lifecycle, reversed."""

    async def run() -> None:
        speaker = audio.Speaker()
        speaking_ability = speaking.Speaking(encoder=RecordingEncoder(), speaker=speaker)
        environment = Environment()

        await start_ability_tree(environment, speaking_ability)
        assert lifecycle.is_started(speaker)

        await speaking_ability.stop(environment)
        assert not lifecycle.is_started(speaker)

    asyncio.run(run())


def test_speaking_never_stops_a_mouth_it_did_not_start() -> None:
    """An externally started speaker keeps running after the ability stops: shared, not owned."""

    async def run() -> None:
        speaker = audio.Speaker()
        speaking_ability = speaking.Speaking(encoder=RecordingEncoder(), speaker=speaker)
        environment = Environment()

        # Started by someone else before the ability ever runs: shared, not owned.
        _ = await mosfet.started(environment, speaker, require_model(speaker.model))
        await start_ability_tree(environment, speaking_ability)

        await speaking_ability.stop(environment)
        assert lifecycle.is_started(speaker)

    asyncio.run(run())


def test_speaking_playout_does_not_attach_optional_speaker() -> None:
    """Temporary: playout elevates ``environment.sound`` without Speaker attachment.

    Mouth/Speaker transduction returns later; an injected speaker must not accumulate
    controller attachments from utterances in the meantime.
    """

    async def run() -> int:
        speaker = audio.Speaker()
        speaking_ability = speaking.Speaking(encoder=RecordingEncoder(audio=b"\x00\x01"), speaker=speaker)
        environment = Environment()
        _ = await mosfet.started(environment, speaker, require_model(speaker.model))
        await start_ability_tree(environment, speaking_ability)

        for _ in range(3):
            _ = await speaking_ability.apply(speaking.InputData(text="Hello."), ctx=environment)
            await _wait_until(lambda: speaking_ability.state().endswith("/idle"))
        return len(device_bots(speaker))

    assert asyncio.run(run()) == 0

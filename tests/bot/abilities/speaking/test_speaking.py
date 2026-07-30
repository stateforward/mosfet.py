"""Speaking output ability: text → encoder → ``environment.sound`` elevation."""

from __future__ import annotations

import asyncio
import typing

import hsm
from bot import lifecycle
from bot.abilities import processing

from bot.abilities import encoding
from bot.abilities import speaking
from bot.devices import audio
from bot.environment import SoundData, SoundEvent, Environment
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


class FailingEncoder(encoding.Encoder[bytes, bytes]):
    @typing.override
    async def encode(self, input: bytes) -> bytes:
        del input
        raise RuntimeError("tts failed")


async def _wait_until(condition: typing.Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met")


def test_speaking_input_is_call_event_for_cognition_selection() -> None:
    assert speaking.InputEvent.name == "bot.ability.speaking.input"
    assert speaking.InputEvent.kind == processing.EventKind


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

    model: typing.ClassVar[hsm.Model] = hsm.define(
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
        _ = await hsm.started(environment, listener, listener.model, hsm.Config(id="environment-ear"))
        environment.join(listener)

        outputs: list[speaking.OutputData] = []
        original = speaking_ability.dispatch

        def capture_terminal(ctx: hsm.Context, event: hsm.Event) -> typing.Awaitable[None]:
            from bot.abilities import ability

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
            from bot.abilities import ability

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


def test_speaking_is_cognition_callable_output_ability() -> None:
    """Speaking is a CallEvent ability suitable for bot.output / cognition tool selection.

    Actor-map assembly is bot-private; public contract is event kind + wiring (see e2e test).
    """

    speaking_ability = speaking.Speaking(encoder=RecordingEncoder())
    assert speaking.InputEvent.kind == processing.EventKind
    assert speaking_ability.input_event is speaking.InputEvent
    assert speaking_ability.input_event.kind == processing.EventKind


def test_cognition_to_speaking_output_end_to_end() -> None:
    """Bot input event → cognition/intuition → select speaking.input → encoder runs."""

    import bot
    from bot.bot import Bot
    from bot.device import Device
    from bot.abilities import processing
    from bot.environment import Environment
    from tests.bot.test_bot import as_cognition

    class SpeakProcessor(processing.Processor):
        inputs: list[processing.InputData]

        def __init__(self) -> None:
            self.inputs = []

        @typing.override
        async def process(self, input: processing.InputData) -> processing.Events:
            self.inputs.append(input)
            names = {event.name for event in input.schemas}
            assert "bot.ability.speaking.input" in names
            assert "speaking" in input.actors
            return (
                processing.SelectedEvent(
                    event="bot.ability.speaking.input",
                    data={"text": "hi from cognition"},
                    target="speaking",
                    reason="test speak selection",
                ),
            )

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
            bot.InputEvent.with_data(bot.InputEventData(target_device="phone", priority=0)),
        )
        await _wait_until(lambda: bool(encoder.calls))
        return encoder.calls, processor.inputs

    calls, inputs = asyncio.run(run())
    assert calls == [b"hi from cognition"]
    assert len(inputs) == 1
    assert "bot.ability.speaking.input" in {event.name for event in inputs[0].schemas}


class OwnerEar(hsm.Instance):
    """Stands where the body stands: attaches to Speaking and records what it is told.

    The efference copy goes to whatever owns this ability, which in a real bot is the body. Here
    it is the only thing this test needs, so the recorded order is exactly the order the body
    would see.
    """

    def __init__(self, seen: list[str]) -> None:
        super().__init__()
        self._seen = seen

    @staticmethod
    def _record(ctx: hsm.Context, instance: "OwnerEar", event: hsm.Event[typing.Any]) -> None:
        del ctx
        if isinstance(event.data, speaking.EfferenceData):
            instance._seen.append("efference")

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "OwnerEar",
        hsm.initial(hsm.target("owning")),
        hsm.state("owning", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
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

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "OrderingSoundListener",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(SoundEvent), hsm.effect(_record))),
    )


async def _speak_with_owner(
    *,
    media_type: str = "audio/pcm",
    audio_bytes: bytes = b"\x00" * 32_000,
    speaker: audio.Speaker | None = None,
) -> tuple[list[str], list[speaking.EfferenceData]]:
    """Say one thing with a real mouth in a real environment; return what the owner saw."""

    from bot.protocols import attachment

    mouth = audio.Speaker() if speaker is None else speaker
    speaking_ability = speaking.Speaking(
        encoder=RecordingEncoder(audio=audio_bytes),
        speaker=mouth,
        sample_rate_hz=16_000,
        channels=1,
        media_type=media_type,
    )
    environment = Environment()
    order: list[str] = []
    copies: list[speaking.EfferenceData] = []

    owner = OwnerEar(order)
    _ = await hsm.started(environment, owner, owner.model)
    _ = await hsm.started(environment, mouth, typing.cast(hsm.Model, mouth.model))
    listener = OrderingSoundListener(order)
    _ = await hsm.started(environment, listener, listener.model, hsm.Config(id="environment-ear"))
    environment.join(listener)

    _ = await hsm.started(environment, speaking_ability, typing.cast(hsm.Model, speaking_ability.model))
    _ = await speaking_ability.attach(
        environment,
        attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
    )

    original = owner.dispatch

    def capture(ctx: hsm.Context, event: hsm.Event[typing.Any]) -> typing.Awaitable[None]:
        if isinstance(event.data, speaking.EfferenceData):
            copies.append(event.data)
        return original(ctx, event)

    owner.dispatch = capture  # type: ignore[method-assign]
    try:
        await speaking_ability.dispatch(
            environment, speaking.InputEvent.with_data(speaking.InputData(text="Hello there."))
        )
        await _wait_until(lambda: "sound" in order)
        await asyncio.sleep(0.02)
    finally:
        owner.dispatch = original  # type: ignore[method-assign]
    return order, copies


def test_the_copy_of_a_command_leaves_before_the_sound_does() -> None:
    """An efference copy that arrived after the sound would be a report, not a prediction.

    Issued on entry to playout: after the audio exists, strictly before it reaches the mouth.
    Not at the request — synthesis takes seconds during which nothing is being produced — and
    not at completion, which means the act was committed, not that the sound stopped.
    """

    order, copies = asyncio.run(_speak_with_owner())

    assert order[:2] == ["efference", "sound"]
    assert len(copies) == 1


def test_the_copy_says_which_mouth_how_long_and_in_what_form_and_never_the_words() -> None:
    """Words on the copy would make this self-recognition, which is the thing being avoided.

    Duration is measured, not guessed: 32000 bytes of 16-bit mono at 16 kHz is one second of
    sound, and that is how long the consequences of this command are expected to last.
    """

    async def run() -> tuple[audio.Speaker, list[speaking.EfferenceData]]:
        mouth = audio.Speaker()
        _, copies = await _speak_with_owner(speaker=mouth)
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

    order, copies = asyncio.run(_speak_with_owner(media_type="audio/opus"))

    assert copies == []
    assert "sound" in order


def test_the_efference_event_is_never_offerable_to_a_model() -> None:
    """A nerve, not a tool. Nothing decides to send one, so nothing may select one."""

    assert speaking.EfferenceEvent.kind != processing.EventKind
    # Contrast: the ability's one front door is offerable, which is what makes the difference
    # between them a decision rather than an oversight.
    assert speaking.InputEvent.kind == processing.EventKind


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
        _ = await hsm.started(environment, listener, listener.model, hsm.Config(id="environment-ear"))
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
        _ = await hsm.started(environment, speaker, require_model(speaker.model))
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
        _ = await hsm.started(environment, speaker, require_model(speaker.model))
        await start_ability_tree(environment, speaking_ability)

        for _ in range(3):
            _ = await speaking_ability.apply(speaking.InputData(text="Hello."), ctx=environment)
            await _wait_until(lambda: speaking_ability.state().endswith("/idle"))
        return len(device_bots(speaker))

    assert asyncio.run(run()) == 0

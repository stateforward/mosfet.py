import json
import os
import pathlib
import re
import shutil
import subprocess
import typing

import pytest


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _example_source() -> str:
    return (_repo_root() / "examples" / "phone_bot" / "src" / "phone_bot_example" / "__init__.py").read_text(
        encoding="utf-8"
    )


def _default_cognition_instructions() -> dict[str, str]:
    """Phone bot uses ability ClassVar defaults (no example-local instruction strings)."""

    from bot.abilities.cognition import intuition, reasoning

    return {
        "intuition_instructions": intuition.DEFAULT_INSTRUCTIONS,
        "reasoning_instructions": reasoning.DEFAULT_INSTRUCTIONS,
    }


def _run_phone_bot_example(*args: str, env: dict[str, str | None] | None = None) -> dict[str, object]:
    run_env = os.environ.copy()
    if env is not None:
        for key, value in env.items():
            if value is None:
                _ = run_env.pop(key, None)
            else:
                run_env[key] = value
    result = subprocess.run(
        ["uv", "run", "--project", "examples/phone_bot", "phone-bot-example", "--json", *args],
        cwd=_repo_root(),
        env=run_env,
        check=True,
        capture_output=True,
        text=True,
    )
    json_start = result.stdout.find("{")
    assert json_start >= 0, result.stdout
    return typing.cast(dict[str, object], json.loads(result.stdout[json_start:]))


def _run_phone_bot_python(code: str) -> str:
    result = subprocess.run(
        ["uv", "run", "--project", "examples/phone_bot", "python", "-c", code],
        cwd=_repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _full_cognition_config_kwargs() -> str:
    return (
        "cognition=phone_bot_example.CognitionConfig("
        "api_key='openai-ok',"
        "intuition_api_key='mercury-ok',"
        "reflection_api_key='openai-ok',"
        ")"
    )


def test_phone_bot_example_is_provider_package_example() -> None:
    pyproject = (_repo_root() / "examples" / "phone_bot" / "pyproject.toml").read_text(encoding="utf-8")
    source = _example_source()

    assert "bot-provider-livekit" in pyproject
    assert "bot-provider-gemini" in pyproject
    assert "bot-provider-sqlite-memory" in pyproject
    assert "bot-provider-openai-compat" in pyproject
    assert "bot-provider-mlx-audio" in pyproject
    assert "bot-provider-pyannote" in pyproject
    assert "bot-provider-elevenlabs" not in pyproject
    assert "from bot.providers.gemini import SpeechDecoder as GeminiSpeechDecoder" in source
    assert "from bot.providers.gemini import SpeechEncoder as GeminiSpeechEncoder" in source
    assert "from bot.providers.mlx_audio import VoiceDetector as SileroVoiceDetector" in source
    assert "from bot.providers.pyannote import Classifier as PyannoteVoiceClassifier" in source
    assert "from bot.providers.openai_compat import Processor as OpenAIProcessor" in source
    assert "AlwaysVoiceDetector" not in source
    assert "PeakEnergyVoiceDetector" not in source
    # The streaming detector consumes raw PCM directly, so no PCM->WAV wrapper stands in front
    # of it any more; it is told the room's audio shape instead.
    assert "PcmAwareVoiceDetector" not in source
    assert "_silero_voice_detector" in source
    assert "sample_rate_hz=config.input_sample_rate_hz" in source
    assert "_pyannote_voice_classifier" in source
    assert "voice_classifier=classifier" in source
    assert "KindSoundClassifier" in source
    assert "from bot.devices import phone as phone_device" in source
    assert "from bot.providers.livekit import PhoneService" in source
    assert "from bot.providers.livekit.audio import PcmWavDecoder" in source
    assert "class LiveKitVoiceDecoder" not in source
    assert "class GeminiVoiceDecoder" in source
    assert "abilities.VoiceDecoder" in source
    assert "RoomAudioEndpoint" not in source
    assert "RoomAudioConnectData" not in source
    assert "create_audio_bridge" not in source
    assert "from bot.providers.sqlite_memory import" not in source
    assert 'DEFAULT_MERCURY_MODEL = "mercury-2"' in source
    assert 'DEFAULT_MERCURY_BASE_URL = "https://api.inceptionlabs.ai/v1"' in source
    assert 'DEFAULT_OPENAI_REASONING_MODEL = "gpt-5.6-terra"' in source
    assert 'DEFAULT_GEMINI_STT_MODEL = "gemini-3.5-flash"' in source
    assert 'DEFAULT_GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"' in source
    assert 'DEFAULT_GEMINI_TTS_VOICE = "Kore"' in source
    assert 'DEFAULT_SILERO_VAD_MODEL = "mlx-community/silero-vad"' in source
    assert "ShortTermMemory()" in source
    assert "class ExampleConversation" in source
    assert "speech_decoder=None" in source
    assert "PhoneService(" in source
    assert "create_phone_gateway" not in source
    assert "_gemini_speech_decoder" in source
    assert "_gemini_speech_encoder" in source
    assert "stt_provider=gemini" in source
    assert "tts_provider=gemini" in source
    assert "vad_provider=silero" in source
    assert "off-device Gemini STT/TTS will fail" in source
    assert "listening=listening_ability" in source
    assert "_auto_answer_phone_ring" not in source
    assert "example auto-answer ring" not in source
    assert "suppress hang_up" not in source
    assert "_mercury_intuition_client" in source
    assert "mercury2_intuition" in source
    assert "openai_terra_reasoning" in source


def test_somebody_in_the_bots_room_is_heard_or_not_by_where_they_are_standing() -> None:
    """Saying something out loud is a sound in a place, subject to the room it is said in.

    The example's own ``Person`` speaks, through the mouth it stands behind, through
    ``Environment.broadcast``. One ear is where the robot's ears are — a metre away with the
    robot's own hearing floor — and one is across a field with the same floor. The near ear hears
    a real utterance at a real level; the far ear hears nothing at all, which is only possible if
    the sound went through the environment rather than at somebody.

    Nothing here asserts that anything acts on what it heard. The ears are ears.
    """

    code = '''
import asyncio
import typing

import hsm
from bot import abilities
from bot.environment import Environment, SoundData, SoundEvent, space
from phone_bot_example import person


class Ear(hsm.Instance):
    """Something in the room with a hearing threshold and no opinions."""

    def __init__(self, heard: list) -> None:
        super().__init__()
        self._heard = heard

    @staticmethod
    def _record(ctx, instance, event) -> None:
        if isinstance(event.data, SoundData):
            instance._heard.append(event.data)

    model = hsm.define(
        "Ear",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(SoundEvent), hsm.effect(_record))),
    )


class WordsAsAudio(abilities.Encoder):
    """A vocal tract with no macOS in it, so this runs anywhere the example imports.

    The real one is ``person.SayEncoder``; this is the same contract with the synthesizer taken
    out, so the audio stays traceable back to the words through every stage.
    """

    async def encode(self, input: bytes) -> bytes:
        return b"spoken:" + input


async def main() -> None:
    environment = Environment()
    near, far = [], []
    # Strong references: environment presence is a WeakValueDictionary, so an ear nobody is
    # holding leaves the room before anybody speaks and both counts come back zero.
    ears = {}
    for name, heard, position in (
        ("near", near, space.Position(x=0.0, y=0.0)),
        ("far", far, space.Position(x=0.0, y=500.0)),
    ):
        ear = Ear(heard)
        ears[name] = ear
        _ = await hsm.started(environment, ear, ear.model, hsm.Config(id=name))
        environment.join(ear, placement=space.Placement(position=position, threshold_db=20.0))

    someone = person.Person(
        encoder=WordsAsAudio(),
        position=space.Position(x=0.0, y=1.0),
        amplitude_db=60.0,
    )
    _ = await someone.enter(environment)
    _ = await someone.say("Call Bob at 555-0142.", ctx=environment)
    for _ in range(500):
        await asyncio.sleep(0.01)
        if near:
            break
    print(len(near), len(far))
    print(near[0].audio.decode(), near[0].amplitude_db, near[0].media_type, near[0].sample_rate_hz)
    _ = await someone.leave(environment)


asyncio.run(main())
'''

    assert _run_phone_bot_python(code) == "\n".join(
        [
            "1 0",
            "spoken:Call Bob at 555-0142. 60.0 audio/wav 16000",
        ]
    )


@pytest.mark.skipif(
    shutil.which("say") is None or shutil.which("afconvert") is None,
    reason="the local synthesizer for talking to a bot is macOS say/afconvert",
)
def test_the_words_somebody_types_are_synthesized_on_this_machine() -> None:
    """The operator's line becomes real audio locally: no key, no network, no provider.

    Only that it is audio is asserted. What it sounds like is the synthesizer's business, and
    what it means is nobody's business on this side of the microphone.
    """

    code = "\n".join(
        [
            "import asyncio",
            "from phone_bot_example import person",
            "audio = asyncio.run(person.SayEncoder().encode(b'Call Bob at phone bot bob.'))",
            "print(audio[:4].decode(), audio[8:12].decode(), len(audio) > 1000)",
        ]
    )

    assert _run_phone_bot_python(code) == "RIFF WAVE True"


def test_the_turnkey_command_hands_the_bot_a_config_and_nothing_else(tmp_path: pathlib.Path) -> None:
    """``phone-bot`` resolves LiveKit and starts the bot. It carries no words to it.

    The turnkey command rewrites its env file on the way through (minted token, resolved room),
    which is the only reason it touches config at all. LiveKit and the bot itself are stubbed
    out; what is pinned is that the operator has no way, on this command line, to put anything
    in a bot's head before it is awake.
    """

    env_path = tmp_path / ".env"
    _ = env_path.write_text("BOT_LIVEKIT_ROOM=a-room\n", encoding="utf-8")
    code = "\n".join(
        [
            "import phone_bot_example.cli as cli",
            "seen = []",
            "async def fake_run(config, **kwargs):",
            "    seen.append(config.livekit.room)",
            "    seen.append([field for field in vars(config) if 'told' in field or 'tell' in field])",
            "    return {'livekit_room_audio_connected': True}",
            "cli.run = fake_run",
            f"cli.main(['--json', '--skip-livekit-start', '--env', {str(env_path)!r}])",
            "print(seen)",
        ]
    )

    assert _run_phone_bot_python(code) == "['a-room', []]"


def test_a_failed_utterance_never_repeats_what_somebody_said() -> None:
    """When the synthesizer fails, the words do not come back out in the failure.

    ``Speaking`` turns any exception from the encoder into ``FailureData(message=str(error))``,
    which the owner logs, so anything the encoder puts in an exception is something that ends up
    in a log file. ``say``'s argv *is* the sentence, so this pins that neither the message nor
    the exception chain behind it carries the utterance anywhere.
    """

    code = """
import asyncio
import subprocess

from phone_bot_example import person

SAID = "Call Bob at 555-0142 and mention the budget"
real = subprocess.run
person.subprocess.run = lambda command, **kwargs: real(["false"], **kwargs)
try:
    asyncio.run(person.SayEncoder().encode(SAID.encode()))
except Exception as error:
    chain = [str(error), repr(error.__cause__), repr(error.__context__)]
    print(str(error))
    print(any(SAID in link for link in chain))
finally:
    person.subprocess.run = real
"""

    assert _run_phone_bot_python(code) == "\n".join(["say failed with exit status 1", "False"])


def test_arriving_ends_only_on_a_correlated_attachment_outcome() -> None:
    """Standing in the doorway ends when the attachment protocol answers, and only then.

    The protocol answers an attach request exactly once — complete or failed — and stamps the
    request id on both, so a person guarding on their arrival id hears whichever one comes. A
    timer of this person's own would be a second deadline racing that answer, able to declare
    somebody voiceless while the reply that settles them is already in flight. There is nothing
    left for it to catch, so ``arriving`` carries no trigger but the two outcomes.
    """

    from bot.protocols import attachment

    code = "\n".join(
        [
            "from phone_bot_example import person",
            "from tests.hsm_model import transition_map",
            "print(sorted(transition_map(person.Person.model)['/Person/arriving']))",
        ]
    )

    assert _run_phone_bot_python(code) == str(
        sorted([attachment.AttachCompleteEvent.name, attachment.AttachFailedEvent.name])
    )


def test_a_person_whose_voice_is_already_taken_stops_waiting() -> None:
    """Being refused a voice is an answer, and ``enter`` has to raise on it rather than hang.

    An ability belongs to one owner, so a voice somebody else holds refuses the next actor who
    asks for it. That refusal is correlated to this arrival, which is the whole reason it can be
    heard at all: the transition that fields it is guarded on the arrival's id. A person who
    cannot speak finds out here, in the call that put them in the room.
    """

    code = '''
import asyncio

import hsm
from bot.environment import Environment, space
from bot.protocols import attachment
from phone_bot_example import person


class Bystander(hsm.Instance):
    """Holds the voice first, and says so once it is really theirs."""

    def __init__(self) -> None:
        super().__init__()
        self.holding = asyncio.Event()

    @staticmethod
    def _hold(ctx, instance, event) -> None:
        instance.holding.set()

    model = hsm.define(
        "Bystander",
        hsm.initial(hsm.target("waiting")),
        hsm.state(
            "waiting",
            hsm.transition(
                hsm.on(attachment.AttachCompleteEvent),
                hsm.effect(_hold),
                hsm.target("../holding"),
            ),
        ),
        hsm.state("holding"),
    )


class Refused(person.Person):
    """Walks in after somebody else has already taken this voice."""

    async def enter(self, environment):
        bystander = Bystander()
        _ = await hsm.started(environment, bystander, bystander.model)
        _ = await self._voice.attach(
            environment,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=bystander)),
        )
        # Wait on the holder's own completion, so the voice is provably taken before this
        # person asks for it — not merely likely to be by the time they do.
        await asyncio.wait_for(bystander.holding.wait(), timeout=10.0)
        return await super().enter(environment)


async def main() -> None:
    refused = Refused(
        encoder=person.SayEncoder(),
        position=space.Position(x=0.0, y=1.0),
        amplitude_db=60.0,
    )
    try:
        _ = await asyncio.wait_for(refused.enter(Environment()), timeout=10.0)
    except asyncio.TimeoutError:
        print("still waiting")
    except RuntimeError:
        print("stopped waiting")
    print(refused.state())


asyncio.run(main())
'''

    assert _run_phone_bot_python(code) == "\n".join(["stopped waiting", "/Person/voiceless"])


def test_a_bot_somebody_spoke_to_is_offered_its_phone_and_nothing_more() -> None:
    """The bot is in a room where it can be spoken to, and it has a phone. The choice is its own.

    ``phone.dial`` is offered because the handset is resting on the hook, which is true of every
    bot in this example whether anybody has said anything to it or not. Nothing about being
    spoken to enables it, and nothing about being spoken to dials it — the assertion here is that
    the capability reached the bot, never that anything caused it to be used.
    """

    code = "\n".join(
        [
            "import asyncio, phone_bot_example",
            "from bot.abilities import processing",
            "from bot.devices import phone",
            "from bot.environment import Environment",
            "async def main():",
            "    for label in ('spoken-to', 'alone'):",
            "        environment = Environment()",
            "        body = await phone_bot_example.start_bot(label, environment=environment)",
            "        offered = tuple(event.name for event in processing.enabled_call_events(body.phone()))",
            "        print(body.label(), phone.DialEvent.name in offered)",
            "asyncio.run(main())",
        ]
    )

    assert _run_phone_bot_python(code) == "\n".join(["spoken-to True", "alone True"])


def test_the_example_never_reads_what_is_said_to_it() -> None:
    """The utterance is opaque: no parse, no number extraction, no "heard X → dial".

    An example that inspected what somebody said would be this example choosing for the bot. All
    it does with the words is turn them into sound and let go of them.
    """

    source = _example_source()
    example_root = _repo_root() / "examples" / "phone_bot" / "src" / "phone_bot_example"
    cli = (example_root / "cli.py").read_text(encoding="utf-8")
    speaker = (example_root / "person.py").read_text(encoding="utf-8")
    harness = (_repo_root() / "examples" / "phone_bot" / "scripts" / "blackbox_livekit_two_bots.py").read_text(
        encoding="utf-8"
    )

    for reading_the_words in (
        "text.startswith",
        "text.lower",
        "text.split",
        "utterance.lower",
        "utterance.split",
        '"call" in',
        "DialEvent",
        "DialData",
        "phone.dial",
    ):
        for reader in (source, cli, speaker, harness):
            assert reading_the_words not in reader, reading_the_words


def test_nothing_in_this_example_can_be_told_anything_before_it_wakes() -> None:
    """Configuration is not conversation, so the flag that pretended it was is gone outright.

    No ``--tell``, no ``BOT_TELL``, no config field, no directive written behind the bot's back —
    and no compatibility path that would let one come back. Talking to a bot happens while it is
    running, out loud, or it does not happen.
    """

    example = _repo_root() / "examples" / "phone_bot"
    files = (
        example / "src" / "phone_bot_example" / "__init__.py",
        example / "src" / "phone_bot_example" / "cli.py",
        example / "src" / "phone_bot_example" / "person.py",
        example / "scripts" / "blackbox_livekit_two_bots.py",
        example / "README.md",
        example / ".env.example",
    )

    for path in files:
        text = path.read_text(encoding="utf-8")
        for gone in ("--tell", "BOT_TELL", "config.told", "told=", "directive_insert_input"):
            assert gone not in text, f"{gone} in {path.name}"


def test_the_example_uses_number_as_identity_and_optional_dial_plan() -> None:
    """Identity is the line number; optional MappingDialPlan is alias-only; bot dials digits only.

    Not the deleted directory, which aliased a *name* to an address — that is a contact list, and
    it belongs on a handset rather than in a room. No identity ever reaches the bot as a dial plan
    requirement: dialing addresses the normalized number.
    """

    source = _example_source()
    env_example = (_repo_root() / "examples" / "phone_bot" / ".env.example").read_text(encoding="utf-8")
    readme = (_repo_root() / "examples" / "phone_bot" / "README.md").read_text(encoding="utf-8")
    harness = (_repo_root() / "examples" / "phone_bot" / "scripts" / "blackbox_livekit_two_bots.py").read_text(
        encoding="utf-8"
    )

    assert "signaling.MappingDialPlan(app_config.livekit.dial_plan)" in source
    assert "BOT_LIVEKIT_IDENTITY=5550141" in env_example
    for text in (source, env_example, readme):
        assert "BOT_LIVEKIT_DIAL_PLAN" in text
        # The name→address alias table is gone for good, not renamed.
        assert "BOT_LIVEKIT_DIRECTORY" not in text
        assert "LIVEKIT_DIRECTORY" not in text
        assert "MappingDirectory" not in text
    assert "BOT_LIVEKIT_DIRECTORY" not in harness
    assert "directory=" not in source
    assert "_directory_entries" not in source
    assert "from bot.providers.livekit import PhoneService" in source
    # Two-bots harness identities are normalized numbers, not pretty names.
    assert "phone-bot-alice" not in harness
    assert "phone-bot-bob" not in harness
    # Fictional 555-01xx only: a harness default that could ring a real subscriber is a defect
    # whether or not anybody notices it dialling. Compared as digits, because the defaults are
    # written the way somebody would say them out loud and the punctuation is not the number.
    for number in re.findall(r'--call(?:er|ee)-number",\s*\n\s*default="([^"]+)"', harness):
        assert re.fullmatch(r"55501\d\d", re.sub(r"\D", "", number)), number
        # Said aloud, not spelled out as a digit run: the sentence has to survive being spoken.
        assert "-" in number, number


def test_phone_cognition_preserves_shared_memory_collaboration() -> None:
    source = _example_source()

    assert "autonomy=cognition.Autonomy(memory=store)" in source
    assert "reasoning=cognition.Reasoning(processor=deliberate, memory=store)" in source
    assert "reflection=cognition.Reflection(processor=reflection_processor, memory=store)" in source


def test_phone_bot_example_loads_local_provider_env_without_secret_output(tmp_path: pathlib.Path) -> None:
    env_path = tmp_path / ".env"
    _ = env_path.write_text(
        "\n".join(
            [
                "BOT_OPENAI_API_KEY=secret-openai-key",
                "BOT_REASONING_MODEL=gpt-test-terra",
                "BOT_MERCURY_API_KEY=secret-mercury-key",
                "BOT_LIVEKIT_TRACK_NAME=test-track",
                "BOT_GEMINI_API_KEY=secret-gemini-key",
                "BOT_GEMINI_STT_MODEL=gemini-stt-test",
                "BOT_GEMINI_TTS_MODEL=gemini-tts-test",
                "BOT_GEMINI_TTS_VOICE=Puck",
                "BOT_SILERO_VAD_MODEL=local/silero-test",
            ]
        ),
        encoding="utf-8",
    )

    summary = _run_phone_bot_example("--env", str(env_path))

    assert summary["status"] == "blocked"
    assert summary["can_talk"] is False
    assert (
        summary["cognition_client"] == "mercury=mercury-2@https://api.inceptionlabs.ai/v1 "
        "openai_terra_reasoning=gpt-test-terra@https://api.openai.com/v1 "
        "openai_terra_reflection=gpt-5.6-terra"
    )
    assert summary["mercury_intuition"] == {
        "model": "mercury-2",
        "base_url": "https://api.inceptionlabs.ai/v1",
        "api_key_loaded": True,
    }
    assert summary["openai_reasoning"] == {
        "model": "gpt-test-terra",
        "api_key_loaded": True,
        "base_url": "https://api.openai.com/v1",
    }
    assert typing.cast(dict[str, object], summary["openai_reflection"])["api_key_loaded"] is True
    assert summary["livekit"] == {
        "url_loaded": False,
        "token_loaded": False,
        "api_key_loaded": False,
        "room": "bot-phone-bot",
        "identity": "5550141",
        "track_name": "test-track",
        "dial_plan_entries": 0,
    }
    assert summary["speech"] == {
        "voice_name": "Puck",
        "tts_model": "gemini-tts-test",
        "stt_model": "gemini-stt-test",
        "vad_model_id": "local/silero-test",
        "voice_identity_model_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "api_key_loaded": True,
        "stt_provider": "gemini",
        "vad_provider": "silero",
        "voice_identity_provider": "pyannote",
        "tts_provider": "gemini",
        "input_sample_rate_hz": 48000,
        "input_channels": 1,
        "output_sample_rate_hz": 24000,
        "output_channels": 1,
    }
    assert "secret-openai-key" not in json.dumps(summary)
    assert "secret-mercury-key" not in json.dumps(summary)
    assert "secret-gemini-key" not in json.dumps(summary)


def test_phone_bot_example_loads_voice_ai_provider_env_aliases_without_secret_output(tmp_path: pathlib.Path) -> None:
    env_path = tmp_path / ".env"
    _ = env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=voice-ai-openai-key",
                "VA_LIVEKIT_URL=wss://livekit.example.test",
                "VA_LIVEKIT_TOKEN=voice-ai-livekit-token",
            ]
        ),
        encoding="utf-8",
    )

    summary = _run_phone_bot_example("--env", str(env_path))

    assert summary["status"] == "blocked"
    assert summary["can_talk"] is False
    assert typing.cast(dict[str, object], summary["openai_reasoning"])["api_key_loaded"] is True
    assert typing.cast(dict[str, object], summary["livekit"])["url_loaded"] is True
    assert typing.cast(dict[str, object], summary["livekit"])["token_loaded"] is True
    assert typing.cast(dict[str, object], summary["speech"])["tts_provider"] == "gemini"
    assert typing.cast(dict[str, object], summary["speech"])["stt_provider"] == "gemini"
    assert "voice-ai-openai-key" not in json.dumps(summary)
    assert "voice-ai-livekit-token" not in json.dumps(summary)


def test_phone_bot_example_processor_instructions_are_schema_led() -> None:
    source = _example_source()
    # Example does not hard-code ability prompts; ClassVar defaults are used as-is.
    assert "intuition_instructions" not in source
    assert "reasoning_instructions" not in source
    assert 'DEFAULT_OPENAI_REASONING_MODEL = "gpt-5.6-terra"' in source
    instructions = _default_cognition_instructions()
    prompt_text = "\n".join(instructions.values())
    assert set(instructions) == {"intuition_instructions", "reasoning_instructions"}
    # Phone-bot example must not inject transport/protocol ceremony beyond ClassVar defaults.
    for protocol_detail in (
        "available operation",
        "source_event",
        "payload",
        "incoming phone call",
        "input.frame",
        "event_schema",
        "phone.dial",
        "result.kind",
    ):
        assert protocol_detail not in prompt_text


def test_phone_bot_example_start_bot_returns_active_bot() -> None:
    code = "\n".join(
        [
            "import asyncio, phone_bot_example",
            "async def main():",
            "    config = phone_bot_example.AppConfig(",
            "        cognition=phone_bot_example.CognitionConfig(",
            "            api_key='openai-ok',",
            "            intuition_api_key='mercury-ok',",
            "            reflection_api_key='openai-ok',",
            "        ),",
            "        speech=phone_bot_example.SpeechConfig(),",
            "    )",
            "    body = await phone_bot_example.start_bot('probe', config=config)",
            "    print(body.state())",
            "    print(body.conversation().state())",
            "asyncio.run(main())",
        ]
    )

    assert _run_phone_bot_python(code) == "\n".join(
        [
            "/PhoneBot/active/unfocused",
            "/ExampleConversationLifecycle/attached/behavior/inactive",
        ]
    )


def test_phone_bot_example_routes_livekit_audio_through_phone_service() -> None:
    source = _example_source()

    assert "async def talk_once" not in source
    assert "--audio-file" not in source
    assert "remote_audio_sink" not in source
    assert "PhoneService(" in source
    assert "_handset(service=phone_service)" in source
    # The handset earpiece and the robot's voice are separate transducers, in separate places.
    # Sharing one object would put the same speaker at the ear and at the mouth, which is how the
    # far end ends up hearing itself.
    assert "_speaking(speaker=voice" in source
    assert "_speaking(speaker=speaker" not in source
    # The voice transducer is the robot's mouth, not a device it owns: Speaking powers it, so it
    # is injected into the ability and never registered with the body. Handing Speaking a speaker
    # nothing starts is what left the bot mute on a live call.
    assert '"voice": self._voice' not in source
    # One value for the robot's voice rate, reaching the encoder, Speaking, and the LiveKit
    # source. Three independent defaults is how a 24 kHz voice met a 48 kHz source and went mute.
    assert "uplink_sample_rate_hz=app_config.speech.output_sample_rate_hz" in source
    assert "ensure_future" not in source


def test_phone_bot_example_does_not_probe_cognition_on_plain_run(tmp_path: pathlib.Path) -> None:
    env_path = tmp_path / ".env"
    _ = env_path.write_text(
        "\n".join(
            [
                "BOT_OPENAI_API_KEY=secret-openai-key",
                "BOT_MERCURY_API_KEY=secret-mercury-key",
            ]
        ),
        encoding="utf-8",
    )
    code = "\n".join(
        [
            "import asyncio, pathlib, phone_bot_example",
            "class FakeClient:",
            "    def generate_content(self, **kwargs):",
            "        raise RuntimeError('generate_content should not be requested during plain run')",
            "    def create_interaction(self, **kwargs):",
            "        raise RuntimeError('create_interaction should not be requested during plain run')",
            "    def chat_completions(self, **kwargs):",
            "        raise RuntimeError('chat should not be requested during plain run')",
            "phone_bot_example._openai_reasoning_client = lambda config: FakeClient()",
            "phone_bot_example._openai_reflection_client = lambda config: FakeClient()",
            "phone_bot_example._mercury_intuition_client = lambda config: FakeClient()",
            "phone_bot_example._mlx_speech_encoder = lambda config: FakeClient()",
            "phone_bot_example._mlx_speech_decoder = lambda config: FakeClient()",
            "async def main():",
            f"    config = phone_bot_example.AppConfig.from_env_file(pathlib.Path({str(env_path)!r}))",
            "    summary = await phone_bot_example.run(config)",
            "    print(summary['livekit_remote_audio_chunks'], summary['livekit_room_audio_connected'])",
            "asyncio.run(main())",
        ]
    )

    assert _run_phone_bot_python(code) == "0 False"


def test_phone_bot_example_does_not_connect_livekit_on_plain_run(tmp_path: pathlib.Path) -> None:
    env_path = tmp_path / ".env"
    _ = env_path.write_text(
        "\n".join(
            [
                "BOT_LIVEKIT_URL=wss://livekit.example.test",
                "BOT_LIVEKIT_TOKEN=secret-livekit-token",
            ]
        ),
        encoding="utf-8",
    )
    code = "\n".join(
        [
            "import asyncio, pathlib, phone_bot_example",
            "async def fail_connect(self, **kwargs):",
            "    raise RuntimeError('LiveKit should not be connected during plain run')",
            "phone_bot_example.PhoneService.connect_room = fail_connect",
            "async def main():",
            f"    config = phone_bot_example.AppConfig.from_env_file(pathlib.Path({str(env_path)!r}))",
            "    summary = await phone_bot_example.run(config)",
            "    print(summary['livekit_room_audio_attempted'], summary['livekit_room_audio_connected'])",
            "asyncio.run(main())",
        ]
    )

    assert _run_phone_bot_python(code) == "False False"


def test_phone_bot_example_can_attempt_fake_livekit_room_audio_when_opted_in() -> None:
    code = "\n".join(
        [
            "import asyncio, phone_bot_example",
            "from bot.providers.livekit import PhoneService, MediaSnapshot",
            "_orig_attach = PhoneService.attach",
            "async def _attach(self, environment, target):",
            "    connect = self._room_connect",
            "    self._room_connect = None",
            "    try:",
            "        await _orig_attach(self, environment, target)",
            "    finally:",
            "        self._room_connect = connect",
            "        if connect is not None:",
            "            self._local_track_sid = 'TR_fake'",
            "PhoneService.attach = _attach",
            "async def main():",
            "    config = phone_bot_example.AppConfig(",
            "        livekit=phone_bot_example.LiveKitConfig(url='wss://livekit.example.test', token='secret-token'),",
            "        speech=phone_bot_example.SpeechConfig(),",
            f"        {_full_cognition_config_kwargs()},",
            "    )",
            "    summary = await phone_bot_example.run(config, connect_livekit=True)",
            "    print(summary['livekit_room_audio_attempted'], summary['livekit_room_audio_connected'])",
            "asyncio.run(main())",
        ]
    )
    assert _run_phone_bot_python(code) == "True True"


def test_phone_bot_example_does_not_report_ready_without_cognition_credentials() -> None:
    code = "\n".join(
        [
            "import asyncio, phone_bot_example",
            "from bot.providers.livekit import PhoneService",
            "_orig_attach = PhoneService.attach",
            "async def _attach(self, environment, target):",
            "    connect = self._room_connect",
            "    self._room_connect = None",
            "    try:",
            "        await _orig_attach(self, environment, target)",
            "    finally:",
            "        self._room_connect = connect",
            "        if connect is not None:",
            "            self._local_track_sid = 'TR_fake'",
            "PhoneService.attach = _attach",
            "async def main():",
            "    config = phone_bot_example.AppConfig(",
            "        livekit=phone_bot_example.LiveKitConfig(url='wss://livekit.example.test', token='secret-token'),",
            "        speech=phone_bot_example.SpeechConfig(),",
            "    )",
            "    summary = await phone_bot_example.run(config, connect_livekit=True)",
            "    print(summary['status'], summary['can_talk'])",
            "asyncio.run(main())",
        ]
    )
    assert _run_phone_bot_python(code) == "blocked False"


def test_phone_bot_example_reports_room_ready_without_remote_chunks() -> None:
    code = "\n".join(
        [
            "import asyncio, phone_bot_example",
            "from bot.providers.livekit import PhoneService",
            "_orig_attach = PhoneService.attach",
            "async def _attach(self, environment, target):",
            "    connect = self._room_connect",
            "    self._room_connect = None",
            "    try:",
            "        await _orig_attach(self, environment, target)",
            "    finally:",
            "        self._room_connect = connect",
            "        if connect is not None:",
            "            self._local_track_sid = 'TR_fake'",
            "PhoneService.attach = _attach",
            "async def main():",
            "    config = phone_bot_example.AppConfig(",
            "        livekit=phone_bot_example.LiveKitConfig(url='wss://livekit.example.test', token='secret-token'),",
            "        speech=phone_bot_example.SpeechConfig(api_key='gemini-ok'),",
            f"        {_full_cognition_config_kwargs()},",
            "    )",
            "    summary = await phone_bot_example.run(config, connect_livekit=True)",
            "    print(summary['status'], summary['can_talk'], summary['livekit_remote_audio_chunks'])",
            "asyncio.run(main())",
        ]
    )
    assert _run_phone_bot_python(code) == "ready True 0"


def test_phone_bot_example_does_not_report_livekit_attempt_without_credentials() -> None:
    code = "\n".join(
        [
            "import asyncio, phone_bot_example",
            "async def main():",
            "    summary = await phone_bot_example.run(phone_bot_example.AppConfig(), connect_livekit=True)",
            "    print(summary['livekit_room_audio_attempted'], summary['livekit_room_audio_connected'])",
            "asyncio.run(main())",
        ]
    )
    assert _run_phone_bot_python(code) == "False False"


def test_phone_bot_powers_the_voice_transducer_it_speaks_through() -> None:
    """Speaking starts the voice speaker with the ability, so playout has something live to attach to.

    Regression: the speaker split gave Speaking its own transducer and nothing started it, so the
    first utterance raised "Device is not started in this environment" and the bot never spoke.
    The mouth is powered but not yet wired: attachment still waits for the first utterance.
    """

    code = """
import asyncio
import bot.lifecycle as lifecycle
import phone_bot_example as example
from bot.environment import Environment


async def main() -> None:
    body = example.PhoneBot("probe")
    environment = Environment()
    _ = await body.attach(environment)
    for _ in range(100):
        await asyncio.sleep(0.05)
        if body.state().startswith("/PhoneBot/active"):
            break
    voice = object.__getattribute__(body, "_voice")
    print(lifecycle.is_started(voice))
    print(voice.state())


asyncio.run(main())
"""

    assert _run_phone_bot_python(code) == "\n".join(["True", "/Device/detached"])


def test_phone_bot_listening_speech_products_carry_source_ids_into_conversation() -> None:
    """Listening labels speech; Communication seed + Conversation.input admit labeled products."""

    code = """
from __future__ import annotations

import asyncio

import hsm
from bot.abilities import listening, memory
from bot.abilities.communication import conversation
from bot.abilities.communication import behaviors
from bot.abilities.hearing import voice
from phone_bot_example import SpeechConfig, _listening, _pyannote_voice_classifier


class FixedInference:
    def __call__(self, waveform):
        del waveform
        return [[0.12, -0.08, 0.31]]


async def wait_until(condition, *, timeout_s: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if condition():
            return
        await asyncio.sleep(0)
    raise TimeoutError("wait_until timeout")


async def main() -> None:
    config = SpeechConfig()
    _ = _listening(
        config,
        voice_classifier=_pyannote_voice_classifier(config, inference=FixedInference()),
    )
    store = memory.Memory()
    installed = behaviors.install_seed_behaviors(store)
    assert installed[0].triggers == (listening.SpeechEvent.name,)

    speech = listening.SpeechData(
        content=bytes([0, 1]) * 160,
        voice_detection=voice.detection.ApplyData(
            segments=(
                voice.detection.VoiceDetectionSegment(
                    start_seconds=0.0,
                    end_seconds=0.02,
                    confidence=0.9,
                ),
            )
        ),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({(0.12, -0.08, 0.31)}),
    )
    assert speech.source_ids == frozenset({(0.12, -0.08, 0.31)})

    conversation_ability = conversation.Conversation()
    ctx = hsm.Context()
    assert conversation_ability.model is not None
    _ = await hsm.started(ctx, conversation_ability, conversation_ability.model)
    from bot.protocols import attachment
    class Owner(hsm.Instance):
        model = hsm.define("Owner", hsm.initial(hsm.target("/Owner/a")), hsm.state("a"))
    owner = Owner()
    _ = await hsm.started(ctx, owner, owner.model)
    _ = await conversation_ability.attach(
        ctx,
        attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
    )
    await wait_until(lambda: "/behavior/inactive" in (conversation_ability.state() or ""))

    operation_id = "seed-admit"
    await hsm.dispatch(
        ctx,
        conversation_ability,
        conversation.InputEvent.with_data_and_id(
            conversation.TurnData(
                source_ids=speech.source_ids,
                target_ids=frozenset(),
                content=speech.content,
                content_type="audio/raw",
            ),
            operation_id,
        ),
    )
    await wait_until(lambda: "/active/" in (conversation_ability.state() or ""))
    print(bool(speech.source_ids))
    print(list(next(iter(speech.source_ids))))
    print("/active/" in (conversation_ability.state() or ""))
    print(installed[0].name)


asyncio.run(main())
"""

    assert _run_phone_bot_python(code) == "\n".join(
        [
            "True",
            "[0.12, -0.08, 0.31]",
            "True",
            "SpeechHeard",
        ]
    )


def test_phone_bot_e2e_cognition_wires_speech_event_to_conversation() -> None:
    """Real PhoneBot + live cognition: time until SpeechEvent→Conversation behavior appears.

    No seeded behaviors. Ambient speech goes through Listening into live bot cognition.
    Reflection must author the wire itself (no FixedProcessor ignore stub).
    """

    code = r"""
from __future__ import annotations

import asyncio
import pathlib
import time

from bot.abilities import listening, memory
from bot.abilities.communication import conversation
from bot.behavior import storage as behavior_storage
from bot.environment import Environment
from phone_bot_example import AppConfig, start_bot, _someone_in_the_room

SPEECH_EVENT = listening.SpeechEvent.name
CONVERSATION_INPUT = conversation.InputEvent.name
MAX_TURNS = 12
TURN_TIMEOUT_S = 180.0
OVERALL_TIMEOUT_S = 900.0


def _wires_speech_heard(item) -> bool:
    triggers = tuple(item.triggers or ())
    if SPEECH_EVENT not in triggers:
        return False
    return CONVERSATION_INPUT in (item.source or "")


def _inventory(store: memory.Memory):
    select = memory.InputData(
        statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())
    )
    out = store.execute(select)
    if len(out.results) < 2:
        return ()
    return behavior_storage.instances_from_behavior_results(
        tuple(row.as_mapping() for row in out.results[0].rows),
        tuple(row.as_mapping() for row in out.results[1].rows),
    )


async def wait_until(condition, *, timeout_s: float) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if condition():
            return
        await asyncio.sleep(0)
    raise TimeoutError(f"condition not met within {timeout_s}s")


async def main() -> None:
    config = AppConfig.from_env_file(pathlib.Path(".env"))
    if not config.cognition.can_process():
        print("SKIP missing cognition credentials")
        return

    store = memory.ShortTermMemory()
    environment = Environment()
    body = await start_bot(
        "e2e-speech-wire",
        config=config,
        environment=environment,
        memory=store,
        connect_livekit=False,
    )
    someone = _someone_in_the_room()
    await someone.enter(environment)

    started = time.perf_counter()
    found = None
    turns = 0
    handoffs_before = len(body.listening_handoffs())
    elapsed = 0.0

    try:
        while turns < MAX_TURNS and (time.perf_counter() - started) < OVERALL_TIMEOUT_S:
            turns += 1
            await someone.say("Call Bob at 555-0142.", ctx=environment)
            await wait_until(
                lambda: len(body.listening_handoffs()) > handoffs_before
                or len(body.outputs()) >= turns
                or len(body.failures()) >= turns,
                timeout_s=TURN_TIMEOUT_S,
            )
            handoffs_before = len(body.listening_handoffs())
            await wait_until(
                lambda: (body.state() or "").endswith("/active/unfocused")
                or (body.state() or "").endswith("/active/focused")
                or (body.state() or "").endswith("/inactive"),
                timeout_s=TURN_TIMEOUT_S,
            )
            for item in _inventory(store):
                if _wires_speech_heard(item):
                    found = item
                    break
            if found is not None:
                break
    finally:
        elapsed = time.perf_counter() - started
        try:
            await someone.leave(environment)
        except Exception:
            pass

    if found is None:
        print(
            f"NEVER turns={turns} elapsed_s={elapsed:.3f} "
            f"handoffs={len(body.listening_handoffs())} "
            f"outputs={len(body.outputs())} failures={len(body.failures())} "
            f"stimulus={SPEECH_EVENT!r} target={CONVERSATION_INPUT!r}"
        )
        raise SystemExit(2)

    print(
        f"WIRED turns={turns} elapsed_s={elapsed:.3f} "
        f"behavior={found.name!r} triggers={list(found.triggers)}"
    )


asyncio.run(main())
"""

    import os
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd="examples/phone_bot",
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=960,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if "SKIP missing cognition credentials" in out:
        import pytest

        pytest.skip("cognition credentials unavailable for live e2e")
    assert proc.returncode == 0, out
    assert "WIRED" in out, out


def test_labeled_phone_bots_share_environment_instances_and_publish_distinct_owners() -> None:
    code = """
import asyncio
import importlib

import hsm
import phone_bot_example as example
from bot.environment import Environment, space


async def main() -> None:
    start = importlib.import_module("bot.start")
    published = []
    start.publish = lambda payload: published.append(payload)
    start.publish_live = lambda payload: published.append(payload)
    environment = Environment()
    alice_placement = space.Placement(position=space.Position(x=0, y=0))
    bob_placement = space.Placement(position=space.Position(x=100, y=0))
    alice = await example.start_bot(
        "Alice", environment=environment, placement=alice_placement
    )
    bob = await example.start_bot(
        "Bob", environment=environment, placement=bob_placement
    )
    assert alice_placement.position.distance_to(bob_placement.position) >= 100
    instances = environment.value(hsm.Keys.Instances)
    assert alice.context().value(hsm.Keys.Instances) is instances
    assert bob.context().value(hsm.Keys.Instances) is instances
    assert environment._placements[alice] == alice_placement
    assert environment._placements[bob] == bob_placement
    assert {payload["name"] for payload in published if payload["name"] in {"/Alice", "/Bob"}} == {"/Alice", "/Bob"}
    assert alice_placement.position.distance_to(bob_placement.position) >= 100
    children = [
        payload
        for payload in published
        if payload["name"] in {"/AlicePhone", "/BobPhone", "/AlicePhoneMicrophone", "/BobPhoneMicrophone"}
    ]
    assert {payload["name"] for payload in children} == {
        "/AlicePhone",
        "/BobPhone",
        "/AlicePhoneMicrophone",
        "/BobPhoneMicrophone",
    }
    assert len({payload["name"] for payload in children}) == len(children)
    assert {
        payload["owner"]
        for payload in children
        if payload["name"].endswith("Microphone")
    } == {"/AlicePhone", "/BobPhone"}


asyncio.run(main())
"""
    assert _run_phone_bot_python(code) == ""

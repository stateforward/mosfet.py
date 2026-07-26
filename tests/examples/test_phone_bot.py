import json
import os
import pathlib
import subprocess
import typing


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
    assert "bot-provider-elevenlabs" not in pyproject
    assert "from bot.providers.gemini import SpeechDecoder as GeminiSpeechDecoder" in source
    assert "from bot.providers.gemini import SpeechEncoder as GeminiSpeechEncoder" in source
    assert "from bot.providers.mlx_audio import VoiceDetector as SileroVoiceDetector" in source
    assert "from bot.providers.openai_compat import Processor as OpenAIProcessor" in source
    assert "AlwaysVoiceDetector" not in source
    assert "PeakEnergyVoiceDetector" not in source
    assert "PcmAwareVoiceDetector" in source
    assert "_silero_voice_detector" in source
    assert "KindSoundClassifier" in source
    assert "from bot.devices import phone as phone_device" in source
    assert "from bot.providers.livekit import PhoneService" in source
    assert "from bot.providers.livekit.audio import PcmWavDecoder" in source
    assert "VoiceDecoder" not in source
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
    assert "class LiveKitVoiceDecoder" not in source
    assert "class ExampleTextConversation" in source
    assert "ExampleVoiceConversation" not in source
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
        summary["cognition_client"]
        == "mercury=mercury-2@https://api.inceptionlabs.ai/v1 "
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
        "identity": "bot-phone-bot",
        "track_name": "test-track",
    }
    assert summary["speech"] == {
        "voice_name": "Puck",
        "tts_model": "gemini-tts-test",
        "stt_model": "gemini-stt-test",
        "vad_model_id": "local/silero-test",
        "api_key_loaded": True,
        "stt_provider": "gemini",
        "vad_provider": "silero",
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
            "/ExampleTextConversationLifecycle/attached/behavior/silent",
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
    # The voice transducer is one of the robot's own devices, so the body powers it. Handing
    # Speaking a speaker nobody starts is what left the bot mute on a live call.
    assert '"voice": self._voice' in source
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
    """The body starts the voice speaker, so Speaking has something live to attach to.

    Regression: the speaker split gave Speaking its own transducer and nothing started it, so the
    first utterance raised "Device is not started in this environment" and the bot never spoke.
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

    assert _run_phone_bot_python(code) == "\n".join(["True", "/Device/attached"])

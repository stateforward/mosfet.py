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


def test_phone_bot_example_is_provider_package_example() -> None:
    pyproject = (_repo_root() / "examples" / "phone_bot" / "pyproject.toml").read_text(encoding="utf-8")
    source = _example_source()

    assert "bot-provider-livekit" in pyproject
    assert "bot-provider-gemini" in pyproject
    assert "bot-provider-sqlite-memory" in pyproject
    assert "bot-provider-elevenlabs" not in pyproject
    assert "bot-provider-openai-compat" not in pyproject
    assert "bot-provider-mlx-audio" not in pyproject
    assert "from bot.providers.gemini import ChatClient, Processor" in source
    assert "from bot.providers.gemini import SpeechDecoder as GeminiSpeechDecoder" in source
    assert "from bot.providers.gemini import SpeechEncoder as GeminiSpeechEncoder" in source
    assert "from bot.devices import phone as phone_device" in source
    assert "from bot.providers.livekit import PhoneService" in source
    assert "from bot.providers.livekit.audio import PcmWavDecoder" in source
    assert "VoiceDecoder" not in source
    assert "RoomAudioEndpoint" not in source
    assert "RoomAudioConnectData" not in source
    assert "create_audio_bridge" not in source
    assert "from bot.providers.sqlite_memory import" not in source
    assert "SqliteMemoryEncoder" not in source
    assert "SqliteMemoryDecoder" not in source
    assert 'DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"' in source
    assert 'DEFAULT_GEMINI_INTUITION_MODEL = "gemini-3.1-flash-lite"' in source
    assert 'DEFAULT_GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"' in source
    assert "ShortTermMemory()" in source
    assert "class LiveKitVoiceDecoder" not in source
    assert "class ExampleTextConversation" in source
    assert "ExampleVoiceConversation" not in source
    assert "PhoneService(" in source
    assert "create_phone_gateway" not in source


def test_phone_cognition_preserves_shared_memory_collaboration() -> None:
    source = _example_source()

    assert "autonomy=cognition.Autonomy(memory=store)" in source
    assert "reasoning=cognition.Reasoning(processor=deliberate, memory=store)" in source
    assert "reflection=cognition.Reflection(processor=deliberate, memory=store)" in source


def test_phone_bot_example_loads_local_provider_env_without_secret_output(tmp_path: pathlib.Path) -> None:
    env_path = tmp_path / ".env"
    _ = env_path.write_text(
        "\n".join(
            [
                "BOT_GEMINI_API_KEY=secret-gemini-key",
                "BOT_GEMINI_MODEL=gemini-test-model",
                "BOT_GEMINI_VOICE_NAME=Puck",
                "BOT_LIVEKIT_TRACK_NAME=test-track",
            ]
        ),
        encoding="utf-8",
    )

    summary = _run_phone_bot_example("--env", str(env_path))

    assert summary["status"] == "blocked"
    assert summary["can_talk"] is False
    assert summary["cognition_client"] == "ChatClient(gemini-test-model)"
    assert summary["gemini"] == {
        "model": "gemini-test-model",
        "api_key_loaded": True,
        "tts_model": "gemini-3.1-flash-tts-preview",
        "stt_model": "gemini-3.5-flash",
        "voice_name": "Puck",
    }
    assert summary["livekit"] == {
        "url_loaded": False,
        "token_loaded": False,
        "api_key_loaded": False,
        "room": "bot-phone-bot",
        "identity": "bot-phone-bot",
        "track_name": "test-track",
    }
    assert summary["speech"] == {
        "api_key_loaded": True,
        "voice_name": "Puck",
        "tts_model": "gemini-3.1-flash-tts-preview",
        "stt_model": "gemini-3.5-flash",
        "input_sample_rate_hz": 48000,
        "input_channels": 1,
        "output_sample_rate_hz": 24000,
        "output_channels": 1,
    }
    assert "secret-gemini-key" not in json.dumps(summary)


def test_phone_bot_example_loads_voice_ai_provider_env_aliases_without_secret_output(tmp_path: pathlib.Path) -> None:
    env_path = tmp_path / ".env"
    _ = env_path.write_text(
        "\n".join(
            [
                "VA_GEMINI_API_KEY=voice-ai-gemini-key",
                "VA_LIVEKIT_URL=wss://livekit.example.test",
                "VA_LIVEKIT_TOKEN=voice-ai-livekit-token",
            ]
        ),
        encoding="utf-8",
    )

    summary = _run_phone_bot_example("--env", str(env_path))

    assert summary["status"] == "blocked"
    assert summary["can_talk"] is False
    assert summary["cognition_client"] == "ChatClient(gemini-3.5-flash)"
    assert typing.cast(dict[str, object], summary["gemini"])["api_key_loaded"] is True
    assert typing.cast(dict[str, object], summary["livekit"])["url_loaded"] is True
    assert typing.cast(dict[str, object], summary["livekit"])["token_loaded"] is True
    assert typing.cast(dict[str, object], summary["speech"])["api_key_loaded"] is True
    assert "voice-ai-gemini-key" not in json.dumps(summary)
    assert "voice-ai-livekit-token" not in json.dumps(summary)


def test_phone_bot_example_processor_instructions_are_schema_led() -> None:
    source = _example_source()
    # Example no longer hard-codes ability prompts; ClassVar defaults stay schema-led.
    assert "intuition_instructions" not in source
    assert "reasoning_instructions" not in source
    assert 'DEFAULT_GEMINI_INTUITION_MODEL = "gemini-3.1-flash-lite"' in source
    instructions = _default_cognition_instructions()
    prompt_text = "\n".join(instructions.values())
    assert set(instructions) == {"intuition_instructions", "reasoning_instructions"}
    for protocol_detail in (
        "available operation",
        "source_event",
        "payload",
        "call_id",
        "incoming phone call",
        "input.frame",
        "event_schema",
        "phone.answer_call",
        "phone.dial",
        "result.kind",
        "kind=",
        "data.",
    ):
        assert protocol_detail not in prompt_text


def test_phone_bot_example_start_bot_returns_active_bot() -> None:
    code = "\n".join(
        [
            "import asyncio, phone_bot_example",
            "async def main():",
            "    config = phone_bot_example.AppConfig()",
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
    assert "phone_device.Phone(service=phone_service, speaker=speaker)" in source
    assert "_speaking(speaker=speaker" in source
    assert "ensure_future" not in source


def test_phone_bot_example_does_not_probe_cognition_on_plain_run(tmp_path: pathlib.Path) -> None:
    env_path = tmp_path / ".env"
    _ = env_path.write_text("BOT_GEMINI_API_KEY=secret-gemini-key\n", encoding="utf-8")
    code = "\n".join(
        [
            "import asyncio, pathlib, phone_bot_example",
            "class FakeClient:",
            "    def generate_content(self, **kwargs):",
            "        raise RuntimeError('generate_content should not be requested during plain run')",
            "    def create_interaction(self, **kwargs):",
            "        raise RuntimeError('create_interaction should not be requested during plain run')",
            "def fake_chat_client(config, *, model=None):",
            "    return FakeClient()",
            "phone_bot_example._chat_client = fake_chat_client",
            "phone_bot_example._gemini_client = lambda **kwargs: FakeClient()",
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
            "async def _attach(self, world, target):",
            "    connect = self._room_connect",
            "    self._room_connect = None",
            "    try:",
            "        await _orig_attach(self, world, target)",
            "    finally:",
            "        self._room_connect = connect",
            "        if connect is not None:",
            "            self._local_track_sid = 'TR_fake'",
            "PhoneService.attach = _attach",
            "async def main():",
            "    config = phone_bot_example.AppConfig(",
            "        livekit=phone_bot_example.LiveKitConfig(url='wss://livekit.example.test', token='secret-token'),",
            "        speech=phone_bot_example.SpeechConfig(api_key='k'),",
            "        cognition=phone_bot_example.CognitionConfig(api_key='ok'),",
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
            "async def _attach(self, world, target):",
            "    connect = self._room_connect",
            "    self._room_connect = None",
            "    try:",
            "        await _orig_attach(self, world, target)",
            "    finally:",
            "        self._room_connect = connect",
            "        if connect is not None:",
            "            self._local_track_sid = 'TR_fake'",
            "PhoneService.attach = _attach",
            "async def main():",
            "    config = phone_bot_example.AppConfig(",
            "        livekit=phone_bot_example.LiveKitConfig(url='wss://livekit.example.test', token='secret-token'),",
            "        speech=phone_bot_example.SpeechConfig(api_key='k'),",
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
            "async def _attach(self, world, target):",
            "    connect = self._room_connect",
            "    self._room_connect = None",
            "    try:",
            "        await _orig_attach(self, world, target)",
            "    finally:",
            "        self._room_connect = connect",
            "        if connect is not None:",
            "            self._local_track_sid = 'TR_fake'",
            "PhoneService.attach = _attach",
            "async def main():",
            "    config = phone_bot_example.AppConfig(",
            "        livekit=phone_bot_example.LiveKitConfig(url='wss://livekit.example.test', token='secret-token'),",
            "        speech=phone_bot_example.SpeechConfig(api_key='k'),",
            "        cognition=phone_bot_example.CognitionConfig(api_key='ok'),",
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

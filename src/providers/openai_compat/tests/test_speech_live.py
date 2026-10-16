"""Live check of ``openai_compat.SpeechEncoder`` and ``SpeechDecoder`` against served OpenAI-compatible speech endpoints.

Point the test at a speech server (for example Breeze TTS) and, optionally, a transcription server (for example a
batch Nemotron endpoint); without a transcription URL the encoded audio is transcribed by the speech server::

    BOT_OPENAI_COMPAT_SPEECH_BASE_URL=http://lab.local:8002/v1 \\
    BOT_OPENAI_COMPAT_TRANSCRIPTION_BASE_URL=http://lab.local:8001/v1 \\
      uv run --package mosfet-provider-openai-compat --group dev python -m pytest \\
      src/providers/openai_compat/tests/test_speech_live.py -m live -v -s

``BOT_OPENAI_COMPAT_SPEECH_MODEL``, ``BOT_OPENAI_COMPAT_SPEECH_VOICE`` and ``BOT_OPENAI_COMPAT_TRANSCRIPTION_MODEL``
name the served models and voice when the server needs them. What was heard is printed, never pinned.
"""

from __future__ import annotations

import asyncio
import os

import pytest

import mosfet.providers.openai_compat as openai_compat
from mosfet.abilities.vocal import speech

pytestmark = pytest.mark.live


def test_served_speech_round_trips_through_encoding_and_transcription() -> None:
    speech_url = os.environ.get("BOT_OPENAI_COMPAT_SPEECH_BASE_URL")
    if not speech_url:
        pytest.skip("BOT_OPENAI_COMPAT_SPEECH_BASE_URL not set")
    encoder = openai_compat.SpeechEncoder(
        model=os.environ.get("BOT_OPENAI_COMPAT_SPEECH_MODEL", "tts"),
        base_url=speech_url,
        voice=os.environ.get("BOT_OPENAI_COMPAT_SPEECH_VOICE"),
    )

    async def encode() -> bytes:
        text = speech.Text()
        text.write(speech.Piece(part=0, words="Hello from the bot."))
        text.close()
        return b"".join([chunk.pcm async for chunk in encoder.encode(text)])

    audio = asyncio.run(encode())

    assert audio
    assert len(audio) % (2 * encoder.format.channels) == 0
    decoder = openai_compat.SpeechDecoder(
        model=os.environ.get("BOT_OPENAI_COMPAT_TRANSCRIPTION_MODEL", "stt"),
        base_url=os.environ.get("BOT_OPENAI_COMPAT_TRANSCRIPTION_BASE_URL", speech_url),
        sample_rate_hz=encoder.format.sample_rate_hz,
        channels=encoder.format.channels,
    )
    transcript = asyncio.run(decoder.decode(audio)).decode("utf-8")
    print(f"heard: {transcript!r}")
    assert transcript.strip()

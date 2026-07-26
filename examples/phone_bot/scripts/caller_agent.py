#!/usr/bin/env python3
"""Join the phone_bot LiveKit room as a second participant and speak.

Default speech path is **off-device Gemini TTS** (no local MLX, no macOS ``say``)
so dual-agent runs do not burn on-device STT/TTS resources. Optional ``--tts say``
falls back to macOS ``say`` for offline smoke.

Optional ``record_dir`` writes a blackbox conversation capture:
- ``agent_b_outbound.wav`` — what agent B published (concatenated turns)
- ``agent_a_inbound.wav`` — what agent B received from agent A over LiveKit
- ``conversation.json`` — timeline of B's lines, peaks, frame counts, timestamps
- ``turns/b_NN_*.wav`` — per-turn outbound clips
"""

from __future__ import annotations

import argparse
import asyncio
import array
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import wave

from livekit import rtc

# Repo example helpers (mint token + Gemini speech config).
_EXAMPLE_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXAMPLE_ROOT / "src"))
from phone_bot_example import (  # noqa: E402
    DEFAULT_GEMINI_TTS_MODEL,
    DEFAULT_GEMINI_TTS_VOICE,
    AppConfig,
    mint_livekit_access_token,
)


def _say_to_pcm(text: str, *, sample_rate_hz: int = 48_000, voice: str = "Samantha") -> bytes:
    """Render speech with macOS say → WAV → raw s16le mono PCM (optional offline fallback)."""

    with tempfile.TemporaryDirectory() as tmp:
        aiff = pathlib.Path(tmp) / "utter.aiff"
        wav = pathlib.Path(tmp) / "utter.wav"
        _ = subprocess.run(
            ["say", "-v", voice, "-o", str(aiff), text],
            check=True,
            capture_output=True,
        )
        _ = subprocess.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{sample_rate_hz}",
                "-c",
                "1",
                str(aiff),
                str(wav),
            ],
            check=True,
            capture_output=True,
        )
        with wave.open(str(wav), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == sample_rate_hz
            return handle.readframes(handle.getnframes())


def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """Extract s16le mono PCM and sample rate from a WAV container."""

    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or width != 2:
        raise RuntimeError(f"Gemini TTS expected mono s16le WAV; got channels={channels} width={width}")
    return frames, rate


async def _gemini_to_pcm(
    text: str,
    *,
    api_key: str,
    model: str,
    voice_name: str,
    sample_rate_hz: int,
) -> tuple[bytes, int]:
    """Render speech with off-device Gemini TTS → raw s16le mono PCM."""

    from bot.providers.gemini import ChatClient, SpeechEncoder

    encoder = SpeechEncoder(
        client=ChatClient(api_key=api_key, model=model),
        model=model,
        voice_name=voice_name,
        sample_rate_hz=sample_rate_hz,
        output_format="wav",
    )
    wav_bytes = await encoder.encode(text.encode("utf-8"))
    return _wav_to_pcm(wav_bytes)


def _pcm_peak(pcm: bytes) -> int:
    """Peak absolute sample across the whole buffer, scanned in bounded chunks.

    Scans everything rather than a leading window: an inbound capture opens while the far
    side is still silent (agent A does not speak until spoken to, ~11s into a run), so a
    truncated window reports silence for audio that is plainly present later.
    """

    if len(pcm) < 2:
        return 0
    usable = pcm if len(pcm) % 2 == 0 else pcm[:-1]
    peak = 0
    for start in range(0, len(usable), 192_000):
        samples = array.array("h")
        samples.frombytes(usable[start : start + 192_000])
        chunk_peak = max((abs(s) for s in samples), default=0)
        if chunk_peak > peak:
            peak = chunk_peak
    return peak


def _write_wav(path: pathlib.Path, pcm: bytes, *, sample_rate_hz: int, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm)


def _pcm_frames(pcm: bytes, *, sample_rate_hz: int, samples_per_frame: int = 480) -> list[rtc.AudioFrame]:
    """Slice s16le mono PCM into LiveKit AudioFrames (10ms @ 48k default)."""

    bytes_per_frame = samples_per_frame * 2
    frames: list[rtc.AudioFrame] = []
    for offset in range(0, len(pcm), bytes_per_frame):
        chunk = pcm[offset : offset + bytes_per_frame]
        if len(chunk) < bytes_per_frame:
            chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))
        frames.append(
            rtc.AudioFrame(
                data=chunk,
                sample_rate=sample_rate_hz,
                num_channels=1,
                samples_per_channel=samples_per_frame,
            )
        )
    return frames


async def _publish_pcm(
    source: rtc.AudioSource,
    pcm: bytes,
    *,
    sample_rate_hz: int,
) -> None:
    for frame in _pcm_frames(pcm, sample_rate_hz=sample_rate_hz):
        await source.capture_frame(frame)
        await asyncio.sleep(frame.samples_per_channel / sample_rate_hz)


def _frame_to_pcm(frame: object) -> tuple[bytes, int, int]:
    """Extract s16le PCM from a LiveKit AudioFrame / AudioFrameEvent."""

    audio = getattr(frame, "frame", frame)
    data = getattr(audio, "data", b"")
    if hasattr(data, "tobytes"):
        pcm = bytes(data.tobytes())
    elif isinstance(data, (bytes, bytearray, memoryview)):
        pcm = bytes(data)
    else:
        pcm = bytes(data)
    sample_rate = int(getattr(audio, "sample_rate", 48_000))
    channels = int(getattr(audio, "num_channels", getattr(audio, "channels", 1)) or 1)
    return pcm, sample_rate, channels


async def run_caller(
    *,
    url: str,
    api_key: str,
    api_secret: str,
    room_name: str,
    identity: str,
    lines: list[str],
    sample_rate_hz: int,
    voice: str,
    listen_seconds: float,
    speak_delay_seconds: float,
    record_dir: pathlib.Path | None = None,
    tts: str = "gemini",
    gemini_api_key: str | None = None,
    gemini_tts_model: str = DEFAULT_GEMINI_TTS_MODEL,
    gemini_tts_voice: str = DEFAULT_GEMINI_TTS_VOICE,
) -> dict[str, object]:
    tts_backend = tts.strip().lower()
    if tts_backend not in {"gemini", "say"}:
        raise ValueError(f"Unsupported tts backend {tts!r}; use gemini or say.")
    if tts_backend == "gemini" and not gemini_api_key:
        raise RuntimeError(
            "Gemini TTS requires BOT_GEMINI_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY "
            "(or pass --gemini-api-key). Use --tts say for offline macOS speech."
        )

    token = mint_livekit_access_token(
        api_key=api_key,
        api_secret=api_secret,
        identity=identity,
        room=room_name,
        name=identity,
    )
    room = rtc.Room()
    remote_audio_frames = 0
    bot_participants: list[str] = []
    t0 = time.perf_counter()
    timeline: list[dict[str, object]] = []
    outbound_pcm = bytearray()
    inbound_pcm = bytearray()
    # Publish rate may differ from the requested rate when Gemini returns native 24 kHz.
    publish_rate_hz = sample_rate_hz
    inbound_rate = sample_rate_hz
    inbound_channels = 1
    record_root = record_dir
    if record_root is not None:
        record_root.mkdir(parents=True, exist_ok=True)
        (record_root / "turns").mkdir(exist_ok=True)

    def _elapsed_ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    @room.on("participant_connected")
    def _on_participant(participant: rtc.RemoteParticipant) -> None:
        print(f"[caller] remote participant connected: {participant.identity}", flush=True)
        bot_participants.append(participant.identity)
        timeline.append(
            {
                "t_ms": round(_elapsed_ms(), 1),
                "side": "room",
                "event": "participant_connected",
                "identity": participant.identity,
            }
        )

    @room.on("track_subscribed")
    def _on_track(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        del publication
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        print(f"[caller] subscribed audio from {participant.identity}", flush=True)
        timeline.append(
            {
                "t_ms": round(_elapsed_ms(), 1),
                "side": "room",
                "event": "track_subscribed",
                "identity": participant.identity,
                "kind": "audio",
            }
        )

        async def _drain() -> None:
            nonlocal remote_audio_frames, inbound_rate, inbound_channels
            stream = rtc.AudioStream(track)
            async for event in stream:
                remote_audio_frames += 1
                if record_root is not None:
                    try:
                        pcm, rate, channels = _frame_to_pcm(event)
                        if pcm:
                            inbound_pcm.extend(pcm)
                            inbound_rate = rate
                            inbound_channels = channels
                    except Exception as error:  # noqa: BLE001 — recording must not kill media drain
                        if remote_audio_frames == 1:
                            print(f"[caller] record frame extract failed: {error!r}", flush=True)
                if remote_audio_frames == 1 or remote_audio_frames % 100 == 0:
                    print(f"[caller] bot audio frames received: {remote_audio_frames}", flush=True)

        _ = asyncio.create_task(_drain())

    print(f"[caller] connecting identity={identity} room={room_name} url={url}", flush=True)
    await room.connect(url, token)
    print(
        f"[caller] joined; remote_participants={list(room.remote_participants.keys())}",
        flush=True,
    )
    timeline.append(
        {
            "t_ms": round(_elapsed_ms(), 1),
            "side": "room",
            "event": "joined",
            "remote_participants": list(room.remote_participants.keys()),
        }
    )

    # First utterance sets the publish rate (Gemini may be 24 kHz; macOS say uses --sample-rate).
    print(f"[caller] tts_backend={tts_backend}", flush=True)
    if speak_delay_seconds > 0:
        print(f"[caller] waiting {speak_delay_seconds:.1f}s before speaking (ring / answer window)", flush=True)
        await asyncio.sleep(speak_delay_seconds)

    source: rtc.AudioSource | None = None
    for index, line in enumerate(lines, start=1):
        print(f"[caller] speaking ({index}/{len(lines)}): {line!r}", flush=True)
        if tts_backend == "gemini":
            assert gemini_api_key is not None
            pcm, utter_rate = await _gemini_to_pcm(
                line,
                api_key=gemini_api_key,
                model=gemini_tts_model,
                voice_name=gemini_tts_voice or voice,
                sample_rate_hz=sample_rate_hz,
            )
        else:
            pcm = await asyncio.to_thread(_say_to_pcm, line, sample_rate_hz=sample_rate_hz, voice=voice)
            utter_rate = sample_rate_hz
        if source is None:
            publish_rate_hz = utter_rate
            source = rtc.AudioSource(publish_rate_hz, 1)
            track = rtc.LocalAudioTrack.create_audio_track("caller-mic", source)
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            _ = await room.local_participant.publish_track(track, options)
            print(
                f"[caller] published microphone track sample_rate_hz={publish_rate_hz}",
                flush=True,
            )
            timeline.append(
                {
                    "t_ms": round(_elapsed_ms(), 1),
                    "side": "b",
                    "event": "mic_published",
                    "sample_rate_hz": publish_rate_hz,
                    "tts": tts_backend,
                }
            )
        elif utter_rate != publish_rate_hz:
            raise RuntimeError(
                f"TTS sample rate changed mid-call ({utter_rate} != {publish_rate_hz}); refusing to publish."
            )
        peak = _pcm_peak(pcm)
        print(f"[caller] pcm_bytes={len(pcm)} peak={peak} rate={publish_rate_hz}", flush=True)
        turn_start = _elapsed_ms()
        speak_event: dict[str, object] = {
            "t_ms": round(turn_start, 1),
            "side": "b",
            "event": "speak",
            "index": index,
            "text": line,
            "pcm_bytes": len(pcm),
            "peak": peak,
            "duration_s": round(len(pcm) / 2 / publish_rate_hz, 3),
            "tts": tts_backend,
            "sample_rate_hz": publish_rate_hz,
        }
        if record_root is not None:
            outbound_pcm.extend(pcm)
            turn_path = record_root / "turns" / f"b_{index:02d}.wav"
            _write_wav(turn_path, pcm, sample_rate_hz=publish_rate_hz)
            speak_event["wav"] = str(turn_path.name)
        timeline.append(speak_event)
        await _publish_pcm(source, pcm, sample_rate_hz=publish_rate_hz)
        # Brief silence between turns.
        silence = b"\x00" * (publish_rate_hz // 2 * 2)
        if record_root is not None:
            outbound_pcm.extend(silence)
        await _publish_pcm(source, silence, sample_rate_hz=publish_rate_hz)
        await asyncio.sleep(0.5)

    if source is None:
        # No lines: still join and listen with the requested rate.
        source = rtc.AudioSource(sample_rate_hz, 1)
        track = rtc.LocalAudioTrack.create_audio_track("caller-mic", source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        _ = await room.local_participant.publish_track(track, options)
        publish_rate_hz = sample_rate_hz

    frames_before_listen = remote_audio_frames
    print(f"[caller] listening for bot audio up to {listen_seconds:.0f}s…", flush=True)
    timeline.append(
        {
            "t_ms": round(_elapsed_ms(), 1),
            "side": "b",
            "event": "listen_start",
            "listen_seconds": listen_seconds,
            "remote_frames_so_far": frames_before_listen,
        }
    )
    await asyncio.sleep(listen_seconds)
    timeline.append(
        {
            "t_ms": round(_elapsed_ms(), 1),
            "side": "b",
            "event": "listen_end",
            "remote_audio_frames": remote_audio_frames,
            "frames_during_listen": remote_audio_frames - frames_before_listen,
        }
    )
    print(
        f"[caller] done remote_audio_frames={remote_audio_frames} bots_seen={bot_participants}",
        flush=True,
    )
    await room.disconnect()

    result: dict[str, object] = {
        "remote_audio_frames": remote_audio_frames,
        # What B actually heard from A. Frame count alone cannot tell silence from speech: a mute
        # bot still publishes a full track of zeros.
        "inbound_peak": _pcm_peak(bytes(inbound_pcm)),
        "bots_seen": bot_participants,
        "timeline": timeline,
        "record_dir": str(record_root) if record_root is not None else None,
        "tts": tts_backend,
        "sample_rate_hz_outbound": publish_rate_hz,
    }

    if record_root is not None:
        out_wav = record_root / "agent_b_outbound.wav"
        in_wav = record_root / "agent_a_inbound.wav"
        _write_wav(out_wav, bytes(outbound_pcm), sample_rate_hz=publish_rate_hz)
        if inbound_pcm:
            _write_wav(
                in_wav,
                bytes(inbound_pcm),
                sample_rate_hz=inbound_rate,
                channels=inbound_channels,
            )
        else:
            # Still write empty placeholder so the directory layout is stable.
            _write_wav(in_wav, b"", sample_rate_hz=inbound_rate, channels=inbound_channels)
        conversation = {
            "room": room_name,
            "agent_b_identity": identity,
            "tts": tts_backend,
            "sample_rate_hz_outbound": publish_rate_hz,
            "sample_rate_hz_inbound": inbound_rate,
            "channels_inbound": inbound_channels,
            "remote_audio_frames": remote_audio_frames,
            "bots_seen": bot_participants,
            "outbound_pcm_bytes": len(outbound_pcm),
            "inbound_pcm_bytes": len(inbound_pcm),
            "outbound_peak": _pcm_peak(bytes(outbound_pcm)),
            "inbound_peak": _pcm_peak(bytes(inbound_pcm)),
            "outbound_duration_s": (round(len(outbound_pcm) / 2 / publish_rate_hz, 3) if outbound_pcm else 0.0),
            "inbound_duration_s": (
                round(len(inbound_pcm) / 2 / max(inbound_rate * inbound_channels, 1), 3) if inbound_pcm else 0.0
            ),
            "files": {
                "agent_b_outbound": out_wav.name,
                "agent_a_inbound": in_wav.name,
            },
            "timeline": timeline,
        }
        conv_path = record_root / "conversation.json"
        conv_path.write_text(json.dumps(conversation, indent=2) + "\n", encoding="utf-8")
        # Human-readable transcript of known B lines (+ note on A audio).
        lines_txt = [
            "# Dual-agent conversation record (LiveKit blackbox)",
            f"room: {room_name}",
            f"agent_b: {identity}",
            f"remote_audio_frames_from_a: {remote_audio_frames}",
            f"inbound_peak: {conversation['inbound_peak']}",
            f"outbound_peak: {conversation['outbound_peak']}",
            "",
            "## Agent B (spoken lines)",
        ]
        for item in timeline:
            if item.get("event") == "speak":
                lines_txt.append(
                    f"- t={item['t_ms']}ms  B: {item.get('text')!r}  "
                    f"({item.get('duration_s')}s peak={item.get('peak')})"
                )
        lines_txt.extend(
            [
                "",
                "## Agent A (audio only at RTC boundary)",
                f"- Received {remote_audio_frames} audio frames over LiveKit "
                f"({conversation['inbound_duration_s']}s captured, peak={conversation['inbound_peak']}).",
                "- Play agent_a_inbound.wav for what B heard from A.",
                "- Agent A STT text is not on the wire; see phone-bot log / optional scrape.",
                "",
            ]
        )
        (record_root / "conversation.md").write_text("\n".join(lines_txt), encoding="utf-8")
        print(f"[caller] recorded conversation under {record_root}", flush=True)
        result["conversation"] = conversation

    return result


def _looks_like_placeholder_secret(value: str) -> bool:
    lowered = value.strip().casefold()
    if not lowered:
        return True
    return any(
        token in lowered
        for token in (
            "your-",
            "changeme",
            "replace",
            "example",
            "todo",
            "xxx",
            "placeholder",
        )
    )


def _resolve_gemini_api_key(explicit: str | None) -> str | None:
    if explicit and not _looks_like_placeholder_secret(explicit):
        return explicit
    # Prefer example/.env + repo .env (AppConfig) over shell placeholders like GEMINI_API_KEY=your-api-key.
    env_path = _EXAMPLE_ROOT / ".env"
    if env_path.is_file():
        speech_key = AppConfig.from_env_file(env_path).speech.api_key
        if speech_key and not _looks_like_placeholder_secret(speech_key):
            return speech_key
    for name in ("BOT_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value and not _looks_like_placeholder_secret(value):
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LiveKit caller agent that speaks to phone_bot (default: Gemini TTS).",
    )
    _ = parser.add_argument("--url", default="ws://127.0.0.1:7880")
    _ = parser.add_argument("--api-key", default="devkey")
    _ = parser.add_argument("--api-secret", default="secret")
    _ = parser.add_argument("--room", default="bot-phone-bot")
    _ = parser.add_argument("--identity", default="caller-agent")
    _ = parser.add_argument(
        "--tts",
        default="gemini",
        choices=("gemini", "say"),
        help="Speech synthesis backend (default: gemini off-device; say = macOS offline).",
    )
    _ = parser.add_argument(
        "--voice",
        default=None,
        help="Gemini TTS voice (default Kore) or macOS say voice when --tts say (default Samantha).",
    )
    _ = parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Gemini API key (default: BOT_GEMINI_API_KEY / GEMINI_API_KEY / example .env).",
    )
    _ = parser.add_argument(
        "--gemini-tts-model",
        default=os.environ.get("BOT_GEMINI_TTS_MODEL", DEFAULT_GEMINI_TTS_MODEL),
        help="Gemini TTS model id.",
    )
    _ = parser.add_argument(
        "--sample-rate",
        type=int,
        default=24_000,
        help="Target sample rate (Gemini native is 24 kHz; say path may use 48 kHz).",
    )
    _ = parser.add_argument("--speak-delay", type=float, default=3.0)
    _ = parser.add_argument("--listen-seconds", type=float, default=45.0)
    _ = parser.add_argument(
        "--record-dir",
        type=pathlib.Path,
        default=None,
        help="Directory for conversation WAVs + conversation.json (optional).",
    )
    _ = parser.add_argument(
        "--line",
        action="append",
        dest="lines",
        default=None,
        help="Utterance to speak (repeatable). Defaults to a short hello script.",
    )
    args = parser.parse_args()
    lines = (
        list(args.lines)
        if args.lines
        else [
            "Hello, this is the caller agent. Can you hear me?",
            "Please answer the phone and say who you are.",
            "What can you help me with today?",
        ]
    )
    sample_rate = int(args.sample_rate)
    tts_backend = str(args.tts)
    if tts_backend == "say" and sample_rate == 24_000:
        # macOS say path historically used 48 kHz LiveKit frames.
        sample_rate = 48_000
    voice = str(args.voice) if args.voice else ("Samantha" if tts_backend == "say" else DEFAULT_GEMINI_TTS_VOICE)
    result = asyncio.run(
        run_caller(
            url=str(args.url),
            api_key=str(args.api_key),
            api_secret=str(args.api_secret),
            room_name=str(args.room),
            identity=str(args.identity),
            lines=lines,
            sample_rate_hz=sample_rate,
            voice=voice,
            listen_seconds=float(args.listen_seconds),
            speak_delay_seconds=float(args.speak_delay),
            record_dir=args.record_dir,
            tts=tts_backend,
            gemini_api_key=_resolve_gemini_api_key(str(args.gemini_api_key) if args.gemini_api_key else None),
            gemini_tts_model=str(args.gemini_tts_model),
            gemini_tts_voice=voice,
        )
    )
    if result.get("record_dir"):
        print(f"[caller] record_dir={result['record_dir']}", flush=True)


if __name__ == "__main__":
    main()

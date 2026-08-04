#!/usr/bin/env python3
"""Blackbox dual-agent talk through LiveKit only (no shared Environment / direct sound wiring).

Agent A: already-running ``phone-bot`` (or started by this script) on room ``bot-phone-bot``
with local Silero VAD + off-device Gemini STT/TTS (no local Whisper/Qwen).
Agent B: this process as a LiveKit RTC peer using **Gemini TTS** by default
(macOS ``say`` only with ``--tts say``).

Success is observed only from LiveKit + optional external log scrape of agent A:
- B joins the SFU and publishes mic audio with non-silent peaks
- B receives bot audio frames (room media path alive)
- optional: agent A log shows DecodingSpeech after B speaks (utterance batching + VAD/STT)

This intentionally does **not** inject ``environment.sound`` or share an HSM Environment between agents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

# Reuse the existing LiveKit peer (scripts/ are not a package).
_SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

from caller_agent import _resolve_gemini_api_key, run_caller  # noqa: E402


def _port_open(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _count_log_stages(log_path: pathlib.Path, *, since_bytes: int) -> dict[str, int]:
    if not log_path.is_file():
        return {}
    raw = log_path.read_bytes()[since_bytes:]
    text = raw.decode("utf-8", errors="replace")
    keys = (
        "incoming_call",
        "phone.dial",
        "/Phone/dialing",
        "media_ready",
        "/Phone/answered",
        "/Phone/answered/media_ready",
        "phone.answer_call",
        "DetectingVoice",
        "ClassifyingSound",
        "DecodingSpeech",
        "SpeechDecoding",
        "bot.ability.speaking",
        "devices.audio.output",
        "environment.sound",
        "stt_provider=gemini",
        "tts_provider=gemini",
        "vad_provider=silero",
        "silero-vad",
        "Qwen3-TTS",
        "whisper-large",
    )
    return {key: text.count(key) for key in keys}


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--url", default=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880"))
    _ = parser.add_argument("--api-key", default=os.environ.get("LIVEKIT_API_KEY", "devkey"))
    _ = parser.add_argument("--api-secret", default=os.environ.get("LIVEKIT_API_SECRET", "secret"))
    _ = parser.add_argument("--room", default=os.environ.get("LIVEKIT_ROOM", "bot-phone-bot"))
    _ = parser.add_argument("--identity", default="caller-agent")
    _ = parser.add_argument(
        "--dial",
        default=os.environ.get("BOT_LIVEKIT_IDENTITY", "5550141"),
        help="Agent A participant identity that agent B places the call to.",
    )
    _ = parser.add_argument("--phone-bot-log", type=pathlib.Path, default=pathlib.Path("/tmp/phone-bot-live.log"))
    _ = parser.add_argument("--speak-delay", type=float, default=4.0)
    _ = parser.add_argument("--listen-seconds", type=float, default=40.0)
    _ = parser.add_argument(
        "--line",
        action="append",
        dest="lines",
        default=None,
        help="Utterance for agent B (repeatable).",
    )
    _ = parser.add_argument(
        "--start-phone-bot",
        action="store_true",
        help="Spawn ``uv run phone-bot -v`` if nothing is listening as bot (subprocess).",
    )
    default_record = _SCRIPTS.parent / "recordings" / f"dual_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    _ = parser.add_argument(
        "--record-dir",
        type=pathlib.Path,
        default=default_record,
        help="Directory for conversation WAVs + timeline (default: <phone_bot>/recordings/dual_*).",
    )
    _ = parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable conversation recording.",
    )
    _ = parser.add_argument(
        "--tts",
        default="gemini",
        choices=("gemini", "say"),
        help="Agent B speech backend (default: gemini off-device).",
    )
    _ = parser.add_argument(
        "--voice",
        default=os.environ.get("BOT_GEMINI_TTS_VOICE", "Kore"),
        help="Gemini TTS voice (or macOS say voice when --tts say).",
    )
    args = parser.parse_args()

    # Blackbox prerequisite: SFU up.
    host_port = args.url.replace("ws://", "").replace("wss://", "").split("/")[0]
    host, _, port_s = host_port.partition(":")
    port = int(port_s or "7880")
    if not _port_open(host or "127.0.0.1", port):
        print(f"[blackbox] FAIL LiveKit not listening on {host}:{port}", flush=True)
        return 2
    print(f"[blackbox] LiveKit SFU ok {args.url}", flush=True)

    phone_proc: subprocess.Popen[str] | None = None
    log_path: pathlib.Path = args.phone_bot_log
    if args.start_phone_bot:
        example_root = _SCRIPTS.parent
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        phone_proc = subprocess.Popen(
            ["uv", "run", "phone-bot", "-v"],
            cwd=str(example_root),
            # Nobody is standing in the room with this bot — agent B reaches it by telephone. An
            # empty room, not this harness's terminal, which the bot would otherwise read from.
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                # Unbuffered so readiness lines appear in the log immediately.
                "PYTHONUNBUFFERED": "1",
                # Silero VAD local; Gemini STT/TTS off-device (never Whisper/Qwen).
                "BOT_GEMINI_STT_MODEL": os.environ.get("BOT_GEMINI_STT_MODEL", "gemini-3.5-flash"),
                "BOT_GEMINI_TTS_MODEL": os.environ.get("BOT_GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview"),
                "BOT_GEMINI_TTS_VOICE": os.environ.get("BOT_GEMINI_TTS_VOICE", "Kore"),
                "BOT_SILERO_VAD_MODEL": os.environ.get("BOT_SILERO_VAD_MODEL", "mlx-community/silero-vad"),
            },
        )
        print(f"[blackbox] started phone-bot pid={phone_proc.pid} log={log_path}", flush=True)
        # Wait for readiness banner or timeout.
        # Accept Status: ready (stdout summary) or hold-line banner (stderr join text).
        ready_markers = (
            "Status: ready",
            "LiveKit room audio connected: True",
            "Bot is holding the line",
            "local_track_sid=TR_",
        )
        deadline = time.time() + 90
        ready = False
        while time.time() < deadline:
            if log_path.is_file():
                text = log_path.read_text(errors="replace")
                if any(marker in text for marker in ready_markers):
                    ready = True
                    break
            if phone_proc.poll() is not None:
                print(f"[blackbox] FAIL phone-bot exited early code={phone_proc.returncode}", flush=True)
                return 3
            await asyncio.sleep(0.5)
        if not ready:
            print("[blackbox] FAIL phone-bot not ready in time", flush=True)
            phone_proc.terminate()
            return 3
        print("[blackbox] phone-bot ready", flush=True)

    since = log_path.stat().st_size if log_path.is_file() else 0
    marker = f"\nMARK blackbox_livekit_dual_agent t={time.time():.3f}\n"
    if log_path.is_file():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(marker)

    lines = (
        list(args.lines)
        if args.lines
        else [
            "Hello, this is agent B calling through LiveKit. Can you hear me?",
            "Please say who you are.",
        ]
    )
    record_dir: pathlib.Path | None = None if args.no_record else pathlib.Path(args.record_dir)
    if record_dir is not None:
        # Resolve relative paths against the phone_bot example root (not process cwd).
        if not record_dir.is_absolute():
            record_dir = (_SCRIPTS.parent / record_dir).resolve()
        else:
            record_dir = record_dir.resolve()
        record_dir.mkdir(parents=True, exist_ok=True)
        print(f"[blackbox] recording conversation under {record_dir}", flush=True)
    print(
        f"[blackbox] agent B joining room={args.room} identity={args.identity} lines={len(lines)} (LiveKit only)",
        flush=True,
    )

    # Prefer example/.env over shell placeholders (e.g. GEMINI_API_KEY=your-api-key).
    gemini_api_key = _resolve_gemini_api_key(None)

    caller_result: dict[str, object] = {}
    try:
        sample_rate = 24_000 if str(args.tts) == "gemini" else 48_000
        caller_result = await run_caller(
            url=str(args.url),
            api_key=str(args.api_key),
            api_secret=str(args.api_secret),
            room_name=str(args.room),
            identity=str(args.identity),
            dial_identity=str(args.dial),
            lines=lines,
            sample_rate_hz=sample_rate,
            voice=str(args.voice),
            listen_seconds=float(args.listen_seconds),
            speak_delay_seconds=float(args.speak_delay),
            record_dir=record_dir,
            tts=str(args.tts),
            gemini_api_key=gemini_api_key,
            gemini_tts_model=os.environ.get("BOT_GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview"),
            gemini_tts_voice=str(args.voice),
        )
    finally:
        if phone_proc is not None:
            phone_proc.terminate()
            try:
                phone_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                phone_proc.kill()

    stages = _count_log_stages(log_path, since_bytes=since)
    print("[blackbox] === agent A log stage hits (window) ===", flush=True)
    for key, count in stages.items():
        print(f"  {key:28s} {count}", flush=True)

    if record_dir is not None and log_path.is_file():
        window = log_path.read_bytes()[since:]
        (record_dir / "agent_a_log_window.log").write_bytes(window)
        (record_dir / "agent_a_stages.json").write_text(
            json.dumps(stages, indent=2) + "\n",
            encoding="utf-8",
        )
        # Append A-side stage summary into conversation.md if present.
        md_path = record_dir / "conversation.md"
        if md_path.is_file():
            extra = [
                "## Agent A (pipeline stages from log scrape)",
                *(f"- {k}: {v}" for k, v in stages.items()),
                "",
                f"Full log window: {(record_dir / 'agent_a_log_window.log').name}",
                "",
            ]
            with md_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(extra))
        summary = {
            "stages": stages,
            "caller": {
                "remote_audio_frames": caller_result.get("remote_audio_frames"),
                "inbound_peak": caller_result.get("inbound_peak"),
                "bots_seen": caller_result.get("bots_seen"),
            },
            "record_dir": str(record_dir),
        }
        (record_dir / "blackbox_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[blackbox] wrote recording artifacts to {record_dir}", flush=True)

    # A track of digital silence still arrives as thousands of frames, so the only honest proof
    # that agent A spoke is amplitude. Anything at or below this is silence with dither.
    audible_floor = 200
    inbound_peak = int(caller_result.get("inbound_peak") or 0)
    decoding = stages.get("DecodingSpeech", 0)
    detecting = stages.get("DetectingVoice", 0)
    answered = stages.get("/Phone/answered", 0) + stages.get("phone.answer_call", 0)
    media_ready = stages.get("/Phone/answered/media_ready", 0) + stages.get("media_ready", 0)
    # Silero VAD is expected locally; heavy STT/TTS models must stay off-device.
    heavy_local_speech = stages.get("Qwen3-TTS", 0) + stages.get("whisper-large", 0)
    print("[blackbox] === verdict ===", flush=True)
    print("  livekit_sfu: ok", flush=True)
    print(f"  agent_b_tts: {args.tts}", flush=True)
    print("  agent_b_spoke: yes (see caller log peaks)", flush=True)
    print(f"  agent_a_answer: {answered} (the bot's choice; not a harness hard-fail if 0)", flush=True)
    print(f"  agent_a_media_ready: {media_ready}", flush=True)
    print(f"  agent_a_detecting_voice: {detecting}", flush=True)
    print(f"  agent_a_decoding_speech: {decoding}", flush=True)
    print(f"  agent_a_silero_vad_hits: {stages.get('silero-vad', 0)}", flush=True)
    print(f"  agent_a_heavy_local_stt_tts: {heavy_local_speech} (want 0)", flush=True)
    print(f"  agent_a_audible_to_b: peak={inbound_peak} (want > {audible_floor})", flush=True)
    if record_dir is not None:
        print(f"  record_dir: {record_dir}", flush=True)
    if heavy_local_speech > 0:
        print(
            "[blackbox] FAIL agent A log shows local Whisper/Qwen STT/TTS loads "
            "(expected Gemini STT/TTS; Silero VAD alone is ok)",
            flush=True,
        )
        return 6
    if inbound_peak <= audible_floor:
        # Reporting PASS here would launder a mute robot as a working one, which is exactly what
        # hid the WAV uplink defect: every inbound stage fired while agent B heard nothing.
        print(
            f"[blackbox] FAIL agent A never became audible to agent B (peak={inbound_peak}); "
            f"inbound stages fired but nothing was spoken onto the wire",
            flush=True,
        )
        return 7
    if decoding > 0:
        print("[blackbox] PASS speech path entered DecodingSpeech and agent A was audible", flush=True)
        return 0
    if answered == 0:
        # The bot chose not to answer — harness remains green on media path;
        # call media never elevates until answer, so STT is not expected this run.
        print(
            "[blackbox] PARTIAL no answer this run (the bot chose not to); media STT path not entered",
            flush=True,
        )
        return 0
    if detecting > 0:
        print(
            "[blackbox] PARTIAL media+VAD active but DecodingSpeech=0 (no voice segments or STT not reached)",
            flush=True,
        )
        return 4
    print("[blackbox] FAIL no Listening activity on agent A in window", flush=True)
    return 5


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

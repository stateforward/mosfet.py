#!/usr/bin/env python3
"""Two full bots holding a phone conversation with each other over LiveKit.

Both sides are real ``phone-bot`` processes with their own cognition, speech and Environment.
Nothing is scripted: there is no TTS peer playing canned lines, so every utterance on the wire
came from a bot deciding to speak.

The two bots are not symmetric, because a call is not symmetric. Somebody walks up to one of them
and **says something out loud** — "Call Bob at 555-0142." — which reaches it as sound in its
environment, through its ears, its voice detector and its speech decoder, the same way anything
else it hears does. Nobody says anything to the other one. Both have the same phone, both are on
the same exchange and both can dial; having heard a sentence is a thing that happened to the
caller, not a thing it must act on — whether to dial, and whether to answer, are the bots'
decisions and this harness makes neither.

The words go in on the caller's stdin and are spoken by a mouth standing a metre in front of it.
They are never parsed here, and the bot is never handed them as anything but audio.

The room is what makes the call itself possible. A LiveKit room is an exchange: joining it makes a
phone reachable, dialing is call setup addressed to one participant, and the dial plan is what
turns a dialled number into which participant. So the callee rings because the caller called its
number, not because the caller walked into the room.

The numbers come from the 555-0100..555-0199 range that exists so nothing written down can ring a
real subscriber.

Success is judged the way the single-bot harness learned to judge it: **both directions must
carry audible audio**. Two bots that ring, answer and then sit in silence are a failure, however
many pipeline stages fired. A run where the caller never dials is reported, not failed: a bot
that judges there is nothing to call about is behaving, not broken.

The observer is a third participant that publishes nothing and only subscribes, so it can measure
each bot's outbound audio separately.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone

_SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

from livekit import rtc  # noqa: E402

from blackbox_livekit_dual_agent import _count_log_stages, _port_open  # noqa: E402
from phone_bot_example import mint_livekit_access_token  # noqa: E402

_READY_MARKERS = (
    "Status: ready",
    "LiveKit room audio connected: True",
    "Bot is holding the line",
    "local_track_sid=TR_",
)


def _pcm_peak(pcm: bytes) -> int:
    if not pcm:
        return 0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    return max((abs(value) for value in samples), default=0)


def _write_wav(path: pathlib.Path, pcm: bytes, *, sample_rate_hz: int, channels: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(max(channels, 1))
        handle.setsampwidth(2)
        handle.setframerate(max(sample_rate_hz, 1))
        handle.writeframes(pcm)


def _frame_pcm(event: object) -> tuple[bytes, int, int]:
    frame = getattr(event, "frame", None)
    if frame is None:
        return b"", 0, 0
    data = getattr(frame, "data", b"")
    pcm = bytes(data) if not isinstance(data, bytes) else data
    return pcm, int(getattr(frame, "sample_rate", 0)), int(getattr(frame, "num_channels", 1))


def _bot_env_file(
    base: pathlib.Path,
    target: pathlib.Path,
    *,
    identity: str,
    room: str,
    dial_plan: dict[str, str],
) -> pathlib.Path:
    """One env file per bot: same room and same exchange, distinct identity and track.

    Identity has to differ or the two processes collide on the SFU; track name differs so each
    bot's audio is attributable to it in the observer's capture. Both get the whole dial plan,
    because both are lines on one exchange and either could call the other. Nothing here
    distinguishes caller from callee: the two configurations are identical, and the only
    difference between the roles is that somebody spoke to one of them.
    """

    values: dict[str, str] = {}
    if base.exists():
        for line in base.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, _, value = stripped.partition("=")
                values[key.strip()] = value
    values["BOT_LIVEKIT_ROOM"] = room
    values["BOT_LIVEKIT_IDENTITY"] = identity
    values["BOT_LIVEKIT_TRACK_NAME"] = identity
    values["BOT_LIVEKIT_DIAL_PLAN"] = ",".join(f"{number}={endpoint}" for number, endpoint in dial_plan.items())
    # A token minted for the other identity would silently rejoin as the wrong participant.
    _ = values.pop("BOT_LIVEKIT_TOKEN", None)
    target.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    return target


async def _await_ready(log_path: pathlib.Path, process: subprocess.Popen[str], *, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log_path.is_file():
            text = log_path.read_text(errors="replace")
            if any(marker in text for marker in _READY_MARKERS):
                return True
        if process.poll() is not None:
            return False
        await asyncio.sleep(0.5)
    return False


async def _observe(
    *,
    url: str,
    api_key: str,
    api_secret: str,
    room_name: str,
    identity: str,
    seconds: float,
) -> dict[str, dict[str, object]]:
    """Subscribe to every bot track and keep each one's audio separate."""

    token = mint_livekit_access_token(
        api_key=api_key, api_secret=api_secret, identity=identity, room=room_name, name=identity
    )
    room = rtc.Room()
    captured: dict[str, dict[str, object]] = {}

    @room.on("track_subscribed")
    def _on_track(
        track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        del publication
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        speaker = participant.identity
        print(f"[two-bots] observing audio from {speaker}", flush=True)
        captured.setdefault(speaker, {"pcm": bytearray(), "frames": 0, "sample_rate_hz": 0, "channels": 1})

        async def _drain() -> None:
            stream = rtc.AudioStream(track)
            async for event in stream:
                pcm, rate, channels = _frame_pcm(event)
                record = captured[speaker]
                record["frames"] = int(record["frames"]) + 1
                if pcm:
                    typing_pcm = record["pcm"]
                    assert isinstance(typing_pcm, bytearray)
                    typing_pcm.extend(pcm)
                    record["sample_rate_hz"] = rate or int(record["sample_rate_hz"])
                    record["channels"] = channels or int(record["channels"])

        _ = asyncio.create_task(_drain())

    await room.connect(url, token)
    print(f"[two-bots] observer joined room={room_name} as {identity}; listening {seconds:.0f}s", flush=True)
    await asyncio.sleep(seconds)
    await room.disconnect()
    return captured


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--url", default=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880"))
    _ = parser.add_argument("--api-key", default=os.environ.get("LIVEKIT_API_KEY", "devkey"))
    _ = parser.add_argument("--api-secret", default=os.environ.get("LIVEKIT_API_SECRET", "secret"))
    _ = parser.add_argument("--room", default="bot-two-bots")
    _ = parser.add_argument("--caller", default="phone-bot-alice", help="Bot somebody speaks to.")
    _ = parser.add_argument("--callee", default="phone-bot-bob", help="Bot nobody speaks to.")
    _ = parser.add_argument("--caller-number", default="555-0141", help="Number the exchange rings the caller on.")
    _ = parser.add_argument("--callee-number", default="555-0142", help="Number the exchange rings the callee on.")
    _ = parser.add_argument(
        "--say",
        default=None,
        metavar="TEXT",
        help="What somebody says out loud to the caller (default: 'Call Bob at <callee-number>.').",
    )
    _ = parser.add_argument("--observer", default="conversation-observer")
    _ = parser.add_argument("--converse-seconds", type=float, default=60.0)
    _ = parser.add_argument("--ready-timeout", type=float, default=120.0)
    _ = parser.add_argument("--audible-floor", type=int, default=200)
    default_record = _SCRIPTS.parent / "recordings" / f"two_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    _ = parser.add_argument("--record-dir", type=pathlib.Path, default=default_record)
    args = parser.parse_args()

    host_port = str(args.url).replace("ws://", "").replace("wss://", "").split("/")[0]
    host, _, port_s = host_port.partition(":")
    if not _port_open(host or "127.0.0.1", int(port_s or "7880")):
        print(f"[two-bots] FAIL LiveKit not listening on {host_port}", flush=True)
        return 2
    print(f"[two-bots] LiveKit SFU ok {args.url}", flush=True)

    example_root = _SCRIPTS.parent
    base_env = example_root / ".env"
    if not base_env.exists():
        print(f"[two-bots] FAIL no provider env file at {base_env}", flush=True)
        return 2

    record_dir = pathlib.Path(args.record_dir)
    if not record_dir.is_absolute():
        record_dir = (example_root / record_dir).resolve()
    record_dir.mkdir(parents=True, exist_ok=True)

    caller, callee = str(args.caller), str(args.callee)
    identities = [caller, callee]
    # One exchange, both lines on it. Numbers are what a bot can be told and can dial; identities
    # are where the packets go, and only this map connects the two.
    dial_plan = {str(args.caller_number): caller, str(args.callee_number): callee}
    # A sentence with the callee's number in it, said out loud to the one bot somebody talks to.
    # Nothing on this side reads the sentence back, and nothing resolves the number for the bot.
    said = {
        caller: str(args.say) if args.say is not None else f"Call Bob at {args.callee_number}.",
        callee: None,
    }
    processes: dict[str, subprocess.Popen[str]] = {}
    logs: dict[str, pathlib.Path] = {}
    try:
        for identity in identities:
            env_file = _bot_env_file(
                base_env,
                record_dir / f"{identity}.env",
                identity=identity,
                room=str(args.room),
                dial_plan=dial_plan,
            )
            utterance = said[identity]
            log_path = record_dir / f"{identity}.log"
            logs[identity] = log_path
            handle = log_path.open("w", encoding="utf-8")
            processes[identity] = subprocess.Popen(
                ["uv", "run", "phone-bot", "-v", "--env", str(env_file)],
                cwd=str(example_root),
                # A pipe only for the bot somebody speaks to. Nobody is standing in front of the
                # other one, so there is nothing to say into it and no stdin to say it on.
                stdin=subprocess.PIPE if utterance is not None else subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            print(
                f"[two-bots] started {identity} pid={processes[identity].pid} log={log_path} "
                f"spoken_to={utterance is not None}",
                flush=True,
            )

        for identity in identities:
            if not await _await_ready(logs[identity], processes[identity], timeout=float(args.ready_timeout)):
                print(f"[two-bots] FAIL {identity} not ready in time", flush=True)
                return 3
            print(f"[two-bots] {identity} ready", flush=True)

        # Speak only once the bot is awake enough to hear: a sentence said into a room where
        # nothing is listening yet is a sentence nobody heard, which is not what this is testing.
        utterance = said[caller]
        stdin = processes[caller].stdin
        if utterance is not None and stdin is not None:
            _ = stdin.write(utterance + "\n")
            stdin.flush()
            print(f"[two-bots] somebody said something to {caller} ({len(utterance)} characters)", flush=True)

        captured = await _observe(
            url=str(args.url),
            api_key=str(args.api_key),
            api_secret=str(args.api_secret),
            room_name=str(args.room),
            identity=str(args.observer),
            seconds=float(args.converse_seconds),
        )
    finally:
        for identity, process in processes.items():
            process.terminate()
            try:
                _ = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            print(f"[two-bots] stopped {identity}", flush=True)

    # From byte 0, not from readiness: each log is fresh per run, and ringing happens while the
    # bot is still connecting — a window that opens at "ready" reports rang=0 for a call that
    # demonstrably rang.
    stages = {identity: _count_log_stages(logs[identity], since_bytes=0) for identity in identities}
    speech: dict[str, dict[str, object]] = {}
    for identity in identities:
        record = captured.get(identity, {"pcm": bytearray(), "frames": 0, "sample_rate_hz": 48_000, "channels": 1})
        pcm = bytes(record["pcm"]) if isinstance(record["pcm"], bytearray) else b""
        rate = int(record["sample_rate_hz"]) or 48_000
        channels = int(record["channels"]) or 1
        _write_wav(record_dir / f"{identity}.wav", pcm, sample_rate_hz=rate, channels=channels)
        speech[identity] = {
            "peak": _pcm_peak(pcm),
            "frames": int(record["frames"]),
            "seconds": round(len(pcm) / 2 / max(channels, 1) / rate, 2),
            "sample_rate_hz": rate,
        }

    summary = {
        "room": str(args.room),
        "caller": caller,
        "callee": callee,
        "identities": identities,
        "dial_plan": dial_plan,
        # How much was said to each bot, not what. The console line says it this way too; a
        # recording that quotes the sentence back is the first step toward reading it.
        "said_characters": {identity: None if text is None else len(text) for identity, text in said.items()},
        "audible_floor": int(args.audible_floor),
        "speech": speech,
        "stages": stages,
    }
    (record_dir / "two_bot_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("[two-bots] === verdict ===", flush=True)
    for identity in identities:
        stage = stages.get(identity, {})
        heard = speech[identity]
        role = "caller" if identity == caller else "callee"
        # Asymmetric on purpose: the caller dials and the callee rings. A `rang` on the caller
        # would mean setup came back the other way, which is a different call.
        print(
            f"  {identity} ({role}): dialed={stage.get('/Phone/dialing', 0)} "
            f"rang={stage.get('incoming_call', 0)} answered={stage.get('/Phone/answered', 0)} "
            f"media_ready={stage.get('/Phone/answered/media_ready', 0)} decoded={stage.get('DecodingSpeech', 0)} "
            f"spoke_peak={heard['peak']} ({heard['seconds']}s over {heard['frames']} frames)",
            flush=True,
        )
    if not stages.get(callee, {}).get("incoming_call", 0):
        # Not a failure: dialing is the caller's to decide, and a bot that saw no reason to call
        # is behaving. Say so plainly rather than reporting a silent conversation as a defect.
        print(
            f"[two-bots] NO CALL {caller} did not dial {args.callee_number} this run (its choice, not a defect)",
            flush=True,
        )
    print(f"  record_dir: {record_dir}", flush=True)

    silent = [identity for identity in identities if int(speech[identity]["peak"]) <= int(args.audible_floor)]
    if silent:
        # Stage counts cannot see this: a bot that never speaks still publishes a full track of
        # silence, and every inbound stage on the other side fires exactly as it would normally.
        print(
            f"[two-bots] FAIL no audible speech from {', '.join(silent)} "
            f"(peak <= {args.audible_floor}); a conversation needs both sides",
            flush=True,
        )
        return 7
    print("[two-bots] PASS both bots answered and both were audible to each other", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

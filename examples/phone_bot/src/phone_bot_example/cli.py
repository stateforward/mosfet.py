"""Turnkey phone bot: ensure local LiveKit, mint token if needed, run the bot."""

from __future__ import annotations

import argparse
import asyncio
import collections.abc
import json
import logging
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import typing

from phone_bot_example import AppConfig, mint_livekit_access_token, run


def _configure_logging(*, verbose: bool) -> None:
    """Emit HSM observe logs to stderr while holding the line."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    # Machines that already opt into hsm.observe(observer) (phone, livekit, bot body).
    logging.getLogger("bot.telemetry.hsm").setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.getLogger("phone_bot_example.hsm").setLevel(logging.INFO)


def _human_join_instructions(*, url: str, room: str, api_key: str, api_secret: str) -> str:
    import urllib.parse

    token = mint_livekit_access_token(
        api_key=api_key,
        api_secret=api_secret,
        identity="human",
        room=room,
        name="human",
    )
    # Meet /custom reads liveKitUrl + token from the query string (no form fields).
    meet_url = "https://meet.livekit.io/custom?" + urllib.parse.urlencode(
        {"liveKitUrl": url, "token": token},
    )
    return (
        "\nYou are the other end of the phone.\n"
        f"Open this link (local LiveKit only works from this machine):\n"
        f"  {meet_url}\n"
        "Allow the microphone, then talk.\n"
        f"Room: {room}  URL: {url}\n"
        "Bot is holding the line (Ctrl+C to hang up).\n"
    )


def _render_summary(summary: dict[str, object]) -> str:
    warnings = "\n".join(f"- {item}" for item in typing.cast(list[str], summary["warnings"]))
    return (
        "stateforward.bot phone bot\n"
        f"Status: {summary['status']}\n"
        f"Bot state: {summary['bot_state']}\n"
        f"Cognition: {summary['cognition_client']}\n"
        f"LiveKit room audio connected: {summary['livekit_room_audio_connected']}\n"
        f"Warnings:\n{warnings or '- none'}\n"
    )


_EXAMPLE_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_URL = "ws://127.0.0.1:7880"
_DEFAULT_API_KEY = "devkey"
_DEFAULT_API_SECRET = "secret"
_DEFAULT_ROOM = "bot-phone-bot"
_DEFAULT_IDENTITY = "bot-phone-bot"
_DEFAULT_TRACK = "bot-phone-bot"


def _example_root() -> pathlib.Path:
    return _EXAMPLE_ROOT


def _port_open(host: str, port: int, *, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_env_file(path: pathlib.Path) -> pathlib.Path:
    if path.exists():
        return path
    example = path.parent / ".env.example"
    if example.exists():
        print(f"No {path.name} found; copying .env.example -> {path.name}", file=sys.stderr)
        _ = path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        return path
    # Minimal local SFU defaults when neither file exists.
    path.write_text(
        "\n".join(
            [
                f"BOT_LIVEKIT_URL={_DEFAULT_URL}",
                f"BOT_LIVEKIT_API_KEY={_DEFAULT_API_KEY}",
                f"BOT_LIVEKIT_API_SECRET={_DEFAULT_API_SECRET}",
                f"BOT_LIVEKIT_ROOM={_DEFAULT_ROOM}",
                f"BOT_LIVEKIT_IDENTITY={_DEFAULT_IDENTITY}",
                f"BOT_LIVEKIT_TRACK_NAME={_DEFAULT_TRACK}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Created {path} with local LiveKit dev defaults.", file=sys.stderr)
    return path


def _ensure_livekit(*, host: str = "127.0.0.1", port: int = 7880) -> None:
    if _port_open(host, port):
        print(f"LiveKit already listening on {host}:{port}", file=sys.stderr)
        return
    if shutil.which("docker") is None:
        raise RuntimeError(
            "LiveKit is not running on :7880 and docker is not available. "
            "Start LiveKit with `docker compose up -d` or `livekit-server --dev`."
        )
    compose = _example_root() / "docker-compose.yml"
    if not compose.exists():
        raise RuntimeError(f"Missing {compose}")
    print("Starting LiveKit via docker compose…", file=sys.stderr)
    _ = subprocess.run(
        ["docker", "compose", "-f", str(compose), "up", "-d"],
        cwd=_example_root(),
        check=True,
    )
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if _port_open(host, port):
            print(f"LiveKit ready on {host}:{port}", file=sys.stderr)
            return
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for LiveKit on {host}:{port}")


def _resolved_livekit_env(config: AppConfig) -> dict[str, str]:
    """Build env keys for LiveKit join, minting a token when only api_key/secret are set."""

    livekit = config.livekit
    url = livekit.url or os.environ.get("BOT_LIVEKIT_URL") or _DEFAULT_URL
    api_key = livekit.api_key or os.environ.get("BOT_LIVEKIT_API_KEY") or _DEFAULT_API_KEY
    api_secret = livekit.api_secret or os.environ.get("BOT_LIVEKIT_API_SECRET") or _DEFAULT_API_SECRET
    room = livekit.room or _DEFAULT_ROOM
    identity = livekit.identity or _DEFAULT_IDENTITY
    track_name = livekit.track_name or _DEFAULT_TRACK
    token = livekit.token or os.environ.get("BOT_LIVEKIT_TOKEN")
    if not token:
        print(
            f"Minting LiveKit token (key={api_key}, room={room}, identity={identity})",
            file=sys.stderr,
        )
        token = mint_livekit_access_token(
            api_key=api_key,
            api_secret=api_secret,
            identity=identity,
            room=room,
        )
    return {
        "BOT_LIVEKIT_URL": url,
        "BOT_LIVEKIT_TOKEN": token,
        "BOT_LIVEKIT_API_KEY": api_key,
        "BOT_LIVEKIT_API_SECRET": api_secret,
        "BOT_LIVEKIT_ROOM": room,
        "BOT_LIVEKIT_IDENTITY": identity,
        "BOT_LIVEKIT_TRACK_NAME": track_name,
    }


def _merge_env_file(base: pathlib.Path, overrides: collections.abc.Mapping[str, str]) -> pathlib.Path:
    lines: list[str] = []
    if base.exists():
        lines.append(base.read_text(encoding="utf-8").rstrip())
        lines.append("")
    for key, value in overrides.items():
        lines.append(f"{key}={value}")
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".env",
        prefix="phone-bot-",
        delete=False,
    )
    with handle as stream:
        stream.write("\n".join(lines) + "\n")
    return pathlib.Path(handle.name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Turnkey phone bot: start local LiveKit if needed, mint token, join the room.",
    )
    _ = parser.add_argument(
        "--env",
        default=str(_example_root() / ".env"),
        help="Provider env file (default: examples/phone_bot/.env).",
    )
    _ = parser.add_argument("--json", action="store_true", help="Print JSON summary and exit (smoke).")
    _ = parser.add_argument(
        "--reasoning-model",
        default=None,
        help="Override OpenAI Terra reasoning model (default gpt-5.6-terra).",
    )
    _ = parser.add_argument(
        "--once",
        action="store_true",
        help="Connect, print readiness, and exit (do not hold the line).",
    )
    _ = parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG HSM observe logs (phone, livekit, bot) via hsm.observe.",
    )
    _ = parser.add_argument(
        "--skip-livekit-start",
        action="store_true",
        help="Do not docker-compose up LiveKit (still joins if URL is reachable).",
    )
    args = parser.parse_args(argv)

    # Default: hold the line so a human can join. --json / --once are smoke-only.
    hold = not bool(args.json) and not bool(args.once)
    # Holding always enables bot/media logs; -v adds DEBUG on all hsm.observe(observer) machines.
    if hold or bool(args.verbose):
        _configure_logging(verbose=bool(args.verbose))

    env_path = _ensure_env_file(pathlib.Path(str(args.env)).expanduser())
    if not bool(args.skip_livekit_start):
        _ensure_livekit()

    base_config = AppConfig.from_env_file(env_path)
    base_config = base_config.with_cognition_overrides(
        model=typing.cast(str | None, args.reasoning_model),
    )
    livekit_env = _resolved_livekit_env(base_config)
    merged_env = _merge_env_file(env_path, livekit_env)

    def _on_ready(summary: dict[str, object]) -> None:
        if bool(args.json):
            print(json.dumps(summary, indent=2))
        else:
            print(_render_summary(summary))
        if hold:
            print(
                _human_join_instructions(
                    url=livekit_env["BOT_LIVEKIT_URL"],
                    room=livekit_env["BOT_LIVEKIT_ROOM"],
                    api_key=livekit_env["BOT_LIVEKIT_API_KEY"],
                    api_secret=livekit_env["BOT_LIVEKIT_API_SECRET"],
                ),
                file=sys.stderr,
            )

    async def _run_session() -> dict[str, object]:
        config = AppConfig.from_env_file(merged_env)
        config = config.with_cognition_overrides(
            model=typing.cast(str | None, args.reasoning_model),
        )
        if not config.livekit.can_connect_room():
            raise RuntimeError("LiveKit URL/token still missing after phone-bot setup.")
        print("Connecting phone bot to LiveKit…", file=sys.stderr)
        return await run(config, connect_livekit=True, hold=hold, on_ready=_on_ready)

    try:
        try:
            summary = asyncio.run(_run_session())
        except KeyboardInterrupt:
            print("\nHanging up.", file=sys.stderr)
            raise SystemExit(0) from None
    finally:
        try:
            merged_env.unlink(missing_ok=True)
        except OSError:
            pass

    if not bool(summary.get("livekit_room_audio_connected")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

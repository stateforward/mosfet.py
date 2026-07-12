"""Console entry for minting LiveKit access tokens."""

from __future__ import annotations

import argparse
import os

from phone_bot_example import mint_livekit_access_token


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mint a LiveKit join JWT (HS256).")
    _ = parser.add_argument("--api-key", default=os.environ.get("BOT_LIVEKIT_API_KEY", "devkey"))
    _ = parser.add_argument("--api-secret", default=os.environ.get("BOT_LIVEKIT_API_SECRET", "secret"))
    _ = parser.add_argument("--room", default=os.environ.get("BOT_LIVEKIT_ROOM", "bot-phone-bot"))
    _ = parser.add_argument(
        "--identity",
        default=os.environ.get("BOT_LIVEKIT_IDENTITY", "bot-phone-bot"),
    )
    _ = parser.add_argument("--name", default=None)
    _ = parser.add_argument("--ttl-seconds", type=int, default=6 * 60 * 60)
    args = parser.parse_args(argv)
    print(
        mint_livekit_access_token(
            api_key=str(args.api_key),
            api_secret=str(args.api_secret),
            identity=str(args.identity),
            room=str(args.room),
            name=None if args.name is None else str(args.name),
            ttl_seconds=int(args.ttl_seconds),
        )
    )


if __name__ == "__main__":
    main()
    raise SystemExit(0)

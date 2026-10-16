#!/usr/bin/env python3
"""Lab listen check (manual, never CI): render each expression the lab declares into a WAV to listen to.

For every sound in ``LAB_EXPRESSIVENESS``, one two-part reply with the sound before its part's words and one with it
after them, plus one two-part reply per declared style (whisper). Each is said in one voice, through exactly the
encoder the bot speaks with on the lab (``LabVoice``), and written to ``<out>/<line>.wav``. Whether each sounds
right is for a person to hear: nothing here judges the audio, and it gates nothing. It follows anygame's
``deploy/lab/bench/vocal_tags_check.py``.

    uv run --project examples/listen_speak_bot python examples/listen_speak_bot/scripts/listen_check.py \\
        --url http://192.168.2.32:8002 --voice warm

``--url`` defaults to ``BOT_SPEECH_SERVER_TTS_URL``. ``--voice`` names a voice the engine holds; ``--description``
is the description it was made from, written before any direction as the bot's own encoder writes it.
"""

from __future__ import annotations

import argparse
import asyncio
import collections.abc
import os
import pathlib
import sys

from mosfet.abilities import vocal

from listen_speak_bot_example import LAB_EXPRESSIVENESS, LabVoice

_OUT = pathlib.Path(__file__).resolve().parents[1] / "assets" / "listen_check"


def replies() -> dict[str, tuple[vocal.speech.Part, ...]]:
    """Each line to render, by name: every declared sound before and after its words, then every declared style."""

    lines: dict[str, tuple[vocal.speech.Part, ...]] = {}
    for sound in sorted(LAB_EXPRESSIVENESS.sounds):
        for at in ("before", "after"):
            lines[f"{sound}_{at}"] = (
                vocal.speech.Part(
                    words="Oh, you got me. ", expression=vocal.speech.Expression(sound=sound, sound_at=at)
                ),
                vocal.speech.Part(words="I did not see that coming."),
            )
    for style in LAB_EXPRESSIVENESS.styles:
        lines[style] = (
            vocal.speech.Part(words="Can you keep a secret? ", expression=vocal.speech.Expression(style=style)),
            vocal.speech.Part(words="I knew you could."),
        )
    return lines


async def _pieces(parts: tuple[vocal.speech.Part, ...]) -> collections.abc.AsyncIterator[vocal.speech.Piece]:
    for index, part in enumerate(parts):
        yield vocal.speech.Piece(part=index, words=part.words, expression=part.expression)


async def render(
    encoder: collections.abc.Callable[[pathlib.Path], vocal.speech.SpeechEncoder], out: pathlib.Path
) -> list[pathlib.Path]:
    """Say every line with an encoder that writes what it says to the line's WAV, one line at a time, in order."""

    out.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for name, parts in replies().items():
        destination = out / f"{name}.wav"
        async for _ in encoder(destination).encode(_pieces(parts)):
            pass
        written.append(destination)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render each lab sound and style to a WAV for a listen check.")
    _ = parser.add_argument("--url", default=os.environ.get("BOT_SPEECH_SERVER_TTS_URL"), help="Lab TTS engine root.")
    _ = parser.add_argument("--voice", required=True, help="A voice the engine holds.")
    _ = parser.add_argument("--description", default=None, help="The description that voice was made from.")
    _ = parser.add_argument("--out", type=pathlib.Path, default=_OUT, help=f"Where the WAVs go (default: {_OUT}).")
    args = parser.parse_args(argv)
    url: str | None = args.url
    if not url:
        parser.error("--url (or BOT_SPEECH_SERVER_TTS_URL) is required.")
    voice: str = args.voice
    description: str | None = args.description
    out: pathlib.Path = args.out

    def encoder(destination: pathlib.Path) -> vocal.speech.SpeechEncoder:
        return LabVoice(url, destination=destination)(voice, description)

    try:
        written = asyncio.run(render(encoder, out))
    except Exception as error:
        print(f"listen check failed: {error}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

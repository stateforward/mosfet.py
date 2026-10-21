"""The listen_speak bot file: ``python -m listen_speak_bot_example`` runs the bot once on this Mac.

Reads its options (``--play``, ``--json``, ``--env``, ``--gemini-model``, ``--trait``, ``--assets-dir``,
``--verbose``), then the owner's :class:`~listen_speak_bot_example.Settings` from the process environment, with the
env file's assignments winning (``--env``, or the example's own ``.env`` when it exists). A bad option exits ``2``;
a missing or malformed setting exits ``1``, naming it, before anything is rendered or started. It needs macOS
``say`` and ``afconvert``.

A few parts come first, as variables. The ear, which the mouth tells what it is saying, so the bot can learn its own
voice. The conversation, shared by communication and the mouth. The mouth, the one ``speaking.Speaking`` the bot
speaks through. And the MLX Audio workers the run closes when it ends: Whisper, and off the lab Silero.

Then one ``bot.define`` constructs every other part inline: the bot's traits, on the lab the source of the voice it is
born with, its cognition on Gemini with its reflex on a decision model (on the decision server
``DECISION_BASE_URL`` names) and the speech seed that takes heard speech into a conversation, its ear, its
mouth and its conversations. :func:`~listen_speak_bot_example.hear` plays the recording and then half a second of
silence to it, and the run's summary is printed: exit ``0`` when the bot said something, ``1`` otherwise.
"""

import argparse
import asyncio
import datetime
import functools
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import typing

import pydantic
from mosfet import bot
from mosfet.abilities import communication, decoding, identification, listening, memory, persona, speaking
from mosfet.abilities.communication import conversation
from mosfet.abilities.communication.conversation import turn_detector
from mosfet.environment import SoundData
from mosfet.providers import decision
from mosfet.providers import gemini
from mosfet.providers import mlx_audio
from mosfet.providers import pyannote
from mosfet.providers import speech_server

import listen_speak_bot_example as example

if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        prog="python -m listen_speak_bot_example",
        description="Play one recording to a bot with no devices, and report what it did (macOS say).",
    )
    _ = parser.add_argument("--play", action="store_true", help="Play the heard WAV, and the reply, with afplay.")
    _ = parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    _ = parser.add_argument(
        "--env", type=pathlib.Path, default=None, help=f"An env file (default: {root / '.env'} if it exists)."
    )
    _ = parser.add_argument("--gemini-model", default=None, help="The Gemini model, over BOT_GEMINI_MODEL.")
    _ = parser.add_argument(
        "--trait",
        action="append",
        default=None,
        help="A trait the owner gives the bot; repeat for each (default: "
        + ", ".join(example.DEFAULT_TRAITS.traits)
        + ").",
    )
    _ = parser.add_argument(
        "--assets-dir",
        type=pathlib.Path,
        default=root / "assets",
        help="Where hey_gabe.wav, reply.wav and the bot's memory.sqlite3 are kept.",
    )
    _ = parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logs (INFO by default), on stderr.")
    args = parser.parse_args()
    try:
        traits = example.DEFAULT_TRAITS if args.trait is None else persona.Traits(traits=tuple(args.trait))
    except pydantic.ValidationError:
        parser.error("--trait: give 1 to 24 traits, each 1 to 200 characters.")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    env_file: pathlib.Path = root / ".env" if args.env is None else args.env
    try:
        settings = example.Settings.model_validate(
            {
                **os.environ,
                **example.load_env(env_file),
                **({} if args.gemini_model is None else {"BOT_GEMINI_MODEL": args.gemini_model}),
            }
        )
    except pydantic.ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}" for problem in error.errors()
        )
        raise SystemExit(f"listen_speak_bot_example: {problems}") from None
    if shutil.which("say") is None or shutil.which("afconvert") is None:
        raise SystemExit("listen_speak_bot_example: this example needs macOS say and afconvert to render speech")
    assets: pathlib.Path = args.assets_dir
    heard = example.say_to_wav(example.HEARD_PHRASE, assets / "hey_gabe.wav")
    reply = assets / "reply.wav"
    reply.unlink(missing_ok=True)
    streaming = settings.stream_url is not None
    # A streaming session hears raw PCM; local hearing takes the WAV as rendered.
    sounds = tuple(
        SoundData(
            audio=example.wav_pcm(wav) if streaming else wav,
            media_type="audio/pcm" if streaming else "audio/wav",
            sample_rate_hz=example.SAMPLE_RATE_HZ,
            channels=example.CHANNELS,
            kind=kind,
        )
        for wav, kind in ((heard, "speech"), (example.silence_wav(), "silence"))
    )
    voices = identification.Registry(
        identifier=identification.Matcher(
            store=identification.InProcessStore(max_ids=example.MAX_VOICE_IDS), policy=example.VOICE_POLICY
        ),
        kind="voice",
        max_unlinked_age=example.VOICE_RETENTION,
        max_ids=example.MAX_VOICE_IDS,
    )
    prints = (
        pyannote.Classifier(model_id=settings.voice_identity_model)
        if settings.voiceprint_url is None
        else speech_server.VoiceEmbedder(base_url=settings.voiceprint_url)
    )
    vad: tuple[mlx_audio.VoiceActivityClassifier, ...]
    if settings.stream_url is None:
        vad = (
            mlx_audio.VoiceActivityClassifier(
                model_id=settings.vad_model, sample_rate_hz=example.SAMPLE_RATE_HZ, channels=example.CHANNELS
            ),
        )
        hearing = listening.Listening(
            voice_activity_classifier=vad[0], speech_decoder=None, voice_classifier=prints, registry=voices
        )
    else:
        vad = ()
        # One streaming session per sound source; the server's end of turn makes each text product.
        hearing = listening.Listening(
            transcription=functools.partial(speech_server.Transcription, url=settings.stream_url),
            voice_classifier=prints,
            registry=voices,
        )
    whisper = (
        mlx_audio.SpeechDecoder()
        if settings.stt_model is None
        else mlx_audio.SpeechDecoder(model_id=settings.stt_model)
    )
    talk = conversation.Conversation(
        turn_detector=turn_detector.TurnDetector(
            # The turn detector decodes only audio stimuli: MLX Whisper's narrower voice decoder is the boundary.
            decoder=typing.cast(
                decoding.Decoder[turn_detector.ParticipationStimulus, str],
                mlx_audio.VoiceDecoder(speech_decoder=whisper),
            ),
            end_of_turn_silence_seconds=example.END_OF_TURN_SILENCE.total_seconds(),
        ),
    )
    run = example.Run()
    mouth = speaking.Speaking(
        encoder=example.SayEncoder(destination=reply, run=run)
        if settings.tts_url is None
        else example.LabVoice(settings.tts_url, destination=reply, run=run),
        listening=(hearing,),
        conversation=(talk,),
        voicing_timeout=example.VOICING_TIMEOUT,
    )
    listener = bot.define(
        "listen_speak",
        bot.traits(*traits.traits),
        *(
            ()
            if settings.tts_url is None
            else (
                bot.voice(
                    speech_server.Catalog(settings.tts_url, source=example.LAB_SOURCE, languages=example.LANGUAGES),
                    designer=speech_server.Designer(settings.tts_url, source=example.LAB_SOURCE),
                    languages=example.LANGUAGES,
                ),
            )
        ),
        bot.cognition(
            reasoning=gemini.Processor(
                client=gemini.ChatClient(
                    model=settings.gemini_model, api_key=settings.gemini_api_key.get_secret_value()
                ),
                provider="gemini_reasoning",
            ),
            reflex=decision.Processor(
                base_url=settings.decision_base_url,
                api_key=settings.decision_key(),
                max_context_tokens=settings.decision_max_context_tokens,
            ),
            # The bot's memory outlives the run: its persona, and on the lab its voice, are its own on every start.
            memory=memory.Memory(database=str(assets / "memory.sqlite3")),
            associative=True,
            consolidation=True,
            autonomy=True,
            seeds=(communication.speech_heard_seed(),),
            routines=functools.partial(datetime.datetime.now, datetime.UTC),
        ),
        bot.input(hearing),
        bot.output(mouth),
        bot.attach(communication.Communication(conversations=(talk,), speaking=mouth)),
    )
    try:
        settled = asyncio.run(example.hear(listener, sounds, run=run, workers=(whisper, *vad)))
    except Exception as error:
        logging.getLogger("listen_speak_bot_example").exception("listen_speak_bot failed")
        raise SystemExit(f"listen_speak_bot failed: {error}") from None
    summary = example.Summary.model_validate(
        settled.model_dump()
        | {
            "heard_audio_bytes": len(heard),
            "reply_audio_bytes": len(example.wav_pcm(reply.read_bytes())) if reply.exists() else 0,
        }
    )
    if args.json:
        print(summary.model_dump_json(indent=2))
    else:
        print("stateforward.mosfet listen_speak (device-free, Gemini cognition)")
        for key, value in summary.model_dump(mode="json").items():
            print(f"  {key}: {value}")
    if args.play:
        for path in (assets / "hey_gabe.wav", reply):
            if path.exists():
                print(f"Playing {path}", file=sys.stderr)
                _ = subprocess.run(["afplay", str(path)], check=False)
    raise SystemExit(0 if summary.status == "ok" else 1)

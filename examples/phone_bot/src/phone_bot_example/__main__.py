"""The phone bot file: ``python -m phone_bot_example`` runs ``alice`` on this host, holding the line until stopped.

Reads the owner's :class:`~phone_bot_example.Settings` from the environment (the OpenAI and Gemini keys, the
decision server reflex decides on and its optional key, the optional LiveKit line and the model
overrides). The process options (``MOSFET_*`` or flags) are ``bot.process``'s. A missing or malformed setting ends
the process with exit ``1``, naming it, before anything starts.

A few parts come first, as variables, because two parts share each of them (one instance handed to two
constructors). The ear, which the mouth tells what it is saying, so the bot can learn its own voice. The
conversation, shared by communication and the mouth. And the mouth, the one ``speaking.Speaking`` the bot speaks
through, out of the speaker at the handset's mouthpiece.

Then one ``bot.define`` constructs every other part inline: its cognition on the OpenAI reasoning model with its
reflex on the decision model and the speech seed that takes heard speech into a conversation, its phone on the LiveKit line (its earpiece at the
robot's ear, its close-talk mouthpiece at the mouth), its ear, its mouth and its conversations. ``bot.run`` starts
the person standing in front of it first, as a shared actor, who says out loud each line typed at this terminal,
and the observer that logs what the bot does.
"""

import datetime
import functools
import logging
import os

import pydantic
from mosfet import bot
from mosfet.abilities import communication, identification, listening, speaking
from mosfet.abilities.communication import conversation
from mosfet.abilities.communication.conversation import turn_detector
from mosfet.abilities.hearing import sound
from mosfet.devices import audio, phone
from mosfet.environment import space
from mosfet.providers import decision
from mosfet.providers import gemini
from mosfet.providers import livekit
from mosfet.providers import mlx_audio
from mosfet.providers import openai_compat
from mosfet.providers import pyannote
from mosfet.providers.livekit import signaling
from mosfet.providers.livekit.audio import PcmWavDecoder

import phone_bot_example as example
from phone_bot_example import person

if __name__ == "__main__":
    process = bot.process()
    try:
        settings = example.Settings.model_validate(os.environ)
    except pydantic.ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}" for problem in error.errors()
        )
        raise SystemExit(f"phone_bot_example: {problems}") from None
    # What the bot does is logged by the run's observer and by every machine's own observer, on stderr.
    for logger in ("phone_bot_example", "bot.telemetry.hsm"):
        logging.getLogger(logger).setLevel(logging.DEBUG if settings.verbose else logging.INFO)
    hearing = listening.Listening(
        # The streaming classifier is told the shape of the raw PCM the line delivers; a WAV (the ring) describes
        # itself.
        voice_activity_classifier=mlx_audio.VoiceActivityClassifier(
            model_id=settings.vad_model,
            sample_rate_hz=settings.input_sample_rate_hz,
            channels=settings.input_channels,
        ),
        # A non-voice sound with a kind (the ring, a busy tone) is heard as that kind.
        sound_classifier=sound.classification.KindSoundClassifier(),
        speech_decoder=None,
        voice_classifier=pyannote.Classifier(model_id=settings.voice_identity_model),
        registry=identification.Registry(
            identifier=identification.Matcher(
                store=identification.InProcessStore(max_ids=example.MAX_VOICE_IDS), policy=example.VOICE_POLICY
            ),
            kind="voice",
            max_unlinked_age=example.VOICE_RETENTION,
            max_ids=example.MAX_VOICE_IDS,
        ),
    )
    talk = conversation.Conversation(
        turn_detector=turn_detector.TurnDetector(
            decoder=example.GeminiVoiceDecoder(
                pcm_decoder=PcmWavDecoder(
                    sample_rate_hz=settings.input_sample_rate_hz, channels=settings.input_channels
                ),
                speech_decoder=gemini.SpeechDecoder(
                    client=gemini.ChatClient(
                        api_key=settings.gemini_api_key.get_secret_value(), model=settings.stt_model
                    ),
                    model=settings.stt_model,
                    mime_type="audio/wav",
                ),
            ),
            end_of_turn_silence_seconds=example.END_OF_TURN_SILENCE_SECONDS,
        ),
    )
    mouth = speaking.Speaking(
        # Raw PCM, at the one rate the speaker, Speaking's label and the LiveKit uplink all take.
        encoder=gemini.SpeechEncoder(
            client=gemini.ChatClient(api_key=settings.gemini_api_key.get_secret_value(), model=settings.stt_model),
            model=settings.tts_model,
            voice_name=settings.tts_voice,
            sample_rate_hz=settings.output_sample_rate_hz,
        ),
        # The robot's own mouth, not a device it owns: Speaking powers it, at the mouthpiece.
        speaker=audio.Speaker(placement=space.Placement(position=example.MOUTH), amplitude_db=example.VOICE_DB),
        listening=(hearing,),
        conversation=(talk,),
    )
    alice = bot.define(
        example.NAME,
        bot.cognition(
            reasoning=openai_compat.Processor(
                client=openai_compat.ChatClient(
                    model=settings.reasoning_model,
                    api_key=settings.openai_api_key.get_secret_value(),
                    base_url=settings.openai_base_url,
                ),
                provider="openai_terra_reasoning",
            ),
            reflex=decision.Processor(
                base_url=settings.decision_base_url,
                api_key=settings.decision_key(),
                max_context_tokens=settings.decision_max_context_tokens,
            ),
            associative=True,
            consolidation=True,
            autonomy=True,
            seeds=(communication.speech_heard_seed(),),
            routines=functools.partial(datetime.datetime.now, datetime.UTC),
        ),
        bot.attach(
            phone.Phone(
                service=livekit.PhoneService(
                    url=settings.livekit_url,
                    token=settings.line_token(),
                    track_name=settings.track_name,
                    dial_plan=None if not settings.dial_plan else signaling.MappingDialPlan(settings.dial_plan),
                    uplink_sample_rate_hz=settings.output_sample_rate_hz,
                ),
                # The earpiece at the robot's ear, quiet; the close-talk mouthpiece at its mouth.
                speaker=audio.Speaker(
                    placement=space.Placement(position=example.BOT_ORIGIN), amplitude_db=example.EARPIECE_DB
                ),
                microphone=audio.Microphone(
                    placement=space.Placement(position=example.MOUTH, threshold_db=example.MOUTHPIECE_THRESHOLD_DB)
                ),
                placement=space.Placement(position=example.BOT_ORIGIN),
            )
        ),
        bot.input(hearing),
        bot.output(mouth),
        bot.attach(communication.Communication(conversations=(talk,), speaking=mouth)),
    )
    bot.run(
        alice,
        actors=(
            person.Person(
                encoder=person.SayEncoder(),
                position=example.ROOM_POSITION,
                amplitude_db=example.VOICE_DB,
                lines=person.terminal,
            ),
        ),
        process=process,
        observe=(bot.observe(example.log),),
    )

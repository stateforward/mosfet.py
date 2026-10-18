"""Willy's bot file: ``python -m willy_example`` runs the household bot on this host.

Reads the owner's :class:`~willy_example.Settings` from the environment (the lab model server's ``WILLY_*_URL``
addresses and models, and the optional room, camera, trait and household variables) and two secrets:
``openai_api_key`` (a placeholder for the lab's keyless servers) and ``computer_token_secret``, the key every
room's far side signs its token with. Each is the Docker secret, the file named by ``<NAME>_FILE``, or
``<NAME>``. The process options (``MOSFET_*`` or flags) are ``bot.process``'s. A missing or malformed setting or
secret, or an unreadable or malformed household file, ends the process with exit ``1``, naming it, before
anything starts.

A few parts come first, as variables, because two parts share each of them (one instance handed to two
constructors). The websocket server every room's computer is served by, because ``bot.run`` starts it first, as a
shared actor. The one file database on ``<data dir>/memory.sqlite3``, shared by memory and the voice (and face)
prints, so a person and the prints of their voice live and are forgotten together: a ``sqlite_memory.Database``
handle that opens nothing here; the first of them to start opens and migrates it, after ``bot.run`` has created
the data dir private (``0700``), because it holds those prints. A database that cannot be opened fails Willy's
start, under ``bot.run``. The memory itself, shared by cognition,
identity and people. The voice registry, owned by the ear that hears with it and asked by people to keep or erase a
print. The ear, which the mouth tells what it is saying so the bot can
learn its own voice. The conversation, shared by communication and the mouth. And the mouth, the one
``speaking.Speaking`` that the bot speaks through, in the voice it is born with: every room's computer hears it
and plays it on that room's speaker, like an intercom.

Then one ``bot.define`` constructs every other part inline: Willy's traits, the source of its voice (the lab's
speech engine, which designs it at Willy's birth), its cognition on the lab's language model with the speech
seed that takes heard speech into a conversation, its ear and its sense of its own name, its mouth, its
conversations, its capacity to know people under the owner's household, and one computer per room, set on the
object the owner names in that room. With ``WILLY_CAMERA=1`` a face registry and an eye join them, seeing through
the rooms' cameras with the OpenCV Zoo's YuNet and SFace.
"""

import datetime
import functools
import os

import pydantic
from mosfet import bot, environment
from mosfet.abilities import (
    communication,
    embedding,
    identification,
    identity,
    listening,
    memory,
    people,
    seeing,
    speaking,
)
from mosfet.abilities.communication import conversation
from mosfet.devices import audio, computer
from mosfet.providers import opencv
from mosfet.providers import openai_compat
from mosfet.providers import speech_server
from mosfet.providers import sqlite_memory
from mosfet.providers import websocket

from willy_example import (
    EMBEDDING_MAX_ITEMS,
    EMBEDDING_SPACE,
    EXPRESSIVENESS,
    FACE_POLICY,
    LANGUAGES,
    MAX_VOICE_IDS,
    SFACE_FILE,
    SPEECH_MODEL,
    TRANSCRIPTION_MODEL,
    VOICE_POLICY,
    VOICE_RETENTION,
    VOICE_SOURCE,
    VOICING_TIMEOUT,
    YUNET_FILE,
    Settings,
)

if __name__ == "__main__":
    process = bot.process()
    try:
        settings = Settings.model_validate(os.environ)
    except pydantic.ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}" for problem in error.errors()
        )
        raise SystemExit(f"willy_example: {problems}") from None
    api_key = process.secrets.get("openai_api_key").get_secret_value()
    server = websocket.Server(
        host=settings.host,
        port=settings.port,
        verifier=websocket.HmacTokens(process.secrets.get("computer_token_secret")),
    )
    # Opened when memory or a print store first starts, inside the data dir bot.run makes private: it holds every print.
    database = sqlite_memory.Database(path=process.data_dir / "memory.sqlite3")
    store = memory.Memory(
        engine=database.open,
        embedder=openai_compat.Embedder(
            client=openai_compat.EmbeddingClient(
                model=settings.embed_model, base_url=settings.embed_url, api_key=api_key
            ),
            capacity=embedding.Capacity(space=EMBEDDING_SPACE, max_items=EMBEDDING_MAX_ITEMS, max_input_tokens=None),
            instructions=embedding.EMBEDDING_GEMMA_2,
        ),
    )
    voices = identification.Registry(
        identifier=identification.Matcher(store=sqlite_memory.IdentityStore(engine=database.open), policy=VOICE_POLICY),
        kind="voice",
        max_unlinked_age=VOICE_RETENTION,
        max_ids=MAX_VOICE_IDS,
    )
    eyes = settings.face_models if settings.camera else None
    faces = (
        ()
        if eyes is None
        else (
            identification.Registry(
                identifier=identification.Matcher(
                    store=sqlite_memory.IdentityStore(engine=database.open), policy=FACE_POLICY
                ),
                kind="face",
                max_unlinked_age=VOICE_RETENTION,
                max_ids=MAX_VOICE_IDS,
            ),
        )
    )
    hearing = listening.Listening(
        transcription=lambda: speech_server.Transcription(url=settings.stt_url),
        voice_classifier=speech_server.VoiceEmbedder(base_url=settings.voiceprint_url),
        registry=voices,
        voice_diarizer=speech_server.Diarizer(base_url=settings.diar_url),
        affect_classifier=speech_server.AffectClassifier(base_url=settings.emotion_url),
    )
    talk = conversation.Conversation()
    mouth = speaking.Speaking(
        encoder=lambda key, description: openai_compat.SpeechEncoder(
            model=SPEECH_MODEL,
            base_url=f"{settings.tts_url.rstrip('/')}/v1",
            voice=key,
            voice_description=description,
            expressiveness=EXPRESSIVENESS,
            markup=openai_compat.ANGLE_TAGS,
        ),
        speaker=audio.Speaker(),
        listening=(hearing,),
        conversation=(talk,),
        voicing_timeout=VOICING_TIMEOUT,
    )
    willy = bot.define(
        "willy",
        bot.traits(*settings.traits.traits),
        bot.voice(
            speech_server.Catalog(settings.tts_url, source=VOICE_SOURCE, languages=LANGUAGES),
            designer=speech_server.Designer(settings.tts_url, source=VOICE_SOURCE),
            languages=LANGUAGES,
        ),
        bot.cognition(
            openai_compat.Processor(
                client=openai_compat.ChatClient(model=settings.model, base_url=settings.llm_url, api_key=api_key),
                provider="willy",
            ),
            memory=store,
            associative=True,
            consolidation=True,
            autonomy=True,
            seeds=(communication.speech_heard_seed(),),
            routines=functools.partial(datetime.datetime.now, datetime.UTC),
        ),
        bot.input(hearing),
        bot.input(
            identity.Identity(
                recognizer=identity.SpeechNameRecognizer(
                    decoder=openai_compat.SpeechDecoder(model=TRANSCRIPTION_MODEL, base_url=settings.transcribe_url)
                ),
                memory=store,
            )
        ),
        *(
            bot.input(
                seeing.Seeing(
                    change=opencv.FrameDiff(),
                    change_threshold=0.03,
                    face_detector=opencv.FaceDetector(
                        models=opencv.Models(yunet=eyes / YUNET_FILE, sface=eyes / SFACE_FILE)
                    ),
                    faces=registry,
                )
            )
            for registry in faces
            if eyes is not None
        ),
        bot.output(mouth),
        bot.attach(communication.Communication(conversations=(talk,), speaking=mouth)),
        bot.attach(people.People(memory=store, registries=(voices, *faces), household=settings.household)),
        *(
            bot.attach(
                computer.Computer(
                    id=room.id,
                    transport=server,
                    peer_id=room.id,
                    placement=environment.Object(id=room.on, placement=room.placement).surface(),
                )
            )
            for room in settings.rooms
        ),
    )
    bot.run(willy, actors=(server,), process=process)

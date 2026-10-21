"""The companion's bot file: ``python -m companion_example`` runs the household bot on this host.

Reads the owner's :class:`~companion_example.Settings` from the environment (the lab model server's ``COMPANION_*_URL``
addresses and models, ``DECISION_BASE_URL``, and the optional ``DECISION_MAX_CONTEXT_TOKENS``,
``COMPANION_THINKING_TOKENS``, ``COMPANION_MAX_CONTEXT_TOKENS``, room, camera and trait variables) and its secrets:
``openai_api_key`` (a placeholder for the lab's keyless servers),
the optional ``decision_api_key`` (the key of the decision server the companion's reflex decides on; without it none is
sent, as the local decide expects) and ``computer_token_secret``, the key every
room's far side signs its token with. Each is the Docker secret, the file named by ``<NAME>_FILE``, or
``<NAME>``. The process options (``MOSFET_*`` or flags) are ``bot.process``'s. A missing or malformed setting or
secret ends the process with exit ``1``, naming it, before anything starts.

A few parts come first, as variables, because two parts share each of them (one instance handed to two
constructors). The websocket server every room's computer is served by, because ``bot.run`` starts it first, as a
shared actor. The one file database on ``<data dir>/memory.sqlite3``, shared by memory and the voice (and face)
prints, so a person and the prints of their voice live and are forgotten together: a ``sqlite_memory.Database``
handle that opens nothing here; the first of them to start opens and migrates it, after ``bot.run`` has created
the data dir private (``0700``), because it holds those prints. A database that cannot be opened fails the companion's
start, under ``bot.run``. The memory itself, shared by cognition and
identity. The voice registry, owned by the ear that hears with it and given to the cognition, whose memory keeps or
erases a print as the companion learns or forgets whose it is. The ear, which the mouth tells what it is saying so the
bot can learn its own voice. The conversation, shared by communication and the mouth. And the mouth, the one
``speaking.Speaking`` that the bot speaks through, in the voice it is born with: every room's computer hears it
and plays it on that room's speaker, like an intercom.

Then one ``bot.define`` constructs every other part inline: the companion's traits, the source of its voice (the lab's
shared speech engine, which designs it at the companion's birth; the companion is allowed none of the engine's other
voices, so it is born only as the one it designs), its cognition in three tiers, each bounded by its own step
timeout: its reflex on the decision model, its intuition on the intuition model with thinking off, and its
reasoning on the reasoning model, which thinks first within the owner's ``COMPANION_THINKING_TOKENS``, the speech seed
that takes heard speech into a conversation and the registries whose prints follow whom it knows (people it learns
itself), its ear and its sense of its own name, its mouth, its conversations, and one computer per room, set on the
object the owner names in that room. With ``COMPANION_CAMERA=1`` a face registry and an eye join them, seeing through
the rooms' cameras with the OpenCV Zoo's YuNet and SFace.
"""

import datetime
import functools
import os

import pydantic
from mosfet import bot, environment, runtime
from mosfet.abilities import (
    communication,
    embedding,
    identification,
    identity,
    listening,
    memory,
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

from companion_example import (
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
from companion_example.tiers import Tiers

if __name__ == "__main__":
    process = bot.process()
    try:
        settings = Settings.model_validate(os.environ)
    except pydantic.ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}" for problem in error.errors()
        )
        raise SystemExit(f"companion_example: {problems}") from None
    api_key = process.secrets.get("openai_api_key").get_secret_value()
    try:
        # The local decision server takes no key; one is sent only when the secret is given.
        decision_key = process.secrets.get("decision_api_key").get_secret_value() or None
    except runtime.MissingSecretError:
        decision_key = None
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
    # The body's sensory register; the ear writes there everything the companion hears, and Conversation everything
    # he says, so it is the same one.
    register = memory.StmMemory()
    hearing = listening.Listening(
        transcription=lambda: speech_server.Transcription(url=settings.stt_url),
        voice_classifier=speech_server.VoiceEmbedder(base_url=settings.voiceprint_url),
        registry=voices,
        voice_diarizer=speech_server.Diarizer(base_url=settings.diar_url),
        affect_classifier=speech_server.AffectClassifier(base_url=settings.emotion_url),
        stm_memory=register,
    )
    tiers = Tiers.of(settings, api_key=api_key, decision_key=decision_key)
    talk = conversation.Conversation(stm_memory=register)
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
    companion = bot.define(
        "companion",
        bot.traits(*settings.traits.traits),
        bot.register(register),
        bot.voice(
            speech_server.Catalog(settings.tts_url, source=VOICE_SOURCE, languages=LANGUAGES),
            designer=speech_server.Designer(settings.tts_url, source=VOICE_SOURCE),
            languages=LANGUAGES,
        ),
        bot.cognition(
            reasoning=tiers.reasoning,
            intuition=tiers.intuition,
            reflex=tiers.reflex,
            memory=store,
            registries=(voices, *faces),
            associative=True,
            consolidation=True,
            autonomy=True,
            seeds=(communication.speech_heard_seed(),),
            routines=functools.partial(datetime.datetime.now, datetime.UTC),
        ),
        Tiers.timeouts(),
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
    bot.run(companion, actors=(server,), process=process)

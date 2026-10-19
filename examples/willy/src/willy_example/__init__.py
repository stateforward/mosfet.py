"""Willy: one bot for the whole house, the people who live there and anyone who visits, that learns who people are.

    python -m willy_example --ready-file /tmp/ready

``__main__`` is the bot file. It reads the owner's :class:`Settings` and the process's secrets, then defines one
bot, ``willy``, with one ``bot.define`` that constructs every part inline, and runs it with ``bot.run``. Each room
the owner lists has a ``computer.Computer`` of its own, named for the room and set on something in it (an
``environment.Object``, a shelf or a counter), and served by one ``websocket.Server`` on this host; a far side in
each room carries the room's microphone and speaker (and camera) to it. Willy hears through ``listening`` on the
lab's streaming speech server, with voiceprints, diarization and the tone of voice; speaks through ``speaking`` in
the one voice it is given when it is born; and knows people through ``people``: a person is one node in its
memory with every voice (and face) the bot has linked to them, each voice one person's.

Nothing here tells Willy what to do. It is given no instruction text and no reply: a voice nobody has named is a
fact it is shown (``known`` false), and whether to ask who that is, what to call them, whom a voice belongs to and
whom to forget are its own choices. What the owner fixes are facts and capacities: who lives in the house (the
household file: each member's id, name, adult or child, and birth year, which Willy can never change), Willy's
traits, and what Willy can do at all, which is what is attached here.

Every model runs locally, on the lab model server (see ``.env.example``): the language model, its embeddings,
streaming speech to text, speech, voice design, voiceprints, diarization and tone.
"""

import collections.abc
import datetime
import json
import pathlib
import typing

import pydantic
from mosfet.abilities import embedding, identification, memory, persona, vocal
from mosfet.environment import space

DEFAULT_TRAITS: typing.Final = persona.Traits(traits=("warm", "patient", "curious", "honest"))
"""The traits this example's owner gives Willy; ``WILLY_TRAITS`` gives it others."""
DEFAULT_ROOMS: typing.Final = "living-room:shelf"
"""The one room Willy is in when the owner names none: the living room, its computer on a shelf."""
ROOM_SPACING_M: typing.Final = 10.0
"""How far apart, in metres along one line, the rooms are placed, in the order the owner lists them."""
MAX_ROOMS: typing.Final = 16
"""Most rooms one Willy serves."""
MAX_HOUSEHOLD_FILE_BYTES: typing.Final = 64 * 1024
"""The largest household file read; a household is a short list."""

LANGUAGES: typing.Final[tuple[vocal.speech.Locale, ...]] = ("en",)
"""The languages Willy speaks, and the lab's speech engine renders."""
VOICE_SOURCE: typing.Final = "lab"
"""The name given the lab's speech engine as a voice source: the voice Willy is born with carries it."""
SPEECH_MODEL: typing.Final = "tts"
"""The model name sent to the lab's OpenAI-compatible speech endpoint (the server ignores it)."""
TRANSCRIPTION_MODEL: typing.Final = "stt"
"""The model name sent to the lab's OpenAI-compatible transcription endpoint."""
EXPRESSIVENESS: typing.Final = vocal.speech.Expressiveness(
    cues=frozenset[vocal.speech.Cue]({"instructions", "emotion", "intensity", "emphasis", "style", "sound"}),
    styles=("whisper",),
    sounds=frozenset[vocal.speech.Sound]({"laugh", "giggle", "chuckle", "gasp", "sigh"}),
)
"""What the lab's speech engine does with an expression, written in its angle tags (``openai_compat.ANGLE_TAGS``)."""
VOICING_TIMEOUT: typing.Final = datetime.timedelta(seconds=120)
"""How long speech waits for Willy's voice: on its first start the lab designs it."""

EMBEDDING_SPACE: typing.Final = embedding.Space(model="google/embeddinggemma-2", dimensions=768)
"""The space of the lab's EmbeddingGemma 2 vectors, which memory recalls by."""
EMBEDDING_MAX_ITEMS: typing.Final = 16
"""Most texts memory sends the lab's embedder in one request."""

VOICE_POLICY: typing.Final = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65)
"""When two voiceprints are one voice: WeSpeaker's operating point on the lab's voiceprint server."""
FACE_POLICY: typing.Final = identification.MatchPolicy(threshold=0.363, margin=0.05, refine_threshold=0.5)
"""When two face prints are one face: SFace's published cosine threshold."""
MAX_VOICE_IDS: typing.Final = 1_000
"""Most voice (and face) ids Willy remembers."""
VOICE_RETENTION: typing.Final = datetime.timedelta(days=30)
"""How long a voice (or face) nobody is known by is remembered after it was last heard (or seen)."""
YUNET_FILE: typing.Final = "face_detection_yunet_2023mar.onnx"
"""YuNet's file in the owner's OpenCV models directory: it finds faces."""
SFACE_FILE: typing.Final = "face_recognition_sface_2021dec.onnx"
"""SFace's file in the owner's OpenCV models directory: it makes a face's print."""

RoomId = typing.Annotated[
    str,
    pydantic.StringConstraints(pattern=r"^[a-z][a-z0-9-]{2,62}$"),
    pydantic.Field(
        description=(
            "A room's name: the id of its computer and the peer its far side connects as. 3 to 63 lower-case "
            "letters, digits or hyphens, starting with a letter."
        ),
        examples=["kitchen", "living-room"],
    ),
]


class Room(pydantic.BaseModel):
    """One room of the house: the computer there is named for it, and sits on ``on`` at ``placement``."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    id: RoomId
    on: space.ObjectId = pydantic.Field(
        description="What the room's computer sits on: an object in the room, such as a shelf or a counter.",
        examples=["shelf", "counter"],
    )
    placement: space.Placement = pydantic.Field(
        description="Where that object is in the house; nothing there is selective about what it hears."
    )


class Settings(pydantic.BaseModel):
    """The owner's settings for Willy, read from the process environment by ``__main__``.

    Each field is read from the variable its alias names; a variable set to the empty string is unset. Every
    address is required: where the owner's model server is has no safe default. A value is never shown in an
    error: the error names the variable and what is wrong with it. ``WILLY_HOUSEHOLD_FILE`` names a JSON file
    (a list of ``memory.Member``), which is read here, at the process boundary, when the settings are.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="ignore", hide_input_in_errors=True
    )

    llm_url: str = pydantic.Field(
        alias="WILLY_LLM_URL",
        min_length=1,
        description="The OpenAI-compatible endpoint of Willy's language model.",
        examples=["http://192.168.2.2:8100/v1"],
    )
    model: str = pydantic.Field(
        alias="WILLY_MODEL", min_length=1, description="The model that endpoint serves.", examples=["qwen3.8-27b"]
    )
    typesafe_base_url: str | None = pydantic.Field(
        default=None,
        alias="TYPESAFE_BASE_URL",
        min_length=1,
        description=("The local system_one server Willy's intuition decides on; unset, it decides on hosted TypeSafe."),
        examples=["http://127.0.0.1:8102"],
    )
    embed_url: str = pydantic.Field(
        alias="WILLY_EMBED_URL",
        min_length=1,
        description="The OpenAI-compatible endpoint of the embedding model memory recalls by.",
        examples=["http://192.168.2.2:8014/v1"],
    )
    embed_model: str = pydantic.Field(
        alias="WILLY_EMBED_MODEL",
        min_length=1,
        description="The embedding model that endpoint serves (EmbeddingGemma 2, 768 dimensions).",
        examples=["embeddinggemma-2"],
    )
    stt_url: str = pydantic.Field(
        alias="WILLY_STT_URL",
        pattern=r"^wss?://",
        description="The speech server's streaming speech-to-text root, as a WebSocket URL with no path.",
        examples=["ws://192.168.2.2:8001"],
    )
    transcribe_url: str = pydantic.Field(
        alias="WILLY_TRANSCRIBE_URL",
        min_length=1,
        description="The OpenAI-compatible transcription endpoint Willy hears its own name through.",
        examples=["http://192.168.2.2:8001/v1"],
    )
    tts_url: str = pydantic.Field(
        alias="WILLY_TTS_URL",
        min_length=1,
        description="The speech engine's root: its voices, its voice designer, and speech under /v1.",
        examples=["http://192.168.2.2:8002"],
    )
    voiceprint_url: str = pydantic.Field(
        alias="WILLY_VOICEPRINT_URL",
        min_length=1,
        description="The voiceprint server's root.",
        examples=["http://192.168.2.2:8003"],
    )
    emotion_url: str = pydantic.Field(
        alias="WILLY_EMOTION_URL",
        min_length=1,
        description="The tone-of-voice server's root.",
        examples=["http://192.168.2.2:8004"],
    )
    diar_url: str = pydantic.Field(
        alias="WILLY_DIAR_URL",
        min_length=1,
        description="The diarization server's root, which tells several voices in one turn apart.",
        examples=["http://192.168.2.2:8005"],
    )
    host: str = pydantic.Field(
        default="127.0.0.1",
        alias="WILLY_HOST",
        min_length=1,
        description="Where the rooms' computers are served; loopback by default.",
        examples=["0.0.0.0"],
    )
    port: int = pydantic.Field(
        default=8765,
        alias="WILLY_PORT",
        ge=1,
        le=65_535,
        description="The port the rooms' computers are served on.",
        examples=[8765],
    )
    rooms: tuple[Room, ...] = pydantic.Field(
        default=DEFAULT_ROOMS,
        alias="WILLY_ROOMS",
        min_length=1,
        max_length=MAX_ROOMS,
        description="Each room and what its computer sits on, as room:object, comma separated.",
        examples=["kitchen:counter,den:shelf"],
        validate_default=True,
    )
    camera: bool = pydantic.Field(
        default=False,
        alias="WILLY_CAMERA",
        description="Whether Willy sees, through the rooms' cameras: 1 to see, 0 (the default) not to.",
        examples=["0"],
    )
    face_models: pathlib.Path | None = pydantic.Field(
        default=None,
        alias="WILLY_FACE_MODELS",
        description=f"With WILLY_CAMERA=1, the directory holding the OpenCV Zoo's {YUNET_FILE} and {SFACE_FILE}.",
        examples=["/opt/opencv-models"],
    )
    traits: persona.Traits = pydantic.Field(
        default=DEFAULT_TRAITS,
        alias="WILLY_TRAITS",
        description="The owner's traits for Willy, comma separated.",
        examples=["warm,patient,curious,honest"],
    )
    household: tuple[memory.Member, ...] = pydantic.Field(
        default=(),
        alias="WILLY_HOUSEHOLD_FILE",
        description="A JSON file listing who lives in the house; unset, nobody is declared and Willy learns everyone.",
        examples=["./household.json"],
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _unset_empty(cls, data: object) -> object:
        if isinstance(data, collections.abc.Mapping):
            given = typing.cast(collections.abc.Mapping[object, object], data)
            return {name: value for name, value in given.items() if value != ""}
        return data

    @pydantic.field_validator("traits", mode="before")
    @classmethod
    def _split_traits(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        given = tuple(trait.strip() for trait in value.split(","))
        if not all(given):
            raise ValueError("must be comma-separated traits, none of them empty")
        return persona.Traits(traits=given)

    @pydantic.field_validator("rooms", mode="before")
    @classmethod
    def _place_rooms(cls, value: object) -> object:
        """``room:object,...`` becomes one room per entry, ``ROOM_SPACING_M`` apart in the order given."""

        if not isinstance(value, str):
            return value
        rooms: list[dict[str, object]] = []
        for index, entry in enumerate(value.split(",")):
            room, colon, on = (part.strip() for part in entry.partition(":"))
            if not colon or not room or not on:
                raise ValueError("must be comma-separated room:object pairs, such as kitchen:counter")
            position = space.Position(x=index * ROOM_SPACING_M, y=0.0)
            rooms.append({"id": room, "on": on, "placement": space.Placement(position=position)})
        if len({room["id"] for room in rooms}) != len(rooms):
            raise ValueError("must name each room once")
        return tuple(rooms)

    @pydantic.field_validator("household", mode="before")
    @classmethod
    def _read_household(cls, value: object) -> object:
        """Read the household file the variable names: JSON, at most ``MAX_HOUSEHOLD_FILE_BYTES`` (file I/O)."""

        if not isinstance(value, str):
            return value
        try:
            with pathlib.Path(value).open("rb") as file:
                content = file.read(MAX_HOUSEHOLD_FILE_BYTES + 1)
        except OSError as error:
            raise ValueError(f"cannot be read ({error.strerror or type(error).__name__})") from None
        if len(content) > MAX_HOUSEHOLD_FILE_BYTES:
            raise ValueError(f"is larger than {MAX_HOUSEHOLD_FILE_BYTES} bytes")
        try:
            return typing.cast(object, json.loads(content))
        except ValueError:
            raise ValueError('is not JSON: it must be a list of members, such as [{"id": "ada", ...}]') from None

    @pydantic.model_validator(mode="after")
    def _see_with_models(self) -> typing.Self:
        if self.camera and self.face_models is None:
            raise ValueError("WILLY_CAMERA=1 needs WILLY_FACE_MODELS, the directory of the face models")
        return self


__all__ = [
    "DEFAULT_ROOMS",
    "DEFAULT_TRAITS",
    "EMBEDDING_MAX_ITEMS",
    "EMBEDDING_SPACE",
    "EXPRESSIVENESS",
    "FACE_POLICY",
    "LANGUAGES",
    "MAX_HOUSEHOLD_FILE_BYTES",
    "MAX_ROOMS",
    "MAX_VOICE_IDS",
    "ROOM_SPACING_M",
    "SFACE_FILE",
    "SPEECH_MODEL",
    "TRANSCRIPTION_MODEL",
    "VOICE_POLICY",
    "VOICE_RETENTION",
    "VOICE_SOURCE",
    "VOICING_TIMEOUT",
    "YUNET_FILE",
    "Room",
    "RoomId",
    "Settings",
]

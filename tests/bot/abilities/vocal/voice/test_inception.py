"""Inception: the voice a bot is given when it is made, from its traits, and the same voice on every start after.

Inception is the bot's making, not its choice (P-6, P-14): the scripted inception model here stands in for its
makers' model, and every test pins a capacity (one conception, made permanent, written down, voiced once, recalled and
checked) and never what voice anything picks.
"""

from __future__ import annotations

import asyncio
import collections
import collections.abc
import datetime
import json
import pathlib
import sqlite3
import typing
from typing import override

import hsm
import pydantic
import pytest

from mosfet import event
from mosfet.abilities import ability
from mosfet.abilities import memory
from mosfet.abilities import persona
from mosfet.abilities import processing
from mosfet.abilities import speaking
from mosfet.abilities import vocal
from mosfet.abilities.vocal import voice
from mosfet.devices import audio
from mosfet.environment import Environment
from mosfet.protocols import attachment

from tests import metric_reader
from tests.bot.abilities.support import EventRecorder, shared_hsm_context, start_abilities_for_test
from tests.hsm_instance_state import ability_terminal_owner, start_ability_tree

_TRAITS = persona.Traits(traits=("curious", "patient with children"))
_DESCRIBED = "A bright, quick young voice with a smile in it, never hurried."
_EXPRESSIVE = vocal.speech.Expressiveness(cues=frozenset({"instructions", "emotion"}))
_FAST = datetime.timedelta(milliseconds=20)


def described(description: str = _DESCRIBED) -> dict[str, object]:
    return {"conception": {"kind": "described", "description": description}}


def shortlisted(key: str, source: str = "lab") -> dict[str, object]:
    return {"conception": {"kind": "shortlisted", "ref": {"source": source, "key": key}}}


class Shelf(voice.Catalog):
    """A voice source's catalog: the voices it holds, which a designer of the same source adds to."""

    def __init__(self, *entries: voice.Entry, source: str = "lab", fail: int = 0) -> None:
        self.source = source
        self.entries: list[voice.Entry] = list(entries)
        self.calls: list[tuple[str, ...]] = []
        self.fail: int = fail

    @override
    async def voices(
        self, *, languages: tuple[vocal.speech.Locale, ...] = (), descriptors: voice.Descriptors | None = None
    ) -> tuple[voice.Entry, ...]:
        del descriptors
        self.calls.append(languages)
        if self.fail:
            self.fail -= 1
            raise voice.Error("unavailable", "the voice source is restarting")
        return tuple(self.entries)


class Studio(voice.Designer):
    """Makes a voice from each description and keeps it on its shelf; refuses the first ``refuse`` requests."""

    def __init__(self, shelf: Shelf, *, max_description: int = 300, refuse: int = 0) -> None:
        self.source = shelf.source
        self.max_description = max_description
        self.shelf: Shelf = shelf
        self.requests: list[voice.DesignRequest] = []
        self.refuse: int = refuse

    @override
    async def create(self, request: voice.DesignRequest) -> voice.Designed:
        self.requests.append(request)
        if self.refuse:
            self.refuse -= 1
            raise voice.Error("overloaded", "the voice store is full")
        ref = voice.Ref(source=self.source, key=f"v-{len(self.requests):012x}")
        self.shelf.entries.append(voice.Entry(ref=ref, name=ref.key, description=request.description))
        return voice.Designed(ref=ref, description=request.description)


class Maker(processing.Processor):
    """The makers' inception model, scripted: answers each request with the next conception (the last repeats)."""

    def __init__(self, *answers: object, hang: bool = False) -> None:
        self.answers: list[object] = list(answers)
        self.inputs: list[processing.InputData] = []
        self.hang: bool = hang
        self.cancelled: bool = False

    @override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.inputs.append(input)
        if self.hang:
            try:
                _ = await asyncio.Event().wait()
            finally:
                self.cancelled = True
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return (processing.SelectedEvent(event=voice.ConceiveEvent.name, data=answer),)


class Mouthpiece(vocal.speech.SpeechEncoder):
    """An encoder in the voice it was constructed with; these tests never make it speak."""

    def __init__(self) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=24000, channels=1)
        self.expressiveness: vocal.speech.Expressiveness = _EXPRESSIVE

    @override
    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        async for _ in input:
            pass
        return
        yield


class Voices:
    """Speaking's encoder factory: constructs the voice it is given and remembers every call."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.refuse: bool = refuse

    def __call__(self, key: str, description: str | None) -> vocal.speech.SpeechEncoder:
        self.calls.append((key, description))
        if self.refuse:
            raise ValueError(f"no voice {key!r} here")
        return Mouthpiece()


class RecordingInception(voice.Inception):
    """Inception that mirrors its terminals for assertions (see tests.bot.abilities.support)."""

    outputs: list[object]
    failures: list[object]

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.outputs = []
        self.failures = []

    def told(self) -> list[voice.VoiceData]:
        return [typing.cast(voice.VoiceData, output) for output in self.outputs]

    def messages(self) -> list[str]:
        return [typing.cast(ability.FailureData, failure).message for failure in self.failures]


class WriteRefusingMemory(memory.Memory):
    """Memory that can be read but not written, as a full or read-only store would be."""

    @override
    def _execute(self, data: memory.InputData) -> memory.OutputData:
        if any(statement.sql.lstrip().upper().startswith("INSERT") for statement in data.statements):
            raise RuntimeError("memory is read-only")
        return super()._execute(data)


class Bot:
    """One run of a bot: its environment, Speaking built with an encoder factory, and its Inception."""

    def __init__(
        self, environment: Environment, voices: Voices, mouth: speaking.Speaking, inception: RecordingInception
    ):
        self.environment: Environment = environment
        self.voices: Voices = voices
        self.speaking: speaking.Speaking = mouth
        self.inception: RecordingInception = inception

    def state(self) -> str:
        return self.inception.state() or ""

    async def stop(self) -> None:
        await hsm.stop(self.inception, self.environment)


def open_store(store_type: type[memory.Memory] = memory.Memory) -> tuple[memory.Memory, sqlite3.Connection]:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    return store_type(engine=memory.store.open_sqlite_engine(connection=connection)), connection


async def wait_until(condition: collections.abc.Callable[[], bool], timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.001)


async def settle() -> None:
    for _ in range(50):
        await asyncio.sleep(0)


async def make(
    *,
    shelf: Shelf,
    maker: Maker,
    studio: Studio | None = None,
    store: memory.Memory | None = None,
    voices: Voices | None = None,
    start_speaking: bool = True,
    retry_after: datetime.timedelta = _FAST,
    inception_timeout: datetime.timedelta = datetime.timedelta(seconds=5),
    allowed: tuple[str, ...] | None = None,
) -> Bot:
    """One bot's inception; ``allowed`` is left to Inception's own default unless given."""

    environment = Environment()
    voices = voices or Voices()
    mouth = speaking.Speaking(encoder=voices, speaker=audio.Speaker())
    if start_speaking:
        await start_ability_tree(environment, mouth)
    if store is not None and not store.state():
        # Stands in for the owner that attaches the bot's memory; inception is only its peer.
        await start_abilities_for_test(environment, store)
    inception = RecordingInception(
        traits=_TRAITS,
        languages=("en",),
        speaking=mouth,
        processor=maker,
        catalog=shelf,
        designer=studio,
        memory=store,
        retry_after=retry_after,
        inception_timeout=inception_timeout,
        **({} if allowed is None else {"allowed": allowed}),
    )
    await start_abilities_for_test(environment, inception)
    return Bot(environment, voices, mouth, inception)


async def made(bot: Bot) -> None:
    await wait_until(lambda: bot.state().endswith("/made"))
    await settle()


async def recalled(store: memory.Memory) -> tuple[voice.Given, ...]:
    return voice.recalled_given(await memory.transact(shared_hsm_context(), store, voice.given_select_input()))


def test_a_bot_made_by_a_designing_source_is_given_one_voice_conceived_from_its_traits() -> None:
    async def run() -> tuple[Bot, Maker, Shelf, Studio, tuple[voice.Given, ...]]:
        store, connection = open_store()
        try:
            shelf = Shelf()
            studio = Studio(shelf)
            maker = Maker(described())
            bot = await make(shelf=shelf, studio=studio, maker=maker, store=store)
            await made(bot)
            remembered = await recalled(store)
            await bot.stop()
            return bot, maker, shelf, studio, remembered
        finally:
            connection.close()

    bot, maker, shelf, studio, remembered = asyncio.run(run())

    # The inception model is shown the traits, the languages and the source's capacity, nothing else.
    (asked,) = maker.inputs
    assert asked.input == voice.InceptionInput(traits=_TRAITS.traits, languages=("en",), description_max_chars=300)
    assert tuple(offered.name for offered in asked.schemas) == (voice.ConceiveEvent.name,)
    assert dict(asked.actors) == {}
    # One conception, one render, kept: no slate and no listing of a designing source.
    assert studio.requests == [voice.DesignRequest(description=_DESCRIBED, languages=("en",))]
    assert shelf.calls == []
    ref = voice.Ref(source="lab", key="v-000000000001")
    assert remembered == (
        voice.Given(ref=ref, origin="designed", description=_DESCRIBED, languages=("en",), traits=_TRAITS.traits),
    )
    # Speaking constructs its encoder once, from the voice's key and the description it was made from.
    assert bot.voices.calls == [("v-000000000001", _DESCRIBED)]
    assert bot.inception.told() == [
        voice.VoiceData(
            ref=ref, origin="designed", description=_DESCRIBED, languages=("en",), expressiveness=_EXPRESSIVE
        )
    ]
    assert bot.inception.failures == []


def _conceive_schema(offered: hsm.Event[typing.Any]) -> dict[str, typing.Any]:
    schema = offered.schema
    assert isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)
    return schema.model_json_schema()


def test_a_designing_sources_frame_bounds_the_described_voice_by_the_designers_limit() -> None:
    """D-R-35 (Willy run 7, D-1): the schema let a description run to 1000 characters while the designer takes 300,
    so the model wrote past the bound it was only told as a number, and every conception was refused. The frame's
    conceive schema now carries the designer's limit itself; nothing else in it changes."""

    async def run() -> processing.InputData:
        shelf = Shelf()
        maker = Maker(described())
        bot = await make(shelf=shelf, studio=Studio(shelf, max_description=300), maker=maker)
        await made(bot)
        await bot.stop()
        (asked,) = maker.inputs
        return asked

    asked = asyncio.run(run())

    (offered,) = asked.schemas
    assert (offered.name, offered.kind) == (voice.ConceiveEvent.name, voice.ConceiveEvent.kind)
    schema = _conceive_schema(offered)
    expected = voice.ConceiveData.model_json_schema()
    expected["$defs"]["Described"]["properties"]["description"]["maxLength"] = 300
    assert schema == expected


def test_a_catalog_sources_frame_offers_the_conceive_event_as_it_is() -> None:
    warm = voice.Entry(ref=voice.Ref(source="lab", key="warm"), name="warm")

    async def run() -> processing.InputData:
        maker = Maker(shortlisted("warm"))
        bot = await make(shelf=Shelf(warm), maker=maker, allowed=("warm",))
        await made(bot)
        await bot.stop()
        (asked,) = maker.inputs
        return asked

    asked = asyncio.run(run())

    assert asked.schemas == (voice.ConceiveEvent,)


def test_a_bot_made_by_a_catalog_source_is_given_the_one_voice_shortlisted_from_its_catalog() -> None:
    warm = voice.Entry(ref=voice.Ref(source="lab", key="warm"), name="warm", description="A warm, unhurried voice.")
    buddy = voice.Entry(
        ref=voice.Ref(source="lab", key="buddy"),
        name="buddy",
        languages=("en-US",),
        preview=voice.Preview(pcm=b"\x01\x00" * 8, format=vocal.speech.Format(sample_rate_hz=24000, channels=1)),
    )

    async def run() -> tuple[Bot, Maker, Shelf]:
        shelf = Shelf(warm, buddy)
        maker = Maker(shortlisted("buddy"))
        bot = await make(shelf=shelf, maker=maker, allowed=("warm", "buddy"))
        await made(bot)
        await bot.stop()
        return bot, maker, shelf

    bot, maker, shelf = asyncio.run(run())

    (asked,) = maker.inputs
    shown = asked.input
    assert isinstance(shown, voice.InceptionInput)
    assert (shown.description_max_chars, shown.omitted) == (None, 0)
    # Every voice its owner allowed is shown, but a model-facing input never carries audio: previews are left out.
    assert shown.entries == (warm, buddy.model_copy(update={"preview": None}))
    assert shelf.calls == [("en",)]
    assert bot.voices.calls == [("buddy", "buddy")]
    (told,) = bot.inception.told()
    assert (told.ref, told.origin, told.description, told.languages) == (buddy.ref, "catalog", "buddy", ("en-US",))


def test_a_long_catalog_is_shown_in_part_and_says_how_much_was_left_out() -> None:
    entries = [voice.Entry(ref=voice.Ref(source="lab", key=f"v{index}"), name=f"v{index}") for index in range(205)]

    async def run() -> Maker:
        maker = Maker(shortlisted("v3"))
        bot = await make(shelf=Shelf(*entries), maker=maker, allowed=tuple(entry.ref.key for entry in entries))
        await made(bot)
        await bot.stop()
        return maker

    shown = asyncio.run(run()).inputs[0].input
    assert isinstance(shown, voice.InceptionInput)
    assert (len(shown.entries), shown.omitted) == (200, 5)


def _catalog_of_many_makers() -> tuple[voice.Entry, ...]:
    """A shared source: a stock voice and the voices another maker designed on it (a family's, a sidekick's)."""

    return (
        voice.Entry(ref=voice.Ref(source="lab", key="warm"), name="warm"),
        voice.Entry(ref=voice.Ref(source="lab", key="v-family"), name="v-family", labels={"custom": "true"}),
        voice.Entry(ref=voice.Ref(source="lab", key="v-sidekick"), name="v-sidekick", labels={"custom": "true"}),
    )


@pytest.mark.parametrize(
    ("allowed", "shown_keys"),
    [
        (("warm",), ("warm",)),
        (("warm", "v-sidekick"), ("warm", "v-sidekick")),
        # A key the source does not hold is no voice to be born as.
        (("v-family", "v-gone"), ("v-family",)),
    ],
)
def test_a_bot_is_born_only_from_the_voices_its_owner_allowed(
    allowed: tuple[str, ...], shown_keys: tuple[str, ...]
) -> None:
    async def run() -> tuple[Bot, Maker]:
        maker = Maker(shortlisted(shown_keys[-1]))
        bot = await make(shelf=Shelf(*_catalog_of_many_makers()), maker=maker, allowed=allowed)
        await made(bot)
        await bot.stop()
        return bot, maker

    bot, maker = asyncio.run(run())

    shown = maker.inputs[0].input
    assert isinstance(shown, voice.InceptionInput)
    assert tuple(entry.ref.key for entry in shown.entries) == shown_keys
    assert shown.omitted == 0
    assert bot.voices.calls == [(shown_keys[-1], shown_keys[-1])]


def test_a_voice_its_owner_did_not_allow_cannot_be_shortlisted() -> None:
    async def run() -> tuple[Bot, Maker]:
        maker = Maker(shortlisted("v-family"), shortlisted("warm"))
        bot = await make(shelf=Shelf(*_catalog_of_many_makers()), maker=maker, allowed=("warm",))
        await made(bot)
        await bot.stop()
        return bot, maker

    bot, maker = asyncio.run(run())

    assert len(maker.inputs) == 2
    (message,) = bot.inception.messages()
    assert "not one of the catalog entries" in message
    assert bot.voices.calls == [("warm", "warm")]


def test_a_bot_allowed_no_voice_of_a_source_that_does_not_design_conceives_nothing_and_says_so() -> None:
    """Unscoped, a bot may be born only as a voice it designed; without a designer that is none."""

    async def run() -> tuple[Bot, Maker]:
        maker = Maker(shortlisted("warm"))
        shelf = Shelf(*_catalog_of_many_makers())
        bot = await make(shelf=shelf, maker=maker, retry_after=datetime.timedelta(seconds=30))
        await wait_until(lambda: bool(bot.inception.failures))
        await bot.stop()
        return bot, maker

    bot, maker = asyncio.run(run())

    assert maker.inputs == []
    assert bot.voices.calls == []
    (message,) = bot.inception.messages()
    assert "catalog holds no voices this bot may be born as" in message


def test_a_bot_that_designs_its_voice_is_never_born_as_one_another_maker_designed_on_its_source() -> None:
    async def run() -> tuple[Bot, Maker, Shelf, Studio]:
        shelf = Shelf(*_catalog_of_many_makers())
        studio = Studio(shelf)
        maker = Maker(described())
        bot = await make(shelf=shelf, studio=studio, maker=maker)
        await made(bot)
        await bot.stop()
        return bot, maker, shelf, studio

    bot, maker, shelf, studio = asyncio.run(run())

    # Shown only what it may describe, never the source's other voices; one voice designed, and spoken in.
    (asked,) = maker.inputs
    assert asked.input == voice.InceptionInput(traits=_TRAITS.traits, languages=("en",), description_max_chars=300)
    assert shelf.calls == []
    assert studio.requests == [voice.DesignRequest(description=_DESCRIBED, languages=("en",))]
    assert bot.voices.calls == [("v-000000000001", _DESCRIBED)]


def test_a_voice_given_before_its_owner_scoped_the_view_is_still_the_bots() -> None:
    """The view bounds what a bot is born as; a voice it was already given stays its own for life (P-5)."""

    async def run() -> Bot:
        store, connection = open_store()
        try:
            shelf = Shelf(*_catalog_of_many_makers())
            first = await make(shelf=shelf, maker=Maker(shortlisted("v-family")), store=store, allowed=("v-family",))
            await made(first)
            await first.stop()
            second = await make(shelf=shelf, maker=Maker(shortlisted("warm")), store=store)
            await made(second)
            await second.stop()
            return second
        finally:
            connection.close()

    second = asyncio.run(run())

    assert second.inception.failures == []
    assert second.voices.calls == [("v-family", "v-family")]


def test_on_every_later_start_the_voice_is_recalled_checked_and_given_without_asking_anyone() -> None:
    async def run() -> tuple[Bot, Bot, Maker, Shelf]:
        store, connection = open_store()
        try:
            shelf = Shelf()
            studio = Studio(shelf)
            first = await make(shelf=shelf, studio=studio, maker=Maker(described()), store=store)
            await made(first)
            await first.stop()
            again = Maker(described("A completely different voice, deep and slow."))
            second = await make(shelf=shelf, studio=studio, maker=again, store=store)
            await made(second)
            await second.stop()
            return first, second, again, shelf
        finally:
            connection.close()

    first, second, again, shelf = asyncio.run(run())

    assert again.inputs == []
    # The second start checks the source still holds the voice, and nothing else.
    assert shelf.calls == [()]
    assert second.voices.calls == first.voices.calls == [("v-000000000001", _DESCRIBED)]
    assert second.inception.told() == first.inception.told()


def test_a_voice_the_source_no_longer_holds_is_lost_and_the_bot_stays_unvoiced() -> None:
    """The lab silently falls back to another voice for an unknown one; a lost voice is never spoken in."""

    async def run() -> tuple[Bot, Maker]:
        store, connection = open_store()
        try:
            await start_abilities_for_test(Environment(), store)
            lost = voice.Given(
                ref=voice.Ref(source="lab", key="v-gone"), origin="designed", description=_DESCRIBED, traits=("bold",)
            )
            _ = await memory.transact(shared_hsm_context(), store, voice.given_insert_input(lost))
            maker = Maker(described())
            shelf = Shelf()
            bot = await make(shelf=shelf, studio=Studio(shelf), maker=maker, store=store)
            await wait_until(lambda: bot.state().endswith("/estranged") and bool(bot.inception.failures))
            await asyncio.sleep(0.05)
            await bot.stop()
            return bot, maker
        finally:
            connection.close()

    bot, maker = asyncio.run(run())

    assert maker.inputs == []
    assert bot.voices.calls == []
    assert bot.inception.told() == []
    (message,) = bot.inception.messages()
    assert "no longer held" in message


def test_a_voice_recorded_only_for_another_source_is_estranged_and_never_made_again() -> None:
    async def run() -> tuple[Bot, Maker]:
        store, connection = open_store()
        try:
            await start_abilities_for_test(Environment(), store)
            elsewhere = voice.Given(
                ref=voice.Ref(source="cloud", key="aria"), origin="catalog", description="aria", traits=("bold",)
            )
            _ = await memory.transact(shared_hsm_context(), store, voice.given_insert_input(elsewhere))
            maker = Maker(described())
            shelf = Shelf()
            bot = await make(shelf=shelf, studio=Studio(shelf), maker=maker, store=store)
            await wait_until(lambda: bot.state().endswith("/estranged") and bool(bot.inception.failures))
            await asyncio.sleep(0.05)
            await bot.stop()
            return bot, maker
        finally:
            connection.close()

    bot, maker = asyncio.run(run())

    assert maker.inputs == []
    assert bot.voices.calls == []
    (message,) = bot.inception.messages()
    assert "another voice source" in message


def test_a_source_that_cannot_answer_the_check_is_asked_again() -> None:
    async def run() -> tuple[Bot, Shelf]:
        store, connection = open_store()
        try:
            shelf = Shelf()
            studio = Studio(shelf)
            first = await make(shelf=shelf, studio=studio, maker=Maker(described()), store=store)
            await made(first)
            await first.stop()
            shelf.fail = 2
            second = await make(shelf=shelf, studio=studio, maker=Maker(described()), store=store)
            await made(second)
            await second.stop()
            return second, shelf
        finally:
            connection.close()

    second, shelf = asyncio.run(run())

    assert shelf.calls == [(), (), ()]
    assert second.inception.failures == []
    assert second.voices.calls == [("v-000000000001", _DESCRIBED)]


@pytest.mark.parametrize(
    ("designing", "answers", "refusal"),
    [
        (True, (described("x" * 40), described("A warm voice, slow to anger.")), "longer than the source's 30"),
        (True, (shortlisted("warm"), described("A warm voice, slow to anger.")), "must be described"),
        (False, (shortlisted("nobody"), shortlisted("warm")), "not one of the catalog entries"),
        (False, (described(), shortlisted("warm")), "shortlist an entry"),
        (False, ({"conception": {"kind": "described"}}, shortlisted("warm")), "invalid voice"),
    ],
)
def test_a_conception_outside_what_was_asked_is_refused_and_conceived_again(
    designing: bool, answers: tuple[object, ...], refusal: str
) -> None:
    warm = voice.Entry(ref=voice.Ref(source="lab", key="warm"), name="warm")

    async def run() -> tuple[Bot, Maker, Studio | None]:
        shelf = Shelf(warm)
        studio = Studio(shelf, max_description=30) if designing else None
        maker = Maker(*answers)
        bot = await make(shelf=shelf, studio=studio, maker=maker, allowed=() if designing else ("warm",))
        await made(bot)
        await bot.stop()
        return bot, maker, studio

    bot, maker, studio = asyncio.run(run())

    assert len(maker.inputs) == 2
    (message,) = bot.inception.messages()
    assert refusal in message
    if studio is not None:
        assert [request.description for request in studio.requests] == ["A warm voice, slow to anger."]
    assert len(bot.voices.calls) == 1


def test_a_voice_the_source_cannot_make_is_conceived_and_made_again() -> None:
    async def run() -> tuple[Bot, Maker, Studio]:
        shelf = Shelf()
        studio = Studio(shelf, refuse=1)
        maker = Maker(described())
        bot = await make(shelf=shelf, studio=studio, maker=maker)
        await made(bot)
        await bot.stop()
        return bot, maker, studio

    bot, maker, studio = asyncio.run(run())

    assert (len(maker.inputs), len(studio.requests)) == (2, 2)
    (message,) = bot.inception.messages()
    assert "could not make the bot's voice" in message
    assert bot.voices.calls == [("v-000000000002", _DESCRIBED)]


def test_a_voice_that_cannot_be_written_down_is_not_spoken_in() -> None:
    """Fail closed: a voice the bot could not remember would be a different voice on its next start."""

    async def run() -> tuple[Bot, str]:
        store, connection = open_store(WriteRefusingMemory)
        try:
            shelf = Shelf()
            bot = await make(
                shelf=shelf,
                studio=Studio(shelf),
                maker=Maker(described()),
                store=store,
                retry_after=datetime.timedelta(seconds=30),
            )
            await wait_until(lambda: bool(bot.inception.failures))
            await settle()
            state = bot.state()
            await bot.stop()
            return bot, state
        finally:
            connection.close()

    bot, state = asyncio.run(run())

    assert state.endswith("/resting")
    assert bot.voices.calls == []
    (message,) = bot.inception.messages()
    assert "could not remember the bot's voice" in message


def test_a_conception_that_takes_too_long_is_stopped_and_tried_again() -> None:
    async def run() -> tuple[Bot, Maker, str]:
        shelf = Shelf()
        maker = Maker(described(), hang=True)
        bot = await make(
            shelf=shelf,
            studio=Studio(shelf),
            maker=maker,
            inception_timeout=datetime.timedelta(milliseconds=50),
            retry_after=datetime.timedelta(seconds=30),
        )
        await wait_until(lambda: bool(bot.inception.failures) and maker.cancelled)
        state = bot.state()
        await bot.stop()
        return bot, maker, state

    bot, maker, state = asyncio.run(run())

    assert state.endswith("/resting")
    assert maker.cancelled
    (message,) = bot.inception.messages()
    assert "in time" in message


def test_speaking_that_cannot_take_the_voice_leaves_the_bot_unvoiced() -> None:
    async def run() -> Bot:
        shelf = Shelf()
        bot = await make(shelf=shelf, studio=Studio(shelf), maker=Maker(described()), voices=Voices(refuse=True))
        await made(bot)
        await bot.stop()
        return bot

    bot = asyncio.run(run())

    assert bot.inception.told() == []
    (message,) = bot.inception.messages()
    assert "Speaking could not take the bot's voice" in message and "no voice 'v-000000000001' here" in message


def test_inception_waits_for_speaking_to_start_before_giving_it_the_voice() -> None:
    async def run() -> tuple[Bot, str]:
        shelf = Shelf()
        bot = await make(shelf=shelf, studio=Studio(shelf), maker=Maker(described()), start_speaking=False)
        await wait_until(lambda: "/voicing/" in bot.state())
        await asyncio.sleep(0.05)
        waiting = bot.state()
        await start_ability_tree(bot.environment, bot.speaking)
        await made(bot)
        await bot.stop()
        return bot, waiting

    bot, waiting = asyncio.run(run())

    assert "/voicing/" in waiting
    assert bot.voices.calls == [("v-000000000001", _DESCRIBED)]
    assert len(bot.inception.told()) == 1
    assert bot.inception.failures == []


def test_a_bot_attached_again_after_its_voice_was_taken_keeps_it_and_is_told_nothing_false() -> None:
    """Speaking takes the voice once (a second one is a counted drop): an Inception attached again does not ask."""

    async def run() -> tuple[Bot, str]:
        shelf = Shelf()
        bot = await make(
            shelf=shelf,
            studio=Studio(shelf),
            maker=Maker(described()),
            inception_timeout=datetime.timedelta(milliseconds=200),
        )
        await made(bot)
        owner = ability_terminal_owner(bot.inception)
        assert owner is not None
        _ = await bot.inception.detach(
            bot.environment, attachment.DetachEvent.with_data(attachment.DetachData(actor=owner))
        )
        await wait_until(lambda: bot.state().endswith("/detached"))
        _ = await bot.inception.attach(
            bot.environment, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner))
        )
        await made(bot)
        # Longer than the voicing bound: a re-voicing Speaking never answers would have failed by now.
        await asyncio.sleep(0.3)
        state = bot.state()
        await bot.stop()
        return bot, state

    bot, state = asyncio.run(run())

    assert state.endswith("/made")
    assert bot.voices.calls == [("v-000000000001", _DESCRIBED)]
    assert len(bot.inception.told()) == 1
    assert bot.inception.failures == []


def test_a_failed_detach_after_the_voice_was_taken_keeps_it_and_is_told_nothing_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def hold_detach(
        member: processing.Processing, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]
    ) -> bool:
        del member, ctx, event
        return True

    async def run() -> tuple[Bot, str, list[hsm.Event[typing.Any]]]:
        shelf = Shelf()
        bot = await make(
            shelf=shelf,
            studio=Studio(shelf),
            maker=Maker(described()),
            inception_timeout=datetime.timedelta(milliseconds=200),
        )
        await made(bot)
        replies = EventRecorder()
        _ = await hsm.started(bot.environment, replies, replies.model)
        owner = ability_terminal_owner(bot.inception)
        assert owner is not None
        # The inception model's processing never answers its detach, so the detach times out and fails.
        monkeypatch.setattr(processing.Processing, "detach", hold_detach)
        _ = await bot.inception.detach(
            bot.environment,
            attachment.DetachEvent.with_data(
                attachment.DetachData(actor=owner, reply_to=replies, timeout=datetime.timedelta(milliseconds=5))
            ),
        )
        await wait_until(lambda: bool(replies.named(attachment.DetachFailedEvent)))
        await made(bot)
        await asyncio.sleep(0.3)
        state = bot.state()
        await bot.stop()
        return bot, state, replies.events

    bot, state, replies = asyncio.run(run())

    assert [reply.name for reply in replies] == [attachment.DetachFailedEvent.name]
    assert state.endswith("/made")
    assert bot.voices.calls == [("v-000000000001", _DESCRIBED)]
    assert len(bot.inception.told()) == 1
    assert bot.inception.failures == []


def test_inception_refuses_a_composition_it_cannot_honor() -> None:
    shelf = Shelf()
    mouth = speaking.Speaking(encoder=Voices(), speaker=audio.Speaker())

    def compose(
        *,
        languages: tuple[str, ...] = ("en",),
        processor: processing.Processor | collections.abc.Callable[[], processing.Processor] | None = None,
        designer: voice.Designer | None = None,
        retry_after: datetime.timedelta = _FAST,
        inception_timeout: datetime.timedelta = _FAST,
        allowed: tuple[str, ...] = (),
    ) -> voice.Inception:
        return voice.Inception(
            traits=_TRAITS,
            languages=languages,
            speaking=mouth,
            processor=processor or Maker(described()),
            catalog=shelf,
            designer=designer,
            retry_after=retry_after,
            inception_timeout=inception_timeout,
            allowed=allowed,
        )

    with pytest.raises(ValueError, match="designer makes voices for 'cloud'"):
        _ = compose(designer=Studio(Shelf(source="cloud")))
    with pytest.raises(ValueError, match="retry_after"):
        _ = compose(retry_after=datetime.timedelta(0))
    with pytest.raises(ValueError, match="inception_timeout"):
        _ = compose(inception_timeout=datetime.timedelta(0))
    with pytest.raises(pydantic.ValidationError):
        _ = compose(languages=("English",))
    with pytest.raises(pydantic.ValidationError):
        _ = compose(allowed=("",))
    # A source that designs gives the bot the one voice it designs: no voice of it is allowed beside that one.
    with pytest.raises(ValueError, match="allowed no voice"):
        _ = compose(designer=Studio(shelf), allowed=("warm",))
    # A processor factory is built once, like any other composed processor.
    built: list[Maker] = []

    def build() -> processing.Processor:
        built.append(Maker(described()))
        return built[-1]

    _ = compose(processor=build)
    assert len(built) == 1


def test_conceiving_is_offered_only_on_the_inception_request_and_offers_nothing_to_copy() -> None:
    async def run() -> tuple[str, ...]:
        shelf = Shelf()
        bot = await make(shelf=shelf, studio=Studio(shelf), maker=Maker(described()))
        await made(bot)
        offered = tuple(call.name for call in processing.enabled_call_events(bot.inception))
        await bot.stop()
        return offered

    offered = asyncio.run(run())

    assert voice.ConceiveEvent.name == "bot.ability.voice.conceive"
    assert voice.ConceiveEvent.kind == event.EventKind
    assert voice.ConceiveEvent.name not in offered
    schema = voice.ConceiveData.model_json_schema()
    assert "examples" not in json.dumps(schema)
    definitions = typing.cast(dict[str, dict[str, object]], schema["$defs"])
    assert definitions["Described"]["description"] == (
        "The voice of a newly made bot with these traits: who is speaking (apparent age and character), pitch, "
        "texture, warmth, pace and energy. Never a real person's voice."
    )
    assert definitions["Shortlisted"]["description"] == "The catalog voice that best fits these traits."


def test_a_conception_is_described_or_shortlisted_and_bounded() -> None:
    adapter = pydantic.TypeAdapter(voice.Conception)

    assert isinstance(adapter.validate_python({"kind": "described", "description": _DESCRIBED}), voice.Described)
    assert isinstance(
        adapter.validate_python({"kind": "shortlisted", "ref": {"source": "lab", "key": "warm"}}), voice.Shortlisted
    )
    for invalid in (
        {"kind": "described", "description": "Too short."},
        {"kind": "described", "description": "x" * 1001},
        {"kind": "cloned", "description": _DESCRIBED},
        {"description": _DESCRIBED},
    ):
        with pytest.raises(pydantic.ValidationError):
            _ = adapter.validate_python(invalid)
    entry = voice.Entry(ref=voice.Ref(source="lab", key="warm"), name="warm")
    with pytest.raises(pydantic.ValidationError):
        _ = voice.InceptionInput(traits=("curious",), entries=(entry,) * 201)
    with pytest.raises(pydantic.ValidationError):
        _ = voice.InceptionInput(traits=())


def test_the_voice_renders_as_given_at_inception_with_what_it_can_do() -> None:
    told = voice.VoiceData(
        ref=voice.Ref(source="lab", key="v-3f2a9c41d07e"),
        origin="designed",
        description="A bright <young> voice & a smile.",
        languages=("en", "en-GB"),
        expressiveness=vocal.speech.Expressiveness(
            cues=frozenset({"instructions", "emotion", "style", "sound"}),
            styles=("whisper",),
            sounds=frozenset({"laugh"}),
        ),
    )
    unspoken = told.model_copy(update={"languages": (), "expressiveness": vocal.speech.Expressiveness()})

    assert told.__model_repr__() == (
        '<voice given_at="inception" languages="en en-GB">A bright &lt;young&gt; voice &amp; a smile.'
        "<expressiveness><cues>emotion instructions sound style</cues><styles>whisper</styles>"
        "<sounds>laugh</sounds></expressiveness></voice>"
    )
    assert unspoken.__model_repr__() == (
        '<voice given_at="inception">A bright &lt;young&gt; voice &amp; a smile.'
        "<expressiveness><cues></cues></expressiveness></voice>"
    )


def _counts(reader: metric_reader.Reader, name: str, *keys: str) -> collections.Counter[tuple[str, ...]]:
    return collections.Counter(
        {tuple(str(point.attributes.get(key)) for key in keys): int(point.value) for point in reader.points(name)}
    )


def test_inception_and_recall_are_counted_by_origin_and_outcome(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        store, connection = open_store()
        try:
            shelf = Shelf()
            studio = Studio(shelf, refuse=1)
            first = await make(shelf=shelf, studio=studio, maker=Maker(described()), store=store)
            await made(first)
            await first.stop()
            second = await make(shelf=shelf, studio=studio, maker=Maker(described()), store=store)
            await made(second)
            await second.stop()
            shelf.entries.clear()
            third = await make(shelf=shelf, studio=studio, maker=Maker(described()), store=store)
            await wait_until(lambda: third.state().endswith("/estranged"))
            await third.stop()
        finally:
            connection.close()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        inceptions_before = _counts(reader, "bot.voice.inception.count", "bot.voice.origin", "bot.outcome")
        recalls_before = _counts(reader, "bot.voice.recall.count", "bot.outcome")
        asyncio.run(run())
        inceptions = _counts(reader, "bot.voice.inception.count", "bot.voice.origin", "bot.outcome") - inceptions_before
        recalls = _counts(reader, "bot.voice.recall.count", "bot.outcome") - recalls_before
        durations = [point for point in reader.points("bot.voice.inception.duration")]

    assert inceptions == collections.Counter({("designed", "made"): 1, ("designed", "failed"): 1})
    assert recalls == collections.Counter({("unmade",): 1, ("recalled",): 1, ("lost",): 1})
    assert durations

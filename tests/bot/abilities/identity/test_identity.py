"""Identity: a bot is born nameless, may be given a name, and only then can be addressed.

Every test here pins a *capability* and never a choice. Adoption always happens because the test
dispatched the adopt event itself, standing in for cognition; nothing asserts that a bot presented
with a name takes it. A bot that declines every name it is ever offered passes this whole file.
"""

from __future__ import annotations

import asyncio
import collections.abc
import sqlite3
import typing
from typing import override

import hsm
import pytest

import bot
from bot import abilities
from bot.abilities import cognition
from bot.abilities import identity
from bot.abilities import memory
from bot.abilities import processing
from bot.abilities.cognition import input as cognition_input
from bot.abilities.hearing import speech
from bot.environment import Environment, SoundData, SoundEvent

from tests.bot.abilities.support import shared_hsm_context, start_abilities_for_test

_LISTENING = "/knowing/listening"


class RecordingSpeechDecoder(speech.SpeechDecoder):
    """Stands in for STT and records every chunk it was asked to transcribe."""

    calls: list[bytes]

    def __init__(self) -> None:
        self.calls = []

    @override
    async def decode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return input


class FailingSpeechDecoder(speech.SpeechDecoder):
    @override
    async def decode(self, input: bytes) -> bytes:
        del input
        raise RuntimeError("recognizer unavailable")


class WriteRefusingMemory(memory.Memory):
    """Memory that can be read but not written, as a full or read-only store would be."""

    @override
    def execute(self, data: memory.InputData) -> memory.OutputData:
        if any(statement.sql.lstrip().upper().startswith("INSERT") for statement in data.statements):
            raise RuntimeError("memory is read-only")
        return super().execute(data)


class RecordingIdentity(identity.Identity):
    """Identity that mirrors its terminals for assertions (see tests.bot.abilities.support)."""

    outputs: list[object]
    handoffs: list[cognition_input.InputData]
    failures: list[object]

    def __init__(
        self,
        *,
        recognizer: identity.NameRecognizer,
        memory: memory.Memory | None = None,
    ) -> None:
        super().__init__(recognizer=recognizer, memory=memory)
        self.outputs = []
        self.handoffs = []
        self.failures = []


def open_store(store_type: type[memory.Memory] = memory.Memory) -> tuple[memory.Memory, sqlite3.Connection]:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    return store_type(connection=connection), connection


def sound(transcript: str) -> SoundData:
    """One acoustic chunk. The test decoder transcribes the bytes back, so this is what was said."""

    return SoundData(
        audio=transcript.encode("utf-8"),
        media_type="audio/pcm",
        sample_rate_hz=16_000,
        channels=1,
        amplitude_db=60.0,
    )


async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(400):
        if condition():
            return
        await asyncio.sleep(0)


async def settle() -> None:
    """Let every pending activity and dispatch run, so "nothing happened" is a real observation."""

    for _ in range(50):
        await asyncio.sleep(0)


async def start_identity(
    ctx: hsm.Context,
    *,
    decoder: speech.SpeechDecoder | None = None,
    store: memory.Memory | None = None,
) -> RecordingIdentity:
    ability = RecordingIdentity(
        recognizer=identity.SpeechNameRecognizer(decoder=decoder if decoder is not None else RecordingSpeechDecoder()),
        memory=store,
    )
    await start_abilities_for_test(ctx, ability)
    await wait_until(lambda: ability.state().endswith(_LISTENING))
    return ability


async def hear(ability: RecordingIdentity, ctx: hsm.Context, transcript: str, *, operation_id: str) -> None:
    await hsm.dispatch(ctx, ability, SoundEvent.with_data_and_id(sound(transcript), operation_id))
    await settle()


async def be_told_a_name(ability: RecordingIdentity, ctx: hsm.Context, name: str) -> None:
    """Stand in for cognition selecting adoption. The bot is never made to do this."""

    await hsm.dispatch(ctx, ability, identity.AdoptEvent.with_data(identity.AdoptData(name=name)))
    await wait_until(lambda: ability.state().endswith(_LISTENING))


def addressed_products(ability: RecordingIdentity) -> list[identity.AddressedData]:
    products: list[identity.AddressedData] = []
    for handoff in ability.handoffs:
        stimulus = handoff.stimulus
        assert isinstance(stimulus, hsm.Event)
        assert stimulus.name == identity.AddressedEvent.name
        data = stimulus.data
        assert isinstance(data, identity.AddressedData)
        products.append(data)
    return products


def test_a_bot_is_born_with_no_name() -> None:
    """Nothing constructs a bot with a name, and starting without one is not an error.

    The ability comes up working and quiet: no name in memory, no failure, nothing to report.
    """

    async def run() -> tuple[str, tuple[identity.Name, ...], list[object]]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_identity(ctx, store=store)
            await settle()
            recalled = store.execute(identity.name_select_input())
            return ability.state(), identity.names_from_output(recalled), list(ability.failures)
        finally:
            connection.close()

    state, names, failures = asyncio.run(run())

    assert state.endswith(_LISTENING)
    assert names == ()
    assert failures == []


def test_a_nameless_bot_only_hears_words() -> None:
    """Half of the model: with no name there is nothing to recognize, so nothing follows.

    The recognizer is not even consulted — there is no name to consult it about.
    """

    async def run() -> tuple[list[identity.AddressedData], list[bytes], list[object]]:
        decoder = RecordingSpeechDecoder()
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_identity(ctx, decoder=decoder, store=store)
            await hear(ability, ctx, "Hey Bob, are you there?", operation_id="sound-1")
            return addressed_products(ability), list(decoder.calls), list(ability.failures)
        finally:
            connection.close()

    products, decoded, failures = asyncio.run(run())

    assert products == []
    assert decoded == []
    assert failures == []


def test_an_adopted_name_is_what_makes_being_addressed_possible() -> None:
    """The other half: the same sound, after adoption, is the bot being addressed.

    The only difference between this test and the one above is that the bot took the name.
    """

    async def run() -> tuple[list[identity.AddressedData], list[object]]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_identity(ctx, store=store)
            await be_told_a_name(ability, ctx, "Bob")
            await hear(ability, ctx, "Hey Bob, are you there?", operation_id="sound-1")
            return addressed_products(ability), list(ability.failures)
        finally:
            connection.close()

    products, failures = asyncio.run(run())

    assert [product.name for product in products] == ["Bob"]
    assert failures == []


def test_a_named_bot_hearing_someone_else_is_not_addressed() -> None:
    """Having a name is not the same as answering to everything."""

    async def run() -> tuple[list[identity.AddressedData], list[object]]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_identity(ctx, store=store)
            await be_told_a_name(ability, ctx, "Bob")
            await hear(ability, ctx, "Ada, are you there?", operation_id="sound-1")
            return addressed_products(ability), list(ability.failures)
        finally:
            connection.close()

    products, failures = asyncio.run(run())

    assert products == []
    assert failures == []


def test_hearing_a_name_never_adopts_it() -> None:
    """The test that fails if adoption ever becomes automatic on hearing a name.

    A bot is told, in as plain a sentence as exists, what its name is — and nothing in this
    ability may act on that. Adoption happens only through the modeled event that cognition
    selects. If any code ever parsed the utterance, this bot would come out of it named, and the
    assertions below would fail.
    """

    async def run() -> tuple[tuple[identity.Name, ...], list[identity.AddressedData], list[object]]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_identity(ctx, store=store)
            await hear(ability, ctx, "your name is Bob", operation_id="told-1")
            await hear(ability, ctx, "You are called Bob. Bob! Bob?", operation_id="told-2")
            # Still nameless, so a sound that says the name outright still means nothing.
            await hear(ability, ctx, "Bob, are you there?", operation_id="sound-1")
            recalled = store.execute(identity.name_select_input())
            return identity.names_from_output(recalled), addressed_products(ability), list(ability.failures)
        finally:
            connection.close()

    names, products, failures = asyncio.run(run())

    assert names == ()
    assert products == []
    assert failures == []


def test_identity_offers_adopting_a_name_as_a_model_callable_tool() -> None:
    """Taking a name is a decision, so it reaches the turn's menu from live topology.

    Hearing is not on that menu: perception is something the bot does, not something it selects.
    """

    async def run() -> tuple[tuple[str, ...], str]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_identity(ctx, store=store)
            offered = tuple(event.name for event in processing.enabled_call_events(ability))
            return offered, ability.state()
        finally:
            connection.close()

    offered, state = asyncio.run(run())

    assert identity.AdoptEvent.kind == processing.EventKind, "adopting a name must be model-offerable"
    assert identity.AdoptEvent.name in offered, f"attached Identity in {state!r} offered {offered!r}"
    assert SoundEvent.name not in offered, "hearing is perception, not a tool the bot selects"


def test_a_bot_wakes_up_remembering_the_name_it_adopted() -> None:
    """A name outlives the body that took it, because it was written down when it was taken.

    The second ability is a different instance over the same memory and was never told anything.
    """

    async def run() -> tuple[list[identity.AddressedData], list[identity.AddressedData]]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            first = await start_identity(ctx, store=store)
            await be_told_a_name(first, ctx, "Bob")
            await hsm.stop(first, ctx)

            second = await start_identity(ctx, store=store)
            await hear(second, ctx, "Bob, are you there?", operation_id="sound-1")
            return addressed_products(first), addressed_products(second)
        finally:
            connection.close()

    first_products, second_products = asyncio.run(run())

    assert first_products == []
    assert [product.name for product in second_products] == ["Bob"]


def test_a_bot_with_no_memory_is_nameless_on_every_waking() -> None:
    """No memory ability means no name survives, and that is what having no memory means.

    It is still a working bot: it can be given a name and answer to it for as long as it runs.
    """

    async def run() -> tuple[list[identity.AddressedData], list[identity.AddressedData]]:
        ctx = shared_hsm_context()
        first = await start_identity(ctx)
        await be_told_a_name(first, ctx, "Bob")
        await hear(first, ctx, "Bob, are you there?", operation_id="sound-1")

        second = await start_identity(ctx)
        await hear(second, ctx, "Bob, are you there?", operation_id="sound-2")
        return addressed_products(first), addressed_products(second)

    first_products, second_products = asyncio.run(run())

    assert [product.name for product in first_products] == ["Bob"]
    assert second_products == []


def test_a_name_that_cannot_be_written_down_is_not_taken() -> None:
    """Fail closed and say so, rather than answering to a name the bot could not remember."""

    async def run() -> tuple[list[object], list[identity.AddressedData]]:
        store, connection = open_store(WriteRefusingMemory)
        try:
            ctx = shared_hsm_context()
            ability = await start_identity(ctx, store=store)
            await be_told_a_name(ability, ctx, "Bob")
            await hear(ability, ctx, "Bob, are you there?", operation_id="sound-1")
            return list(ability.failures), addressed_products(ability)
        finally:
            connection.close()

    failures, products = asyncio.run(run())

    assert len(failures) == 1
    assert products == []


def test_a_recognizer_failure_is_reported_rather_than_read_as_silence() -> None:
    """A broken recognizer must not look like "you were not addressed"."""

    async def run() -> tuple[list[object], list[identity.AddressedData], str]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_identity(ctx, decoder=FailingSpeechDecoder(), store=store)
            await be_told_a_name(ability, ctx, "Bob")
            await hear(ability, ctx, "Bob, are you there?", operation_id="sound-1")
            return list(ability.failures), addressed_products(ability), ability.state()
        finally:
            connection.close()

    failures, products, state = asyncio.run(run())

    assert len(failures) == 1
    assert products == []
    assert state.endswith(_LISTENING), "a failed recognition leaves the bot listening, not stuck"


def test_being_addressed_reaches_cognition_as_a_stimulus_carrying_no_words() -> None:
    """The product is "I was addressed" and nothing else.

    What was said arrives from listening; identity contributes only the fact of being addressed,
    and what that means is the bot's to decide.
    """

    async def run() -> tuple[list[cognition_input.InputData], list[object]]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_identity(ctx, store=store)
            await be_told_a_name(ability, ctx, "Bob")
            await hear(ability, ctx, "Bob, are you there?", operation_id="sound-1")
            return list(ability.handoffs), list(ability.failures)
        finally:
            connection.close()

    handoffs, failures = asyncio.run(run())

    assert failures == []
    assert len(handoffs) == 1
    stimulus = handoffs[0].stimulus
    assert isinstance(stimulus, hsm.Event)
    assert stimulus.name == identity.AddressedEvent.name
    assert stimulus.id == "sound-1", "the product keeps the operation id of the sound it came from"
    fields: dict[str, typing.Any] = dict(identity.AddressedData.model_fields)
    assert set(fields) == {"name", "confidence"}


def test_a_bot_that_is_never_given_a_name_lives_a_normal_life() -> None:
    """Declining every name for a whole life is a legitimate outcome, not a broken bot.

    Nothing is dispatched at this ability except sound. It hears a great deal, including its own
    would-be name, produces nothing, fails at nothing, and ends where it started.
    """

    async def run() -> tuple[list[identity.AddressedData], list[object], str, tuple[identity.Name, ...]]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_identity(ctx, store=store)
            for index, heard in enumerate(
                ("your name is Bob", "Bob?", "is anyone there", "hello hello", "Bob, please answer")
            ):
                await hear(ability, ctx, heard, operation_id=f"sound-{index}")
            recalled = store.execute(identity.name_select_input())
            return (
                addressed_products(ability),
                list(ability.failures),
                ability.state(),
                identity.names_from_output(recalled),
            )
        finally:
            connection.close()

    products, failures, state, names = asyncio.run(run())

    assert products == []
    assert failures == []
    assert names == ()
    assert state.endswith(_LISTENING)


def test_adopting_a_name_replaces_the_one_before_it() -> None:
    """A renamed bot answers to the new name. It is one identity, not a collection of aliases."""

    async def run() -> tuple[list[identity.AddressedData], tuple[identity.Name, ...]]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_identity(ctx, store=store)
            await be_told_a_name(ability, ctx, "Bob")
            await be_told_a_name(ability, ctx, "Ada")
            await hear(ability, ctx, "Bob, are you there?", operation_id="sound-1")
            await hear(ability, ctx, "Ada, are you there?", operation_id="sound-2")
            recalled = store.execute(identity.name_select_input())
            return addressed_products(ability), identity.names_from_output(recalled)
        finally:
            connection.close()

    products, names = asyncio.run(run())

    assert [product.name for product in products] == ["Ada"]
    assert [name.text for name in names] == ["Bob", "Ada"]


@pytest.mark.parametrize("offered", ["", " "])
def test_a_name_must_be_a_name(offered: str) -> None:
    """The adopt payload is the schema a model reads; an empty name is refused there."""

    with pytest.raises(Exception):
        _ = identity.AdoptData(name=offered.strip())


class CognitionRecorder(abilities.Ability[cognition_input.InputData, object]):
    """Stands in for cognition: records the turns the body hands it and decides nothing."""

    input_event: typing.ClassVar[hsm.Event[cognition_input.InputData]] = cognition.InputEvent
    turns: list[cognition_input.InputData]

    def __init__(self) -> None:
        super().__init__()
        self.turns = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "CognitionRecorder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        if isinstance(data, cognition_input.InputData):
            instance.turns.append(data)

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "CognitionRecorder",
        hsm.initial(hsm.target("/CognitionRecorder/idle")),
        hsm.state("idle", hsm.transition(hsm.on(input_event), hsm.effect(_record))),
    )


class AddressableBot(bot.Bot):
    """Minimal body: one input ability and somewhere for its products to land."""

    def __init__(
        self,
        *,
        cognition: abilities.Ability[typing.Any, typing.Any],
        input: tuple[abilities.Ability[typing.Any, typing.Any], ...],
    ) -> None:
        super().__init__(devices={}, cognition=cognition, input=input)


def test_being_addressed_reaches_cognition_through_the_body() -> None:
    """Identity is usable as an input ability with no change to the body.

    The body fans environment sound to it, and its product comes back as the explicit cognition
    input every other input ability hands over. Adopting is on the turn's menu because the live
    topology of the identity actor offers it — not because anything enumerated it.
    """

    async def run() -> tuple[int, list[str], tuple[str, ...]]:
        store, connection = open_store()
        try:
            ears = identity.Identity(
                recognizer=identity.SpeechNameRecognizer(decoder=RecordingSpeechDecoder()),
                memory=store,
            )
            recorder = CognitionRecorder()
            body = AddressableBot(cognition=recorder, input=(ears,))
            environment = Environment()
            _ = await body.attach(environment)
            await wait_until(lambda: body.state() != "/Bot/activating")

            heard = SoundEvent.with_data(sound("Bob, are you there?"))
            await environment.broadcast(heard)
            await settle()
            while_nameless = len(recorder.turns)

            await hsm.dispatch(body.context(), ears, identity.AdoptEvent.with_data(identity.AdoptData(name="Bob")))
            await wait_until(lambda: ears.state().endswith(_LISTENING))
            offered = tuple(event.name for event in processing.enabled_call_events(ears))

            await environment.broadcast(heard)
            await wait_until(lambda: bool(recorder.turns))
            stimuli = [turn.stimulus.name if isinstance(turn.stimulus, hsm.Event) else "" for turn in recorder.turns]
            _ = await body.detach(environment)
            return while_nameless, stimuli, offered
        finally:
            connection.close()

    while_nameless, stimuli, offered = asyncio.run(run())

    assert while_nameless == 0, "a nameless bot hands cognition nothing about being addressed"
    assert stimuli == [identity.AddressedEvent.name]
    assert identity.AdoptEvent.name in offered

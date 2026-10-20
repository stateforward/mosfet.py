"""People: the bot's capacity to know who people are, which it learns itself.

Every test pins a *capacity* and never a choice. A link, an unlink, a name or a forget happens only
because the test dispatched it itself, standing in for cognition selecting the tool; nothing here
asserts that a bot hearing a stranger asks who they are, or links anyone. A bot that never touches
these tools passes this whole file.

The parts are real: a file-backed ``memory.Memory`` and a voice ``identification.Registry`` over a
``Matcher`` and an ``InProcessStore``, all reached by message.
"""

import asyncio
import collections.abc
import dataclasses
import datetime
import inspect
import pathlib
import typing
import uuid

import hsm
import mosfet
import pydantic
import pytest

from mosfet import environment, event, lifecycle
from mosfet.abilities import ability, cognition, identification, memory, people, processing
from mosfet.protocols import attachment

from tests import metric_reader
from tests.bot.abilities.support import EventRecorder

_WAIT = 5.0
_POLICY = identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65)
# Two orthogonal voiceprints: similarity 0, so they are never one voice.
_A = identification.Print(vector=(0.6180339887, 0.7861513778), model="voice-model")
_B = identification.Print(vector=(-0.7861513778, 0.6180339887), model="voice-model")
_TOOLS = {people.LinkEvent.name, people.UnlinkEvent.name, people.NameEvent.name, people.ForgetEvent.name}
_FAR_FUTURE = datetime.datetime(2100, 1, 1, tzinfo=datetime.UTC)


async def _until(predicate: collections.abc.Callable[[], bool]) -> None:
    async with asyncio.timeout(_WAIT):
        while not predicate():
            await asyncio.sleep(0.005)


def _offered(instance: hsm.Instance) -> set[str]:
    return {offered.name for offered in processing.enabled_call_events(instance)}


@dataclasses.dataclass
class Harness:
    ctx: environment.Environment
    owner: EventRecorder
    store: memory.Memory
    prints: identification.InProcessStore
    voices: identification.Registry
    capacity: people.People

    async def attach(self, part: ability.Ability[typing.Any, typing.Any]) -> None:
        """Stand in for the owner that attaches ``part``; wait for it to say it is attached."""

        _ = await part.attach(
            self.ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=self.owner), uuid.uuid4().hex)
        )
        await _until(
            lambda: any(
                seen.name == attachment.AttachCompleteEvent.name and seen.source == hsm.id(part)
                for seen in self.owner.events
            )
        )

    async def serving(self) -> None:
        await _until(lambda: _offered(self.capacity) == _TOOLS)

    async def ask(
        self, child: hsm.Instance, request: hsm.Event[typing.Any], data: object, *answers: hsm.Event[typing.Any]
    ) -> object:
        """Send one request to a peer by message and return its one correlated reply's data."""

        terminal = await ability.run_terminal_operation(
            self.ctx,
            child=child,
            request=request.with_data_and_id(data, uuid.uuid4().hex),
            terminals=answers,
            timeout=datetime.timedelta(seconds=_WAIT),
        )
        return terminal.data

    async def heard(self, voiceprint: identification.Print) -> str:
        """The ref of the voice ``voiceprint`` is, as the ear would learn it from the registry."""

        identified = await self.ask(
            self.voices,
            identification.IdentifyEvent,
            identification.IdentifyData(prints=(voiceprint,)),
            identification.IdentifiedEvent,
            identification.FailedEvent,
        )
        assert isinstance(identified, identification.IdentifiedData), identified
        (recognition,) = identified.recognitions
        assert recognition is not None
        return identification.ref("voice", recognition.id)

    async def own(self, voiceprint: identification.Print) -> str:
        """The ref of the bot's own voice, enrolled the way speaking enrolls it."""

        enrolled = await self.ask(
            self.voices,
            identification.EnrollEvent,
            identification.EnrollData(id=None, print=voiceprint, own=True),
            identification.EnrolledEvent,
            identification.FailedEvent,
        )
        assert isinstance(enrolled, identification.Recognition), enrolled
        return identification.ref("voice", enrolled.id)

    async def who(self, ref: str) -> memory.Present:
        """Who memory says ``ref`` is on the next turn it takes part in."""

        found = await self.ask(
            self.store,
            memory.SearchEvent,
            memory.SearchData(
                cue=(),
                participants=(mosfet.Participant(ref=ref, role=mosfet.ParticipantRole.SOURCE),),
                nearest=0,
                recent=0,
                present=0,
                linked=0,
            ),
            memory.FoundEvent,
            memory.FailedEvent,
        )
        assert isinstance(found, memory.FoundData), found
        (present,) = found.people
        return present

    async def persons(self) -> tuple[memory.Person, ...]:
        listed = await self.ask(
            self.store, memory.PersonEvent, memory.PersonListData(), memory.PersonDoneEvent, memory.FailedEvent
        )
        assert isinstance(listed, memory.PersonDoneData), listed
        return listed.persons

    def observations(self) -> list[hsm.Event[typing.Any]]:
        """What People told the bot: the stimulus of each cognition input it handed its owner."""

        observed: list[hsm.Event[typing.Any]] = []
        for seen in self.owner.events:
            if seen.name != cognition.InputEvent.name or not isinstance(seen.data, cognition.InputData):
                continue
            stimulus = seen.data.stimulus
            if isinstance(stimulus, hsm.Event):
                observed.append(stimulus)
        return observed

    async def select(self, request: hsm.Event[typing.Any], data: pydantic.BaseModel) -> hsm.Event[typing.Any]:
        """Stand in for cognition selecting one People tool, and return what the bot is told about it."""

        request_id = uuid.uuid4().hex
        _ = await hsm.dispatch(self.ctx, self.capacity, request.with_data_and_id(data, request_id))
        await _until(lambda: any(observed.id == request_id for observed in self.observations()))
        (observed,) = [observed for observed in self.observations() if observed.id == request_id]
        return observed

    def prune_unlinked(self) -> int:
        """Retention, run far in the future: every print no person holds is pruned."""

        return self.prints.prune("voice", unseen_before=_FAR_FUTURE, max_ids=64)

    async def stop(self) -> None:
        for part in (self.capacity, self.voices, self.store):
            if lifecycle.is_started(part):
                await part.stop(self.ctx)


async def _harness(tmp_path: pathlib.Path, *, timeout: datetime.timedelta | None = None) -> Harness:
    ctx = environment.Environment()
    owner = EventRecorder()
    _ = await mosfet.started(ctx, owner, EventRecorder.model)
    prints = identification.InProcessStore(max_ids=64)
    voices = identification.Registry(
        identifier=identification.Matcher(store=prints, policy=_POLICY),
        kind="voice",
        max_unlinked_age=datetime.timedelta(days=30),
        max_ids=64,
    )
    store = memory.Memory(database=str(tmp_path / "memory.db"))
    harness = Harness(
        ctx=ctx,
        owner=owner,
        store=store,
        prints=prints,
        voices=voices,
        capacity=people.People(memory=store, registries=(voices,))
        if timeout is None
        else people.People(memory=store, registries=(voices,), timeout=timeout),
    )
    await harness.attach(store)
    await harness.attach(voices)
    await harness.attach(harness.capacity)
    return harness


def _run[T](
    tmp_path: pathlib.Path,
    scenario: collections.abc.Callable[[Harness], collections.abc.Awaitable[T]],
) -> T:
    async def run() -> T:
        harness = await _harness(tmp_path)
        try:
            await harness.serving()
            return await scenario(harness)
        finally:
            await harness.stop()

    return asyncio.run(run())


def _changed(observed: hsm.Event[typing.Any]) -> people.ChangedData:
    assert observed.name == people.ChangedEvent.name, observed
    assert isinstance(observed.data, people.ChangedData)
    return observed.data


def _failed(observed: hsm.Event[typing.Any]) -> people.FailureData:
    assert observed.name == people.FailedEvent.name, observed
    assert isinstance(observed.data, people.FailureData)
    return observed.data


# --- construction ---------------------------------------------------------------------------------


def test_constructing_people_starts_nothing_and_refuses_what_it_cannot_serve() -> None:
    store = memory.Memory()
    voices = identification.Registry(
        identifier=identification.Matcher(store=identification.InProcessStore(max_ids=4), policy=_POLICY),
        kind="voice",
        max_unlinked_age=datetime.timedelta(days=30),
        max_ids=4,
    )
    other = identification.Registry(
        identifier=identification.Matcher(store=identification.InProcessStore(max_ids=4), policy=_POLICY),
        kind="voice",
        max_unlinked_age=datetime.timedelta(days=30),
        max_ids=4,
    )

    built = people.People(memory=store, registries=(voices,))

    assert built.identity() == "people"
    assert built.memory() is store
    assert not lifecycle.is_started(built)
    assert not lifecycle.is_started(store)
    assert not lifecycle.is_started(voices)
    with pytest.raises(ValueError, match="voice"):
        _ = people.People(memory=store, registries=(voices, other))
    with pytest.raises(ValueError, match="timeout"):
        _ = people.People(memory=store, timeout=datetime.timedelta(0))


# --- the schemas the bot is offered ---------------------------------------------------------------


def test_the_tools_are_model_callable_and_never_offer_a_role_or_birth_year() -> None:
    requests = (people.LinkEvent, people.UnlinkEvent, people.NameEvent, people.ForgetEvent)
    observations = (people.ChangedEvent, people.FailedEvent)

    assert all(request.kind is event.EventKind for request in requests)
    assert all(observation.kind is not event.EventKind for observation in observations)
    for data in (people.LinkData, people.UnlinkData, people.NameData, people.ForgetData):
        schema = data.model_json_schema()
        properties = typing.cast(dict[str, dict[str, object]], schema["properties"])
        assert schema.get("description"), data
        assert not {"role", "born", "declared"} & set(properties), data
        assert all(field.get("description") for field in properties.values()), data
    # A name and a ref are complete answers: an example of either names nobody, or someone else.
    link = typing.cast(dict[str, dict[str, object]], people.LinkData.model_json_schema()["properties"])
    assert "examples" not in link["ref"]
    assert all("examples" not in branch for branch in typing.cast(list[dict[str, object]], link["name"]["anyOf"]))


def test_people_takes_no_household() -> None:
    """R-11: nobody lists the household; who people are is what the bot learns."""

    assert "household" not in inspect.signature(people.People.__init__).parameters


def test_a_link_names_a_new_person_only() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = people.LinkData(ref="voice:3f2a9c", person="ada", name="Ada")


# --- serving ----------------------------------------------------------------------------------------


def test_linking_an_unknown_voice_to_a_new_named_person_makes_the_next_turn_know_them(
    tmp_path: pathlib.Path,
) -> None:
    async def scenario(harness: Harness) -> tuple[memory.Present, people.ChangedData, memory.Present, str]:
        ref = await harness.heard(_A)
        stranger = await harness.who(ref)
        changed = _changed(await harness.select(people.LinkEvent, people.LinkData(ref=ref, name="Sam")))
        known = await harness.who(ref)
        _ = harness.prune_unlinked()
        return stranger, changed, known, ref if await harness.heard(_A) == ref else ""

    stranger, changed, known, kept = _run(tmp_path, scenario)

    assert stranger.known is False
    assert changed.op == "link"
    assert changed.person.name == "Sam"
    assert changed.person.refs == (stranger.ref,)
    assert (known.person, known.known) == (changed.person.id, True)
    # The print of a voice someone holds is kept through retention.
    assert kept == stranger.ref


def test_a_person_can_have_two_voices(tmp_path: pathlib.Path) -> None:
    async def scenario(harness: Harness) -> tuple[str, str, people.ChangedData, memory.Present]:
        first, second = await harness.heard(_A), await harness.heard(_B)
        sam = _changed(await harness.select(people.LinkEvent, people.LinkData(ref=first, name="Sam"))).person
        both = _changed(await harness.select(people.LinkEvent, people.LinkData(ref=second, person=sam.id)))
        return first, second, both, await harness.who(second)

    first, second, both, present = _run(tmp_path, scenario)

    assert first != second
    assert set(both.person.refs) == {first, second}
    assert (present.person, present.known) == (both.person.id, True)


def test_a_voice_that_is_someone_elses_is_a_conflict_and_their_link_stands(tmp_path: pathlib.Path) -> None:
    async def scenario(harness: Harness) -> tuple[str, str, people.FailureData, memory.Present, bool]:
        ref = await harness.heard(_A)
        sam = _changed(await harness.select(people.LinkEvent, people.LinkData(ref=ref, name="Sam"))).person
        refused = _failed(await harness.select(people.LinkEvent, people.LinkData(ref=ref, name="Max")))
        present = await harness.who(ref)
        _ = harness.prune_unlinked()
        return ref, sam.id, refused, present, await harness.heard(_A) == ref

    ref, sam, refused, present, kept = _run(tmp_path, scenario)

    assert (refused.op, refused.kind) == ("link", people.FailureKind.CONFLICT)
    assert (present.ref, present.person, present.known) == (ref, sam, True)
    assert kept


def test_the_bots_own_voice_is_never_linked_to_anyone(tmp_path: pathlib.Path) -> None:
    async def scenario(harness: Harness) -> tuple[people.FailureData, tuple[memory.Person, ...]]:
        own = await harness.own(_A)
        refused = _failed(await harness.select(people.LinkEvent, people.LinkData(ref=own, name="Sam")))
        return refused, await harness.persons()

    refused, persons = _run(tmp_path, scenario)

    assert (refused.op, refused.kind) == ("link", people.FailureKind.CONFLICT)
    assert [person.name for person in persons if person.name is not None] == []


def test_a_link_memory_refuses_changes_no_mark(tmp_path: pathlib.Path) -> None:
    async def scenario(harness: Harness) -> tuple[str, people.FailureData, int, str, people.FailureData, int, bool]:
        ref = await harness.heard(_A)
        refused = _failed(await harness.select(people.LinkEvent, people.LinkData(ref=ref, person="nobody")))
        pruned = harness.prune_unlinked()
        heard_again = await harness.heard(_A)
        _ = _changed(await harness.select(people.LinkEvent, people.LinkData(ref=heard_again, name="Sam")))
        moved = _failed(await harness.select(people.LinkEvent, people.LinkData(ref=heard_again, person="nobody")))
        return (
            ref,
            refused,
            pruned,
            heard_again,
            moved,
            harness.prune_unlinked(),
            await harness.heard(_A) == heard_again,
        )

    ref, refused, pruned, heard_again, moved, kept_pruned, kept = _run(tmp_path, scenario)

    assert (refused.op, refused.kind) == ("link", people.FailureKind.NOT_FOUND)
    # Nothing was marked, so the voice nobody holds is pruned and is new when heard again.
    assert pruned == 1
    assert heard_again != ref
    # A refused link of a voice someone named leaves its mark: their print is still kept.
    assert moved.kind is people.FailureKind.NOT_FOUND
    assert (kept_pruned, kept) == (0, True)


def test_a_voice_linked_to_a_person_nobody_has_named_is_not_kept(tmp_path: pathlib.Path) -> None:
    """The mark follows known (D-W8-18, D-R-39): a link into a person nobody named keeps no print."""

    async def scenario(harness: Harness) -> tuple[str, memory.Present, int, str]:
        ref = await harness.heard(_A)
        _ = _changed(await harness.select(people.LinkEvent, people.LinkData(ref=ref)))
        present = await harness.who(ref)
        return ref, present, harness.prune_unlinked(), await harness.heard(_A)

    ref, present, pruned, heard_again = _run(tmp_path, scenario)

    assert present.known is False
    assert pruned == 1
    assert heard_again != ref


def test_naming_a_person_keeps_every_one_of_their_prints(tmp_path: pathlib.Path) -> None:
    """Naming makes a person known, so each voice of theirs is kept through retention (D-W8-18)."""

    async def scenario(harness: Harness) -> tuple[tuple[str, str], memory.Present, int, tuple[str, str]]:
        first, second = await harness.heard(_A), await harness.heard(_B)
        stranger = await harness.who(first)
        _ = _changed(await harness.select(people.LinkEvent, people.LinkData(ref=second, person=stranger.person)))
        _ = _changed(await harness.select(people.NameEvent, people.NameData(person=stranger.person, name="Sam")))
        known = await harness.who(second)
        return (first, second), known, harness.prune_unlinked(), (await harness.heard(_A), await harness.heard(_B))

    before, known, pruned, after = _run(tmp_path, scenario)

    assert known.known is True
    assert pruned == 0
    assert after == before


def test_unlinking_a_voice_gives_it_back_to_nobody(tmp_path: pathlib.Path) -> None:
    async def scenario(harness: Harness) -> tuple[str, str, people.ChangedData, memory.Present, int]:
        ref = await harness.heard(_A)
        sam = _changed(await harness.select(people.LinkEvent, people.LinkData(ref=ref, name="Sam"))).person
        unlinked = _changed(await harness.select(people.UnlinkEvent, people.UnlinkData(ref=ref)))
        present = await harness.who(ref)
        return ref, sam.id, unlinked, present, harness.prune_unlinked()

    ref, sam, unlinked, present, pruned = _run(tmp_path, scenario)

    assert unlinked.op == "unlink"
    assert (unlinked.person.name, unlinked.person.refs) == (None, (ref,))
    assert unlinked.person.id != sam
    assert (present.person, present.known) == (unlinked.person.id, False)
    assert pruned == 1


def test_naming_a_person_the_bot_learned(tmp_path: pathlib.Path) -> None:
    async def scenario(harness: Harness) -> tuple[people.ChangedData, memory.Present]:
        ref = await harness.heard(_A)
        person = _changed(await harness.select(people.LinkEvent, people.LinkData(ref=ref))).person
        named = _changed(await harness.select(people.NameEvent, people.NameData(person=person.id, name="Sam")))
        return named, await harness.who(ref)

    named, present = _run(tmp_path, scenario)

    assert (named.op, named.person.name) == ("name", "Sam")
    assert (present.person, present.known) == (named.person.id, True)


def test_forgetting_a_person_erases_their_prints(tmp_path: pathlib.Path) -> None:
    async def scenario(
        harness: Harness,
    ) -> tuple[tuple[str, str], people.ChangedData, tuple[str, str], tuple[memory.Person, ...]]:
        first, second = await harness.heard(_A), await harness.heard(_B)
        sam = _changed(await harness.select(people.LinkEvent, people.LinkData(ref=first, name="Sam"))).person
        _ = await harness.select(people.LinkEvent, people.LinkData(ref=second, person=sam.id))
        forgotten = _changed(await harness.select(people.ForgetEvent, people.ForgetData(person=sam.id)))
        return (first, second), forgotten, (await harness.heard(_A), await harness.heard(_B)), await harness.persons()

    before, forgotten, after, persons = _run(tmp_path, scenario)

    assert (forgotten.op, forgotten.person.name) == ("forget", "Sam")
    assert set(forgotten.person.refs) == set(before)
    # Their prints are gone: the same voices are strangers now, under ids never seen before.
    assert not set(after) & set(before)
    assert all(person.id != forgotten.person.id for person in persons)


def test_the_tools_are_offered_at_once_and_a_memory_that_cannot_answer_fails_the_statement(
    tmp_path: pathlib.Path,
) -> None:
    """Nothing is declared first (R-11): People serves as soon as it is attached."""

    async def run() -> tuple[set[str], people.FailureData]:
        harness = await _harness(tmp_path, timeout=datetime.timedelta(milliseconds=200))
        try:
            await harness.serving()
            offered = _offered(harness.capacity)
            await harness.store.stop(harness.ctx)
            refused = _failed(await harness.select(people.LinkEvent, people.LinkData(ref="voice:3f2a9c", name="Sam")))
            return offered, refused
        finally:
            await harness.stop()

    offered, refused = asyncio.run(run())

    assert offered == _TOOLS
    assert refused.op == "link"
    assert refused.kind in {people.FailureKind.UNAVAILABLE, people.FailureKind.TIMEOUT}


def test_a_detached_people_answers_a_selection_unavailable() -> None:
    async def run() -> hsm.Event[typing.Any]:
        ctx = environment.Environment()
        owner = EventRecorder()
        _ = await mosfet.started(ctx, owner, EventRecorder.model)
        detached = people.People(memory=memory.Memory())
        model = detached.model
        assert model is not None
        _ = await mosfet.started(ctx, detached, model)
        request = dataclasses.replace(
            people.ForgetEvent.with_data_and_id(people.ForgetData(person=memory.SELF), "forget"),
            source=hsm.id(owner),
            target=hsm.id(detached),
        )
        _ = await hsm.dispatch(ctx, detached, request)
        await _until(lambda: bool(owner.named(people.FailedEvent)))
        await detached.stop(ctx)
        return owner.named(people.FailedEvent)[0]

    answered = asyncio.run(run())

    assert answered.id == "forget"
    failure = _failed(answered)
    assert (failure.op, failure.kind) == ("forget", people.FailureKind.UNAVAILABLE)


def test_telemetry_counts_each_statement_with_closed_attributes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run(reader: metric_reader.Reader) -> dict[tuple[tuple[str, object], ...], float]:
        before = reader.points("bot.people.count")
        harness = await _harness(tmp_path)
        try:
            await harness.serving()
            ref = await harness.heard(_A)
            _ = await harness.select(people.LinkEvent, people.LinkData(ref=ref, name="Sam"))
            _ = await harness.select(people.LinkEvent, people.LinkData(ref=ref, name="Max"))
        finally:
            await harness.stop()
        after = reader.points("bot.people.count")
        counted = {tuple(sorted(point.attributes.items())): point.value for point in after}
        for point in before:
            key = tuple(sorted(point.attributes.items()))
            counted[key] = counted.get(key, 0.0) - point.value
        return {key: value for key, value in counted.items() if value}

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        counted = asyncio.run(run(reader))

    def key(op: str, outcome: str, kind: str) -> tuple[tuple[str, object], ...]:
        return (("bot.failure.kind", kind), ("bot.outcome", outcome), ("bot.people.op", op))

    assert counted == {
        key("link", "done", "none"): 1.0,
        key("link", "failed", "conflict"): 1.0,
    }

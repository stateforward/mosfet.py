"""HSM runtime tests for ``memory.Associative`` over a real ``memory.Memory`` peer."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import itertools
import typing

import hsm
import mosfet
import pytest

from mosfet import environment, media
from mosfet.abilities import ability, embedding, memory, processing
from mosfet.abilities.ability import Effort
from mosfet.abilities.cognition import event as projection
from mosfet.abilities.language import text
from mosfet.protocols import attachment

_PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"
_WAIT = 3.0
_NOW = datetime.datetime(2026, 10, 14, 12, 0, tzinfo=datetime.UTC)
_ADA = mosfet.Participant(ref="+15555550100", role=mosfet.ParticipantRole.SOURCE)
_SPACE = embedding.Space(model="test/embedder", dimensions=2)


class GatedEmbedder(embedding.Embedder):
    """Plain-object embedder that holds every request until ``gate`` is set."""

    def __init__(self, gate: asyncio.Event) -> None:
        self.capacity = embedding.Capacity(space=_SPACE, max_items=8, max_input_tokens=2048)
        self.attachments = ()
        self.gate = gate

    async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
        del ctx
        _ = await self.gate.wait()
        return embedding.OutputData(
            space=_SPACE,
            role=input.role,
            vectors=tuple(embedding.Vector(values=(1.0, 0.0)) for _ in input.items),
        )


class Owner(hsm.Instance):
    """Attachment owner and reply address that records every event delivered to it."""

    events: list[hsm.Event[typing.Any]]
    changed: asyncio.Event

    @staticmethod
    def _record(ctx: hsm.Context, instance: Owner, event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)
        instance.changed.set()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "AssociativeTestOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.changed = asyncio.Event()

    def replies(self, request_id: str) -> list[hsm.Event[typing.Any]]:
        names = {memory.AssociatedEvent.name, memory.RecollectedEvent.name, ability.FailedEvent.name}
        return [event for event in self.events if event.id == request_id and event.name in names]

    async def until(self, predicate: collections.abc.Callable[[], bool]) -> None:
        async with asyncio.timeout(_WAIT):
            while not predicate():
                self.changed.clear()
                _ = await self.changed.wait()


@dataclasses.dataclass
class Harness:
    ctx: environment.Environment
    memory: memory.Memory
    associative: memory.Associative
    owner: Owner
    memory_owner: Owner

    async def attach_memory(self) -> None:
        _ = await self.memory.attach(
            self.ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=self.memory_owner))
        )
        await self.memory_owner.until(
            lambda: any(e.name == attachment.AttachCompleteEvent.name for e in self.memory_owner.events)
        )

    async def attach(self) -> None:
        _ = await self.associative.attach(
            self.ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=self.owner), "attach")
        )
        await self.owner.until(lambda: any(e.name == attachment.AttachCompleteEvent.name for e in self.owner.events))

    async def detach(self) -> None:
        _ = await self.associative.detach(
            self.ctx, attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=self.owner), "detach")
        )

    async def send(self, cue: memory.CueData, request_id: str) -> None:
        _ = await hsm.dispatch(
            self.ctx,
            self.associative,
            dataclasses.replace(
                memory.AssociateEvent.with_data_and_id(cue, request_id),
                source=hsm.id(self.owner),
                target=hsm.id(self.associative),
            ),
        )

    async def recall(self, recall: memory.RecallData, request_id: str) -> None:
        _ = await hsm.dispatch(
            self.ctx,
            self.associative,
            dataclasses.replace(
                memory.RecallEvent.with_data_and_id(recall, request_id),
                source=hsm.id(self.owner),
                target=hsm.id(self.associative),
            ),
        )

    async def episodes(self) -> tuple[memory.Episode, ...]:
        reply = await ability.run_terminal_operation(
            self.ctx,
            child=self.memory,
            request=memory.RecentEvent.with_data_and_id(memory.RecentData(limit=16), "recent"),
            terminals=(memory.RecentOutputEvent, memory.FailedEvent),
            timeout=datetime.timedelta(seconds=_WAIT),
        )
        assert isinstance(reply.data, memory.EpisodesData), reply.data
        return reply.data.episodes

    async def reply(self, request_id: str) -> hsm.Event[typing.Any]:
        await self.owner.until(lambda: bool(self.owner.replies(request_id)))
        return self.owner.replies(request_id)[0]

    async def person(self, statement: memory.PersonData) -> None:
        reply = await ability.run_terminal_operation(
            self.ctx,
            child=self.memory,
            request=memory.PersonEvent.with_data_and_id(statement, f"person-{statement.op}"),
            terminals=(memory.PersonDoneEvent, memory.FailedEvent),
            timeout=datetime.timedelta(seconds=_WAIT),
        )
        assert isinstance(reply.data, memory.PersonDoneData), reply.data

    async def remember(self, episode: memory.EpisodeData) -> None:
        reply = await ability.run_terminal_operation(
            self.ctx,
            child=self.memory,
            request=memory.RememberEvent.with_data_and_id(episode, f"remember-{episode.turn_id}"),
            terminals=(memory.RememberedEvent, memory.FailedEvent),
            timeout=datetime.timedelta(seconds=_WAIT),
        )
        assert isinstance(reply.data, memory.RememberedData), reply.data


async def _harness(
    *,
    times: collections.abc.Iterator[datetime.datetime] | None = None,
    embedder: embedding.Embedder | None = None,
    budget_tokens: int = 1024,
    timeout: datetime.timedelta = datetime.timedelta(seconds=2),
) -> Harness:
    ctx = environment.Environment()
    owner, memory_owner = Owner(), Owner()
    for actor in (owner, memory_owner):
        _ = await mosfet.started(ctx, actor, Owner.model)
    # Memory stamps the bot's own person when it opens, before the episodes the test times, and
    # every search after them (who is present is resolved, and stamped, in the search).
    stamps = itertools.chain((_NOW,), times, itertools.repeat(_NOW)) if times is not None else iter(lambda: _NOW, None)
    store = memory.Memory(
        engine=memory.store.open_sqlite_engine(),
        embedder=embedder,
        now=lambda: next(stamps),
        embed_timeout=datetime.timedelta(seconds=5),
        operation_timeout=datetime.timedelta(seconds=10),
    )
    associative = memory.Associative(memory=store, budget_tokens=budget_tokens, now=lambda: _NOW, timeout=timeout)
    return Harness(ctx=ctx, memory=store, associative=associative, owner=owner, memory_owner=memory_owner)


def _episode(words: str, turn_id: str, *, participants: tuple[mosfet.Participant, ...] = ()) -> memory.EpisodeData:
    return memory.EpisodeData(
        turn_id=turn_id,
        stimulus="bot.ability.conversation.messages",
        parts=(text.TextPart(text=words),),
        participants=participants,
        outcome=memory.Outcome.COMPLETED,
    )


# --- lifecycle and contract ---------------------------------------------------------------------


def test_associative_attaches_through_its_empty_group_and_owns_a_detaching_state() -> None:
    async def scenario() -> str:
        harness = await _harness()
        await harness.attach()
        return harness.associative.state()

    assert asyncio.run(scenario()) == "/AssociativeLifecycle/attached/behavior/idle"
    model = memory.Associative.model
    assert model is not None
    for state in ("initializing", "idle", "busy", "detaching", "degraded"):
        assert f"/AssociativeLifecycle/attached/behavior/{state}" in model.members
    assert memory.Associative.effort is Effort.M
    assert memory.Associative.detached_requests == (memory.AssociateEvent, memory.RecallEvent)
    # Recall is a model-callable tool; associating is not.
    assert memory.RecallEvent.kind == processing.EventKind
    assert memory.AssociateEvent.kind != processing.EventKind


# --- scoring and budget -------------------------------------------------------------------------


def test_associations_rank_by_the_injected_weights_and_presence_lifts_who_is_here() -> None:
    """Golden values without an embedder: recency halves every two days, presence adds 0.5."""

    times = iter(
        (
            _NOW - datetime.timedelta(days=2),  # ada-old: 0.35 * 0.5 + 0.5 = 0.675
            _NOW,  # newest: 0.35 * 1.0 = 0.35
            _NOW - datetime.timedelta(days=4),  # ada-older: 0.35 * 0.25 + 0.5 = 0.5875
        )
    )

    async def scenario() -> hsm.Event[typing.Any]:
        harness = await _harness(times=times)
        await harness.attach_memory()
        await harness.remember(_episode("Ada asked about noon.", "ada-old", participants=(_ADA,)))
        await harness.remember(_episode("The kettle boiled.", "newest"))
        await harness.remember(_episode("Ada called last week.", "ada-older", participants=(_ADA,)))
        await harness.attach()
        await harness.send(memory.CueData(cue=(text.TextPart(text="Ada again"),), participants=(_ADA,)), "a-1")
        return await harness.reply("a-1")

    reply = asyncio.run(scenario())

    assert reply.name == memory.AssociatedEvent.name
    surfaced = reply.data
    assert isinstance(surfaced, memory.AssociationsData)
    assert [item.text for item in surfaced.items] == [
        "Ada asked about noon.",
        "Ada called last week.",
        "The kettle boiled.",
    ]
    assert surfaced.omitted == 0
    assert surfaced.items[0].involves == (_ADA.ref,)
    assert memory.Lane.PRESENT in surfaced.items[0].lanes


def test_the_token_budget_admits_best_first_and_counts_what_it_left_out() -> None:
    times = iter((_NOW - datetime.timedelta(minutes=3), _NOW - datetime.timedelta(minutes=2), _NOW))

    async def scenario() -> memory.AssociationsData:
        harness = await _harness(times=times, budget_tokens=25)
        await harness.attach_memory()
        for index in range(3):
            # 40 UTF-8 bytes: 10 approximate tokens each, so two fit a budget of 25.
            await harness.remember(_episode(f"memory {index} " + "x" * 31, f"turn-{index}"))
        await harness.attach()
        await harness.send(memory.CueData(), "a-1")
        reply = await harness.reply("a-1")
        assert isinstance(reply.data, memory.AssociationsData)
        return reply.data

    surfaced = asyncio.run(scenario())

    assert [item.text.split(" ")[1] for item in surfaced.items] == ["2", "1"]
    assert surfaced.omitted == 1


def test_a_surfaced_image_is_a_memory_placeholder_never_bytes() -> None:
    async def scenario() -> memory.AssociationsData:
        harness = await _harness()
        await harness.attach_memory()
        await harness.remember(
            memory.EpisodeData(
                turn_id="photo",
                parts=(
                    text.TextPart(text="the gate"),
                    text.ImagePart(image=media.Image(content=_PNG, media_type="image/png")),
                ),
                outcome=memory.Outcome.COMPLETED,
            )
        )
        await harness.attach()
        await harness.send(memory.CueData(), "a-1")
        reply = await harness.reply("a-1")
        assert isinstance(reply.data, memory.AssociationsData)
        return reply.data

    surfaced = asyncio.run(scenario())

    (item,) = surfaced.items
    assert [image.reason for image in item.images] == ["memory"]
    rendered = "".join(
        part.text
        for part in projection.model_facing_content(surfaced, capacity=media.ImageCapacity(max_images=8))
        if isinstance(part, text.TextPart)
    )
    assert 'omitted="memory"' in rendered
    assert 'omitted="retention"' not in rendered


# --- failure, timeout, exactly once -------------------------------------------------------------


def test_a_memory_that_cannot_be_reached_is_a_typed_failure() -> None:
    async def scenario() -> hsm.Event[typing.Any]:
        harness = await _harness()
        await harness.attach()
        await harness.send(memory.CueData(), "a-1")
        return await harness.reply("a-1")

    reply = asyncio.run(scenario())

    assert reply.name == ability.FailedEvent.name
    assert isinstance(reply.data, ability.FailureData)
    assert reply.data.message.startswith("unavailable")


def test_a_search_that_outlives_the_timeout_is_answered_timeout_once() -> None:
    async def scenario() -> tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]]]:
        gate = asyncio.Event()
        harness = await _harness(embedder=GatedEmbedder(gate), timeout=datetime.timedelta(milliseconds=200))
        await harness.attach_memory()
        await harness.attach()
        await harness.send(memory.CueData(cue=(text.TextPart(text="noon"),)), "a-1")
        reply = await harness.reply("a-1")
        gate.set()
        await harness.send(memory.CueData(), "a-2")
        _ = await harness.reply("a-2")
        assert reply.name == ability.FailedEvent.name
        return harness.owner.replies("a-1"), harness.owner.replies("a-2")

    replies, later = asyncio.run(scenario())

    assert len(replies) == 1
    assert isinstance(replies[0].data, ability.FailureData)
    assert replies[0].data.message.startswith("timeout")
    # The late search reply is ignored, and the next request is served with its own id.
    assert [reply.name for reply in later] == [memory.AssociatedEvent.name]


def test_detach_answers_the_request_in_flight_and_the_one_waiting_exactly_once() -> None:
    """Review finding: a request deferred behind the one in flight is answered on detach, not dropped."""

    async def scenario() -> dict[str, list[hsm.Event[typing.Any]]]:
        gate = asyncio.Event()
        harness = await _harness(embedder=GatedEmbedder(gate))
        await harness.attach_memory()
        await harness.attach()
        await harness.send(memory.CueData(cue=(text.TextPart(text="noon"),)), "in-flight")
        await harness.owner.until(lambda: "/busy/" in harness.associative.state())
        await harness.send(memory.CueData(), "waiting")
        await harness.detach()
        await harness.owner.until(lambda: harness.associative.state() == "/AssociativeLifecycle/detached")
        await harness.send(memory.CueData(), "after")
        _ = await harness.reply("after")
        gate.set()
        await asyncio.sleep(0.2)
        return {request: harness.owner.replies(request) for request in ("in-flight", "waiting", "after")}

    replies = asyncio.run(scenario())

    for request in ("in-flight", "waiting", "after"):
        assert len(replies[request]) == 1, (request, replies[request])
        assert replies[request][0].name == ability.FailedEvent.name
    # Each one is answered unavailable: the in-flight one by busy's exit, the waiting one where hsm
    # re-delivers it (detaching, or the detached lifecycle), the late one by the detached lifecycle.
    messages = {request: replies[request][0].data for request in ("in-flight", "waiting", "after")}
    for request, failure in messages.items():
        assert isinstance(failure, ability.FailureData), (request, failure)
        assert failure.message.startswith("unavailable"), (request, failure)
    after = messages["after"]
    assert isinstance(after, ability.FailureData)
    assert after.message == "unavailable: Associative is not attached."


@pytest.mark.parametrize("budget_tokens", [0, -1])
def test_budget_tokens_must_be_positive(budget_tokens: int) -> None:
    with pytest.raises(ValueError, match="budget_tokens"):
        _ = memory.Associative(memory=memory.Memory(), budget_tokens=budget_tokens)


# --- deliberate recall --------------------------------------------------------------------------


def _offered(associative: memory.Associative) -> set[str]:
    return {event.name for event in processing.enabled_call_events(associative)}


def test_recall_is_offered_while_idle_and_only_then() -> None:
    async def scenario() -> tuple[set[str], set[str]]:
        gate = asyncio.Event()
        harness = await _harness(embedder=GatedEmbedder(gate))
        await harness.attach_memory()
        await harness.attach()
        idle = _offered(harness.associative)
        await harness.send(memory.CueData(cue=(text.TextPart(text="noon"),)), "a-1")
        await harness.owner.until(lambda: "/busy" in harness.associative.state())
        busy = _offered(harness.associative)
        gate.set()
        _ = await harness.reply("a-1")
        return idle, busy

    idle, busy = asyncio.run(scenario())

    assert idle == {memory.RecallEvent.name}
    assert busy == set()


def test_recalling_a_node_tells_the_requester_in_full_with_image_bytes_and_reinforces_it() -> None:
    image = media.Image(content=_PNG, media_type="image/png")

    async def scenario() -> tuple[hsm.Event[typing.Any], tuple[memory.Episode, ...], str, str]:
        harness = await _harness()
        await harness.attach_memory()
        await harness.remember(
            memory.EpisodeData(
                turn_id="photo",
                parts=(text.TextPart(text="the gate"), text.ImagePart(image=image)),
                participants=(_ADA,),
                outcome=memory.Outcome.COMPLETED,
            )
        )
        (episode,) = await harness.episodes()
        await harness.attach()
        await harness.recall(memory.RecallData(node=episode.node_id), "r-1")
        reply = await harness.reply("r-1")
        return reply, await harness.episodes(), hsm.id(harness.owner), hsm.id(harness.associative)

    reply, after, requester, associative = asyncio.run(scenario())

    assert reply.name == memory.RecollectedEvent.name
    # The recollection goes to the owner (in a bot, Cognition), not to the stage that selected it.
    assert (reply.source, reply.target) == (associative, requester)
    recollection = reply.data
    assert isinstance(recollection, memory.RecollectionData)
    assert recollection.about is None
    (node,) = recollection.nodes
    assert node.text == "the gate"
    assert node.episode is not None
    assert node.episode.images == (image,)
    # Deliberate recall reinforces what it brought to mind.
    (episode,) = after
    assert episode.recall_count == 1
    assert episode.recalled_at is not None


def test_recalling_about_something_brings_the_best_memories_to_mind_within_the_image_bound() -> None:
    """At most ``limit`` memories, best first; image bytes for at most MAX_TURN_IMAGES, newest first."""

    image = media.Image(content=_PNG, media_type="image/png")
    # Three remembers, then the recall's reinforcement stamp.
    times = itertools.chain(
        (_NOW - datetime.timedelta(days=4), _NOW - datetime.timedelta(days=2)), itertools.repeat(_NOW)
    )

    async def scenario() -> hsm.Event[typing.Any]:
        harness = await _harness(times=times)
        await harness.attach_memory()
        await harness.remember(_episode("The kettle boiled.", "oldest"))
        for words, turn_id in (("the gate at dusk", "older"), ("the gate at dawn", "newest")):
            await harness.remember(
                memory.EpisodeData(
                    turn_id=turn_id,
                    parts=(text.TextPart(text=words), *(text.ImagePart(image=image),) * 5),
                    outcome=memory.Outcome.COMPLETED,
                )
            )
        await harness.attach()
        await harness.recall(memory.RecallData(about="the gate", limit=2), "r-1")
        return await harness.reply("r-1")

    reply = asyncio.run(scenario())

    assert reply.name == memory.RecollectedEvent.name, reply.data
    recollection = reply.data
    assert isinstance(recollection, memory.RecollectionData)
    assert recollection.about == "the gate"
    assert [node.text for node in recollection.nodes] == ["the gate at dawn", "the gate at dusk"]
    newest, older = (node.episode for node in recollection.nodes)
    assert newest is not None and older is not None
    assert all(isinstance(kept, media.Image) for kept in newest.images)
    assert [isinstance(kept, media.Image) for kept in older.images] == [True, True, True, False, False]
    assert all(kept.reason == "memory" for kept in older.images if isinstance(kept, media.ReleasedImage))


def test_a_recall_memory_cannot_serve_is_a_typed_failure_to_the_owner() -> None:
    async def scenario() -> hsm.Event[typing.Any]:
        harness = await _harness()
        await harness.attach()
        await harness.recall(memory.RecallData(about="noon"), "r-1")
        return await harness.reply("r-1")

    reply = asyncio.run(scenario())

    assert reply.name == ability.FailedEvent.name
    assert reply.target
    assert isinstance(reply.data, ability.FailureData)
    assert reply.data.message.startswith("unavailable")


def test_a_reply_goes_where_its_payload_says_never_where_its_name_says() -> None:
    """A recall's outcome is the owner's observation; any other request is answered at its requester.

    The routing reads the payload: a request sent under the recall event's name that carries a
    cue is not a recall, so its invalid-input failure goes back to the requester that sent it.
    """

    async def scenario() -> dict[str, tuple[list[hsm.Event[typing.Any]], list[hsm.Event[typing.Any]]]]:
        harness = await _harness()
        await harness.attach()
        requester = Owner()
        _ = await mosfet.started(harness.ctx, requester, Owner.model)
        requests: tuple[hsm.Event[typing.Any], ...] = (
            memory.RecallEvent.with_data_and_id(memory.RecallData(about="noon"), "recall"),
            memory.RecallEvent.with_data_and_id(memory.CueData(), "cue"),
        )
        for request in requests:
            _ = await hsm.dispatch(
                harness.ctx,
                harness.associative,
                dataclasses.replace(request, source=hsm.id(requester), target=hsm.id(harness.associative)),
            )
        await harness.owner.until(lambda: bool(harness.owner.replies("recall")))
        await requester.until(lambda: bool(requester.replies("cue")))
        await asyncio.sleep(0.05)
        return {request: (harness.owner.replies(request), requester.replies(request)) for request in ("recall", "cue")}

    replies = asyncio.run(scenario())

    to_owner, to_requester = replies["recall"]
    assert [reply.name for reply in to_owner] == [ability.FailedEvent.name]
    assert to_requester == []
    to_owner, to_requester = replies["cue"]
    assert to_owner == []
    (invalid,) = to_requester
    assert isinstance(invalid.data, ability.FailureData)
    assert invalid.data.message.startswith("invalid_input")


def test_a_recall_waiting_at_detach_and_one_after_it_are_answered_unavailable_once() -> None:
    async def scenario() -> dict[str, list[hsm.Event[typing.Any]]]:
        gate = asyncio.Event()
        harness = await _harness(embedder=GatedEmbedder(gate))
        await harness.attach_memory()
        await harness.attach()
        await harness.recall(memory.RecallData(about="noon"), "in-flight")
        await harness.owner.until(lambda: harness.associative.state().endswith("/busy/recalling"))
        await harness.recall(memory.RecallData(about="dusk"), "waiting")
        await harness.detach()
        await harness.owner.until(lambda: harness.associative.state() == "/AssociativeLifecycle/detached")
        await harness.recall(memory.RecallData(about="dawn"), "after")
        _ = await harness.reply("after")
        gate.set()
        await asyncio.sleep(0.2)
        return {request: harness.owner.replies(request) for request in ("in-flight", "waiting", "after")}

    replies = asyncio.run(scenario())

    for request, answered in replies.items():
        assert len(answered) == 1, (request, answered)
        assert answered[0].name == ability.FailedEvent.name
        failure = answered[0].data
        assert isinstance(failure, ability.FailureData)
        assert failure.message.startswith("unavailable"), (request, failure)


# --- who is here (W8-7) -------------------------------------------------------------------------


def test_a_turn_says_who_is_speaking_who_is_present_and_who_the_bot_is() -> None:
    """Pins what the bot is told, never what it does with it: asking who a stranger is stays its choice."""

    child = mosfet.Participant(ref="voice:ada", role=mosfet.ParticipantRole.SOURCE)
    stranger = mosfet.Participant(ref="voice:3f2a9c", role=mosfet.ParticipantRole.SOURCE)
    seen = mosfet.Participant(ref="face:7b1e04", role=mosfet.ParticipantRole.SOURCE)
    mouth = mosfet.Participant(ref="voice:tts-warm", role=mosfet.ParticipantRole.SOURCE, own=True)

    async def scenario() -> hsm.Event[typing.Any]:
        harness = await _harness()
        await harness.attach_memory()
        # Nobody declares Ada (R-11): the bot learned her voice and her name.
        await harness.person(memory.PersonLinkData(ref=child.ref, name="Ada"))
        await harness.remember(_episode("Hello, I am here.", "spoke", participants=(mouth,)))
        await harness.attach()
        cue = memory.CueData(cue=(text.TextPart(text="who is that?"),), participants=(child, stranger, seen))
        await harness.send(cue, "a-1")
        return await harness.reply("a-1")

    reply = asyncio.run(scenario())

    assert reply.name == memory.AssociatedEvent.name
    surfaced = reply.data
    assert isinstance(surfaced, memory.AssociationsData)
    assert [(present.ref, present.part, present.known) for present in surfaced.people] == [
        (child.ref, mosfet.ParticipantRole.SOURCE, True),
        (stranger.ref, mosfet.ParticipantRole.SOURCE, False),
        (seen.ref, mosfet.ParticipantRole.SOURCE, False),
    ]
    ada = surfaced.people[0].person
    unknown = surfaced.people[1].person
    assert unknown not in (ada, memory.SELF) and unknown != surfaced.people[2].person
    persons = {person.id: person for person in surfaced.persons}
    assert persons[ada] == memory.Person(id=ada, name="Ada", refs=(child.ref,))
    assert persons[unknown] == memory.Person(id=unknown, refs=(stranger.ref,))
    assert surfaced.you == memory.Person(id=memory.SELF, refs=(mouth.ref,))
    assert persons[memory.SELF] == surfaced.you

    rendered = "".join(
        part.text
        for part in projection.model_facing_content(surfaced, capacity=media.ImageCapacity())
        if isinstance(part, text.TextPart)
    )
    for shown in ("voice:3f2a9c", "Ada", "voice:tts-warm", "known"):
        assert shown in rendered
    for hidden in ("similarity", "print", "embedding", "vector"):
        assert hidden not in rendered


def test_the_bots_own_voice_on_the_turn_is_the_bot() -> None:
    mouth = mosfet.Participant(ref="voice:tts-warm", role=mosfet.ParticipantRole.SOURCE, own=True)

    async def scenario() -> hsm.Event[typing.Any]:
        harness = await _harness()
        await harness.attach_memory()
        await harness.attach()
        await harness.send(memory.CueData(cue=(text.TextPart(text="echo"),), participants=(mouth,)), "a-1")
        return await harness.reply("a-1")

    surfaced = asyncio.run(scenario()).data
    assert isinstance(surfaced, memory.AssociationsData)
    (present,) = surfaced.people
    assert (present.ref, present.person, present.known) == (mouth.ref, memory.SELF, True)
    assert surfaced.you.refs == (mouth.ref,)

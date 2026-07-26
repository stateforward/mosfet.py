import asyncio
import collections.abc
import typing

import hsm
import pydantic
import pytest

from bot.world import World


OBSERVED_EVENT = hsm.Event[str](
    name="world.observed",
    schema=typing.cast(
        pydantic.TypeAdapter[object],
        pydantic.TypeAdapter(typing.Annotated[str, pydantic.Field(description="Observed world broadcast payload.")]),
    ),
)


def _consume_observed_event(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event) -> None:
    del ctx, instance, event


class BroadcastRecorder(hsm.Instance):
    seen: list[tuple[str, str | None]]

    def __init__(self) -> None:
        super().__init__()
        self.seen = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == OBSERVED_EVENT.name:
            self.seen.append((typing.cast(str, event.data), event.target))
        return super().dispatch(ctx, event)

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "BroadcastRecorder",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(OBSERVED_EVENT), hsm.effect(_consume_observed_event))),
    )


def test_world_is_hsm_context() -> None:
    async def run() -> list[tuple[str, str | None]]:
        world = World()
        inside = BroadcastRecorder()
        assert isinstance(world, hsm.Context)

        _ = await hsm.started(world, inside, inside.model, hsm.Config(id="inside"))
        await hsm.dispatch_all(world, OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    inside_seen = asyncio.run(run())
    assert inside_seen == [("hello", "inside")]


def test_world_broadcast_dispatches_to_started_instances_in_scope() -> None:
    async def run() -> tuple[list[tuple[str, str | None]], list[tuple[str, str | None]]]:
        world = World()
        inside = BroadcastRecorder()
        outside = BroadcastRecorder()

        _ = await hsm.started(world, inside, inside.model, hsm.Config(id="inside"))
        world.join(inside)
        _ = await hsm.started(None, outside, outside.model, hsm.Config(id="outside"))

        await world.broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen, outside.seen

    inside_seen, outside_seen = asyncio.run(run())

    assert inside_seen == [("hello", "inside")]
    assert outside_seen == []


def test_world_from_done_context_preserves_broadcast_scope() -> None:
    async def run() -> list[tuple[str, str | None]]:
        world = World()
        inside = BroadcastRecorder()
        done_context = world.with_value("probe", "done")

        _ = await hsm.started(world, inside, inside.model, hsm.Config(id="inside"))
        world.join(inside)
        done_context.cancel()

        await World.from_context(done_context).broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    inside_seen = asyncio.run(run())

    assert inside_seen == [("hello", "inside")]


def test_world_from_canceled_world_preserves_broadcast_scope() -> None:
    """Cancel is not stop, so a canceled scope must not silently mute live participants.

    ``hsm.dispatch_to`` returns early on a canceled context. Resolution therefore falls back to a
    live World over the *same* instance and participant maps, which are shared by reference — the
    one production path that reads presence off a context rather than off the World object.
    """

    async def run() -> list[tuple[str, str | None]]:
        world = World()
        inside = BroadcastRecorder()

        _ = await hsm.started(world, inside, inside.model, hsm.Config(id="inside"))
        world.join(inside)
        world.cancel()

        resolved = World.from_context(inside.context())
        assert resolved is not world
        assert not resolved.is_done()
        await resolved.broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    assert asyncio.run(run()) == [("hello", "inside")]


def test_world_broadcast_without_participants_reaches_nobody() -> None:
    """An empty presence set means nobody — never "everything in the addressing map"."""

    async def run() -> list[tuple[str, str | None]]:
        world = World()
        inside = BroadcastRecorder()

        _ = await hsm.started(world, inside, inside.model, hsm.Config(id="inside"))

        await world.broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    assert asyncio.run(run()) == []


def test_world_broadcast_skips_stopped_participant() -> None:
    async def run() -> list[tuple[str, str | None]]:
        world = World()
        stopped = BroadcastRecorder()
        listening = BroadcastRecorder()

        _ = await hsm.started(world, stopped, stopped.model, hsm.Config(id="stopped"))
        world.join(stopped)
        _ = await hsm.started(world, listening, listening.model, hsm.Config(id="listening"))
        world.join(listening)
        await hsm.stop(stopped)

        await world.broadcast(OBSERVED_EVENT.with_data("hello"))

        assert listening.seen == [("hello", "listening")]
        return stopped.seen

    assert asyncio.run(run()) == []


def test_world_leave_removes_a_citizen() -> None:
    async def run() -> list[tuple[str, str | None]]:
        world = World()
        inside = BroadcastRecorder()

        _ = await hsm.started(world, inside, inside.model, hsm.Config(id="inside"))
        world.join(inside)
        world.leave(inside)
        world.leave(inside)

        await world.broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    assert asyncio.run(run()) == []


def test_world_join_rejects_an_unstarted_instance() -> None:
    world = World()

    with pytest.raises(RuntimeError, match="must be started"):
        world.join(BroadcastRecorder())


def test_world_join_rejects_an_instance_from_another_world() -> None:
    async def run() -> None:
        world = World()
        other = World()
        outsider = BroadcastRecorder()

        _ = await hsm.started(other, outsider, outsider.model, hsm.Config(id="outsider"))

        with pytest.raises(RuntimeError, match="already started in another world"):
            world.join(outsider)

    asyncio.run(run())


def test_world_from_context_reuses_the_world_of_a_descendant_context() -> None:
    """Resolution must be O(1) and allocation-free; every context allocation leaks a callback."""

    async def run() -> None:
        world = World()
        inside = BroadcastRecorder()

        _ = await hsm.started(world, inside, inside.model, hsm.Config(id="inside"))
        world.join(inside)

        assert World.from_context(inside.context()) is world
        assert World.from_context(inside.context()) is world
        assert World.from_context(world.with_value("probe", "child")) is world
        assert World.from_context(world) is world

    asyncio.run(run())


def test_world_from_context_without_a_world_creates_one() -> None:
    context = hsm.Context()

    world = World.from_context(context)

    assert isinstance(world, World)
    assert World.from_context(world) is world


def test_world_does_not_mediate_attachment() -> None:
    world = World()

    assert not hasattr(world, "attach")
    assert not hasattr(world, "detach")
    assert not hasattr(world, "agents")
    assert not hasattr(world, "devices")
    assert not hasattr(world, "get")
    # Presence is write-only. A readable participant set would make World the registry it is not,
    # so neither an accessor nor the context key it is published under is reachable from here.
    assert not hasattr(world, "participants")
    assert not hasattr(world, "citizens")
    assert not hasattr(world, "instances")
    assert not hasattr(World, "Participants")

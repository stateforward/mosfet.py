import asyncio
import collections.abc
import typing

import hsm
import pydantic

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
        done_context.cancel()

        await World.from_context(done_context).broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    inside_seen = asyncio.run(run())

    assert inside_seen == [("hello", "inside")]


def test_world_does_not_mediate_attachment() -> None:
    world = World()

    assert not hasattr(world, "attach")
    assert not hasattr(world, "detach")
    assert not hasattr(world, "agents")
    assert not hasattr(world, "devices")
    assert not hasattr(world, "get")

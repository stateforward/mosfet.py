import asyncio
import collections.abc
import typing

import hsm
import pydantic
import pytest

from bot.environment import SoundData, SoundEvent, Environment, space


OBSERVED_EVENT = hsm.Event[str](
    name="environment.observed",
    schema=typing.cast(
        pydantic.TypeAdapter[object],
        pydantic.TypeAdapter(
            typing.Annotated[str, pydantic.Field(description="Observed environment broadcast payload.")]
        ),
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


def test_environment_is_hsm_context() -> None:
    async def run() -> list[tuple[str, str | None]]:
        environment = Environment()
        inside = BroadcastRecorder()
        assert isinstance(environment, hsm.Context)

        _ = await hsm.started(environment, inside, inside.model, hsm.Config(id="inside"))
        await hsm.dispatch_all(environment, OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    inside_seen = asyncio.run(run())
    assert inside_seen == [("hello", "inside")]


def test_environment_broadcast_dispatches_to_started_instances_in_scope() -> None:
    async def run() -> tuple[list[tuple[str, str | None]], list[tuple[str, str | None]]]:
        environment = Environment()
        inside = BroadcastRecorder()
        outside = BroadcastRecorder()

        _ = await hsm.started(environment, inside, inside.model, hsm.Config(id="inside"))
        environment.join(inside)
        _ = await hsm.started(None, outside, outside.model, hsm.Config(id="outside"))

        await environment.broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen, outside.seen

    inside_seen, outside_seen = asyncio.run(run())

    assert inside_seen == [("hello", "inside")]
    assert outside_seen == []


def test_environment_from_done_context_preserves_broadcast_scope() -> None:
    async def run() -> list[tuple[str, str | None]]:
        environment = Environment()
        inside = BroadcastRecorder()
        done_context = environment.with_value("probe", "done")

        _ = await hsm.started(environment, inside, inside.model, hsm.Config(id="inside"))
        environment.join(inside)
        done_context.cancel()

        await Environment.from_context(done_context).broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    inside_seen = asyncio.run(run())

    assert inside_seen == [("hello", "inside")]


def test_environment_from_canceled_environment_preserves_broadcast_scope() -> None:
    """Cancel is not stop, so a canceled scope must not silently mute live participants.

    ``hsm.dispatch_to`` returns early on a canceled context. Resolution therefore falls back to a
    live Environment over the *same* instance and participant maps, which are shared by reference — the
    one production path that reads presence off a context rather than off the Environment object.
    """

    async def run() -> list[tuple[str, str | None]]:
        environment = Environment()
        inside = BroadcastRecorder()

        _ = await hsm.started(environment, inside, inside.model, hsm.Config(id="inside"))
        environment.join(inside)
        environment.cancel()

        resolved = Environment.from_context(inside.context())
        assert resolved is not environment
        assert not resolved.is_done()
        await resolved.broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    assert asyncio.run(run()) == [("hello", "inside")]


def test_environment_broadcast_without_participants_reaches_nobody() -> None:
    """An empty presence set means nobody — never "everything in the addressing map"."""

    async def run() -> list[tuple[str, str | None]]:
        environment = Environment()
        inside = BroadcastRecorder()

        _ = await hsm.started(environment, inside, inside.model, hsm.Config(id="inside"))

        await environment.broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    assert asyncio.run(run()) == []


def test_environment_broadcast_skips_stopped_participant() -> None:
    async def run() -> list[tuple[str, str | None]]:
        environment = Environment()
        stopped = BroadcastRecorder()
        listening = BroadcastRecorder()

        _ = await hsm.started(environment, stopped, stopped.model, hsm.Config(id="stopped"))
        environment.join(stopped)
        _ = await hsm.started(environment, listening, listening.model, hsm.Config(id="listening"))
        environment.join(listening)
        await hsm.stop(stopped)

        await environment.broadcast(OBSERVED_EVENT.with_data("hello"))

        assert listening.seen == [("hello", "listening")]
        return stopped.seen

    assert asyncio.run(run()) == []


def test_environment_leave_removes_a_citizen() -> None:
    async def run() -> list[tuple[str, str | None]]:
        environment = Environment()
        inside = BroadcastRecorder()

        _ = await hsm.started(environment, inside, inside.model, hsm.Config(id="inside"))
        environment.join(inside)
        environment.leave(inside)
        environment.leave(inside)

        await environment.broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    assert asyncio.run(run()) == []


def test_environment_join_rejects_an_unstarted_instance() -> None:
    environment = Environment()

    with pytest.raises(RuntimeError, match="must be started"):
        environment.join(BroadcastRecorder())


def test_environment_join_rejects_an_instance_from_another_environment() -> None:
    async def run() -> None:
        environment = Environment()
        other = Environment()
        outsider = BroadcastRecorder()

        _ = await hsm.started(other, outsider, outsider.model, hsm.Config(id="outsider"))

        with pytest.raises(RuntimeError, match="already started in another environment"):
            environment.join(outsider)

    asyncio.run(run())


def test_environment_from_context_reuses_the_environment_of_a_descendant_context() -> None:
    """Resolution must be O(1) and allocation-free; every context allocation leaks a callback."""

    async def run() -> None:
        environment = Environment()
        inside = BroadcastRecorder()

        _ = await hsm.started(environment, inside, inside.model, hsm.Config(id="inside"))
        environment.join(inside)

        assert Environment.from_context(inside.context()) is environment
        assert Environment.from_context(inside.context()) is environment
        assert Environment.from_context(environment.with_value("probe", "child")) is environment
        assert Environment.from_context(environment) is environment

    asyncio.run(run())


def test_environment_from_context_without_a_environment_creates_one() -> None:
    context = hsm.Context()

    environment = Environment.from_context(context)

    assert isinstance(environment, Environment)
    assert Environment.from_context(environment) is environment


def test_environment_does_not_mediate_attachment() -> None:
    environment = Environment()

    assert not hasattr(environment, "attach")
    assert not hasattr(environment, "detach")
    assert not hasattr(environment, "agents")
    assert not hasattr(environment, "devices")
    assert not hasattr(environment, "get")
    # Presence is write-only. A readable participant set would make Environment the registry it is not,
    # so neither an accessor nor the context key it is published under is reachable from here.
    assert not hasattr(environment, "participants")
    assert not hasattr(environment, "citizens")
    assert not hasattr(environment, "instances")
    assert not hasattr(Environment, "Participants")


class SoundRecorder(hsm.Instance):
    """Environment citizen that records the ``environment.sound`` stimuli that actually reach it."""

    heard: list[tuple[bytes, str | None]]

    def __init__(self) -> None:
        super().__init__()
        self.heard = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "SoundRecorder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        if isinstance(data, SoundData):
            instance.heard.append((bytes(data.audio), event.target))

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "SoundRecorder",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(SoundEvent), hsm.effect(_record))),
    )


def _sound(amplitude_db: float | None) -> hsm.Event[typing.Any]:
    return SoundEvent.with_data(
        SoundData(
            audio=b"chunk",
            media_type="audio/pcm",
            sample_rate_hz=16_000,
            channels=1,
            amplitude_db=amplitude_db,
        )
    )


def test_environment_broadcast_reaches_a_participant_above_its_threshold() -> None:
    """A placed listener hears a source that is still loud enough by the time it arrives."""

    async def run() -> list[tuple[bytes, str | None]]:
        environment = Environment()
        near = SoundRecorder()

        _ = await hsm.started(environment, near, near.model, hsm.Config(id="near"))
        environment.join(near, placement=space.Placement(position=space.Position(x=1.0, y=0.0), threshold_db=20.0))

        await environment.broadcast(_sound(60.0), origin=space.Position(x=0.0, y=0.0))

        return near.heard

    assert asyncio.run(run()) == [(b"chunk", "near")]


def test_environment_broadcast_skips_a_participant_below_its_threshold() -> None:
    """Distance is what silences it: the same sound at the same level, further away."""

    async def run() -> tuple[list[tuple[bytes, str | None]], float]:
        environment = Environment()
        far = SoundRecorder()
        position = space.Position(x=500.0, y=0.0)

        _ = await hsm.started(environment, far, far.model, hsm.Config(id="far"))
        environment.join(far, placement=space.Placement(position=position, threshold_db=20.0))

        await environment.broadcast(_sound(60.0), origin=space.Position(x=0.0, y=0.0))

        return far.heard, space.received_level_db(60.0, position.distance_to(space.Position(x=0.0, y=0.0)))

    heard, level = asyncio.run(run())

    assert heard == []
    assert level < 20.0


def test_environment_broadcast_without_geometry_reaches_everyone() -> None:
    """Geometry is opt-in on three counts, and any one of them missing means "deliver"."""

    async def run() -> tuple[list[tuple[bytes, str | None]], ...]:
        environment = Environment()
        unplaced = SoundRecorder()
        no_threshold = SoundRecorder()
        placed = SoundRecorder()

        _ = await hsm.started(environment, unplaced, unplaced.model, hsm.Config(id="unplaced"))
        _ = await hsm.started(environment, no_threshold, no_threshold.model, hsm.Config(id="no-threshold"))
        _ = await hsm.started(environment, placed, placed.model, hsm.Config(id="placed"))
        far = space.Position(x=500.0, y=0.0)
        environment.join(unplaced)
        environment.join(no_threshold, placement=space.Placement(position=far))
        environment.join(placed, placement=space.Placement(position=far, threshold_db=20.0))

        # No origin: the emitter did not say where it was.
        await environment.broadcast(_sound(60.0))
        # No amplitude: the emitter did not say how loud it was.
        await environment.broadcast(_sound(None), origin=space.Position(x=0.0, y=0.0))

        return unplaced.heard, no_threshold.heard, placed.heard

    unplaced_heard, no_threshold_heard, placed_heard = asyncio.run(run())

    assert len(unplaced_heard) == 2
    assert len(no_threshold_heard) == 2
    # Placed and selective, but each broadcast was missing one of origin or amplitude.
    assert len(placed_heard) == 2


def test_environment_broadcast_short_circuits_when_attenuation_silences_everyone() -> None:
    """The guard runs on the narrowed list.

    If narrowing ran after it, an empty audible list would reach ``hsm.dispatch_to`` as no ids at
    all, which means "every instance in scope" — the exact defect the guard exists to prevent,
    restored in the case that matters most: a quiet sound in a large environment.
    """

    async def run() -> tuple[list[tuple[bytes, str | None]], list[tuple[bytes, str | None]]]:
        environment = Environment()
        far = SoundRecorder()
        bystander = SoundRecorder()

        _ = await hsm.started(environment, far, far.model, hsm.Config(id="far"))
        _ = await hsm.started(environment, bystander, bystander.model, hsm.Config(id="bystander"))
        environment.join(far, placement=space.Placement(position=space.Position(x=500.0, y=0.0), threshold_db=20.0))

        await environment.broadcast(_sound(60.0), origin=space.Position(x=0.0, y=0.0))

        return far.heard, bystander.heard

    far_heard, bystander_heard = asyncio.run(run())

    assert far_heard == []
    # Never joined, so it must not hear anything either way.
    assert bystander_heard == []


def test_environment_join_rejects_a_conflicting_placement() -> None:
    """One object cannot be in two places.

    A shared speaker joined twice with different placements would silently take the second, which
    is how a phone earpiece ends up at the mouth and the echo comes back.
    """

    async def run() -> None:
        environment = Environment()
        speaker = SoundRecorder()

        _ = await hsm.started(environment, speaker, speaker.model, hsm.Config(id="speaker"))
        ear = space.Placement(position=space.Position(x=0.0, y=0.0), threshold_db=20.0)
        mouth = space.Placement(position=space.Position(x=0.15, y=0.0), threshold_db=20.0)
        environment.join(speaker, placement=ear)
        # Same placement again is fine; presence is idempotent.
        environment.join(speaker, placement=ear)

        with pytest.raises(RuntimeError, match="already placed"):
            environment.join(speaker, placement=mouth)

    asyncio.run(run())

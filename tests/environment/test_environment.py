import asyncio
import collections.abc
import typing
import xml.etree.ElementTree

import hsm
import mosfet
import pydantic
import pytest

from mosfet.device import Device
from mosfet.environment import SoundData, SoundEvent, Environment, space


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
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        if event.name == OBSERVED_EVENT.name:
            self.seen.append((typing.cast(str, event.data), event.target))
        return super().dispatch(ctx, event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "BroadcastRecorder",
        hsm.initial(hsm.target("listening")),
        hsm.state("listening", hsm.transition(hsm.on(OBSERVED_EVENT), hsm.effect(_consume_observed_event))),
    )


def test_environment_is_hsm_context() -> None:
    async def run() -> list[tuple[str, str | None]]:
        environment = Environment()
        inside = BroadcastRecorder()
        assert isinstance(environment, hsm.Context)

        _ = await mosfet.started(environment, inside, inside.model, hsm.Config(id="inside"))
        await hsm.dispatch_all(environment, OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    inside_seen = asyncio.run(run())
    assert inside_seen == [("hello", "inside")]


def test_environment_broadcast_dispatches_to_started_instances_in_scope() -> None:
    async def run() -> tuple[list[tuple[str, str | None]], list[tuple[str, str | None]]]:
        environment = Environment()
        inside = BroadcastRecorder()
        outside = BroadcastRecorder()

        _ = await mosfet.started(environment, inside, inside.model, hsm.Config(id="inside"))
        environment.join(inside)
        _ = await mosfet.started(None, outside, outside.model, hsm.Config(id="outside"))

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

        _ = await mosfet.started(environment, inside, inside.model, hsm.Config(id="inside"))
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

        _ = await mosfet.started(environment, inside, inside.model, hsm.Config(id="inside"))
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

        _ = await mosfet.started(environment, inside, inside.model, hsm.Config(id="inside"))

        await environment.broadcast(OBSERVED_EVENT.with_data("hello"))

        return inside.seen

    assert asyncio.run(run()) == []


def test_environment_broadcast_skips_stopped_participant() -> None:
    async def run() -> list[tuple[str, str | None]]:
        environment = Environment()
        stopped = BroadcastRecorder()
        listening = BroadcastRecorder()

        _ = await mosfet.started(environment, stopped, stopped.model, hsm.Config(id="stopped"))
        environment.join(stopped)
        _ = await mosfet.started(environment, listening, listening.model, hsm.Config(id="listening"))
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

        _ = await mosfet.started(environment, inside, inside.model, hsm.Config(id="inside"))
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

        _ = await mosfet.started(other, outsider, outsider.model, hsm.Config(id="outsider"))

        with pytest.raises(RuntimeError, match="already started in another environment"):
            environment.join(outsider)

    asyncio.run(run())


def test_environment_from_context_reuses_the_environment_of_a_descendant_context() -> None:
    """Resolution must be O(1) and allocation-free; every context allocation leaks a callback."""

    async def run() -> None:
        environment = Environment()
        inside = BroadcastRecorder()

        _ = await mosfet.started(environment, inside, inside.model, hsm.Config(id="inside"))
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
    levels: list[float | None]
    received: list[SoundData]

    def __init__(self) -> None:
        super().__init__()
        self.heard = []
        self.levels = []
        self.received = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "SoundRecorder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        if isinstance(data, SoundData):
            instance.heard.append((bytes(data.audio), event.target))
            instance.levels.append(data.received_level_db)
            instance.received.append(data)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
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

        _ = await mosfet.started(environment, near, near.model, hsm.Config(id="near"))
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

        _ = await mosfet.started(environment, far, far.model, hsm.Config(id="far"))
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

        _ = await mosfet.started(environment, unplaced, unplaced.model, hsm.Config(id="unplaced"))
        _ = await mosfet.started(environment, no_threshold, no_threshold.model, hsm.Config(id="no-threshold"))
        _ = await mosfet.started(environment, placed, placed.model, hsm.Config(id="placed"))
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

        _ = await mosfet.started(environment, far, far.model, hsm.Config(id="far"))
        _ = await mosfet.started(environment, bystander, bystander.model, hsm.Config(id="bystander"))
        environment.join(far, placement=space.Placement(position=space.Position(x=500.0, y=0.0), threshold_db=20.0))

        await environment.broadcast(_sound(60.0), origin=space.Position(x=0.0, y=0.0))

        return far.heard, bystander.heard

    far_heard, bystander_heard = asyncio.run(run())

    assert far_heard == []
    # Never joined, so it must not hear anything either way.
    assert bystander_heard == []


def test_environment_broadcast_stamps_the_level_each_participant_receives() -> None:
    """The same sound arrives at two listeners at two different levels, and each is told which.

    ``amplitude_db`` is a property of the sound at its source and is the same for both;
    ``received_level_db`` is a property of the sound *here* and is what a listener can act on.
    """

    async def run() -> tuple[list[float | None], list[float | None]]:
        environment = Environment()
        near = SoundRecorder()
        far = SoundRecorder()

        _ = await mosfet.started(environment, near, near.model, hsm.Config(id="near"))
        _ = await mosfet.started(environment, far, far.model, hsm.Config(id="far"))
        environment.join(near, placement=space.Placement(position=space.Position(x=1.0, y=0.0), threshold_db=0.0))
        environment.join(far, placement=space.Placement(position=space.Position(x=10.0, y=0.0), threshold_db=0.0))

        await environment.broadcast(_sound(60.0), origin=space.Position(x=0.0, y=0.0))

        return near.levels, far.levels

    near_levels, far_levels = asyncio.run(run())

    assert near_levels == [pytest.approx(60.0)]
    assert far_levels == [pytest.approx(space.received_level_db(60.0, 10.0))]
    assert far_levels[0] != near_levels[0]


def test_environment_broadcast_leaves_the_source_amplitude_alone() -> None:
    """Received level is a second quantity, never a rewrite of the first."""

    async def run() -> list[float | None]:
        environment = Environment()
        listener = SoundRecorder()
        amplitudes: list[float | None] = []

        _ = await mosfet.started(environment, listener, listener.model, hsm.Config(id="listener"))
        environment.join(listener, placement=space.Placement(position=space.Position(x=10.0, y=0.0), threshold_db=0.0))

        original = _sound(60.0)
        await environment.broadcast(original, origin=space.Position(x=0.0, y=0.0))

        data = original.data
        assert isinstance(data, SoundData)
        amplitudes.append(data.amplitude_db)
        return amplitudes

    assert asyncio.run(run()) == [60.0]


def test_environment_broadcast_leaves_the_level_unstamped_without_geometry() -> None:
    """Geometry is opt-in, so an unplaced listener is told nothing about level rather than zero."""

    async def run() -> tuple[list[float | None], list[float | None]]:
        environment = Environment()
        unplaced = SoundRecorder()
        placed = SoundRecorder()

        _ = await mosfet.started(environment, unplaced, unplaced.model, hsm.Config(id="unplaced"))
        _ = await mosfet.started(environment, placed, placed.model, hsm.Config(id="placed"))
        environment.join(unplaced)
        environment.join(placed, placement=space.Placement(position=space.Position(x=1.0, y=0.0)))

        # Placed, but the emitter never said how loud it was.
        await environment.broadcast(_sound(None), origin=space.Position(x=0.0, y=0.0))
        # Loud, but the emitter never said where it was.
        await environment.broadcast(_sound(60.0))

        return unplaced.levels, placed.levels

    unplaced_levels, placed_levels = asyncio.run(run())

    assert unplaced_levels == [None, None]
    assert placed_levels == [None, None]


def test_environment_broadcast_stamps_a_participant_without_a_threshold() -> None:
    """Hearing everything is not the same as knowing nothing: level is still measured."""

    async def run() -> list[float | None]:
        environment = Environment()
        listener = SoundRecorder()

        _ = await mosfet.started(environment, listener, listener.model, hsm.Config(id="listener"))
        environment.join(listener, placement=space.Placement(position=space.Position(x=10.0, y=0.0)))

        await environment.broadcast(_sound(60.0), origin=space.Position(x=0.0, y=0.0))

        return listener.levels

    assert asyncio.run(run()) == [pytest.approx(space.received_level_db(60.0, 10.0))]


class ElevatedSoundData(SoundData):
    """A domain elevation of a sound, standing in for the phone's ring stimulus."""

    caller: str


def test_environment_broadcast_stamps_a_domain_elevation_without_flattening_it() -> None:
    """Stamping the level must not cost a listener the domain fields it was going to read."""

    async def run() -> tuple[list[float | None], list[str]]:
        environment = Environment()
        listener = SoundRecorder()
        callers: list[str] = []

        _ = await mosfet.started(environment, listener, listener.model, hsm.Config(id="listener"))
        environment.join(listener, placement=space.Placement(position=space.Position(x=2.0, y=0.0)))

        elevated = SoundEvent.with_data(
            ElevatedSoundData(audio=b"ring", kind="phone.ringing", amplitude_db=70.0, caller="+15551234567")
        )
        await environment.broadcast(elevated, origin=space.Position(x=0.0, y=0.0))

        for data in listener.received:
            assert isinstance(data, ElevatedSoundData)
            callers.append(data.caller)
        return listener.levels, callers

    levels, callers = asyncio.run(run())

    assert levels == [pytest.approx(space.received_level_db(70.0, 2.0))]
    assert callers == ["+15551234567"]


def test_environment_join_rejects_a_conflicting_placement() -> None:
    """One object cannot be in two places.

    A shared speaker joined twice with different placements would silently take the second, which
    is how a phone earpiece ends up at the mouth and the echo comes back.
    """

    async def run() -> None:
        environment = Environment()
        speaker = SoundRecorder()

        _ = await mosfet.started(environment, speaker, speaker.model, hsm.Config(id="speaker"))
        ear = space.Placement(position=space.Position(x=0.0, y=0.0), threshold_db=20.0)
        mouth = space.Placement(position=space.Position(x=0.15, y=0.0), threshold_db=20.0)
        environment.join(speaker, placement=ear)
        # Same placement again is fine; presence is idempotent.
        environment.join(speaker, placement=ear)

        with pytest.raises(RuntimeError, match="already placed"):
            environment.join(speaker, placement=mouth)

    asyncio.run(run())


class _OwnerActor(hsm.Instance):
    """Minimal perspective actor that declares owned devices on its own snapshot."""

    _owned: dict[str, str]

    def __init__(self, owned: dict[str, str] | None = None) -> None:
        super().__init__()
        self._owned = dict(owned) if owned else {}

    @staticmethod
    def _note_owned_devices(ctx: hsm.Context, instance: "_OwnerActor", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _ = instance.set("owned_devices", dict(instance._owned))

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "OwnerActor",
        hsm.attribute("owned_devices"),
        hsm.initial(hsm.target("/OwnerActor/active")),
        hsm.state("active", hsm.entry(_note_owned_devices)),
    )


def test_model_snapshot_returns_none_for_an_unstarted_perspective() -> None:
    """No honest snapshot to render → no envelope invented around nothing."""

    environment = Environment()

    assert environment.model_snapshot(_OwnerActor()) is None


def test_model_snapshot_composes_an_environment_root_around_the_perspective() -> None:
    """The environment takes the snapshot: root is ``<environment id="…">``, self is the perspective."""

    async def run() -> str | None:
        environment = Environment()
        owner = _OwnerActor()
        _ = await mosfet.started(environment, owner, typing.cast(hsm.Model, owner.model))
        return environment.model_snapshot(owner)

    instructions = asyncio.run(run())
    assert instructions is not None
    root = xml.etree.ElementTree.fromstring(instructions)
    assert root.tag == "environment"
    # This environment's own identity, not any HSM actor id and never Python id().
    assert root.get("id")
    assert root.get("state") is None
    self_element = root.find("self")
    assert self_element is not None
    assert self_element.get("state") == "/OwnerActor/active"


def test_model_snapshot_reports_the_same_environment_identity_across_perspectives() -> None:
    """Identity is the environment's own, not derived from whichever perspective is asked."""

    async def run() -> tuple[str | None, str | None]:
        environment = Environment()
        first = _OwnerActor()
        second = _OwnerActor()
        _ = await mosfet.started(environment, first, typing.cast(hsm.Model, first.model))
        _ = await mosfet.started(environment, second, typing.cast(hsm.Model, second.model))
        return environment.model_snapshot(first), environment.model_snapshot(second)

    first_block, second_block = asyncio.run(run())
    assert first_block is not None and second_block is not None
    first_id = xml.etree.ElementTree.fromstring(first_block).get("id")
    second_id = xml.etree.ElementTree.fromstring(second_block).get("id")
    assert first_id == second_id


def test_model_snapshot_resolves_owned_devices_from_the_environments_own_scope() -> None:
    """Owned devices resolve by runtime id in the environment's own addressing scope.

    No actor map is passed in anywhere: the environment looks the id up itself.
    """

    async def run() -> tuple[str | None, str]:
        environment = Environment()
        device = Device()
        _ = await mosfet.started(environment, device, typing.cast(hsm.Model, device.model))
        owner = _OwnerActor({"phone": hsm.id(device)})
        _ = await mosfet.started(environment, owner, typing.cast(hsm.Model, owner.model))
        return environment.model_snapshot(owner), hsm.id(device)

    instructions, device_id = asyncio.run(run())
    assert instructions is not None
    root = xml.etree.ElementTree.fromstring(instructions)
    device_element = root.find("self/owned_devices/device")
    assert device_element is not None
    assert device_element.get("ref") == "phone"
    assert device_element.get("id") == device_id


def test_model_snapshot_drops_an_owned_reference_the_environment_cannot_resolve() -> None:
    """A device reference the perspective names but the environment cannot resolve is dropped."""

    async def run() -> str | None:
        environment = Environment()
        owner = _OwnerActor({"phone": "not-a-live-runtime-id"})
        _ = await mosfet.started(environment, owner, typing.cast(hsm.Model, owner.model))
        return environment.model_snapshot(owner)

    instructions = asyncio.run(run())
    assert instructions is not None
    root = xml.etree.ElementTree.fromstring(instructions)
    owned_devices = root.find("self/owned_devices")
    assert owned_devices is not None
    assert owned_devices.findall("device") == []


def test_model_snapshot_returns_none_for_a_perspective_started_in_another_environment() -> None:
    """A perspective addressable only in a foreign scope has no honest block for this environment
    to compose: this environment cannot see into another environment's addressing map, so it has
    nothing true to render — the same "nothing honest to show" as an unstarted perspective."""

    async def run() -> str | None:
        environment = Environment()
        other = Environment()
        foreign = _OwnerActor()
        _ = await mosfet.started(other, foreign, typing.cast(hsm.Model, foreign.model))
        return environment.model_snapshot(foreign)

    assert asyncio.run(run()) is None


def test_model_snapshot_omits_an_owned_device_that_has_since_stopped() -> None:
    """A device the perspective still names but that has since stopped has no honest snapshot to
    give — it must be dropped from ``owned_devices`` rather than crash the whole block, even though
    it remains addressable (a strong reference keeps it in this environment's own scope)."""

    async def run() -> str | None:
        environment = Environment()
        device = Device()
        _ = await mosfet.started(environment, device, typing.cast(hsm.Model, device.model))
        owner = _OwnerActor({"phone": hsm.id(device)})
        _ = await mosfet.started(environment, owner, typing.cast(hsm.Model, owner.model))
        await hsm.stop(device)
        return environment.model_snapshot(owner)

    instructions = asyncio.run(run())
    assert instructions is not None
    root = xml.etree.ElementTree.fromstring(instructions)
    owned_devices = root.find("self/owned_devices")
    assert owned_devices is not None
    assert owned_devices.findall("device") == []

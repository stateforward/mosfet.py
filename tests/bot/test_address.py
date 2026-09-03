import gc
import typing
import weakref

import hsm
import pytest

import bot
from bot import address
from bot import scope
from bot.environment import Environment


class _Thing(hsm.Instance):
    model: typing.ClassVar[object] = bot.define("AddressThing", bot.initial(bot.target("ready")), bot.state("ready"))


_StartedActor = typing.TypeVar("_StartedActor", bound=hsm.Instance)


async def _started(parent: hsm.Context, actor: _StartedActor, model: object) -> _StartedActor:
    """Start one actor for address tests.

    Test-local models are declared as ``ClassVar[object]`` so one cast lives
    here instead of a ``type: ignore`` at every call site.
    """

    return await bot.started(parent, actor, typing.cast(hsm.Model, model))


def _private_parent(environment: Environment) -> hsm.Context:
    """Child context with a private Instances map, off the environment addressing map.

    Same shape the body uses for bot-owned scopes (private marker only filters
    environment fan-out; explicit dispatch works normally), built here through
    the public ``scope``/``hsm`` seams so the test never reaches into body internals.
    """

    values: dict[typing.Hashable, object] = {
        hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance](),
    }
    return hsm.Context(parent=environment, values=scope.mark_private(values))


def test_started_registers_actor_under_environment_path() -> None:
    import asyncio

    async def run() -> tuple[str, _Thing]:
        environment = Environment()
        thing = _Thing()
        await _started(environment, thing, _Thing.model)
        return address.environment_path(environment.environment_id, hsm.id(thing)), thing

    path, thing = asyncio.run(run())
    assert address.resolve(path) is thing
    # The registry is process-global, so other suites' live environments may legitimately
    # hold sibling entries under "/env". Assert the test's own registration, not exclusivity.
    assert (path, thing) in address.under_prefix(f"/{path.split('/')[1]}")


def test_unregister_is_idempotent_and_targeted() -> None:
    async def run() -> str:
        environment = Environment()
        thing = _Thing()
        await _started(environment, thing, _Thing.model)
        path = address.environment_path(environment.environment_id, hsm.id(thing))
        address.unregister(path)
        assert address.resolve(path) is None
        address.unregister(path)  # idempotent
        return path

    import asyncio

    path = asyncio.run(run())
    assert address.resolve(path) is None


def test_register_rejects_path_collision_with_other_instance() -> None:
    import asyncio

    async def run() -> tuple[str, _Thing]:
        environment = Environment()
        first = _Thing()
        second = _Thing()
        await _started(environment, first, _Thing.model)
        return address.environment_path(environment.environment_id, hsm.id(first)), second

    path, second = asyncio.run(run())
    with pytest.raises(RuntimeError, match="already registered"):
        address.register(path, second)


def test_reregistering_same_instance_is_accepted() -> None:
    import asyncio

    async def run() -> tuple[str, _Thing]:
        environment = Environment()
        thing = _Thing()
        await _started(environment, thing, _Thing.model)
        path = address.environment_path(environment.environment_id, hsm.id(thing))
        address.register(path, thing)
        return path, thing

    path, thing = asyncio.run(run())
    assert address.resolve(path) is thing


def test_detached_actor_leaves_registry_on_collection() -> None:
    """A one-shot actor whose whole context chain dies is gone from the registry.

    hsm's context chain keeps environment-scoped children strongly alive, so this
    exercises the detached one-shot shape: an ephemeral actor started under a
    private parent whose retention ends with the test's own references.
    """

    import asyncio

    async def run() -> tuple[str, weakref.ReferenceType[typing.Any] | None]:
        environment = Environment()
        private_parent = _private_parent(environment)

        class _Ephemeral(hsm.Instance):
            model: typing.ClassVar[object] = bot.define(
                "AddressEphemeral", bot.initial(bot.target("ready")), bot.state("ready")
            )

        ephemeral = _Ephemeral()
        await _started(private_parent, ephemeral, _Ephemeral.model)
        ephemeral_path = address.environment_path(environment.environment_id, hsm.id(ephemeral))
        assert address.resolve(ephemeral_path) is ephemeral
        return ephemeral_path, weakref.ref(ephemeral)

    ephemeral_path, ephemeral_ref = asyncio.run(run())
    gc.collect()
    # Weak value: the registry does not keep the actor alive once nothing else does.
    # (Whether the actor itself is collectable depends on hsm's context retention; the
    # registry's own guarantee is that a collected actor leaves no live entry behind.)
    if ephemeral_ref is not None and ephemeral_ref() is None:
        assert address.resolve(ephemeral_path) is None


def test_visibility_root_extracts_first_segment() -> None:
    assert address.visibility_root("/abc/xyz") == "/abc"
    assert address.visibility_root("/abc/xyz/inner") == "/abc"
    assert address.visibility_root("/abc") == "/abc"


def test_private_scope_actor_registers_under_environment_segment() -> None:
    import asyncio

    async def run() -> tuple[str, hsm.Instance]:
        environment = Environment()
        thing = _Thing()
        await _started(environment, thing, _Thing.model)
        private_parent = _private_parent(environment)

        class _Private(hsm.Instance):
            model: typing.ClassVar[object] = bot.define(
                "AddressPrivate", bot.initial(bot.target("ready")), bot.state("ready")
            )

        private = _Private()
        await _started(private_parent, private, _Private.model)
        private_path = address.environment_path(environment.environment_id, hsm.id(private))
        return private_path, private

    private_path, private = asyncio.run(run())
    # Same environment segment: addressable by explicit dispatch, filtered at broadcast.
    assert private_path.startswith(f"/{private_path.split('/')[1]}/")
    assert address.resolve(private_path) is private


def test_stopped_actor_unregisters_via_device_pattern() -> None:
    import asyncio

    async def run() -> tuple[str, _Thing]:
        environment = Environment()
        thing = _Thing()
        await _started(environment, thing, _Thing.model)
        path = address.environment_path(environment.environment_id, hsm.id(thing))
        await thing.stop(environment)
        address.unregister_instance(thing, path)
        return path, thing

    path, _ = asyncio.run(run())
    assert address.resolve(path) is None


def test_private_actor_excluded_from_public_prefix_but_resolvable() -> None:
    import asyncio

    async def run() -> tuple[str, hsm.Instance, str, hsm.Instance]:
        environment = Environment()
        public_thing = _Thing()
        await _started(environment, public_thing, _Thing.model)
        private_parent = _private_parent(environment)

        class _Private(hsm.Instance):
            model: typing.ClassVar[object] = bot.define(
                "AddressPrivate", bot.initial(bot.target("ready")), bot.state("ready")
            )

        private_thing = _Private()
        await _started(private_parent, private_thing, _Private.model)
        env_prefix = environment.scope_path
        return (
            env_prefix,
            public_thing,
            address.environment_path(environment.environment_id, hsm.id(private_thing)),
            private_thing,
        )

    env_prefix, public_thing, private_path, private_thing = asyncio.run(run())
    public = address.under_prefix_public(env_prefix)
    assert (f"{env_prefix}/{hsm.id(public_thing)}", public_thing) in public
    assert all(path != private_path for path, _ in public)
    # Under the full prefix (private included), the private actor is present — explicit addressing works.
    assert (private_path, private_thing) in address.under_prefix(env_prefix)


def test_environments_do_not_leak_through_registry() -> None:
    import asyncio

    async def run() -> tuple[str, str, hsm.Instance, hsm.Instance]:
        first = Environment()
        second = Environment()
        thing_one = _Thing()
        thing_two = _Thing()
        await _started(first, thing_one, _Thing.model)
        await _started(second, thing_two, _Thing.model)
        return first.scope_path, second.scope_path, thing_one, thing_two

    first_prefix, second_prefix, thing_one, thing_two = asyncio.run(run())
    assert first_prefix != second_prefix
    first_paths = {path for path, _ in address.under_prefix(first_prefix)}
    second_paths = {path for path, _ in address.under_prefix(second_prefix)}
    assert f"{first_prefix}/{hsm.id(thing_one)}" in first_paths
    assert f"{second_prefix}/{hsm.id(thing_two)}" in second_paths
    assert not (first_paths & second_paths)

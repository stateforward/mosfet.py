import pytest

from mosfet.abilities import sandbox
from mosfet.providers import docker

_MOST = sandbox.Resources(cpus=4.0, memory_bytes=8 * 1024**3, pids=1024, workspace_bytes=2 * 1024**3)


def test_runc_is_a_container_that_offers_both_networks_and_files() -> None:
    capacity = docker.capacity(max_resources=_MOST)

    assert capacity == sandbox.Capacity(
        isolation=sandbox.Isolation.CONTAINER,
        networks=frozenset({sandbox.Network.NONE, sandbox.Network.EGRESS}),
        max_resources=_MOST,
        files=True,
    )


def test_runsc_declares_a_user_space_kernel() -> None:
    capacity = docker.capacity(max_resources=_MOST, runtime="runsc")

    assert capacity.isolation is sandbox.Isolation.USER_KERNEL
    assert capacity.networks == frozenset({sandbox.Network.NONE, sandbox.Network.EGRESS})
    assert capacity.files is True


def test_a_spec_beyond_the_declared_ceiling_is_refused() -> None:
    capacity = docker.capacity(max_resources=sandbox.Resources(cpus=1.0))

    capacity.check(sandbox.Spec(image="python:3.13-slim"))
    with pytest.raises(ValueError, match="cpus"):
        capacity.check(sandbox.Spec(image="python:3.13-slim", resources=sandbox.Resources(cpus=2.0)))

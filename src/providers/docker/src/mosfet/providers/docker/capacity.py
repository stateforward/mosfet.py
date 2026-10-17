"""What a Docker Engine host can offer, declared once by the factory and injected into the host and the ability."""

import typing

from mosfet.abilities import sandbox

Runtime = typing.Literal["runc", "runsc"]
"""The OCI runtime a host's containers run on: ``runc`` by default, or gVisor's ``runsc`` when the owner sets it."""


def capacity(*, max_resources: sandbox.Resources, runtime: Runtime = "runc") -> sandbox.Capacity:
    """The ``sandbox.Capacity`` of a Docker Engine host on ``runtime`` whose ceilings are ``max_resources``.

    A container on ``runc`` is ``Isolation.CONTAINER``; on ``runsc`` it runs on a user-space kernel,
    ``Isolation.USER_KERNEL``. Both offer no network and the owner's egress bridge, and move whole files.
    """

    return sandbox.Capacity(
        isolation=sandbox.Isolation.USER_KERNEL if runtime == "runsc" else sandbox.Isolation.CONTAINER,
        networks=frozenset({sandbox.Network.NONE, sandbox.Network.EGRESS}),
        max_resources=max_resources,
        files=True,
    )


__all__ = ["Runtime", "capacity"]

"""A sandbox host on the Docker Engine API over a rootless Docker or Podman socket, modeled as an HSM.

Usage::

    from mosfet.abilities import sandbox
    from mosfet.providers import docker

    capacity = docker.capacity(max_resources=sandbox.Resources(cpus=2.0, memory_bytes=2 * 1024**3))
    bench_host = docker.Host(
        socket=config.socket,  # the rootless socket, e.g. $XDG_RUNTIME_DIR/docker.sock
        capacity=capacity,
        images=frozenset({"ghcr.io/stateforward/builder:1.0"}),
        owner=f"{config.name}/{config.instance_id}",
    )
    bench = sandbox.Sandbox(
        host=bench_host, capacity=capacity, spec=sandbox.Spec(image="ghcr.io/stateforward/builder:1.0")
    )

``Host`` answers the canonical ``sandbox.host`` event protocol; the ability owns it unstarted and reaches it only by
events. ``mapping`` and ``stream`` are the pure Engine API shapes and the attach-stream demux the host uses.
"""

from . import host, mapping, stream
from .capacity import Runtime, capacity
from .host import Host, Step

__version__ = "0.1.0"

__all__ = ["Host", "Runtime", "Step", "capacity", "host", "mapping", "stream"]

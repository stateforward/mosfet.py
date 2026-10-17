# mosfet-provider-docker

A sandbox host for `sandbox.Sandbox` on the Docker Engine API, over a **rootless** daemon's unix socket. `docker.Host`
is a state machine that answers the canonical `sandbox.host` event protocol; the ability owns it unstarted and
reaches it only by events. Every Engine API exchange is one state whose activity makes one request on an `httpx`
client with no timeout, no retries and no redirects: the state's own `hsm.after` bounds it.

```python
from mosfet.abilities import sandbox
from mosfet.providers import docker

capacity = docker.capacity(max_resources=sandbox.Resources(cpus=2.0, memory_bytes=2 * 1024**3))
bench = sandbox.Sandbox(
    host=docker.Host(
        socket=config.sandbox_socket,  # for example $XDG_RUNTIME_DIR/docker.sock
        capacity=capacity,
        images=frozenset({"ghcr.io/stateforward/builder:1.0"}),
        owner=f"{config.name}/{config.instance_id}",
    ),
    capacity=capacity,
    spec=sandbox.Spec(image="ghcr.io/stateforward/builder:1.0"),
)
```

The ability starts the host under its own lifetime context, so the host announces `serving` to it each time it
opens and `down` each time it fails to open or loses the daemon (W5-17). The ability makes its machine only after
`serving`, and resumes on the next `serving` after a `down`.

The factory builds one `capacity` and injects it into both the host and the ability (W5-7). `runtime="runsc"`
runs containers on gVisor and declares `Isolation.USER_KERNEL` (W5-9); the daemon must list `runsc`.

## What the host refuses

At open it reads `/_ping`, `/version` and `/info`, and refuses (logged once, retried every `reopen_max`):

| Refusal | When |
| --- | --- |
| `rootful` | `SecurityOptions` lacks `name=rootless`: a rootful Linux daemon, Docker Desktop's VM daemon (W5-8), colima |
| `cgroup_v1` | `CgroupVersion` is not `2` |
| `limit_unsupported` | `MemoryLimit` or `PidsLimit` is false: the limits would be silently ignored |
| `runtime_missing` | `runtime="runsc"` and the daemon has no `runsc` |
| `api_too_old` | the daemon's API is older than `api_version` (default `v1.41`) |

A create is refused `unsupported`, without a request, for an image off the owner's `images` allowlist (nothing is
ever pulled), a spec beyond `capacity`, or `Network.EGRESS` without the owner's `egress_network`. When that bridge is
configured the host creates it while opening (a bridge without inter-container traffic) if it is missing.

`/info` does not prove the limits hold (Podman reports `CpuCfsQuota: false` while its machines enforce `cpu.max`),
so each new machine is checked before it is answered (W5-14): an exec reads its cgroup `cpu.max`, `pids.max` and
`memory.max`, and unless each is a number no greater than the spec asked, the machine is removed and the create is
answered `unsupported`. A machine that could not be checked is removed too.

## A machine

One container per machine, created from the spec and only sleeping; all work arrives by exec:

| Canonical | Docker | Value |
| --- | --- | --- |
| `image` | `Image` | an allowlisted reference, never pulled |
| — | `Entrypoint`, `Cmd` | `["sleep", "infinity"]`, `[]` |
| `user` | `User` (container and every exec) | `65532:65532` |
| `workspace` | `WorkingDir`, `HostConfig.Tmpfs[workspace]` | `rw,nosuid,nodev,size=<workspace_bytes>,mode=1777`: root-owned and sticky, never `uid=`/`gid=` (Podman refuses them, W5-15); only uid 65532 runs inside |
| — | `HostConfig.Tmpfs["/tmp"]` | `rw,nosuid,nodev,size=64m` |
| `resources` | `NanoCpus`, `Memory` = `MemorySwap`, `PidsLimit` | as given, without swap |
| `network` | `NetworkMode` | `none`, or the owner's bridge for `egress` |
| — | `ReadonlyRootfs`, `Privileged`, `CapDrop`, `CapAdd`, `SecurityOpt` | `true`, `false`, `["ALL"]`, `[]`, `["no-new-privileges:true"]` |
| — | `Binds`, `Mounts`, `Devices`, `VolumesFrom` | empty: no host path, never a socket or `/run/secrets` |
| — | `Init`, `Runtime` | `true`, `runc` or `runsc` |
| `env` | `Env` | the spec's (and each run's) variables only, never the host's |
| — | `Labels` | `org.stateforward.mosfet.sandbox=1`, `org.stateforward.mosfet.owner=<owner>` (the sweep's filter) |

## Running, writing and reading

The Engine archive API cannot reach a tmpfs on a read-only root, so files move by exec through fixed `sh` wrappers
(W5-13). Paths and text are only ever positional arguments, never interpolated into a script.

* **run**: `timeout -s KILL <seconds> <argv>`. Stdin, when given, is first staged in `/tmp` by the write wrapper and
  read by a fixed wrapper that removes the file before the command starts. Stdout and stderr come back on the
  multiplexed attach stream, kept within the asked bound with their head and tail. Whether the exec ended before
  the timeout is the host's own `hsm.after(timeout)` timer, not a measured duration. Exit 124, or 137 at the
  timeout, is `timed_out`; 137 before it reads the machine's `memory.events`: an `oom_kill` makes it
  `out_of_memory`, otherwise `killed`. Any other exit is `exited`. A run still going at its timeout plus
  `kill_grace` is answered `failed{timeout}`, and the ability replaces the machine.
* **write**: the text travels as base64 argv chunks; the wrapper creates the parent directories and decodes them
  into the file.
* **read**: the wrapper prints a regular file that is not a link (anything else is `not_found`); a file over
  `mapping.MAX_READ_BYTES` (8 MiB) is `too_large`.

The image must provide `sh`, `timeout`, `mkdir`, `cat` and `base64`.

## Telemetry

Scope `bot.providers.docker`: `bot.provider.docker.exchange.count{step, status}`,
`bot.provider.docker.exchange.duration{step}`, `bot.provider.docker.open.count{outcome}`,
`bot.provider.docker.deferred.count`, and a `bot.docker.exchange` span per exchange. Values are closed sets; no
argv, path, output, environment value, image reference or id is ever an attribute or a message.

## Live check

`scripts/live_check.py` is a manual check against a real rootless socket (not CI). It needs an image already on the
daemon that also provides `python`:

```sh
python src/providers/docker/scripts/live_check.py --socket "$XDG_RUNTIME_DIR/docker.sock" \
    --image python:3.13-slim-bookworm --rootful-socket /var/run/docker.sock
```

Podman's Docker-compatible socket is not claimed until its own live check passes (W5-6).

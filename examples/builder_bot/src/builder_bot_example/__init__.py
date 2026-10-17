"""The builder bot example: one person, reached through a computer, and the bot that builds on its bench.

    mosfet run builder_bot_example:build --ready-file /tmp/ready
    builder-terminal --url ws://127.0.0.1:8765 --peer builder-desk --secret-file secrets/computer_token_secret

``build`` assembles one bot, ``builder``, for one person. The person reaches it through a ``computer.Computer``
served by a ``websocket.Server`` on this host; :mod:`builder_bot_example.terminal` is that computer's far side
in a terminal. The bot builds and tests on a ``sandbox.Sandbox`` bench backed by a ``docker.Host`` on a rootless
Docker or Podman socket. Its mind is the full cognition (memory, reflection, its persona and the routines it
authors) on one OpenAI-compatible local model.

Nothing here tells the bot what to do. It is given no seeded behavior, no skill and no instruction text: what
it can do comes from the tools its computer and its bench offer on each turn, and what it does with them is
its own to choose, including nothing. The owner fixes only who it is (its traits) and its bench (``bench``).
"""

import collections.abc
import datetime
import functools
import logging
import pathlib
import typing

import hsm
import mosfet
import pydantic
from mosfet import runtime
from mosfet.abilities import cognition, memory, persona, processing, sandbox
from mosfet.devices import computer
from mosfet.providers import docker
from mosfet.providers import openai_compat
from mosfet.providers import websocket

DEFAULT_IMAGE: typing.Final = "localhost/mosfet-bench:1"
"""The bench image ``docker/bench/Dockerfile`` builds into the rootless daemon; nothing is pulled."""
DEFAULT_TRAITS: typing.Final = persona.Traits(traits=("careful", "curious", "candid"))
"""The traits this example's owner gives the bot; ``BUILDER_TRAITS`` gives it others."""
MAX_OUTPUT_CHARS: typing.Final = 8_192
"""Most characters the bench keeps of each output stream and file read, so a turn's observations fit a local
model's context."""

_LOG = logging.getLogger(__name__)
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765
_DEFAULT_PEER = "builder-desk"
_DEFAULT_RUNTIME: docker.Runtime = "runc"
_RUNTIMES: tuple[docker.Runtime, ...] = ("runc", "runsc")
_WORKSPACE = "/workspace"
_IMAGE: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(sandbox.ImageRef)


class BuilderBot(mosfet.Bot):
    """A bot that holds one person's computer, ``desk``, and keeps a sandbox ``bench`` to build and test on.

    The bot knows the computer by the id it was built with (``computer.Computer(id="desk", ...)``).

    It adds no behavior to ``mosfet.Bot``: what reaches it from the computer and the bench wakes its cognition,
    and what it does next is that cognition's to choose.
    """

    def __init__(self, *, desk: computer.Computer, bench: sandbox.Sandbox, cognition: cognition.Cognition) -> None:
        super().__init__((desk,), cognition=cognition, acquired_abilities=(bench,))


def bench(*, image: str, network: sandbox.Network) -> sandbox.Spec:
    """The owner's bench: ``image`` on the default limits, ``network``, and a writable home in the workspace.

    The limits are one CPU, 1 GiB without swap, 256 pids and a 512 MiB workspace at ``/workspace``, which the
    spec makes ``HOME``, since the machine's root is read-only. Python writes no bytecode and pip never prompts.
    No secret goes into this environment. A machine lives at most the spec's default eight hours.
    """

    return sandbox.Spec(
        image=image,
        resources=sandbox.Resources(),
        network=network,
        workspace=_WORKSPACE,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_INPUT": "1"},
    )


def compose(
    *,
    processor: processing.Processor,
    server: websocket.Server,
    peer: str,
    person_id: str | None,
    host: hsm.Instance,
    capacity: sandbox.Capacity,
    spec: sandbox.Spec,
    traits: persona.Traits,
    database: str,
    now: collections.abc.Callable[[], datetime.datetime],
) -> runtime.Assembly:
    """Assemble the builder from constructed collaborators; nothing is started here.

    ``processor`` is the one model every faculty thinks with. ``server`` carries the person's computer, ``desk``,
    whose remote machine the server knows as ``peer``; ``person_id`` is who uses it, when the owner knows.
    ``host`` is an unstarted sandbox host the bench owns from here on, declaring ``capacity``; ``spec`` is the
    owner's bench on it.
    ``traits`` are the owner's seed of who the bot is, and ``database`` is where its memory is kept. ``now``
    is the UTC wall time the routines the bot authors are scheduled by.

    Raises:
        ValueError: ``spec`` asks for a network or limit beyond ``capacity``, or ``host`` is already started.
    """

    desk = computer.Computer(id="desk", transport=server, peer_id=peer, person_id=person_id)
    workbench = sandbox.Sandbox(host=host, capacity=capacity, spec=spec, max_output_chars=MAX_OUTPUT_CHARS)
    store = memory.Memory(database=database)
    self_knowledge = persona.Persona(traits=traits, memory=store)
    mind = cognition.Cognition(
        autonomy=cognition.Autonomy(memory=store),
        intuition=cognition.Intuition(processor=processor),
        reasoning=cognition.Reasoning(processor=processor),
        reflection=cognition.Reflection(processor=processor, memory=store, persona=self_knowledge),
        routines=cognition.Routines(memory=store, now=now),
        memory=store,
        associative=memory.Associative(memory=store),
        consolidation=memory.Consolidation(memory=store, processor=processor),
        persona=self_knowledge,
    )
    builder = BuilderBot(desk=desk, bench=workbench, cognition=mind)
    return runtime.Assembly(bots={"builder": builder}, actors=(server,))


def _required(config: runtime.Config, name: str) -> str:
    value = config.environ.get(name)
    if not value:
        raise ValueError(f"set {name}")
    return value


def _port(config: runtime.Config) -> int:
    text = config.environ.get("BUILDER_COMPUTER_PORT") or str(_DEFAULT_PORT)
    if not text.isdecimal() or not 1 <= int(text) <= 65_535:
        raise ValueError(f"BUILDER_COMPUTER_PORT must be a port number from 1 to 65535, got {text!r}")
    return int(text)


def _runtime(config: runtime.Config) -> docker.Runtime:
    text = config.environ.get("BUILDER_SANDBOX_RUNTIME") or _DEFAULT_RUNTIME
    for known in _RUNTIMES:
        if text == known:
            return known
    raise ValueError(f"BUILDER_SANDBOX_RUNTIME must be one of {', '.join(_RUNTIMES)}, got {text!r}")


def _traits(config: runtime.Config) -> persona.Traits:
    text = config.environ.get("BUILDER_TRAITS")
    if text is None:
        return DEFAULT_TRAITS
    given = tuple(trait.strip() for trait in text.split(","))
    if not all(given):
        raise ValueError("BUILDER_TRAITS must be comma-separated traits, none of them empty")
    try:
        return persona.Traits(traits=given)
    except pydantic.ValidationError as error:
        raise ValueError(f"BUILDER_TRAITS is not a set of traits: {error.errors()[0]['msg']}") from error


def _image(config: runtime.Config) -> str:
    text = config.environ.get("BUILDER_SANDBOX_IMAGE") or DEFAULT_IMAGE
    try:
        return _IMAGE.validate_python(text)
    except pydantic.ValidationError as error:
        raise ValueError(f"BUILDER_SANDBOX_IMAGE must be an image reference, got {text!r}") from error


def build(config: runtime.Config) -> runtime.Assembly:
    """Build the builder from the process configuration, for ``mosfet run``.

    Reads ``config.environ`` and ``config.secrets`` only:

    * ``OPENAI_BASE_URL`` and ``OPENAI_MODEL`` (required): the local OpenAI-compatible endpoint and its model;
    * the ``openai_api_key`` secret (required; a placeholder for a keyless local server);
    * ``BUILDER_SANDBOX_SOCKET`` (required): the rootless Docker or Podman socket the bench runs on;
    * ``BUILDER_SANDBOX_IMAGE`` (default ``localhost/mosfet-bench:1``): the one image the bench may use;
    * ``BUILDER_SANDBOX_EGRESS_NETWORK``: the owner's egress bridge; unset, the bench has no network;
    * ``BUILDER_SANDBOX_RUNTIME`` (default ``runc``): ``runc``, or ``runsc`` for gVisor;
    * ``BUILDER_COMPUTER_HOST`` and ``BUILDER_COMPUTER_PORT`` (default ``127.0.0.1`` and ``8765``): where the
      person's computer is served;
    * ``BUILDER_COMPUTER_PEER`` (default ``builder-desk``): the computer's device and peer id;
    * the ``computer_token_secret`` secret (required): the key the terminal's token is signed with;
    * ``BUILDER_PERSON_ID``: who uses the computer, when the owner knows;
    * ``BUILDER_TRAITS`` (default ``careful,curious,candid``): the owner's traits for the bot, comma separated.

    Memory is kept in ``config.data_dir / "memory.sqlite3"``.

    Raises:
        ValueError: a required setting is unset or empty, or a setting is malformed; the message names it.
        runtime.MissingSecretError: no source provides ``openai_api_key`` or ``computer_token_secret``.
    """

    base_url = _required(config, "OPENAI_BASE_URL")
    model = _required(config, "OPENAI_MODEL")
    socket = pathlib.Path(_required(config, "BUILDER_SANDBOX_SOCKET"))
    image = _image(config)
    egress_network = config.environ.get("BUILDER_SANDBOX_EGRESS_NETWORK") or None
    oci_runtime = _runtime(config)
    listen_host = config.environ.get("BUILDER_COMPUTER_HOST") or _DEFAULT_HOST
    port = _port(config)
    peer = config.environ.get("BUILDER_COMPUTER_PEER") or _DEFAULT_PEER
    person_id = config.environ.get("BUILDER_PERSON_ID") or None
    traits = _traits(config)
    api_key = config.secrets.get("openai_api_key")
    token_secret = config.secrets.get("computer_token_secret")

    network = sandbox.Network.NONE if egress_network is None else sandbox.Network.EGRESS
    capacity = docker.capacity(max_resources=sandbox.Resources(), runtime=oci_runtime)
    assembly = compose(
        processor=openai_compat.Processor(
            client=openai_compat.ChatClient(model=model, base_url=base_url, api_key=api_key.get_secret_value()),
            provider="builder_bot",
        ),
        server=websocket.Server(host=listen_host, port=port, verifier=websocket.HmacTokens(token_secret)),
        peer=peer,
        person_id=person_id,
        host=docker.Host(
            socket=socket,
            capacity=capacity,
            images=frozenset({image}),
            owner=f"{config.name}/{config.instance_id}",
            egress_network=egress_network,
            runtime=oci_runtime,
        ),
        capacity=capacity,
        spec=bench(image=image, network=network),
        traits=traits,
        database=str(config.data_dir / "memory.sqlite3"),
        now=functools.partial(datetime.datetime.now, datetime.UTC),
    )
    _LOG.info(
        "builder wired: model=%s bench_image=%s network=%s runtime=%s computer=ws://%s:%d/v1/%s",
        model,
        image,
        network.value,
        oci_runtime,
        listen_host,
        port,
        peer,
    )
    return assembly


__all__ = ["DEFAULT_IMAGE", "DEFAULT_TRAITS", "MAX_OUTPUT_CHARS", "BuilderBot", "bench", "build", "compose"]

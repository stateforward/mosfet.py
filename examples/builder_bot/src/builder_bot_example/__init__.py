"""The builder bot: one person, reached through a computer, and the bot that builds on its bench.

    python -m builder_bot_example --ready-file /tmp/ready
    builder-terminal --url ws://127.0.0.1:8765 --peer builder-desk --secret-file secrets/computer_token_secret

:func:`builder` defines one bot, ``builder``, for one person, with ``bot.define``. The person reaches it through a
``computer.Computer``, ``desk``, served by a ``websocket.Server`` on this host; :mod:`builder_bot_example.terminal`
is that computer's far side in a terminal. The bot builds and tests on a ``sandbox.Sandbox`` bench backed by a
``docker.Host`` on a rootless Docker or Podman socket. Its mind is the full cognition (memory, reflection, its
persona and the routines it authors) on one OpenAI-compatible local model. ``__main__`` reads the owner's
:class:`Settings` and the process's secrets, builds those parts, and runs the bot with ``bot.run``.

Nothing here tells the bot what to do. It is given no seeded behavior, no skill and no instruction text: what
it can do comes from the tools its computer and its bench offer on each turn, and what it does with them is
its own to choose, including nothing. The owner fixes only who it is (its traits) and its bench (``bench``).
"""

import collections.abc
import pathlib
import typing

import hsm
import pydantic
from mosfet import bot
from mosfet.abilities import memory, persona, processing, sandbox
from mosfet.behavior import schedule
from mosfet.devices import computer
from mosfet.providers import docker
from mosfet.providers import websocket

DEFAULT_IMAGE: typing.Final = "localhost/mosfet-bench:2"
"""Bench image 2, which ``docker/bench/Dockerfile`` builds into the rootless daemon; nothing is pulled. It has
mosfet installed, so the bot can write a bot file and test it there."""
DEFAULT_TRAITS: typing.Final = persona.Traits(traits=("careful", "curious", "candid"))
"""The traits this example's owner gives the bot; ``BUILDER_TRAITS`` gives it others."""
MAX_OUTPUT_CHARS: typing.Final = 8_192
"""Most characters the bench keeps of each output stream and file read, so a turn's observations fit a local
model's context."""

_WORKSPACE = "/workspace"


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


def builder(
    *,
    processor: processing.Processor,
    server: websocket.Server,
    host: hsm.Instance,
    capacity: sandbox.Capacity,
    spec: sandbox.Spec,
    peer: str,
    person_id: str | None,
    traits: tuple[persona.Trait, ...],
    store: memory.Memory,
    clock: schedule.WallClock,
) -> bot.Model:
    """The builder: one person's computer as ``desk``, a bench to build on, and the full cognition.

    ``processor`` is the one model every faculty thinks with. ``server`` carries the person's computer, ``desk``,
    whose remote machine the server knows as ``peer``; ``person_id`` is who uses it, when the owner knows. ``host``
    is an unstarted sandbox host the bench owns from here on, declaring ``capacity``; ``spec`` is the owner's bench
    on it. ``traits`` are the owner's seed of who the bot is, ``store`` is where its memory is kept, and ``clock``
    is the UTC wall time the routines the bot authors are scheduled by. Nothing is started here: ``bot.run``
    starts the server as a shared actor, then the bot.

    Raises:
        ValueError: ``spec`` asks for a network or limit beyond ``capacity``, or ``host`` is already started.
    """

    return bot.define(
        "builder",
        bot.traits(*traits),
        bot.cognition(processor, memory=store, associative=True, consolidation=True, autonomy=True, routines=clock),
        bot.attach(computer.Computer(id="desk", transport=server, peer_id=peer, person_id=person_id)),
        bot.attach(sandbox.Sandbox(host=host, capacity=capacity, spec=spec, max_output_chars=MAX_OUTPUT_CHARS)),
    )


class Settings(pydantic.BaseModel):
    """The owner's settings for the builder, read from the process environment by ``__main__``.

    Each field is read from the variable its alias names; a variable set to the empty string is unset. A value
    is never shown in an error: the error names the variable and what is wrong with it.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="ignore", hide_input_in_errors=True
    )

    base_url: str = pydantic.Field(
        alias="OPENAI_BASE_URL",
        min_length=1,
        description="The local OpenAI-compatible endpoint.",
        examples=["http://127.0.0.1:8080/v1"],
    )
    model: str = pydantic.Field(
        alias="OPENAI_MODEL", min_length=1, description="The model that endpoint serves.", examples=["qwen3-coder"]
    )
    socket: pathlib.Path = pydantic.Field(
        alias="BUILDER_SANDBOX_SOCKET",
        description="The rootless Docker or Podman API socket the bench runs on; it has no safe default.",
        examples=["/run/user/1000/podman/podman.sock"],
    )
    image: sandbox.ImageRef = pydantic.Field(
        default=DEFAULT_IMAGE,
        alias="BUILDER_SANDBOX_IMAGE",
        description="The one image the bench may use.",
        examples=[DEFAULT_IMAGE],
    )
    egress_network: str | None = pydantic.Field(
        default=None,
        alias="BUILDER_SANDBOX_EGRESS_NETWORK",
        description="The owner's egress bridge network; unset, the bench has no network at all.",
        examples=["builder-egress"],
    )
    runtime: docker.Runtime = pydantic.Field(
        default="runc",
        alias="BUILDER_SANDBOX_RUNTIME",
        description="The OCI runtime: `runc`, or `runsc` for gVisor.",
        examples=["runc"],
    )
    host: str = pydantic.Field(
        default="127.0.0.1",
        alias="BUILDER_COMPUTER_HOST",
        min_length=1,
        description="Where the person's computer is served; loopback by default.",
        examples=["127.0.0.1"],
    )
    port: int = pydantic.Field(
        default=8765,
        alias="BUILDER_COMPUTER_PORT",
        ge=1,
        le=65_535,
        description="The port the person's computer is served on.",
        examples=[8765],
    )
    peer: computer.PeerId = pydantic.Field(
        default="builder-desk",
        alias="BUILDER_COMPUTER_PEER",
        description="The name the computer's far side connects as.",
        examples=["builder-desk"],
    )
    person_id: str | None = pydantic.Field(
        default=None,
        alias="BUILDER_PERSON_ID",
        description="Who uses the computer, stamped on what it observes, when the owner knows.",
        examples=["person-1"],
    )
    traits: persona.Traits = pydantic.Field(
        default=DEFAULT_TRAITS,
        alias="BUILDER_TRAITS",
        description="The owner's traits for the bot, comma separated.",
        examples=["careful,curious,candid"],
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _unset_empty(cls, data: object) -> object:
        if isinstance(data, collections.abc.Mapping):
            given = typing.cast(collections.abc.Mapping[object, object], data)
            return {name: value for name, value in given.items() if value != ""}
        return data

    @pydantic.field_validator("traits", mode="before")
    @classmethod
    def _split_traits(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        given = tuple(trait.strip() for trait in value.split(","))
        if not all(given):
            raise ValueError("must be comma-separated traits, none of them empty")
        return persona.Traits(traits=given)


__all__ = ["DEFAULT_IMAGE", "DEFAULT_TRAITS", "MAX_OUTPUT_CHARS", "Settings", "bench", "builder"]

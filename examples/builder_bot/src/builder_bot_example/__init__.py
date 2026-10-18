"""The builder bot: one person, reached through a computer, and the bot that builds on its bench.

    python -m builder_bot_example --ready-file /tmp/ready
    builder-terminal --url ws://127.0.0.1:8765 --peer builder-desk --secret-file secrets/computer_token_secret

``__main__`` is the bot file. It reads the owner's :class:`Settings` and the process's secrets, then defines one
bot, ``builder``, for one person, with one ``bot.define`` that constructs every part inline, and runs it with
``bot.run``. The person reaches the bot through a ``computer.Computer``, ``computer``, which sits on a desk (an
``environment.Object``) and is served by a ``websocket.Server`` on this host; :mod:`builder_bot_example.terminal`
is that computer's far side in a terminal. The bot builds and tests on a ``sandbox.Sandbox`` bench backed by a
``docker.Host`` on a rootless Docker or Podman socket. Its mind is the full cognition (memory, reflection, its
persona and the routines it authors) on one OpenAI-compatible local model.

Nothing here tells the bot what to do. It is given no seeded behavior, no skill and no instruction text: what
it can do comes from the tools its computer and its bench offer on each turn, and what it does with them is
its own to choose, including nothing. The owner fixes only who it is (its traits) and its bench.
"""

import collections.abc
import pathlib
import typing

import pydantic
from mosfet.abilities import persona, sandbox
from mosfet.devices import computer
from mosfet.providers import docker

DEFAULT_IMAGE: typing.Final = "localhost/mosfet-bench:2"
"""Bench image 2, which ``docker/bench/Dockerfile`` builds into the rootless daemon; nothing is pulled. It has
mosfet installed, so the bot can write a bot file and test it there."""
DEFAULT_TRAITS: typing.Final = persona.Traits(traits=("careful", "curious", "candid"))
"""The traits this example's owner gives the bot; ``BUILDER_TRAITS`` gives it others."""
MAX_OUTPUT_CHARS: typing.Final = 8_192
"""Most characters the bench keeps of each output stream and file read, so a turn's observations fit a local
model's context."""


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


__all__ = ["DEFAULT_IMAGE", "DEFAULT_TRAITS", "MAX_OUTPUT_CHARS", "Settings"]

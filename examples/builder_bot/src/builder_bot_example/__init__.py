"""The builder bot: one person, reached through a computer, and the bot that builds on its bench.

    python -m builder_bot_example --ready-file /tmp/ready
    builder-terminal --url ws://127.0.0.1:8765 --peer builder-desk --secret-file secrets/computer_token_secret

``__main__`` is the bot file. It reads the owner's :class:`Settings` and the process's secrets, then defines one
bot, ``builder``, for one person, with one ``bot.define`` that constructs every part inline, and runs it with
``bot.run``. Before anything starts it checks the bench image (:func:`check_bench`): bench image 2 is built for
one mosfet commit and labeled with it, and a stale image is refused. The person reaches the bot through a
``computer.Computer``, ``computer``, which sits on a desk (an ``environment.Object``) and is served by a
``websocket.Server`` on this host; :mod:`builder_bot_example.terminal` is that computer's far side in a terminal.
The bot builds and tests on a ``sandbox.Sandbox`` bench backed by a ``docker.Host`` on a rootless Docker or Podman
socket. Its mind is the full cognition (memory, reflection, its persona and the routines it authors) on one
OpenAI-compatible local model.

Nothing here tells the bot what to do. It is given no seeded behavior, no skill and no instruction text: what
it can do comes from the tools its computer and its bench offer on each turn, each bench request described with
what the bench image says it holds, and what it does with them is its own to choose, including nothing. The
owner fixes only who it is (its traits) and its bench.
"""

import collections.abc
import http.client
import pathlib
import socket as sockets
import subprocess
import typing
import urllib.parse

import mosfet
import pydantic
from mosfet.abilities import persona, sandbox
from mosfet.abilities.language import text
from mosfet.devices import computer
from mosfet.providers import docker

DEFAULT_IMAGE: typing.Final = "localhost/mosfet-bench:2"
"""Bench image 2, which ``docker/bench/Dockerfile`` builds into the rootless daemon; nothing is pulled. It has
mosfet installed, so the bot can write a bot file and test it there."""
REVISION_LABEL: typing.Final = "org.opencontainers.image.revision"
"""The label bench image 2 carries: the mosfet commit it was built from (``--build-arg MOSFET_COMMIT``)."""
DESCRIPTION_LABEL: typing.Final = "org.opencontainers.image.description"
"""The label that says what the bench image holds; the bench's ``sandbox.Spec`` is described with it."""
_REBUILD: typing.Final = (
    "rebuild it from the repository root: podman build --build-arg MOSFET_COMMIT=$(git rev-parse HEAD)"
    + " --tag {image} --file docker/bench/Dockerfile ."
)
_CHECK_TIMEOUT_SECONDS: typing.Final = 10.0
_MAX_INSPECT_BYTES: typing.Final = 1024 * 1024
DEFAULT_TRAITS: typing.Final = persona.Traits(traits=("careful", "curious", "candid"))
"""The traits this example's owner gives the bot; ``BUILDER_TRAITS`` gives it others."""
MAX_OUTPUT_CHARS: typing.Final = 8_192
"""Most characters the bench keeps of each output stream and file read, so a turn's observations fit a local
model's context."""


class BenchError(Exception):
    """The bench cannot be used with the mosfet this host runs: its image is missing, stale or unlabeled, or the
    check itself could not be made. The message says which, and how to rebuild the image."""


def mosfet_commit(source: pathlib.Path | None = None) -> str:
    """The commit of the mosfet checkout this process imports: ``git rev-parse HEAD`` in ``source``, by default
    the directory ``mosfet`` is imported from (the example installs it editable from the repository).

    Raises:
        BenchError: If ``source`` is not in a git checkout, or git cannot be run.
    """

    where = source if source is not None else pathlib.Path(mosfet.__file__).parent
    try:
        found = subprocess.run(
            ["git", "-C", str(where), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_CHECK_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BenchError(
            f"cannot tell which mosfet commit this host runs: git failed ({type(error).__name__})"
        ) from None
    if found.returncode != 0:
        raise BenchError(f"cannot tell which mosfet commit this host runs: {where} is not in a git checkout")
    return found.stdout.strip()


class _ImageConfig(pydantic.BaseModel):
    labels: dict[str, str] | None = pydantic.Field(default=None, alias="Labels")


class _Image(pydantic.BaseModel):
    """The part of the Engine API's image inspect (``GET /images/{name}/json``) the check reads."""

    config: _ImageConfig = pydantic.Field(default_factory=_ImageConfig, alias="Config")


def check_bench(*, socket: pathlib.Path, image: str, commit: str) -> str | None:
    """Check that the daemon on ``socket`` has the bench ``image``, built for mosfet ``commit``, and return what
    the image says it holds (its ``DESCRIPTION_LABEL``), or ``None`` when it says nothing.

    The image's ``REVISION_LABEL`` must be ``commit``: a bench built from another commit has another mosfet,
    and a bot file the bot writes there is tested against an API this host does not run. The daemon is asked
    once, by image inspect; nothing is pulled.

    Raises:
        BenchError: If the daemon cannot be reached or answers otherwise than an image inspect, the image is not
            there, or it was built for another commit (or carries no commit at all).
    """

    rebuild = _REBUILD.format(image=image)
    # One HTTP exchange with the Engine API on the daemon's unix socket: the connection uses the socket it is given.
    connection = http.client.HTTPConnection("localhost", timeout=_CHECK_TIMEOUT_SECONDS)
    try:
        connected = sockets.socket(sockets.AF_UNIX, sockets.SOCK_STREAM)
        connection.sock = connected  # closed with the connection
        connected.settimeout(_CHECK_TIMEOUT_SECONDS)
        connected.connect(str(socket))
        connection.request("GET", f"/images/{urllib.parse.quote(image, safe='/:@')}/json")
        response = connection.getresponse()
        status, body = response.status, response.read(_MAX_INSPECT_BYTES + 1)
    except (OSError, http.client.HTTPException) as error:
        raise BenchError(
            f"cannot check bench image {image}: the daemon at {socket} did not answer ({type(error).__name__});"
            + " start it, then start the builder"
        ) from None
    finally:
        connection.close()
    if status == 404:
        raise BenchError(f"bench image {image} is not in the daemon at {socket}; {rebuild}")
    if status != 200 or len(body) > _MAX_INSPECT_BYTES:
        raise BenchError(f"cannot check bench image {image}: the daemon at {socket} answered {status}")
    try:
        labels = _Image.model_validate_json(body).config.labels or {}
    except pydantic.ValidationError:
        raise BenchError(f"cannot check bench image {image}: the daemon at {socket} answered no image") from None
    built_for = labels.get(REVISION_LABEL)
    if built_for is None:
        raise BenchError(
            f"bench image {image} carries no mosfet commit ({REVISION_LABEL}), so it predates this host's"
            + f" mosfet {commit}; {rebuild}"
        )
    if built_for != commit:
        raise BenchError(
            f"bench image {image} was built for mosfet {built_for}, but this host runs mosfet {commit}; {rebuild}"
        )
    return labels.get(DESCRIPTION_LABEL) or None


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
    stream: bool = pydantic.Field(
        default=True,
        alias="OPENAI_STREAM",
        description=(
            "Whether the model's calls are streamed, so what the bot is writing reaches the screen as it writes;"
            + " off for an endpoint that cannot stream a tool call."
        ),
        examples=[True, False],
    )
    tool_choice: typing.Literal[text.ToolSelectionPolicy.REQUIRED, text.ToolSelectionPolicy.AUTO] = pydantic.Field(
        default=text.ToolSelectionPolicy.REQUIRED,
        alias="OPENAI_TOOL_CHOICE",
        description=(
            "The tool choice sent with each of the model's calls: `required`, or `auto` for an endpoint whose"
            + " guided decoding of a required call degrades what the model writes (drops the line breaks in code)."
        ),
        examples=["required", "auto"],
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


__all__ = [
    "DEFAULT_IMAGE",
    "DESCRIPTION_LABEL",
    "DEFAULT_TRAITS",
    "MAX_OUTPUT_CHARS",
    "REVISION_LABEL",
    "BenchError",
    "Settings",
    "check_bench",
    "mosfet_commit",
]

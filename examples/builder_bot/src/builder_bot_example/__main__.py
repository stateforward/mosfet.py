"""The builder bot file: ``python -m builder_bot_example`` runs the builder for one person on this host.

Reads the owner's :class:`~builder_bot_example.Settings` from the environment (``OPENAI_BASE_URL``,
``OPENAI_MODEL``, ``BUILDER_SANDBOX_SOCKET`` and the optional ``OPENAI_STREAM``, ``OPENAI_TOOL_CHOICE``,
``TYPESAFE_BASE_URL`` and ``BUILDER_*`` variables) and its secrets: ``openai_api_key`` (a placeholder for a
keyless local server), ``typesafe_api_key`` (hosted TypeSafe's key, for the decision model intuition decides on;
read only when ``TYPESAFE_BASE_URL`` is unset, since a local system_one server takes none and is never sent it)
and ``computer_token_secret``, the key the terminal's token is signed with. Each is the Docker secret, the file named by ``<NAME>_FILE``, or
``<NAME>``. The process options (``MOSFET_*`` or flags) are ``bot.process``'s; memory is kept in
``<data dir>/memory.sqlite3``. A missing or malformed setting or secret ends the process with exit ``1``,
naming it, before anything starts. So does a bench image that was not built for the mosfet commit this host
runs (``check_bench``), naming both commits and the rebuild.

Then one ``bot.define`` constructs every part of the bot inline: its model, its intuition's decision model, its
memory, the person's computer
``computer`` and its bench. The bench is the owner's: the settings' image on the default limits (one CPU,
1 GiB without swap, 256 pids and a 512 MiB workspace at ``/workspace``, which is ``HOME``, since the machine's
root is read-only), no network unless the owner names an egress network, Python writing no bytecode and pip
never prompting, no secret in its environment, and described with what the image's own description label says
it holds. Two variables come first. The websocket server the computer is served by, because ``bot.run`` starts
it first, as a shared actor, and then the bot. And the desk the computer sits on, an ``environment.Object``: it
does nothing and is never given to the bot, and the computer is placed on it (``placement=desk.surface()``), so
the bot sees the desk as its computer's ``on="desk"``.
"""

import datetime
import functools
import os

import pydantic
from mosfet import bot, environment
from mosfet.abilities import memory, sandbox
from mosfet.devices import computer
from mosfet.providers import docker
from mosfet.providers import openai_compat
from mosfet.providers import typesafe
from mosfet.providers import websocket

from builder_bot_example import MAX_OUTPUT_CHARS, BenchError, Settings, check_bench, mosfet_commit

if __name__ == "__main__":
    process = bot.process()
    try:
        settings = Settings.model_validate(os.environ)
    except pydantic.ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}" for problem in error.errors()
        )
        raise SystemExit(f"builder_bot_example: {problems}") from None
    try:
        holds = check_bench(socket=settings.socket, image=settings.image, commit=mosfet_commit())
    except BenchError as error:
        raise SystemExit(f"builder_bot_example: {error}") from None
    server = websocket.Server(
        host=settings.host,
        port=settings.port,
        verifier=websocket.HmacTokens(process.secrets.get("computer_token_secret")),
    )
    desk = environment.Object(id="desk")
    builder = bot.define(
        "builder",
        bot.traits(*settings.traits.traits),
        bot.cognition(
            openai_compat.Processor(
                client=openai_compat.ChatClient(
                    model=settings.model,
                    base_url=settings.base_url,
                    api_key=process.secrets.get("openai_api_key").get_secret_value(),
                ),
                provider="builder_bot",
                stream=settings.stream,
                tool_choice=settings.tool_choice,
            ),
            intuition=typesafe.Processor(
                base_url=settings.typesafe_base_url,
                # A local system_one server takes no key and is never sent hosted TypeSafe's.
                api_key=None
                if settings.typesafe_base_url is not None
                else process.secrets.get("typesafe_api_key").get_secret_value(),
            ),
            memory=memory.Memory(database=str(process.data_dir / "memory.sqlite3")),
            associative=True,
            consolidation=True,
            autonomy=True,
            routines=functools.partial(datetime.datetime.now, datetime.UTC),
        ),
        bot.attach(
            computer.Computer(
                id="computer",
                transport=server,
                peer_id=settings.peer,
                person_id=settings.person_id,
                placement=desk.surface(),
            )
        ),
        bot.attach(
            sandbox.Sandbox(
                host=docker.Host(
                    socket=settings.socket,
                    images=frozenset({settings.image}),
                    owner=f"{process.name}/{process.instance_id}",
                    egress_network=settings.egress_network,
                    runtime=settings.runtime,
                ),
                capacity=docker.capacity(max_resources=sandbox.Resources(), runtime=settings.runtime),
                spec=sandbox.Spec(
                    image=settings.image,
                    network=sandbox.Network.NONE if settings.egress_network is None else sandbox.Network.EGRESS,
                    env={"PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_INPUT": "1"},
                    description=holds,
                ),
                max_output_chars=MAX_OUTPUT_CHARS,
            )
        ),
    )
    bot.run(builder, actors=(server,), process=process)

"""The builder bot file: ``python -m builder_bot_example`` runs the builder for one person on this host.

Reads the owner's :class:`~builder_bot_example.Settings` from the environment (``OPENAI_BASE_URL``,
``OPENAI_MODEL``, ``BUILDER_SANDBOX_SOCKET`` and the optional ``BUILDER_*`` variables) and two secrets:
``openai_api_key`` (a placeholder for a keyless local server) and ``computer_token_secret``, the key the
terminal's token is signed with. Each is the Docker secret, the file named by ``<NAME>_FILE``, or
``<NAME>``. The process options (``MOSFET_*`` or flags) are ``bot.process``'s; memory is kept in
``<data dir>/memory.sqlite3``. A missing or malformed setting or secret ends the process with exit ``1``,
naming it, before anything starts.

Then one ``bot.define`` constructs every part of the bot inline: its model, its memory, the person's computer
``desk`` and its bench. The bench is the owner's: the settings' image on the default limits (one CPU, 1 GiB
without swap, 256 pids and a 512 MiB workspace at ``/workspace``, which is ``HOME``, since the machine's root
is read-only), no network unless the owner names an egress network, Python writing no bytecode and pip never
prompting, and no secret in its environment. The websocket server the computer is served by is the one
variable, because ``bot.run`` starts it first, as a shared actor, and then the bot.
"""

import datetime
import functools
import os

import pydantic
from mosfet import bot
from mosfet.abilities import memory, sandbox
from mosfet.devices import computer
from mosfet.providers import docker
from mosfet.providers import openai_compat
from mosfet.providers import websocket

from builder_bot_example import MAX_OUTPUT_CHARS, Settings

if __name__ == "__main__":
    process = bot.process()
    try:
        settings = Settings.model_validate(os.environ)
    except pydantic.ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}" for problem in error.errors()
        )
        raise SystemExit(f"builder_bot_example: {problems}") from None
    server = websocket.Server(
        host=settings.host,
        port=settings.port,
        verifier=websocket.HmacTokens(process.secrets.get("computer_token_secret")),
    )
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
            ),
            memory=memory.Memory(database=str(process.data_dir / "memory.sqlite3")),
            associative=True,
            consolidation=True,
            autonomy=True,
            routines=functools.partial(datetime.datetime.now, datetime.UTC),
        ),
        bot.attach(computer.Computer(id="desk", transport=server, peer_id=settings.peer, person_id=settings.person_id)),
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
                ),
                max_output_chars=MAX_OUTPUT_CHARS,
            )
        ),
    )
    bot.run(builder, actors=(server,), process=process)

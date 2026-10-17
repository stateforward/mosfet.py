"""Run the builder for one person on this host: ``python -m builder_bot_example``.

Reads the owner's :class:`~builder_bot_example.Settings` from the environment (``OPENAI_BASE_URL``,
``OPENAI_MODEL``, ``BUILDER_SANDBOX_SOCKET`` and the optional ``BUILDER_*`` variables) and two secrets:
``openai_api_key`` (a placeholder for a keyless local server) and ``computer_token_secret``, the key the
terminal's token is signed with. Each is the Docker secret, the file named by ``<NAME>_FILE``, or
``<NAME>``. The process options (``MOSFET_*`` or flags) are ``bot.process``'s; memory is kept in
``<data dir>/memory.sqlite3``. A missing or malformed setting or secret ends the process with exit ``1``,
naming it, before anything starts.
"""

import datetime
import functools
import os

import pydantic
from mosfet import bot
from mosfet.abilities import memory, sandbox
from mosfet.providers import docker
from mosfet.providers import openai_compat
from mosfet.providers import websocket

from builder_bot_example import Settings, bench, builder

if __name__ == "__main__":
    process = bot.process()
    try:
        settings = Settings.model_validate(os.environ)
    except pydantic.ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}" for problem in error.errors()
        )
        raise SystemExit(f"builder_bot_example: {problems}") from None
    capacity = docker.capacity(max_resources=sandbox.Resources(), runtime=settings.runtime)
    network = sandbox.Network.NONE if settings.egress_network is None else sandbox.Network.EGRESS
    server = websocket.Server(
        host=settings.host,
        port=settings.port,
        verifier=websocket.HmacTokens(process.secrets.get("computer_token_secret")),
    )
    model = builder(
        processor=openai_compat.Processor(
            client=openai_compat.ChatClient(
                model=settings.model,
                base_url=settings.base_url,
                api_key=process.secrets.get("openai_api_key").get_secret_value(),
            ),
            provider="builder_bot",
        ),
        server=server,
        host=docker.Host(
            socket=settings.socket,
            capacity=capacity,
            images=frozenset({settings.image}),
            owner=f"{process.name}/{process.instance_id}",
            egress_network=settings.egress_network,
            runtime=settings.runtime,
        ),
        capacity=capacity,
        spec=bench(image=settings.image, network=network),
        peer=settings.peer,
        person_id=settings.person_id,
        traits=settings.traits.traits,
        store=memory.Memory(database=str(process.data_dir / "memory.sqlite3")),
        clock=functools.partial(datetime.datetime.now, datetime.UTC),
    )
    bot.run(model, actors=(server,), process=process)

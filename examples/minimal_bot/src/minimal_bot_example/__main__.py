"""The minimal bot file: ``python -m minimal_bot_example`` runs it on an OpenAI-compatible endpoint.

Reads ``OPENAI_BASE_URL``, ``OPENAI_MODEL`` and the ``openai_api_key`` secret (the Docker secret, the file named
by ``OPENAI_API_KEY_FILE``, or ``OPENAI_API_KEY``). The process options (``MOSFET_*`` or flags) are
``bot.process``'s. A missing endpoint variable or secret ends the process with exit ``1`` before it starts.
The bot has nothing to perceive yet, which ``bot.allow("no_devices")`` says is deliberate; memory is kept in
``<data dir>/memory.sqlite3``.
"""

import os

from mosfet import bot
from mosfet.abilities import memory
from mosfet.providers import openai_compat

if __name__ == "__main__":
    process = bot.process()
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL")
    if not base_url or not model:
        raise SystemExit("set OPENAI_BASE_URL and OPENAI_MODEL to the OpenAI-compatible endpoint and its model")
    minimal = bot.define(
        "minimal",
        bot.cognition(
            openai_compat.Processor(
                client=openai_compat.ChatClient(
                    model=model,
                    base_url=base_url,
                    api_key=process.secrets.get("openai_api_key").get_secret_value(),
                ),
                provider="minimal_bot",
            ),
            memory=memory.Memory(database=str(process.data_dir / "memory.sqlite3")),
            associative=True,
            consolidation=True,
        ),
        bot.allow("no_devices"),
    )
    bot.run(minimal, process=process)

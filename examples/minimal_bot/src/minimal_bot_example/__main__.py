"""Run the minimal bot on the OpenAI-compatible endpoint this process is configured for.

Reads ``OPENAI_BASE_URL``, ``OPENAI_MODEL`` and the ``openai_api_key`` secret (the Docker secret, the file named
by ``OPENAI_API_KEY_FILE``, or ``OPENAI_API_KEY``). The process options (``MOSFET_*`` or flags) are
``bot.process``'s. A missing endpoint variable or secret ends the process with exit ``1`` before it starts.
"""

import os

from mosfet import bot
from mosfet.abilities import memory
from mosfet.providers import openai_compat

from minimal_bot_example import minimal

if __name__ == "__main__":
    process = bot.process()
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL")
    if not base_url or not model:
        raise SystemExit("set OPENAI_BASE_URL and OPENAI_MODEL to the OpenAI-compatible endpoint and its model")
    api_key = process.secrets.get("openai_api_key").get_secret_value()
    processor = openai_compat.Processor(
        client=openai_compat.ChatClient(model=model, base_url=base_url, api_key=api_key), provider="minimal_bot"
    )
    store = memory.Memory(database=str(process.data_dir / "memory.sqlite3"))
    bot.run(minimal(processor=processor, store=store), process=process)

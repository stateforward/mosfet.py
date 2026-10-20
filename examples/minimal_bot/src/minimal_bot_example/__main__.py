"""The minimal bot file: ``python -m minimal_bot_example`` runs it on an OpenAI-compatible endpoint.

Reads ``OPENAI_BASE_URL``, ``OPENAI_MODEL`` and the ``openai_api_key`` secret (the Docker secret, the file named
by ``OPENAI_API_KEY_FILE``, or ``OPENAI_API_KEY``). Its intuition decides on a decision model, never a chat
model: the decision server ``DECISION_BASE_URL`` names, which speaks the decisions wire format (anygame's decide,
or a hosted server with the same format). The optional ``decision_api_key`` secret, read the same way, is its key;
without it no key is sent. The optional ``DECISION_MAX_CONTEXT_TOKENS`` is that model's context, which each
decision request is fitted to. The process options (``MOSFET_*`` or flags) are ``bot.process``'s. A missing
endpoint variable or required secret ends the process with exit ``1`` before it starts.
The bot has nothing to perceive yet, which ``bot.allow("no_devices")`` says is deliberate; memory is kept in
``<data dir>/memory.sqlite3``.
"""

import os

from mosfet import bot, runtime
from mosfet.abilities import memory
from mosfet.providers import decision
from mosfet.providers import openai_compat

if __name__ == "__main__":
    process = bot.process()
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL")
    if not base_url or not model:
        raise SystemExit("set OPENAI_BASE_URL and OPENAI_MODEL to the OpenAI-compatible endpoint and its model")
    decider_url = os.environ.get("DECISION_BASE_URL")
    if not decider_url:
        raise SystemExit("set DECISION_BASE_URL to the decision server intuition decides on")
    context = os.environ.get("DECISION_MAX_CONTEXT_TOKENS") or None
    if context is not None and (not context.isdecimal() or int(context) == 0):
        raise SystemExit("set DECISION_MAX_CONTEXT_TOKENS to the decision model's context, a positive whole number")
    try:
        # A local decision server takes no key; one is sent only when the secret is given.
        decider_key = process.secrets.get("decision_api_key").get_secret_value() or None
    except runtime.MissingSecretError:
        decider_key = None
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
            intuition=decision.Processor(
                base_url=decider_url,
                api_key=decider_key,
                max_context_tokens=None if context is None else int(context),
            ),
            memory=memory.Memory(database=str(process.data_dir / "memory.sqlite3")),
            associative=True,
            consolidation=True,
        ),
        bot.allow("no_devices"),
    )
    bot.run(minimal, process=process)

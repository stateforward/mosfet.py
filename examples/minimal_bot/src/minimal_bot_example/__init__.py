"""The smallest bot factory ``mosfet run`` runs: one device-free bot with a full cognition.

    mosfet run minimal_bot_example:build --ready-file /tmp/ready

The bot has no devices, so nothing reaches it until a device or an acquired ability is added; what it
does with what it perceives is its cognition's to decide.
"""

import mosfet
from mosfet import runtime
from mosfet.abilities import cognition, memory
from mosfet.providers import openai_compat


class MinimalBot(mosfet.Bot):
    """A bot with no devices and no abilities beyond the cognition it is given."""

    def __init__(self, *, cognition: cognition.Cognition) -> None:
        super().__init__({}, cognition=cognition)


def build(config: runtime.Config) -> runtime.Assembly:
    """Build one bot whose cognition runs on the OpenAI-compatible endpoint the process is configured for.

    Reads from the process configuration:

    * ``OPENAI_BASE_URL``: the endpoint, for example ``https://api.openai.com/v1`` or a local server;
    * ``OPENAI_MODEL``: the model that endpoint serves;
    * the ``openai_api_key`` secret: the Docker secret ``openai_api_key``, the file named by
      ``OPENAI_API_KEY_FILE``, or ``OPENAI_API_KEY``.

    Raises:
        ValueError: ``OPENAI_BASE_URL`` or ``OPENAI_MODEL`` is unset or empty.
        runtime.MissingSecretError: no source provides ``openai_api_key``.
    """

    base_url = config.environ.get("OPENAI_BASE_URL")
    model = config.environ.get("OPENAI_MODEL")
    if not base_url or not model:
        raise ValueError("set OPENAI_BASE_URL and OPENAI_MODEL to the OpenAI-compatible endpoint and its model")
    processor = openai_compat.Processor(
        client=openai_compat.ChatClient(
            model=model,
            base_url=base_url,
            api_key=config.secrets.get("openai_api_key").get_secret_value(),
        ),
        provider="minimal_bot",
    )
    store = memory.Memory()
    bot = MinimalBot(
        cognition=cognition.Cognition(
            intuition=cognition.Intuition(processor=processor),
            reasoning=cognition.Reasoning(processor=processor),
            reflection=cognition.Reflection(processor=processor, memory=store),
            memory=store,
            associative=memory.Associative(memory=store),
            consolidation=memory.Consolidation(memory=store, processor=processor),
        ),
    )
    return runtime.Assembly(bots={"minimal": bot})


__all__ = ["MinimalBot", "build"]

# minimal_bot

The smallest bot factory `mosfet run` runs: `minimal_bot_example:build` assembles one bot,
`minimal`, with no devices and a full cognition (intuition, reasoning, reflection) on any
OpenAI-compatible endpoint. With no devices nothing reaches the bot, so it activates, stays
ready, and stops cleanly; add a device or an acquired ability to give it something to perceive.

The factory reads:

* `OPENAI_BASE_URL`: the endpoint, for example `https://api.openai.com/v1` or a local server;
* `OPENAI_MODEL`: the model that endpoint serves;
* the `openai_api_key` secret: the Docker secret `/run/secrets/openai_api_key`, the file named by
  `OPENAI_API_KEY_FILE`, or `OPENAI_API_KEY`.

Run it from the repository root:

```bash
uv sync --project examples/minimal_bot
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_MODEL=gpt-5.4-mini OPENAI_API_KEY=... \
  examples/minimal_bot/.venv/bin/mosfet run minimal_bot_example:build --ready-file /tmp/ready
```

The process stays up until SIGTERM or SIGINT and exits `0`. A missing variable or secret exits
`78`. [`docs/runtime.md`](../../docs/runtime.md) has every flag and exit code.

Test it, including the command above run as a child process:

```bash
uv run --project examples/minimal_bot python -m pytest examples/minimal_bot/tests -q
```

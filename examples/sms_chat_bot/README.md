SMS chat-bot example
====================

A minimal SMS chat surface:

* `SMSPhone` handles incoming and outgoing SMS as its own state machine.
* `SMSChatBotBody` turn-stores user messages and sends replies through the same phone surface.
* `ReplyProvider` drives replies through `bot-provider-openai-compat`.

Run it:

```bash
uv run --project examples/sms_chat_bot sms-chat-bot
```

Set `BOT_OPENAI_API_KEY` (or `OPENAI_API_KEY`) plus `BOT_OPENAI_MODEL` and
`BOT_OPENAI_BASE_URL` in either the repo root `.env` or this example's `.env`.
The example reads the repo root file first and the example-local file last.

This is deliberately small enough to hold in your head as a working example of a phone SMS
surface and a chatbot body, not an ambient bot framework proof. It uses a sharp internal shape
for a future phone SMS method rather than a fake full phone stack.

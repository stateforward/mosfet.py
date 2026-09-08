SMS chat-bot example
====================

A minimal SMS chat surface:

* `SMSPhone` handles incoming and outgoing SMS as its own state machine.
* `SMSChatBotBody` uses the real `TextGeneration` ability through
  `ability.run_terminal_operation`.
* `TextGenerationProvider` builds the provider-backed generation ability.

Event flow:

1. An `SMSMessageEvent` moves the body from `idle` to `generating`.
2. The `generating` activity calls the attached `TextGeneration` with a typed input event.
3. Only a real output terminal emits `SMSMessageSentEvent` to the phone.
4. A failure terminal returns the body to `idle`; it sends no SMS and no fallback answer.
5. Body terminals are operation correlated, so a late or duplicate terminal from an old turn
   cannot overwrite a later turn.

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

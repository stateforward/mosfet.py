SMS chat-bot example
====================

The example intentionally has a small SMS chat surface:

* `SMSPhone` owns incoming and outgoing message histories. It is a small SMS state surface,
  not itself an HSM state machine.
* `SMSChatBotBody` turns an `SMSMessageData` into one provider-backed reply through
  `bot.abilities.language.text.TextGenerator`.
* The runner constructs that `TextGenerator` from `bot.providers.openai_compat.ChatClient`
  using `BOT_OPENAI_API_KEY` (or `OPENAI_API_KEY`), `BOT_OPENAI_MODEL`, and
  `BOT_OPENAI_BASE_URL`.
* On successful generation with non-empty output, `SMSChatBotBody` calls `SMSPhone.send`.
* On generation failure or empty output, no SMS is sent: there are no retries and no
  fallback replies.
* `SMSMessageData` is the phone-domain `SmsTextData` from `bot.devices.phone`. The example
  has no `SMSMessageEvent`, `SMSMessageSentEvent`, or idle/generating HSM flow.

Run it:

```bash
uv run --project examples/sms_chat_bot sms-chat-bot
```

A successful reply prints with a `Bot> ` prefix.

Set `BOT_OPENAI_API_KEY` (or `OPENAI_API_KEY`) plus `BOT_OPENAI_MODEL` and
`BOT_OPENAI_BASE_URL` in either the repo root `.env` or this example's `.env`.
The example reads the repo root file first and the example-local file last.

This is deliberately small enough to hold in your head as a working example of a phone SMS
surface and a chatbot body, not an ambient bot framework proof. It uses a sharp internal shape
for a future phone SMS method rather than a fake full phone stack.

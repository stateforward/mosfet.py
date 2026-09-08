SMS chat-bot example
====================

A minimal SMS conversation surface:

* `SMSPhone` handles incoming and outgoing SMS as its own state machine.
* `SMSChatBotBody` turn-stores user messages and sends replies through the same phone surface.
* `DeskNoteReplyProvider` is the local reply policy; any provider can stand in here.

Run it:

```bash
uv run --project examples/sms_chat_bot sms-chat-bot
```

This is deliberately small enough to hold in your head as a working example of a phone SMS
surface and a chatbot body, not an ambient bot framework proof. It uses a sharp internal shape
for a future phone SMS method rather than a fake full phone stack.

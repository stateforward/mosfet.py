# bot-provider-typesafe

TypeSafe AI provider for stateforward.bot: the `system_one` label-tier
(Choice/Score/Noul) wrapped for the bot's Intuition reflex tier.

- `AsyncSystemOneClient` — async `system_one` transport; SDK errors surface as
  `SystemOneError`.
- `Processor` — maps a cognition turn to one Choice over the offered event menu
  (schema descriptions as criteria) plus a reserved pass criterion, and returns a
  `processing.SelectedEvent` with the SDK's own confidence. Payloads stay `None`:
  System One does not author payload fields; the host's dispatch validation is the
  boundary for payload-requiring events (typed rejection → host cascade).

Requires `TYPESAFE_API_KEY` (pass `api_key=`, or let the SDK read the env var).

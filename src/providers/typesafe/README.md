# mosfet-provider-typesafe

TypeSafe AI provider for stateforward.mosfet: the `system_one` label-tier
(Choice/Score/Noul) wrapped for the bot's Intuition reflex tier.

- `AsyncSystemOneClient` — async `system_one` transport to hosted TypeSafe, or to a
  local system_one server at `base_url`; SDK errors surface as `SystemOneError`.
- `Processor` — a decision processor (`processing.DecisionProcessor`), the kind
  intuition runs on. It maps a cognition turn to one batch of questions: a Choice
  over the offered event menu (schema descriptions as criteria) plus a reserved pass criterion
  (the docs' own "none of the above" pattern), and per-payload-key Choices whose
  criteria are value options derived mechanically from the turn's own evidence
  (stimulus data mapping, then schema enum options). The answer label *is* the value,
  so payload data can never leave the menu ("your code never has to recover a value
  from generated prose"). An event with a required key that has no grounded candidates
  is left unoffered — fail-closed at the menu, before any API call. Confidence rides
  the SDK's ChoiceAnswer confidence on the same scale as every other provider.

## Servers and keys

`Processor(base_url=..., api_key=...)` and `AsyncSystemOneClient(base_url=..., api_key=...)`
take the same two arguments:

- No `base_url` is hosted TypeSafe (`https://api.typesafe.ai`, named explicitly, so a
  `TYPESAFE_BASE_URL` in the environment never reroutes it). It needs a key: pass
  `api_key=`, or let the SDK read `TYPESAFE_API_KEY`.
- A `base_url` is a local system_one server that keeps to the same wire format
  (`POST <base_url>/v1/systemone`). With no `api_key` it is sent the placeholder bearer
  `Bearer local`, so the hosted key from the environment never reaches it (an empty bearer would
  be `Bearer ` with a trailing space, an illegal header value that never leaves the client); pass
  `api_key=` when the server wants one.
  Its answer must carry a `model` field, which typesafe-sdk 0.6.0 requires of every
  `/v1/systemone` response.

The address never reaches telemetry: spans and metrics carry `provider="typesafe"` only.

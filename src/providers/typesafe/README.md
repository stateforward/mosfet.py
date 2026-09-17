# mosfet-provider-typesafe

TypeSafe AI provider for stateforward.mosfet: the `system_one` label-tier
(Choice/Score/Noul) wrapped for the bot's Intuition reflex tier.

- `AsyncSystemOneClient` — async `system_one` transport; SDK errors surface as
  `SystemOneError`.
- `Processor` — maps a cognition turn to one batch of questions: a Choice over the
  offered event menu (schema descriptions as criteria) plus a reserved pass criterion
  (the docs' own "none of the above" pattern), and per-payload-key Choices whose
  criteria are value options derived mechanically from the turn's own evidence
  (stimulus data mapping, then schema enum options). The answer label *is* the value,
  so payload data can never leave the menu ("your code never has to recover a value
  from generated prose"). An event with a required key that has no grounded candidates
  is left unoffered — fail-closed at the menu, before any API call. Confidence rides
  the SDK's ChoiceAnswer confidence on the same scale as every other provider.

Requires `TYPESAFE_API_KEY` (pass `api_key=`, or let the SDK read the env var).

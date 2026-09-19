# mosfet-provider-needle

Cactus Compute [Needle 3](https://cactuscompute.com/needle) for stateforward.mosfet: a free,
local, Apache-2.0 tool-calling model wrapped for the bot's Intuition reflex tier.

- `Processor` — offers each selectable event as one Needle tool (the event's payload schema is
  the tool's parameters), completes the turn locally, and maps calls back to `SelectedEvent`s
  with engine-authored payloads. Needle's grammar constrains arguments to each schema; calls
  the engine flags as ungrounded are withheld; no calls (refusal, or everything moved to
  `suppressed_calls`) is Intuition's explicit unhandled envelope, which cascades to reasoning.
  The engine's calibrated 0-1 confidence is normalized to 0-100 on every returned call.
- `LocalEngine` — the in-process `cactus-needle` SDK session; `Engine` is the injectable
  protocol (tests stub it).

Unlike `mosfet-provider-typesafe` (label tier, never authors payloads), Needle authors
payloads the way `mosfet-provider-openai-compat` does, but offline.

## Model weights

Nothing is vendored. On first use the SDK downloads the native engine library (<1 MB) and
`needle3.cact` (8-29 MB) from the `Cactus-Compute` Hugging Face organization into
`~/.cache/cactus-needle`. Pass `LocalEngine(weights=...)` for a fine-tuned `.cact` (tuned
weights report no confidence, so Intuition's gate escalates them).

The SDK and native engine send anonymous usage counts by default; `LocalEngine` sets
`NEEDLE_TELEMETRY=0` and `DO_NOT_TRACK=1` unless the process already set them.

## Tests

Offline tests stub the engine. `tests/test_needle_live.py` (marker `live`) runs one real turn
and is skipped unless `NEEDLE_LIVE=1` and `cactus-needle` is installed.

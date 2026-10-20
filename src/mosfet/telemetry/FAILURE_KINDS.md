# Failure-kind convention

How a failure is named once and reused everywhere it is reported: on the typed
failure event, on the span, and as a metric dimension. Prior art this codifies:
`_stable_failure_kind` in `src/providers/livekit/src/bot/providers/livekit/room_audio.py`.
Rule exceptions for this package live in [`EXCEPTIONS.md`](EXCEPTIONS.md).

## The kind is the typed failure event's field

A failure kind is not a telemetry-only concept. It is the `failure_kind` field
already carried by typed failure event data (`RoomAudioFailureData` and peers).
Telemetry **reports** that kind; it never invents a parallel one.

- Producing code derives the kind once, at the point it converts an exception
  into a typed failure event.
- `bot.telemetry.span.failure_kind(error, fallback)` is that derivation: it
  prefers a `failure_kind` string carried by the error, else the caller's
  fallback, and normalizes the result.
- The same string goes on the event data and on the span. If a span says
  `connect_failed` and the event says something else, one of them is lying.

## Shape

`lower_snake_case`, ASCII, at most 64 characters, from a **closed vocabulary the
code chooses** — never derived from an exception message, a URL, an ID, or any
runtime value. `bot.telemetry.span.normalized_kind` enforces the shape;
it cannot enforce the closed vocabulary, so call sites must.

Prefer `<subject>_<verb-ed>` or `<verb>_failed` naming that says which step
failed, not which library raised:

| Kind | Means |
| --- | --- |
| `connect_failed` | establishing the transport/session did not succeed |
| `disconnect_failed` | teardown did not complete cleanly |
| `publish_failed` | producing/emitting outward failed |
| `subscribe_failed` | attaching to an inbound stream failed |
| `decode_failed` | transforming received bytes into a product failed |
| `encode_failed` | transforming a product into bytes failed |
| `embed_failed` | an embedder failed for a reason no narrower kind names |
| `timeout` | a modeled deadline elapsed before completion |
| `cancelled` | the operation was cancelled, not broken |
| `unauthorized` | credentials/permissions rejected |
| `unavailable` | dependency reachable-but-not-serving |
| `invalid_response` | a peer answered in a shape the contract does not allow |
| `invalid_input` | the request itself is malformed (for example an undecodable image) |
| `unsupported_input` | the request is well formed but outside the provider's declared capacity |
| `too_long` | an input exceeds the model's token bound and was not truncated; split it |
| `overloaded` | a bounded admission queue was full; retry later or shed load |
| `store_failed` | a storage transaction failed and rolled back; nothing was changed |
| `stale_lease` | a decision or release named a leased unit that is no longer leased |
| `measure_failed` | a change measure could not measure a view; nothing was handed on; check the change measure's provider |
| `stage_failed` | a perception stage (objects, faces, description) answered with a failure; the view's field for it is empty; the span's `bot.seeing.stage` says which stage |
| `design_failed` | a voice source could not make a voice from its description for a reason no narrower kind names; nothing was kept |
| `unsupported` | a sandbox request is well formed but beyond what the host offers (its capacity, its image allowlist, or files at all); nothing was run |
| `not_found` | a named thing is missing: a path to read in a sandbox machine is missing or is not a regular file, an identity id to mark is not in its registry, or a person or ref a person statement (memory's, or a `memory.know` / `memory.forget` fact) names is not in memory |
| `conflict` | the request contradicts what already holds: for example a link mark on an id that is the bot's own, a ref linked to a person while it belongs to another, or a statement that would change the bot itself or its own refs (memory's, or a `memory.know` / `memory.forget` fact); nothing was changed |
| `too_large` | a file to read from a sandbox machine is larger than the host will move |
| `busy` | the sandbox's request queue was full; the request was not taken |
| `failed` | a sandbox host failed for a reason no narrower kind names |
| `unknown` | derivation produced nothing usable — a bug in the call site |

Add a kind only when an operator would act differently on it. Two kinds that
lead to the same response should be one kind.

## Where it is recorded

Per `METRICS > SPANS > Logs`:

1. **Metrics** — failure counters carry the kind as a dimension
   (`bot.failure.kind`). This is the load-bearing signal: it answers "is this
   happening, and how often" without a trace search.
2. **Spans** — `bot.telemetry.span.record_failure(span, kind)` sets
   `bot.outcome=failed`, `bot.failure.kind=<kind>`, and an `ERROR` status. The
   `operation()` context manager does this automatically for an escaping
   exception.
3. **Logs** — only when the human-readable reason (the failure event's
   `message`) adds something the kind cannot carry. The message may be
   high-cardinality; the kind never is.

## What must never become a kind

Exception messages, stack frames, provider error codes with embedded IDs,
hostnames, URLs, participant/track/room identifiers, file paths, model output.
These are unbounded-cardinality and belong in a log line at most.

## Cancellation is an outcome, not a failure

`span.operation()` records an `asyncio.CancelledError` as `bot.outcome=cancelled` with no
`ERROR` status and no `bot.failure.kind`. An HSM behavior that dispatches an event to its own
machine may be cancelled by the transition that event takes. Every such dispatch goes through
`telemetry.deliver` (the stamping path of `telemetry.Traced`), which marks the stages open in
that behavior (`span.expect_handoff`); if one of them is then cancelled while the machine is
still running, the span reads `bot.outcome=handed_off` with `bot.handoff.event=<event name>`.
A cancellation while the machine is stopping, or of a stage that never handed off, stays
`cancelled` — so `cancelled` means real cancellation only. Only real exceptions produce `failed`.

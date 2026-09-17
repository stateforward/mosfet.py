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
| `timeout` | a modeled deadline elapsed before completion |
| `cancelled` | the operation was cancelled, not broken |
| `unauthorized` | credentials/permissions rejected |
| `unavailable` | dependency reachable-but-not-serving |
| `invalid_response` | a peer answered in a shape the contract does not allow |
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

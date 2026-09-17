# Telemetry rule exceptions

Documented per [CORE-EXC-001](../../../rules/core.rules.md#core-exc-001-must-document-rule-exceptions).

## Unredacted generator payloads in local JSONL (PY-LOG-002)

| Field | Value |
| --- | --- |
| Violated rule | [`rules/python.rules.md#PY-LOG-002`](../../../rules/python.rules.md#py-log-002-should-use-stable-structured-logs) — logs MUST NOT include secrets or unredacted sensitive payloads |
| Owner | stateforward.mosfet maintainers |
| Rationale | The OTEL goal allows seeing exact text-generator messages (system/user content, tools) during local development when the runtime opts in. The JSONL file exporter may write full request bodies only under that explicit opt-in; the default omits prompt/tool content. This is not a production collector path. |
| Risk tests | Payload capture opt-in (`BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD`); enablement opt-out (`BOT_OTEL_DISABLED` / `enabled=False`); cwd confinement on configure; file mode `0o600`; symlink / `O_NOFOLLOW` open rejection (`tests/telemetry/`) |
| Expiration | When local file export is removed, or replaced by a redacting production exporter |
| Removal plan | Delete the `JsonlFileLogRecordExporter` path (and default-on configure wiring), flip the default to disabled, and remove this exception entry |

The span file (`otel-spans.jsonl`) shares the same confined, `O_NOFOLLOW`, `0o600`
writer but is **not** covered by this exception: span attributes are
low-cardinality and payload-free by contract (see
[`FAILURE_KINDS.md`](FAILURE_KINDS.md) and `bot.telemetry.span`), so no prompt,
transcript, audio, or credential is written there.

## Process-global `_RUNTIME` (PY-OBJ-002)

| Field | Value |
| --- | --- |
| Violated rule | [`rules/python.rules.md#PY-OBJ-002`](../../../rules/python.rules.md#py-obj-002-must-avoid-mutable-global-business-state) — global mutable state MUST NOT be used for runtime configuration |
| Owner | stateforward.mosfet maintainers |
| Rationale | OpenTelemetry installs a process-global `LoggerProvider` via `set_logger_provider`. This package therefore keeps a process-global `_RUNTIME` in `bot.telemetry.configure` as the single source of truth for whether *this* module's JSONL export is enabled, matching that process-global install model. |
| Risk tests | `_RUNTIME_LOCK` around configure / ensure_configured / reset; `reset()` clears state and shuts down the prior provider; enablement SoT is `_RUNTIME.provider` only (readers never fall back to `_logs.get_logger_provider()`) — covered in `tests/telemetry/` |
| Expiration | When JSONL configure stops retaining a process-global provider (or OTEL stops requiring a process-global `LoggerProvider`) |
| Removal plan | Drop `_RUNTIME` / `_RUNTIME_LOCK`; have callers pass or inject an exporting `LoggerProvider`; delete this exception entry |

Single writer: `configure()` / `ensure_configured()` / `reset()` under `_RUNTIME_LOCK`.
`set_logger_provider` remains an install side-effect only so other OTEL code in the
process can observe the same provider; enablement decisions never re-read the
global API slot.

## Dependency floor: matching OpenTelemetry API

**User-approved:** `opentelemetry-sdk>=1.42.1` plus a matching
`opentelemetry-api>=1.42.1` floor (approved together with the sdk addition in
`pyproject.toml`).

## Dependency floor: OTLP gRPC span exporter

**User-approved:** `opentelemetry-exporter-otlp-proto-grpc>=1.42.1,<2.0.0`
(same 1.42 floor as api/sdk). Transitive `grpcio` is the Control subscribe
transport. Dashboard live ingest is OTLP gRPC on `:4317`; JSONL file export
remains the local recording artifact.

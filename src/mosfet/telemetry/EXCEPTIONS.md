# Telemetry rule exceptions

Documented per [CORE-EXC-001](../../../rules/core.rules.md#core-exc-001-must-document-rule-exceptions).

## Unredacted generator payloads in local JSONL (PY-LOG-002)

| Field | Value |
| --- | --- |
| Violated rule | [`rules/python.rules.md#PY-LOG-002`](../../../rules/python.rules.md#py-log-002-should-use-stable-structured-logs) — logs MUST NOT include secrets or unredacted sensitive payloads |
| Owner | stateforward.mosfet maintainers |
| Rationale | The OTEL goal allows seeing exact text-generator messages (system/user content, tools) and responses (tool calls with arguments, text, reasoning summary, status) during local development when the runtime opts in. `BOT_OTEL_CAPTURE=full` additionally logs every observed HSM event with its serialized data, ids, and consuming transition, plus dispatch outcomes, so one turn can be reconstructed end to end (`scripts/otel_turns.py`). The JSONL file exporter writes these bodies only under that explicit opt-in; the default omits them. Credentials are masked at every depth (`mosfet.telemetry.capture.jsonable`: credential-named keys and bearer/API-key-shaped values). Embedding vectors (`mosfet.abilities.embedding.OutputData`) are exported under full capture like any other event data: they are derived only from text and images the same capture already holds, while image bytes themselves are always replaced by `<image:<media_type>:<N> bytes>`. Biometric values are never exported, even under full capture: any `mosfet.telemetry.capture.Biometric` instance (a voiceprint or face print), at any depth, is written as `<biometric:<ClassName>>`. This is not a production collector path. |
| Risk tests | Payload capture opt-in (`BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD`, `BOT_OTEL_CAPTURE=full`); credential redaction, biometric redaction at every depth, and default-off HSM capture (`tests/telemetry/test_capture.py`); enablement opt-out (`BOT_OTEL_DISABLED` / `enabled=False`); cwd confinement on configure; file mode `0o600`; symlink / `O_NOFOLLOW` open rejection (`tests/telemetry/`) |
| Expiration | When local file export is removed, or replaced by a redacting production exporter |
| Removal plan | Delete the `JsonlFileLogRecordExporter` path (and default-on configure wiring), flip the default to disabled, and remove this exception entry |

The span file (`otel-spans.jsonl`) and the metric file (`otel-metrics.jsonl`) share
the same confined, `O_NOFOLLOW`, `0o600` writer but are **not** covered by this
exception: span and metric attributes are low-cardinality and payload-free by
contract, including under `BOT_OTEL_CAPTURE=full` (see
[`FAILURE_KINDS.md`](FAILURE_KINDS.md) and `bot.telemetry.span`), so no prompt,
transcript, audio, or credential is written there.

## Process-global `_RUNTIME` (PY-OBJ-002)

| Field | Value |
| --- | --- |
| Violated rule | [`rules/python.rules.md#PY-OBJ-002`](../../../rules/python.rules.md#py-obj-002-must-avoid-mutable-global-business-state) — global mutable state MUST NOT be used for runtime configuration |
| Owner | stateforward.mosfet maintainers |
| Rationale | OpenTelemetry installs process-global providers via `set_logger_provider`, `set_tracer_provider` and `set_meter_provider`; the meter provider can be set only once per process, and import-time meters (`metrics.get_meter(...)`) bind to whichever one is set. This package therefore keeps a process-global `_RUNTIME` in `mosfet.telemetry.configure` as the single source of truth for whether *this* module's export is enabled, and it retains the once-installed `MeterProvider` with its routing exporter so every later configuration can bind its own metric exporters. |
| Risk tests | `_RUNTIME_LOCK` around configure / ensure_configured / reset; `reset()` clears state, shuts down the prior logger and tracer providers, flushes metrics through the bound exporters, then unbinds and shuts those exporters down, and never shuts down the `MeterProvider` (`meter_provider()` is the same object across cycles); a module-level meter exports into each configuration's metric file across `configure()` / `reset()` cycles; enablement SoT is `_RUNTIME.provider` only (readers never fall back to `_logs.get_logger_provider()`) — covered in `tests/telemetry/` |
| Expiration | When configure stops retaining process-global providers (or OTEL stops requiring a process-global `LoggerProvider` and a set-once `MeterProvider`) |
| Removal plan | Drop `_RUNTIME` / `_RUNTIME_LOCK`; have callers pass or inject exporting providers; delete this exception entry |

Single writer: `configure()` / `ensure_configured()` / `reset()` under `_RUNTIME_LOCK`.
`set_logger_provider` / `set_tracer_provider` remain install side-effects only so other
OTEL code in the process can observe the same providers; enablement decisions never
re-read the global API slot. `set_meter_provider` runs once, on the first enabled
`configure()`.

## Dependency floor: matching OpenTelemetry API

**User-approved:** `opentelemetry-sdk>=1.42.1` plus a matching
`opentelemetry-api>=1.42.1` floor (approved together with the sdk addition in
`pyproject.toml`).

## Dependency floor: OTLP gRPC span and metric exporters

**User-approved:** `opentelemetry-exporter-otlp-proto-grpc>=1.42.1,<2.0.0`
(same 1.42 floor as api/sdk). Dashboard live ingest is OTLP gRPC on `:4317`;
the studio observes only and sends nothing back to the bot. The same package's
`OTLPMetricExporter` sends metrics to the configured endpoint, so metrics add no
dependency. JSONL file export remains the local recording artifact alongside OTLP.

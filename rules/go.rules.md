# GO-BUILD-001 MUST Pin Go Toolchain

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

Go modules MUST pin the Go language version and toolchain version.

Toolchain upgrades MUST be intentional and tested.

# GO-CGO-001 SHOULD Avoid Cgo

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Go code SHOULD build with `CGO_ENABLED=0`.

CGO, C toolchains, and `import "C"` MAY be used only behind documented platform boundaries.

# GO-CONC-001 MUST Own Goroutines

See:
- [CORE-CONC-001](core.rules.md#core-conc-001-must-thread-safety)
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)

Every goroutine MUST have an owner, stop signal, and observation or join path.

Fire-and-forget goroutines are forbidden.

# GO-CTX-001 MUST Use Context At Blocking Boundaries

See:
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Blocking, request-scoped, and external-system operations MUST accept `context.Context` as the first parameter.

Contexts MUST NOT be stored in structs for dependency injection.

# GO-CHAN-001 MUST Bound Channel Operations

See:
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Channels used for runtime work MUST be bounded or otherwise backpressured.

Blocking sends and receives MUST respect cancellation or deadlines.

Only the sending owner MAY close a channel.

# GO-ERR-001 MUST Return Explicit Errors

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)

Expected failures MUST be returned as `error` values or typed results.

Errors MUST be wrapped with `%w` when adding context.

The same error MUST NOT be both logged and returned at the same boundary.

# GO-API-001 SHOULD Keep Interfaces Small

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Interfaces SHOULD be small and consumer-defined.

Package names SHOULD be short, lowercase, and non-stuttering.

# GO-MEM-001 SHOULD Bound Hot Path Allocation

See:
- [CORE-PERF-001](core.rules.md#core-perf-001-must-measure-performance-claims)

Hot paths SHOULD pre-size slices and maps when bounds are known.

Hot paths SHOULD avoid avoidable `fmt`, `any`, reflection, and repeated `[]byte` to `string` conversions.

Escape-sensitive changes SHOULD be verified with compiler escape diagnostics.

# GO-TIME-001 SHOULD Reuse Timers Carefully

See:
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Runtime loops SHOULD use reusable timers instead of repeatedly allocating with `time.After`.

Timer reset and drain behavior MUST be correct and tested when reused.

# GO-TEST-001 MUST Use Deterministic Go Tests

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Go tests MUST use the standard `testing` package.

Concurrent tests MUST use deterministic synchronization instead of sleeps.

Parsers and protocol boundaries SHOULD include fuzz tests.

# GO-OBS-001 SHOULD Use Structured Logs

See:
- [CORE-OBS-001](core.rules.md#core-obs-001-should-bounded-observability)

Production Go services SHOULD use structured logging.

Logs SHOULD include stable component, operation, and error fields.

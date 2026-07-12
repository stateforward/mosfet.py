# DART-TYPE-001 MUST Enforce Sound Null Safety

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Dart code MUST use sound null safety.

Absent values MUST be represented with nullable types instead of sentinel values.

# DART-ANALYZE-001 MUST Enable Strict Analysis

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

Dart projects MUST enable strict casts, strict inference, and strict raw types.

CI MUST fail on formatting errors, analyzer warnings, and test failures.

# DART-DYN-001 MUST NOT Leak Dynamic Into Core Logic

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Public APIs MUST NOT expose `dynamic` unless the API is an explicit interop boundary.

Dynamic input MUST be narrowed or converted at the boundary.

# DART-CAST-001 MUST Avoid Unchecked Casts

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Unchecked `as T` casts are forbidden in core logic.

Use explicit type checks, pattern matching, or boundary validators.

# DART-API-001 MUST Protect Package Boundaries

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

`lib/src/` MUST remain private to its package.

Public APIs MUST be exported from stable library entrypoints.

# DART-ASYNC-001 MUST Handle Futures Explicitly

See:
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Every `Future` MUST be awaited, returned, or explicitly marked as intentionally unawaited.

Functions MUST NOT be marked `async` when they do not `await`.

# DART-RESOURCE-001 MUST Dispose Async Resources

See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)

Streams, subscriptions, sinks, controllers, timers, and isolates MUST have deterministic cancellation or disposal.

Finalizers MUST NOT be the primary cleanup mechanism.

# DART-ERR-001 MUST Preserve Error Context

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)

Catch blocks MUST capture stack traces when handling errors.

Code that rethrows MUST preserve the original stack trace.

Empty catch blocks are forbidden.

# DART-DATA-001 MUST Validate External Data

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

JSON, platform messages, FFI data, JS interop data, and network payloads MUST be validated at the boundary.

Core business logic MUST NOT receive unvalidated `Map<String, dynamic>` payloads.

# DART-PERF-001 SHOULD Avoid Repeated Hot Path Allocation

See:
- [CORE-PERF-001](core.rules.md#core-perf-001-must-measure-performance-claims)

Hot paths SHOULD use `StringBuffer` for repeated string construction and cache expensive reusable objects.

Collection allocation SHOULD use known sizes when available.

# DART-TEST-001 MUST Use Deterministic Dart Tests

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Dart tests MUST avoid real time, live I/O, and ambient external state unless explicitly classified as integration tests.

Bug fixes MUST include targeted regression coverage.

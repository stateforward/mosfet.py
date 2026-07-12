# FLUTTER-BASE-001 MUST Apply Dart Rules

See:
- [DART-TYPE-001](dart.rules.md#dart-type-001-must-enforce-sound-null-safety)
- [DART-ASYNC-001](dart.rules.md#dart-async-001-must-handle-futures-explicitly)

Flutter code MUST comply with the Dart rules unless a Flutter framework boundary explicitly requires otherwise.

# FLUTTER-UI-001 MUST Keep Build Methods Pure

See:
- [CORE-DET-001](core.rules.md#core-det-001-must-deterministic-behavior)

`build()` methods MUST be pure, fast render functions.

Network calls, database or file I/O, platform channel calls, heavy parsing, and synchronous blocking work are forbidden inside `build()`.

# FLUTTER-STATE-001 MUST Keep State Ownership Explicit

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Flutter state MUST live as close as practical to its owner.

Global state MUST be used only for values that are genuinely shared across independent UI areas.

# FLUTTER-ASYNC-001 MUST Model Async UI States

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)

Async UI flows MUST model loading, success, empty, and failure states explicitly.

Errors MUST NOT be swallowed by UI callbacks or builders.

# FLUTTER-LIST-001 MUST Virtualize Large Lists

See:
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Dynamic or large child lists MUST use lazy builders or slivers.

Eager widget lists are forbidden for unbounded data.

# FLUTTER-PLATFORM-001 MUST Isolate Platform Differences

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Platform-specific capabilities MUST be isolated behind adapters.

Feature UI MUST NOT scatter `kIsWeb`, `Platform.isX`, or host-specific checks.

# FLUTTER-NAV-001 SHOULD Make Routes Addressable

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

User-facing screens SHOULD have stable, addressable routes.

Route parameters SHOULD be typed or validated before use.

# FLUTTER-ACCESS-001 MUST Preserve Accessibility

Interactive controls MUST be keyboard reachable where the platform supports keyboards.

Controls MUST expose semantic labels and logical focus order.

Gesture-only controls MUST NOT replace platform button semantics.

# FLUTTER-SEC-001 MUST Protect Client Secrets

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Flutter clients MUST NOT embed secrets or long-lived private credentials.

Sensitive tokens MUST NOT be stored in plain preferences or logged.

# FLUTTER-TEST-001 MUST Test At The Right Boundary

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Pure logic SHOULD be unit tested without Flutter.

Widget tests SHOULD cover UI behavior.

Integration tests SHOULD cover critical user journeys.

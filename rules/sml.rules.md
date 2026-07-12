# SML-BASE-001 MUST Apply Cpp And State Machine Pattern Rules

See:
- [CPP-MEM-001](cpp.rules.md#cpp-mem-001-must-use-raii-ownership)
- [PAT-RTC-001](patterns.rules.md#pat-rtc-001-must-run-to-completion-dispatch)

C++ SML code MUST comply with C++ rules and run-to-completion state machine pattern rules.

# SML-RTC-001 MUST Dispatch Synchronously To Quiescence

See:
- [PAT-RTC-001](patterns.rules.md#pat-rtc-001-must-run-to-completion-dispatch)

A top-level SML dispatch MUST execute all selected internal and anonymous transitions to quiescence before returning.

Queueing and post-for-later mechanisms MUST NOT replace run-to-completion dispatch semantics.

# SML-CONC-001 MUST Serialize Machine Dispatch

See:
- [CORE-CONC-001](core.rules.md#core-conc-001-must-thread-safety)

Only one thread MAY execute inside a machine instance's dispatch at a time.

Cross-actor synchronous dispatch MAY be used only when the call graph is acyclic and orchestrator-ordered.

# SML-MEM-001 MUST Bound Dispatch Allocation

See:
- [CPP-MEM-001](cpp.rules.md#cpp-mem-001-must-use-raii-ownership)

Dispatch paths classified as real-time safe MUST NOT allocate from the heap.

Events SHOULD be small, immutable, and trivially copyable.

# SML-CONTEXT-001 MUST Store Persistent Data In Context

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

SML states MUST be treated as control-flow labels.

Persistent machine-owned data MUST live in explicit context objects with stable lifetimes.

# SML-EVENT-001 MUST Keep Event Payload Lifetime Valid

See:
- [PAT-EVENT-001](patterns.rules.md#pat-event-001-must-typed-event-boundaries)

Event payloads MUST remain valid for the complete top-level dispatch, including nested synchronous dispatch.

Runtime event IDs MUST be validated before table indexing.

# SML-GUARD-001 MUST Keep Guards Pure

See:
- [PAT-GUARD-001](patterns.rules.md#pat-guard-001-must-pure-guards)

SML guards MUST be pure predicates returning `bool`.

Context mutation MUST be restricted to actions.

# SML-ACTION-001 MUST NOT Hide Behavioral Routing In Actions

See:
- [PAT-HSM-001](patterns.rules.md#pat-hsm-001-must-explicit-hierarchical-state-modeling)

Actions and helper functions MUST NOT contain hidden runtime routing logic that decides behavioral progression.

Behavioral branching MUST be represented in transition tables.

# SML-ERROR-001 MUST Handle Unexpected Events Explicitly

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)

Unhandled external events SHOULD be caught explicitly.

Wildcard guards MUST NOT consume internal lifecycle events unintentionally.

# SML-TIME-001 MUST Inject Time Through Events

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Time MUST be injected through event payloads or boundary adapters.

Guards and hot actions MUST NOT read clocks directly.

# SML-OBS-001 MUST Observe Through Explicit APIs

See:
- [PAT-SNAPSHOT-001](patterns.rules.md#pat-snapshot-001-must-snapshot-observation)

External observation MUST use current-state visitors, snapshots, or explicit query APIs.

Observation MUST NOT require steady-state locks unless synchronization is documented.

# SML-TEST-001 MUST Test Deterministic Dispatch

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Tests MUST assert deterministic dispatch, bounded anonymous transitions, and expected current states.

Real-time-safe machines SHOULD test for zero allocation during dispatch.

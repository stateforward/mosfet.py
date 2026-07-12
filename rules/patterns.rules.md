# PAT-ACTOR-001 MUST Actor State Ownership

See:
- [CORE-CONC-001](core.rules.md#core-conc-001-must-thread-safety)
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)

Actors MUST exclusively own their mutable state.

Actors MUST NOT directly mutate another actor's state.

Cross-actor communication MUST occur through explicit messages, events, or snapshots.

# PAT-RTC-001 MUST Run To Completion Dispatch

See:
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)
- [PAT-ACTOR-001](#pat-actor-001-must-actor-state-ownership)

A run-to-completion dispatch MUST process one event to quiescence before the next event mutates the same actor or machine.

Internal transitions and completion events MUST be bounded.

# PAT-HSM-001 MUST Explicit Hierarchical State Modeling

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)
- [PAT-RTC-001](#pat-rtc-001-must-run-to-completion-dispatch)

Hierarchical state machines MUST represent behavioral modes as states and transitions.

Behavioral branching MUST be modeled with transitions, guards, choices, or explicit events rather than hidden routing inside actions.

# PAT-GUARD-001 MUST Pure Guards

See:
- [CORE-DET-001](core.rules.md#core-det-001-must-deterministic-behavior)

Guards MUST be pure predicates.

Guards MUST NOT perform I/O, mutate state, allocate unexpectedly, or trigger side effects.

# PAT-EVENT-001 MUST Typed Event Boundaries

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Events crossing actor, machine, process, or transport boundaries MUST be typed or validated before dispatch.

Event payload ownership and lifetime MUST be explicit.

# PAT-ASYNC-001 MUST Async Work Return Events

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)
- [PAT-RTC-001](#pat-rtc-001-must-run-to-completion-dispatch)
- [HSM-CONTEXT-001](hsm.rules.md#hsm-context-001-must-not-treat-context-is_done-as-machine-liveness)

Asynchronous work started by a state machine or actor MUST report completion, cancellation, and failure through explicit events or messages.

Async work MUST have an owner and cancellation path.

Host-language cancel tokens (including `hsm.Context.is_done()`) MUST NOT be confused with machine liveness when deciding whether deferred ingress or callbacks may still deliver events to a started machine.

# PAT-SNAPSHOT-001 MUST Snapshot Observation

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

External observers MUST observe actors and machines through snapshots, subscriptions, or query APIs.

External code MUST NOT inspect internal state to drive hidden workflow progression.

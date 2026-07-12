# XSTATE-BASE-001 MUST Apply TypeScript And State Machine Pattern Rules

See:
- [TS-STRICT-001](typescript.rules.md#ts-strict-001-must-enable-strict-type-checking)
- [PAT-HSM-001](patterns.rules.md#pat-hsm-001-must-explicit-hierarchical-state-modeling)

XState v5 code MUST comply with TypeScript rules and state machine pattern rules.

# XSTATE-STATE-001 MUST Model Modes As States

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Finite workflow modes MUST be represented as states.

Context MUST NOT store boolean flags or status values that duplicate the machine state value.

# XSTATE-CONTEXT-001 MUST Keep Context Minimal

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

XState context MUST store durable data needed for future transitions.

Derived UI state and large domain objects SHOULD live outside context unless the machine owns them.

# XSTATE-ASSIGN-001 MUST Mutate Context Through Assign

See:
- [CORE-CONC-001](core.rules.md#core-conc-001-must-thread-safety)

Machine context MUST NOT be mutated directly.

Context changes MUST use XState transition mechanisms such as `assign`.

# XSTATE-SETUP-001 SHOULD Use Setup For Typed Implementations

See:
- [TS-STRICT-001](typescript.rules.md#ts-strict-001-must-enable-strict-type-checking)

New XState v5 machines SHOULD use `setup(...)` for typed actions, guards, actors, and delays when it improves type safety or reuse.

Local edits MAY preserve existing valid structure.

# XSTATE-EVENT-001 SHOULD Use Domain Event Names

See:
- [PAT-EVENT-001](patterns.rules.md#pat-event-001-must-typed-event-boundaries)

Events SHOULD describe domain occurrences.

Dot-separated event names MAY be used for grouping.

# XSTATE-GUARD-001 MUST Keep Guards Pure

See:
- [PAT-GUARD-001](patterns.rules.md#pat-guard-001-must-pure-guards)

XState guards MUST be pure.

Side effects MUST live in actions, actors, or boundary adapters.

# XSTATE-ACTOR-001 MUST Choose Actor Lifecycle Intentionally

See:
- [PAT-ASYNC-001](patterns.rules.md#pat-async-001-must-async-work-return-events)

Invoked actors SHOULD be used for work tied to a specific state.

Spawned actors SHOULD be used for dynamic actors requiring manual lifecycle control.

# XSTATE-COMM-001 SHOULD Keep Actor Communication Explicit

See:
- [PAT-ACTOR-001](patterns.rules.md#pat-actor-001-must-actor-state-ownership)

Reusable actors SHOULD receive required actor references through input or emit events to their owner.

`sendParent()` SHOULD be avoided when it hides ownership.

# XSTATE-WILDCARD-001 SHOULD Keep Wildcards Narrow

Wildcard transitions SHOULD be narrow and lowest priority.

Wildcard transitions MUST NOT mask more specific events or internal lifecycle events.

# XSTATE-PERSIST-001 MUST Persist Required Snapshots

See:
- [PAT-SNAPSHOT-001](patterns.rules.md#pat-snapshot-001-must-snapshot-observation)

When persistence must preserve state value, child actors, or history, code MUST persist machine snapshots rather than context alone.

# XSTATE-TEST-001 MUST Test Transitions And Actors

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Tests MUST cover important transitions, guards, actor success paths, and actor failure paths.

UI tests SHOULD assert behavior from snapshots instead of duplicating machine internals.

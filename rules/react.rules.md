# REACT-BASE-001 MUST Apply TypeScript Rules

See:
- [TS-STRICT-001](typescript.rules.md#ts-strict-001-must-enable-strict-type-checking)
- [TS-DATA-001](typescript.rules.md#ts-data-001-must-validate-unknown-data)

React code written in TypeScript MUST comply with the TypeScript rules.

# REACT-PURE-001 MUST Keep Render Pure

See:
- [CORE-DET-001](core.rules.md#core-det-001-must-deterministic-behavior)

Components and hooks MUST be pure and idempotent during render.

Render code MUST NOT mutate props, state, captured objects, or external systems.

# REACT-STATE-001 MUST Keep Source State Minimal

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

React components MUST store only source state.

Values derived from props, state, or cached data SHOULD be computed during render or through selectors.

# REACT-STATE-002 MUST Represent Exclusive Modes Explicitly

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Mutually exclusive UI modes MUST be represented with a single status value, reducer state, discriminated union, or state machine.

Contradictory boolean matrices are forbidden.

# REACT-EFFECT-001 MUST Use Effects Only For External Synchronization

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

`useEffect` MUST synchronize React with external systems such as DOM APIs, timers, subscriptions, storage, or network clients.

Effects MUST NOT be used to derive render state that can be computed directly.

# REACT-EFFECT-002 MUST Respect Effect Dependencies

See:
- [CORE-DET-001](core.rules.md#core-det-001-must-deterministic-behavior)

Effect dependency rules MUST NOT be disabled to hide stale closure bugs.

Code SHOULD be restructured when dependency rules expose design problems.

# REACT-REF-001 MUST Not Store Render State In Refs

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Refs MAY hold imperative handles, DOM nodes, timers, subscriptions, and other non-rendering mutable values.

Refs MUST NOT hold state that drives rendering.

# REACT-COMP-001 SHOULD Prefer Explicit Composition

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Components SHOULD use explicit props, children, slots, or composition over inheritance and opaque configuration objects.

Boolean style matrices SHOULD be replaced with constrained variants.

# REACT-BOUND-001 MUST Separate Client And Server Boundaries

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Server-only and client-only code MUST remain separated unless the framework explicitly supports a shared module.

Browser-only APIs MUST NOT be imported by server-only code.

# REACT-TEST-001 SHOULD Test Behavior Through The UI

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

React tests SHOULD assert user-visible behavior and state transitions rather than implementation details.

External systems SHOULD be mocked at module or boundary adapters.

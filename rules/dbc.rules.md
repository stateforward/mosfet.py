# DBC-SCOPE-001 MUST Define Contracts At Reliance Boundaries

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

A contract MUST define behavior that another module, package, service, job, script, plugin, or operator may rely on.

Implementation details MUST NOT be promoted to public contracts unless consumers are allowed to depend on them.

# DBC-SCOPE-002 MUST Keep Contracts Verifiable

A contract MUST be verifiable by code review, static checking, runtime checking, contract testing, model checking, property testing, or documented manual review.

Aspirational guidance MUST NOT be labeled as a contract.

# DBC-OWN-001 MUST Assign Contract Ownership

See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)

Each public contract MUST have an owner with authority to approve breaking changes, deprecations, waivers, and compatibility exceptions.

Contract ownership MUST be visible to maintainers and consumers.

# DBC-SOURCE-001 MUST Maintain One Canonical Contract Source

A public boundary MUST have one canonical contract source such as a language interface, schema, IDL, executable contract, formal specification, API documentation, or generated artifact.

Generated code, human documentation, examples, and tests MUST be derived from that source or checked for drift.

# DBC-SOURCE-002 MUST Keep Contracts Near Governed Code

Contract text MUST be stored with the code, schema, interface definition, or service definition it governs.

Public contracts MUST NOT exist only in tickets, chats, wikis, slide decks, or tribal knowledge.

# DBC-PRE-001 MUST Define Caller Preconditions

Public operations MUST define caller obligations as preconditions when the operation cannot safely or meaningfully proceed without them.

Preconditions MUST describe only what the caller must establish before invocation.

# DBC-PRE-002 MUST Keep Preconditions Satisfiable

Preconditions MUST be weak enough for ordinary valid callers to satisfy.

Preconditions MUST NOT require callers to know private implementation details.

# DBC-PRE-003 MUST Validate Untrusted Input At Boundaries

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Preconditions over untrusted input MUST be enforced by boundary validation before domain logic runs.

Invalid external input MUST be treated as validation failure, not as an internal contract defect.

# DBC-POST-001 MUST Define Provider Guarantees

Public operations MUST define postconditions when callers rely on returned values, state changes, emitted events, persisted data, side effects, or error behavior.

Postconditions MUST describe externally observable outcomes.

# DBC-POST-002 MUST Define Side Effect Outcomes

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Postconditions MUST state whether an operation mutates state, emits messages, schedules work, persists data, creates resources, deletes resources, or performs no externally visible side effects.

Ordering, durability, aliasing, idempotency, and precision guarantees MUST be stated when they affect correctness.

# DBC-INV-001 MUST Define Stable Invariants

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Objects, aggregates, modules, records, protocols, caches, workflows, state machines, and resources MUST define invariants when validity depends on relationships across fields, states, resources, or events.

Invariants MUST hold at stable public boundaries.

# DBC-INV-002 MUST Establish Invariants Before Exposure

Constructors, factories, loaders, deserializers, migrations, and rehydration paths MUST establish invariants before exposing values.

Public mutating operations MUST restore invariants before returning, yielding, awaiting externally visible work, committing, publishing, or releasing locks.

# DBC-FAIL-001 MUST Define Failure Contracts

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)

Contracts MUST distinguish normal success, expected domain failure, validation failure, authorization failure, transient infrastructure failure, cancellation, timeout, and internal defect.

Expected failures MUST be part of the public contract.

# DBC-FAIL-002 MUST Define Failure Effects

Failure contracts MUST state which side effects are committed, rolled back, retried, partially applied, or unknown after failure.

Resource-owning operations MUST define cleanup behavior on failure, cancellation, timeout, and partial success.

# DBC-BLAME-001 MUST Preserve Contract Blame Semantics

If a trusted caller violates a precondition, the caller is at fault.

If a provider violates a postcondition after preconditions were satisfied, the provider is at fault.

If an invariant is broken at a stable boundary, the component that last held mutation authority is at fault.

# DBC-BLAME-002 MUST Report Violations Safely

Contract violation reports MUST identify the violated contract, boundary, expected condition, safe observed summary, and blame category.

Violation reports MUST NOT expose secrets, credentials, raw personal data, regulated data, or full untrusted payloads.

# DBC-FRAME-001 MUST Declare Effects For Impure Operations

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Every impure public operation MUST declare the externally visible state it may read, write, create, delete, emit, schedule, lock, or mutate.

An operation MUST NOT mutate state outside its declared frame.

# DBC-FRAME-002 MUST Make Dependencies Explicit

Operations that read clocks, randomness, environment variables, process state, registries, filesystems, networks, databases, queues, feature flags, or credentials MUST declare those dependencies.

An operation declared pure MUST NOT perform I/O, mutate reachable state, depend on nondeterminism, or acquire externally visible locks.

# DBC-ALIAS-001 MUST Define Ownership And Aliasing

See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)

Public contracts MUST state whether inputs are borrowed, consumed, copied, retained, mutated, shared, or transferred.

Public contracts MUST state whether returned values are immutable, mutable, owned, shared, pooled, lazy, live, or snapshots.

# DBC-SUBTYPE-001 MUST Preserve Behavioral Substitutability

A subtype, implementation, plugin, adapter, mock, fake, or replacement MUST be behaviorally substitutable for the contract it implements.

Subtypes MUST NOT strengthen preconditions or weaken postconditions, invariants, failure behavior, or frame limits.

# DBC-API-001 MUST Define API And Message Contracts

Public APIs and messages MUST define shape, authentication, authorization, errors, idempotency, pagination, ordering, filtering, versioning, compatibility, and rate limits when applicable.

Independently maintained consumers SHOULD receive machine-readable contracts such as schemas, IDLs, or OpenAPI documents.

# DBC-DATA-001 MUST Define Data Schemas At Boundaries

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Data exchanged across trust, deployment, package, or process boundaries MUST have explicit schemas or equivalent data contracts.

Schemas MUST specify required fields, optional fields, nullability, additional-field policy, bounds, formats, enum values, defaults, and version identifiers when applicable.

# DBC-DATA-002 MUST Distinguish Semantic Values

Data contracts MUST distinguish absent, null, empty, default, redacted, masked, and unknown values.

Field omission MUST NOT be treated as equivalent to explicit null, empty string, empty collection, zero, or default unless the contract says so.

# DBC-CONFIG-001 MUST Define Configuration Contracts

Configuration contracts MUST state source precedence, required keys, defaults, types, units, reload behavior, validation, and failure mode.

Environment variables MUST be parsed into typed configuration values before use.

# DBC-SECRET-001 MUST Define Secret Handling Contracts

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Secrets MUST NOT be stored in source code, examples, contract text, logs, telemetry, snapshots, generated documentation, or error messages.

Secret contracts MUST state access scope, rotation expectation, lifetime, storage category, and redaction policy.

# DBC-CONC-001 MUST Define Concurrency Contracts

See:
- [CORE-CONC-001](core.rules.md#core-conc-001-must-thread-safety)

Concurrent operations MUST state whether they are thread-safe, task-safe, process-safe, reentrant, single-threaded, single-owner, or externally synchronized.

Shared mutable state MUST have an ownership, synchronization, or single-writer contract.

# DBC-ASYNC-001 MUST Define Async Cancellation Contracts

Async operations MUST state cancellation behavior at externally visible await, yield, callback, or continuation boundaries.

Async operations MUST define whether cancellation rolls back work, commits partial work, leaves unknown state, or requires compensating action.

# DBC-DIST-001 MUST Define Distributed Delivery Semantics

Distributed contracts MUST distinguish exactly-once, at-least-once, at-most-once, best-effort, and idempotent behavior.

Message contracts MUST define ordering, deduplication, replay, poison-message handling, retention, and schema version when applicable.

# DBC-CHECK-001 SHOULD Check Trust Boundary Contracts At Runtime

Public trust-boundary contracts SHOULD be checked at runtime in production.

Runtime contract checks MUST be deterministic, side-effect-free, and bounded unless explicitly documented otherwise.

# DBC-CHECK-002 MUST Keep Runtime Checks Safe

Contract checks MUST NOT perform network calls, persistent writes, irreversible operations, or hidden dependency initialization.

Contract checks MUST NOT be the only defense against malicious input.

# DBC-STATIC-001 SHOULD Use Static Verification For Critical Contracts

Static verification SHOULD be used for contracts involving safety, money movement, access control, concurrency, critical invariants, protocol state machines, or complex mutations.

Proof assumptions MUST be recorded as assumptions, not presented as proven properties.

# DBC-TEST-001 MUST Test Public Contracts

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Every public contract MUST have tests for accepted inputs, rejected inputs, boundary values, expected failures, and representative success paths.

Contract tests MUST be organized by public boundary.

# DBC-TEST-002 SHOULD Use Property And Consumer Contract Tests

Property-based tests SHOULD be used for algebraic laws, parsers, serializers, state machines, validators, idempotency, ordering, and boundary-heavy input spaces.

Consumer-provider service pairs SHOULD use contract tests when independently deployed integrations must evolve safely.

# DBC-OBS-001 MUST Define Contract Failure Observability

See:
- [CORE-OBS-001](core.rules.md#core-obs-001-should-bounded-observability)

Operationally relevant public contracts MUST define observability for contract failures.

Contract failure diagnostics MUST be structured and safe for logs, metrics, traces, alerts, or crash reports.

# DBC-PERF-001 MUST Bound Contract Checking Cost

See:
- [CORE-PERF-001](core.rules.md#core-perf-001-must-measure-performance-claims)

Contract checks MUST have bounded cost relative to the operation or an explicitly documented cost class.

Contract checking overhead MUST be measured before checks are disabled for performance.

# DBC-VERSION-001 MUST Classify Contract Changes

Public contract changes MUST be classified as additive, compatible clarification, behavior-preserving refactor, deprecation, or breaking change.

Breaking changes MUST require versioning, migration, or compatibility approval.

# DBC-DEPRECATE-001 MUST Document Contract Deprecations

Deprecation notices MUST include replacement, first deprecated version, planned removal version or condition, owner, and migration guidance.

Removing a public contract element MUST be preceded by a documented deprecation period unless security, legal, or severe operational risk requires immediate removal.

# DBC-DOC-001 MUST Document Public Contracts

Public contract documentation MUST identify caller obligations, provider guarantees, failure behavior, side effects, resource ownership, version, and examples.

Contract documentation MUST distinguish normative rules from examples, commentary, and implementation notes.

# DBC-CI-001 MUST Check Contract Drift In CI

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

CI MUST fail when public contract documentation, schemas, generated clients, generated servers, or examples drift from implementation.

CI SHOULD run contract tests, schema checks, static checks where available, dependency audits, and documentation drift checks before merge.

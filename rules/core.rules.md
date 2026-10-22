# CORE-DET-001 MUST Deterministic Behavior

Systems MUST produce identical observable behavior when provided identical inputs, initial state, configuration, and runtime environment.

Runtime code MUST NOT depend on hidden mutable global state, ambient clocks, random sources, or external systems except through explicit boundaries.

# CORE-MEM-001 MUST Explicit Ownership

Ownership and lifetime MUST be obvious from code.

Mutable state, resources, tasks, subscriptions, and handles MUST have a clearly defined owner.

Hidden ownership transfer is forbidden.

# CORE-WORK-001 MUST Bounded Runtime Work

Runtime execution paths MUST have a provable upper bound.

Unbounded loops, recursion, retries, queues, waits, or retries-until-success are forbidden.

# CORE-CONC-001 MUST Thread Safety

Mutable state MUST have exactly one logical writer or a documented synchronization contract.

Data races are release-blocking defects.

# CORE-STATE-001 MUST Single Source Of Truth

Each fact MUST have one source of truth.

Derived state SHOULD be computed from source state instead of stored and synchronized.

Impossible states SHOULD be made unrepresentable.

# CORE-ERR-001 MUST Explicit Failure Handling

Expected failures MUST be represented explicitly with result types, status values, error objects, or domain failure states.

Silent failure handling is forbidden.

# CORE-API-001 MUST Explicit API Contracts

Public APIs MUST document inputs, outputs, ownership, lifetime, concurrency safety, failure modes, and units.

Boundary APIs MUST classify initialization-only, runtime-safe, and external-system behavior.

# CORE-BOUND-001 MUST Explicit Platform Boundaries

Filesystems, networks, clocks, random sources, environment variables, operating systems, hardware, and external services MUST be isolated behind explicit boundary layers.

Core logic MUST receive external data as inputs instead of reading external systems directly.

# CORE-SEC-001 MUST Validate Untrusted Input

All untrusted input MUST be validated before use.

Secrets MUST NOT be stored in source control, logged, embedded in client code, or exposed through diagnostics.

# CORE-BUILD-001 MUST Reproducible Toolchains

Toolchains, language versions, generators, and package managers MUST be pinned or otherwise reproducible.

Build outputs SHOULD be reproducible from committed sources.

# CORE-TEST-001 MUST Deterministic Tests

Tests MUST be deterministic and hermetic.

Tests MUST NOT depend on wall-clock sleeps, live networks, ambient filesystems, or external mutable state unless the test is explicitly classified as an integration boundary test.

# CORE-PERF-001 MUST Measure Performance Claims

Performance claims MUST be measured.

Optimization decisions MUST be justified with profiling, benchmarks, or production telemetry.

# CORE-OBS-001 SHOULD Bounded Observability

Logging, tracing, metrics, and diagnostics SHOULD be bounded, non-blocking, and cardinality-controlled.

Diagnostics MUST NOT change functional behavior.

# CORE-EXC-001 MUST Document Rule Exceptions

Rule exceptions MUST identify the violated rule, owner, rationale, tests covering the risk, expiration condition, and removal plan.

Temporary exceptions MUST expire.

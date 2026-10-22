# RS-EDITION-001 SHOULD Use Rust 2024 For New Crates

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

New Rust crates SHOULD use `edition = "2024"` unless a supported toolchain or dependency constraint requires an older edition.

Edition migrations MUST be run with `cargo fix --edition` or an equivalent reviewed migration process.

# RS-MSRV-001 MUST Declare Minimum Supported Rust Version

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

Crates MUST declare `rust-version` when they have a minimum supported Rust version.

CI MUST test the declared minimum supported Rust version when the crate promises one.

# RS-CARGO-001 MUST Keep Cargo Metadata Authoritative

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

`Cargo.toml` MUST be the authoritative source for package name, version, edition, Rust version, features, dependencies, license, repository, and published metadata.

Package metadata MUST NOT be duplicated in unrelated source constants without automated synchronization.

# RS-LOCK-001 SHOULD Commit Cargo Lockfiles

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

Applications, services, CLIs, examples, and workspaces SHOULD commit `Cargo.lock`.

Libraries MAY commit `Cargo.lock` for CI reproducibility, but published compatibility MUST be governed by `Cargo.toml` dependency requirements.

# RS-FEATURE-001 MUST Keep Cargo Features Additive

Cargo features MUST be additive.

Disabling a feature MUST NOT be required to make another feature correct.

Default features MUST stay minimal for libraries.

# RS-DEPS-001 MUST Minimize Runtime Dependencies

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Runtime dependencies MUST be justified by real functionality, maintenance value, or boundary requirements.

Optional integrations SHOULD be gated behind clearly named Cargo features.

# RS-API-001 MUST Encode Meaning In Types

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Public APIs MUST use domain-specific types, newtypes, enums, builders, or trait bounds when primitive parameters would permit invalid or ambiguous calls.

Boolean arguments SHOULD NOT select unrelated behaviors in public APIs.

# RS-API-002 SHOULD Preserve Semver-Compatible Public APIs

Public crates SHOULD treat exported items, trait implementations, feature flags, error types, and documented behavior as semver-relevant API.

Breaking changes MUST be intentional and documented.

# RS-OWN-001 MUST Express Ownership In Types

See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)

Rust APIs MUST make ownership, borrowing, mutation, and lifetime behavior explicit through types.

Functions SHOULD accept borrowed values when they do not need ownership.

# RS-STATE-001 MUST Avoid Hidden Interior Mutability

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Interior mutability MUST be justified by ownership, caching, synchronization, or API ergonomics.

Types using `Cell`, `RefCell`, `Mutex`, `RwLock`, atomics, or unsafe interior mutability MUST document mutation and sharing behavior.

# RS-CONC-001 MUST Make Thread Safety Explicit

See:
- [CORE-CONC-001](core.rules.md#core-conc-001-must-thread-safety)

Shared mutable state MUST use explicit synchronization or message passing.

APIs that cross thread boundaries MUST respect `Send` and `Sync` requirements intentionally.

# RS-ASYNC-001 MUST Keep Blocking Work Out Of Async Tasks

See:
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Async Rust tasks MUST NOT perform blocking filesystem, network, synchronization, or CPU-heavy work on an async executor thread.

Blocking work MUST move to a blocking pool, worker thread, process, or synchronous boundary.

# RS-ASYNC-002 MUST Own Spawned Tasks

See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)

Spawned tasks MUST have an owner, cancellation path, and observation path for failures.

Fire-and-forget tasks are forbidden.

# RS-ERR-001 MUST Use Result For Expected Failures

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)

Expected failures MUST be returned with `Result`.

`panic!` MUST be reserved for programmer defects, impossible invariants, or process-fatal conditions.

# RS-ERR-002 MUST Preserve Error Context

Error types MUST preserve enough context for callers and operators to diagnose failures.

Library errors SHOULD be typed and stable enough for intended downstream handling.

Errors and panic messages MUST NOT include secrets.

# RS-UNSAFE-001 MUST Minimize Unsafe Scope

See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)

Unsafe operations MUST be isolated to the smallest practical `unsafe` block.

Unsafe operations inside `unsafe fn` MUST still use explicit `unsafe` blocks.

# RS-UNSAFE-002 MUST Document Safety Invariants

Unsafe functions, unsafe traits, unsafe trait implementations, FFI boundaries, and unsafe blocks MUST document the invariants that make the code sound.

Callers of unsafe APIs MUST be told what obligations they must uphold.

# RS-FFI-001 MUST Isolate FFI Boundaries

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

FFI code MUST be isolated behind safe Rust APIs where practical.

FFI boundary types MUST have explicit layout, ownership, allocation, lifetime, and error contracts.

# RS-SERIAL-001 MUST Validate Serialized Input

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Deserialized external input MUST be validated before it reaches core logic.

Serde implementations for public data structures SHOULD be feature-gated when serialization is optional.

# RS-FMT-001 MUST Use Rustfmt

Rust code MUST be formatted with `rustfmt` or an approved rustfmt-compatible configuration.

Manual formatting exceptions MUST be local and justified.

# RS-LINT-001 MUST Run Clippy

Clippy MUST run in CI for Rust crates.

Clippy suppressions MUST be local, lint-specific, and justified.

Whole-category `clippy::restriction` enables MUST NOT be used.

# RS-TEST-001 MUST Test Rust Crates At Multiple Levels

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Rust crates MUST include unit tests for core behavior.

Public APIs SHOULD include integration tests or doc tests for supported usage.

Unsafe code MUST have tests that exercise documented invariants where practical.

# RS-TEST-002 SHOULD Use Property Or Fuzz Tests At Boundaries

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Parsers, decoders, protocol handlers, unsafe abstractions, and serialization boundaries SHOULD use property tests, fuzz tests, or generated hostile inputs.

# RS-PERF-001 MUST Measure Rust Performance Claims

See:
- [CORE-PERF-001](core.rules.md#core-perf-001-must-measure-performance-claims)

Rust performance claims MUST be supported by benchmarks, profiles, or production telemetry.

Allocation reductions, unsafe optimizations, and concurrency changes MUST preserve correctness evidence.

# RS-DOC-001 MUST Document Public APIs

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Public crates MUST document public modules, types, traits, functions, feature flags, examples, error behavior, and safety requirements.

Documentation examples SHOULD compile as doc tests when practical.

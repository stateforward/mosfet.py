# PY-RUNTIME-001 MUST Declare Supported Python Versions

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

Python projects MUST declare supported Python versions in package metadata, documentation, and CI.

The declared version policy MUST agree across those locations.

# PY-RUNTIME-002 MUST Treat Minimum Python As Syntax Baseline

Source code MUST NOT use syntax or standard-library features unavailable on the minimum supported Python version.

Compatibility shims MUST be isolated and tested on every supported Python minor version.

# PY-RUNTIME-003 MUST NOT Assume Free Threaded Semantics

See:
- [CORE-CONC-001](core.rules.md#core-conc-001-must-thread-safety)

Python code MUST NOT rely on free-threaded interpreter behavior unless CI tests that interpreter configuration.

CPU-bound pure Python code MUST NOT assume parallel bytecode execution on standard GIL builds.

# PY-STRUCT-001 SHOULD Use Src Layout For Packages

Packaged Python projects SHOULD use a `src/` layout so tests import installed code rather than the repository root copy.

Flat scripts MAY be used only for one-file programs with no reusable package logic.

# PY-STRUCT-002 MUST Keep Entry Points Thin

Application entry points, console wrappers, and `if __name__ == "__main__"` blocks MUST call package code.

Domain logic MUST NOT live in entry point wrappers.

# PY-STRUCT-003 MUST Avoid Production Sys Path Mutation

Production code MUST NOT mutate `sys.path` to make imports work.

Import structure MUST be solved through packaging, module layout, or explicit test tooling configuration.

# PY-PROJECT-001 MUST Use Pyproject Metadata

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

Packaged Python projects MUST use `pyproject.toml` for build-system configuration and standardized project metadata.

Runtime dependencies MUST be declared in project metadata.

# PY-PROJECT-002 MUST Keep Version Metadata Single Sourced

Version metadata MUST have one source of truth.

Package constants, SCM tags, and project metadata MUST NOT drift without automated synchronization.

# PY-DEPS-001 MUST Lock Deployable Applications

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

Applications, services, CLIs, and automation tools MUST use a committed lockfile or equivalent reproducible install plan.

Libraries SHOULD NOT force consumer lockfiles through published package metadata.

# PY-DEPS-002 MUST Minimize Runtime Dependencies

Runtime dependencies MUST be as narrow as practical.

A dependency MUST NOT be added for a small standard-library feature.

Development-only tools MUST NOT be included in runtime dependencies.

# PY-DEPS-003 MUST Audit Deployable Dependencies

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Deployable projects MUST run dependency vulnerability audits before release and on a recurring schedule.

Vulnerability exceptions MUST identify advisory, impact, compensating controls, owner, and expiry.

# PY-STYLE-001 MUST Enforce One Formatter

Python projects MUST have one authoritative formatter.

Formatting MUST be enforced in CI with a check-only command.

Conflicting formatters or import sorters MUST NOT be mixed.

# PY-LINT-001 MUST Run Static Linting

Lint configuration MUST be committed.

Lint exceptions MUST be local, specific, and justified.

Blanket lint suppressions are forbidden.

# PY-TYPE-001 MUST Type Public APIs

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Public functions, methods, classes, constants with non-obvious types, and attributes MUST have type annotations.

Typed packages MUST include a `py.typed` marker in built artifacts.

# PY-TYPE-002 MUST Contain Any At Boundaries

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

`Any` MAY appear only at untyped external boundaries, dynamic deserialization before validation, compatibility shims, test doubles, or deliberately dynamic plugin registries.

`Any` MUST be converted to precise types before reaching domain logic.

# PY-TYPE-003 MUST Justify Type Escapes

See:
- [CORE-EXC-001](core.rules.md#core-exc-001-must-document-rule-exceptions)

`cast()` and `# type: ignore` MUST express documented invariants or isolated third-party typing defects.

Type ignores MUST include an error code where the checker supports codes.

# PY-IMPORT-001 MUST Keep Imports Acyclic

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Runtime imports MUST be acyclic.

Circular imports MUST be fixed by moving shared types, lowering dependencies, or using type-only imports.

# PY-IMPORT-002 MUST Keep Import Time Pure

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Modules MUST NOT perform network I/O, filesystem writes, database connections, process spawning, logging configuration, environment mutation, or background task startup at import time.

# PY-API-001 MUST Export Public APIs Intentionally

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Public APIs MUST be exported through documented modules, `__all__`, package docs, or entry point declarations.

Public APIs MUST NOT require callers to import from private modules.

# PY-API-002 MUST Separate Domain From Framework Types

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Framework request objects, ORM sessions, CLI parser objects, and transport-specific types SHOULD be converted to domain types at boundaries.

Domain logic MUST NOT require framework-specific objects unless the domain is the framework integration itself.

# PY-OBJ-001 SHOULD Prefer Dataclasses For Simple Records

Dataclasses SHOULD be used for simple typed records, immutable value objects, and internal message structures.

Mutable default values MUST use `default_factory`.

# PY-OBJ-002 MUST Avoid Mutable Global Business State

See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)

Global mutable state MUST NOT be used for business logic, request state, user state, runtime configuration, or feature flags.

Bounded process-wide caches MAY be used only when ownership, thread safety, observability, and test reset behavior are documented.

# PY-FUNC-001 MUST Make Function Inputs Explicit

See:
- [CORE-DET-001](core.rules.md#core-det-001-must-deterministic-behavior)

Functions MUST have explicit inputs and outputs.

Hidden inputs from globals, environment variables, current working directory, wall-clock time, or random state MUST be injected or isolated at boundaries.

# PY-FUNC-002 MUST NOT Use Mutable Defaults

Function defaults MUST NOT be mutable unless the object is intentionally shared and documented.

# PY-FUNC-003 MUST Bound Recursion

See:
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Recursion MUST have bounded depth or documented input constraints.

Iteration SHOULD be preferred for untrusted or arbitrarily deep inputs.

# PY-ERR-001 MUST Define Package Error Boundaries

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)

Libraries SHOULD define a package-level exception base class for expected package-specific failures.

Public functions MUST document package-specific exceptions they raise.

# PY-ERR-002 MUST Raise Specific Exceptions

Code MUST raise the most specific built-in or custom exception that accurately describes the failure.

Bare `Exception` and `BaseException` MUST NOT be raised for ordinary application errors.

# PY-ERR-003 MUST Preserve Exception Context

Broad `except Exception` blocks MUST re-raise, wrap with exception chaining, or handle a documented boundary failure.

Cleanup code MUST NOT suppress cleanup failures silently.

# PY-ASYNC-001 MUST Own Async Tasks

See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)

Async tasks MUST have explicit ownership.

Fire-and-forget tasks MUST be registered, named when practical, monitored, and cancelled during shutdown.

# PY-ASYNC-002 MUST Keep Blocking Work Off Event Loops

Blocking file, network, subprocess, DNS, or CPU-heavy work MUST NOT run directly on an event loop.

Blocking work MUST move to an executor, process, native library, or async-compatible library.

# PY-ASYNC-003 MUST Keep Async Boundaries Explicit

Libraries MUST NOT call `asyncio.run()` inside public APIs.

Applications and CLIs MAY call it at top-level process boundaries.

# PY-SUBPROCESS-001 MUST Avoid Shell Injection

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Subprocess calls MUST pass argument lists with `shell=False` by default.

Shell invocation MUST NOT interpolate untrusted input.

# PY-IO-001 MUST Isolate IO Boundaries

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Domain logic MUST be isolated from filesystem, network, database, subprocess, clock, randomness, environment, and terminal I/O.

Functions that perform I/O MUST make that behavior clear through name, module placement, docstring, or type.

# PY-IO-002 MUST Manage Files Safely

Files MUST be opened with context managers.

Text file I/O MUST specify an encoding unless using a documented binary protocol.

Temporary files and directories MUST use secure temporary-file APIs.

# PY-CONFIG-001 MUST Load Configuration At Boundaries

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Configuration MUST be loaded at application startup or explicit factory boundaries, not at package import time.

Configuration MUST be represented as typed objects before reaching domain logic.

# PY-CONFIG-002 MUST Protect Secrets

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Secret values MUST NOT have non-empty production defaults.

Secrets MUST NOT be committed, logged, included in exceptions, printed in diagnostics, or embedded in generated artifacts.

# PY-LOG-001 MUST Keep Library Logging Passive

See:
- [CORE-OBS-001](core.rules.md#core-obs-001-should-bounded-observability)

Libraries MUST create module loggers with `logging.getLogger(__name__)`.

Libraries MUST NOT configure process logging or write diagnostics with `print()` except for explicit CLI output paths.

# PY-LOG-002 SHOULD Use Stable Structured Logs

Service and automation boundaries SHOULD use structured logs with stable event names and key-value context.

Logs MUST NOT include secrets or unredacted sensitive payloads.

# PY-DATA-001 MUST Validate External Data

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Untrusted input MUST be validated at the first boundary where enough context exists to validate it.

Validation MUST produce typed domain objects before data reaches core logic.

# PY-DATA-002 MUST Use Safe Parsers

Pickle, marshal, shelve, unsafe YAML loaders, and dynamic code evaluation MUST NOT be used with untrusted input.

Parsers for JSON, YAML, TOML, XML, archives, and compressed data MUST enforce appropriate size or expansion limits.

# PY-DATA-003 MUST Preserve Boundary Semantics

Datetime values crossing boundaries MUST include timezone semantics.

Decimal, money, and identifier fields MUST NOT be parsed through floating-point conversions when precision matters.

# PY-PERF-001 MUST Profile Before Optimizing

See:
- [CORE-PERF-001](core.rules.md#core-perf-001-must-measure-performance-claims)

Python performance changes MUST be based on profiling, benchmarks, production traces, or workload analysis.

Micro-optimizations without evidence are forbidden.

# PY-PERF-002 MUST Bound Caches

Caches MUST have explicit maximum size, invalidation strategy, and tests for stale data behavior.

Unbounded caches MUST NOT be used on unbounded input domains.

# PY-MEM-001 MUST Avoid Unbounded Accumulation

See:
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Code MUST avoid unbounded in-memory accumulation for streams, logs, request bodies, archive contents, and query results.

Large pipelines SHOULD process iterators, chunks, cursors, or streams.

# PY-TEST-001 MUST Commit Tests With Code Changes

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Production code changes MUST include tests or a documented exception.

Bug fixes MUST include targeted regression tests.

# PY-TEST-002 MUST Keep Unit Tests Hermetic

Unit tests MUST cover pure domain logic without network, database, filesystem, wall-clock, or random dependencies.

Integration tests MAY use real I/O but MUST isolate resources and be marked or separated from fast unit tests.

# PY-TEST-003 MUST Test Installed Packages

Packaged projects SHOULD run at least one test or smoke-test path against the installed package, not only the source tree.

Typed public APIs SHOULD include checker-visible examples or typing tests.

# PY-CI-001 MUST Run Python Quality Gates

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

CI MUST run formatting checks, linting, type checking, tests, package build checks, install smoke tests, and dependency or security audits for deployable projects.

CI MUST fail on required gate failures.

# PY-PACKAGE-001 MUST Build Standards Compliant Artifacts

Distributed Python projects MUST build standards-compliant wheels and source distributions through a declared build backend.

Release workflows MUST verify artifacts by installing the wheel in a clean environment.

# PY-PACKAGE-002 MUST Access Package Data Through Resource APIs

Package data MUST be included through backend-supported package data configuration.

Runtime code MUST access package data through import-resource APIs rather than repository-relative paths.

# PY-CLI-001 MUST Keep CLI Output Stable

CLI stdout MUST be reserved for requested command output.

Diagnostics, progress, and human-oriented errors MUST go to stderr or logging.

Scripted CLI output MUST remain stable unless a breaking change is documented.

# PY-DOC-001 MUST Document Public Python Interfaces

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Public modules, classes, functions, exceptions, CLI commands, configuration, and supported Python versions MUST be documented.

Documentation MUST be updated with public behavior changes.

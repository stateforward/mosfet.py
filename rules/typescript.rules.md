# TS-STRICT-001 MUST Enable Strict Type Checking

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

TypeScript projects MUST enable `strict`.

Projects SHOULD also enable `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, `useUnknownInCatchVariables`, `noPropertyAccessFromIndexSignature`, and `noImplicitOverride`.

# TS-ANY-001 MUST NOT Use Unsafe Any

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Explicit `any`, `as any`, and broad unsafe assertions are forbidden in core logic.

Unavoidable interop escapes MUST be isolated to boundary modules and documented.

# TS-NULL-001 MUST Model Absence Explicitly

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Nullability MUST be represented explicitly in types.

Non-null assertions are forbidden unless protected by a documented invariant.

# TS-UNION-001 MUST Check Exhaustiveness

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Discriminated unions MUST be handled exhaustively.

Unreachable branches SHOULD be checked with a `never` helper.

# TS-MODULE-001 MUST Define Module Boundaries

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Packages MUST define public entrypoints explicitly.

Deep imports across package boundaries are forbidden unless the imported path is part of the public contract.

# TS-RUNTIME-001 MUST Separate Runtime Targets

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Node, browser, edge, worker, and test runtimes MUST have explicit configuration boundaries.

Runtime-specific APIs MUST NOT leak into code compiled for other runtimes.

# TS-ASYNC-001 MUST Handle Promises Explicitly

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Promises MUST be awaited, returned, or explicitly marked as handled.

Long-running async work MUST support cancellation through `AbortSignal` or an equivalent owner-controlled mechanism.

# TS-ERR-001 MUST Throw Error Objects

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)

Thrown values MUST be `Error` objects or subclasses.

Catch variables MUST be treated as `unknown` until narrowed.

# TS-DATA-001 MUST Validate Unknown Data

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Network, environment, IPC, storage, and user input data MUST be treated as `unknown` at the boundary.

Static types SHOULD be derived from runtime validation schemas when schemas are used.

# TS-BUILD-001 MUST Typecheck Separately From Transpilation

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

Fast transpilers MUST NOT be the only correctness check.

CI MUST run a dedicated typecheck step.

# TS-SUPPRESS-001 MUST Justify Type Suppressions

See:
- [CORE-EXC-001](core.rules.md#core-exc-001-must-document-rule-exceptions)

`@ts-ignore`, `@ts-nocheck`, broad `eslint-disable`, and strictness relaxations MUST be temporary, documented exceptions.

Generated code MAY use isolated suppressions when the generator contract is documented.

# TS-PERF-001 SHOULD Keep Types Tractable

See:
- [CORE-PERF-001](core.rules.md#core-perf-001-must-measure-performance-claims)

Types SHOULD avoid excessive union expansion and deeply nested conditional generics that degrade editor or compiler performance.

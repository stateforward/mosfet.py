# HONO-BASE-001 MUST Apply TypeScript Rules

See:
- [TS-STRICT-001](typescript.rules.md#ts-strict-001-must-enable-strict-type-checking)
- [TS-RUNTIME-001](typescript.rules.md#ts-runtime-001-must-separate-runtime-targets)

Hono applications written in TypeScript MUST comply with the TypeScript rules.

# HONO-ROUTE-001 MUST Preserve Route Type Inference

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Routes SHOULD keep path, validation, and handler definitions close enough to preserve Hono type inference.

Standalone controller extraction MUST NOT erase route types.

# HONO-ROUTE-002 MUST Mount Routes Intentionally

Routes mounted as sub-apps MUST be defined before mounting.

`mount()` SHOULD be reserved for non-Hono handlers or foreign framework integration.

# HONO-CONTEXT-001 MUST Keep Context Request Scoped

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Hono context variables MUST be request-scoped.

Business state MUST NOT be stored in `Context`.

# HONO-MIDDLEWARE-001 SHOULD Scope Middleware Narrowly

See:
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Middleware SHOULD be path-scoped when only some routes require it.

Global middleware MUST be safe and relevant for every route.

# HONO-VALIDATE-001 MUST Validate Transport Input

See:
- [TS-DATA-001](typescript.rules.md#ts-data-001-must-validate-unknown-data)

Request params, query strings, headers, cookies, forms, and JSON bodies MUST be validated before handler logic consumes them.

Transport schemas MUST NOT become the domain model by accident.

# HONO-ERR-001 MUST Standardize HTTP Errors

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)

Applications MUST map internal errors to a consistent HTTP error shape at the boundary.

Stack traces and raw exception objects MUST NOT be exposed to clients.

# HONO-RPC-001 MUST Export Exact Route Types For RPC

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Typed Hono RPC clients MUST consume the exact route tree type exported by the server boundary.

Client code MUST NOT hand-duplicate server route types.

# HONO-TEST-001 MUST Test HTTP Boundaries

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Hono tests MUST assert status codes, headers, and response body shapes for success and failure cases.

Typed client tests and black-box HTTP tests SHOULD both be used when RPC types are part of the contract.

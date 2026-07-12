# CPP-MEM-001 MUST Use RAII Ownership

See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)

C++ resources MUST be acquired and released through RAII.

Raw owning `new` and `delete` are forbidden in application code.

# CPP-MEM-002 MUST Express Borrowed Memory

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Borrowed memory MUST be represented with references, `std::span`, `std::string_view`, or explicit pointer and length contracts.

Borrowed memory APIs MUST document lifetime.

# CPP-UB-001 MUST NOT Depend On Undefined Behavior

Undefined behavior is forbidden.

Type punning MUST use defined mechanisms such as `std::bit_cast` or `std::memcpy`.

# CPP-INT-001 MUST Check Risky Numeric Operations

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Integer conversion, indexing, size calculation, and arithmetic that can overflow or truncate MUST be checked or represented with safer types.

# CPP-CONC-001 MUST Document Synchronization

See:
- [CORE-CONC-001](core.rules.md#core-conc-001-must-thread-safety)

Shared mutable state MUST have a documented synchronization contract.

Locks MUST NOT be held across blocking I/O or callbacks into unknown code.

# CPP-ATOM-001 MUST Justify Atomic Ordering

See:
- [CORE-CONC-001](core.rules.md#core-conc-001-must-thread-safety)

Atomic operations MUST use memory ordering intentionally.

Default sequential consistency SHOULD be justified when used in hot paths.

# CPP-ERR-001 MUST Represent Expected Failures Explicitly

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)

Expected failures MUST be represented with error codes, status values, `std::expected`, or domain result types.

Error-bearing results SHOULD be marked `[[nodiscard]]`.

# CPP-ABI-001 SHOULD Keep ABI Boundaries Plain

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Shared-library and foreign-function boundaries SHOULD use C ABI compatible functions and standard-layout data.

STL types SHOULD NOT cross unstable ABI boundaries.

# CPP-BUILD-001 MUST Pin Compiler Configuration

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

Compiler versions, language standard, warning policy, and sanitizer configuration MUST be pinned.

Warnings MUST be treated as errors in CI.

# CPP-TEST-001 SHOULD Run Memory And Race Tooling

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Memory-sensitive and concurrency-sensitive C++ code SHOULD run under ASan, UBSan, TSan, or equivalent tooling.

# CPP-PERF-001 SHOULD Prefer Predictable Hot Path Data

See:
- [CORE-PERF-001](core.rules.md#core-perf-001-must-measure-performance-claims)

Hot paths SHOULD prefer contiguous storage, stable bounds, and predictable iteration.

Pointer-heavy structures SHOULD require profiling or algorithmic justification.

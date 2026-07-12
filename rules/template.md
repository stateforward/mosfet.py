# Rules Framework Specification

Version: 1.0

## Philosophy

Rules are plain Markdown.

A rule is a heading.

A rule may reference other rules using standard Markdown links.

No custom syntax, metadata formats, parsers, databases, or dependency engines are required.

The rule graph emerges naturally from Markdown links.

---

## Repository Structure

```text
rules/
├── core.rules.md
├── patterns.rules.md
├── organization.rules.md
├── python.rules.md
├── cpp.rules.md
├── rust.rules.md
├── typescript.rules.md
├── go.rules.md
├── dart.rules.md
└── java.rules.md
```

---

## Rule Categories

### Core Rules

Universal engineering principles.

Examples:

```text
CORE-DET-001
CORE-MEM-001
CORE-CONC-001
CORE-ERR-001
CORE-TEST-001
```

### Pattern Rules

Reusable architecture and design patterns.

Examples:

```text
PAT-ACTOR-001
PAT-HSM-001
PAT-ECS-001
PAT-PIPELINE-001
```

### Organization Rules

Repository and codebase organization.

Examples:

```text
ORG-PKG-001
ORG-MOD-001
ORG-TEST-001
ORG-DOC-001
```

### Language Rules

Language-specific implementations and guidance.

Examples:

```text
PY-TYPE-001
CPP-MEM-001
RS-CONC-001
TS-API-001
```

---

## Rule Format

A rule is defined by a Markdown heading.

Format:

```markdown
# <RULE-ID> <RFC-2119-KEYWORD> <Rule Title>
```

Examples:

```markdown
# CORE-MEM-001 MUST Explicit Ownership
# CORE-GLOB-001 MUST NOT Hidden Global State
# CORE-ERR-001 SHOULD Explicit Error Handling
# CORE-DI-001 MAY Dependency Injection
```

---

## Allowed Keywords

Use RFC 2119 keywords only:

* MUST
* MUST NOT
* SHOULD
* SHOULD NOT
* MAY

These keywords carry their standard normative meanings.

---

## Rule References

Rules reference other rules using standard Markdown links.

Within the same document:

```markdown
See:
- [CORE-MEM-001](#core-mem-001-must-explicit-ownership)
```

Across documents:

```markdown
See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)
```

Multiple references:

```markdown
See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)
- [CORE-CONC-001](core.rules.md#core-conc-001-must-thread-safety)
- [PAT-ACTOR-001](patterns.rules.md#pat-actor-001-must-actor-state-ownership)
```

---

## Example Core Rule

```markdown
# CORE-MEM-001 MUST Explicit Ownership

Ownership MUST be obvious from code.

Objects MUST have a clearly defined owner.

Hidden ownership transfer is forbidden.
```

---

## Example Language Rule

```markdown
# PY-TYPE-001 MUST Type Hint Public APIs

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Public APIs MUST include type annotations.

Modern Python typing syntax MUST be used.
```

---

## Example Pattern Rule

```markdown
# PAT-ACTOR-001 MUST Actor State Ownership

See:
- [CORE-CONC-001](core.rules.md#core-conc-001-must-thread-safety)

Actors MUST exclusively own their mutable state.

Actors MUST NOT directly mutate another actor's state.

Cross-actor communication MUST occur through messages.
```

---

## Example Organization Rule

```markdown
# ORG-PKG-001 MUST Acyclic Package Dependencies

Packages MUST form an acyclic dependency graph.

Circular dependencies are forbidden.
```

---

## Authoring Guidelines

Rules SHOULD:

* be concise
* be normative
* be independently understandable
* reference related rules when useful

Rules SHOULD NOT:

* duplicate requirements from other rules
* contain tutorials
* contain extensive examples
* contain project-specific guidance

Rules MAY contain:

* rationale
* implementation notes
* references to external standards

when such information improves clarity.

---

## Design Goals

The framework MUST remain:

* Plain Markdown
* Human readable
* AI readable
* Git friendly
* Searchable
* Linkable
* Language agnostic
* Tool agnostic

The framework MUST NOT require:

* Custom parsers
* Custom metadata schemas
* Databases
* Build-time preprocessing
* Special tooling

Markdown headings and links are the canonical source of truth.

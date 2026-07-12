# Rules

This repository contains plain Markdown rule files for coding and engineering guidance.

The authoring specification is [template.md](template.md). In short:

- a rule is a Markdown heading
- every rule heading uses `# <RULE-ID> <RFC-2119-KEYWORD> <Rule Title>`
- rules reference related rules with standard Markdown links
- rule files do not use custom metadata, YAML frontmatter, parser directives, or dependency schemas
- language and framework rules link to core or pattern rules instead of duplicating them

## Rule Files

<!-- all top-level *.rules.md files in this repository, grouped by category -->

Core and reusable pattern rules:

- [core.rules.md](core.rules.md)
- [dbc.rules.md](dbc.rules.md)
- [patterns.rules.md](patterns.rules.md)

Language rules:

- [cpp.rules.md](cpp.rules.md)
- [dart.rules.md](dart.rules.md)
- [go.rules.md](go.rules.md)
- [python.rules.md](python.rules.md)
- [rust.rules.md](rust.rules.md)
- [typescript.rules.md](typescript.rules.md)

Framework and tooling rules:

- [flutter.rules.md](flutter.rules.md)
- [hono.rules.md](hono.rules.md)
- [hsm.rules.md](hsm.rules.md)
- [pulumi.rules.md](pulumi.rules.md)
- [react.rules.md](react.rules.md)
- [sml.rules.md](sml.rules.md)
- [webcomponents.rules.md](webcomponents.rules.md)
- [xstate.rules.md](xstate.rules.md)

## Writing Rules

Use one top-level heading per rule:

```markdown
# TS-ASYNC-001 MUST Handle Promises Explicitly

See:
- [CORE-ERR-001](core.rules.md#core-err-001-must-explicit-failure-handling)

Promises MUST be awaited, returned, or explicitly marked as handled.
```

Rules should be concise, independently understandable, and normative. Prefer links to existing rules when a requirement already exists elsewhere.

## Scope

This repository is for coding, language, framework, infrastructure-as-code, and reusable engineering pattern rules.

Project-specific workflow rules, merge policies, agent instructions, and repository-local process rules do not belong in this shared rule set.

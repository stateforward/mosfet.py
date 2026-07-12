# PULUMI-STACK-001 MUST Separate Project And Stack Boundaries

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

A Pulumi project MUST represent a deployment unit.

A Pulumi stack MUST represent an environment, region, tenant, or other explicit deployment boundary.

# PULUMI-COMP-001 SHOULD Keep Entry Points Compositional

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Pulumi entry points SHOULD compose resources and components.

Reusable infrastructure logic SHOULD live in language modules or component resources with explicit inputs and outputs.

# PULUMI-NAME-001 MUST Preserve Resource Identity

Logical resource names MUST be stable.

Resource renames, moves, and reparenting MUST use aliases or documented state migration.

# PULUMI-PROVIDER-001 MUST Make Provider Scope Explicit

See:
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Provider configuration MUST make account, region, tenant, and credential scope explicit.

Multi-account or multi-region stacks MUST pass providers intentionally.

# PULUMI-CONFIG-001 MUST Manage Configuration Through Pulumi

Stack configuration MUST be changed through Pulumi tooling or automation.

Generated stack configuration MAY be committed when it is part of the reviewed deployment contract.

# PULUMI-SECRET-001 MUST Use Secret Channels

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Sensitive values MUST use Pulumi secrets or an approved secret manager.

Plain stack configuration MUST NOT contain secrets.

# PULUMI-VERSION-001 MUST Pin Pulumi Versions

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

Pulumi CLI, provider, package, and language dependency versions MUST be pinned or otherwise reproducible.

# PULUMI-PREVIEW-001 MUST Review Plans Before Updates

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Infrastructure changes MUST be reviewed with `pulumi preview` or an equivalent generated plan before update.

Protected environments SHOULD use saved update plans or equivalent change-control gates.

# PULUMI-DRIFT-001 SHOULD Detect Drift

Production infrastructure SHOULD be checked for drift regularly.

Unexpected drift SHOULD be treated as an incident or tracked remediation item.

# PULUMI-POLICY-001 SHOULD Enforce Policy Checks

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Infrastructure previews SHOULD run policy checks for security, cost, ownership, and operational constraints.

# PULUMI-TEST-001 MUST Test Infrastructure Logic

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Reusable infrastructure logic MUST have unit tests with mocks or fakes.

Critical stacks SHOULD have lifecycle or integration tests with post-deploy assertions.

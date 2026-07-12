# WC-BASE-001 MUST Apply TypeScript And DOM Boundary Rules

See:
- [TS-STRICT-001](typescript.rules.md#ts-strict-001-must-enable-strict-type-checking)
- [TS-DATA-001](typescript.rules.md#ts-data-001-must-validate-unknown-data)
- [CORE-BOUND-001](core.rules.md#core-bound-001-must-explicit-platform-boundaries)

Web Component code written in TypeScript MUST comply with the TypeScript rules.

Browser, DOM, storage, network, history, timer, and observer APIs MUST be treated as platform boundaries.

# WC-NAME-001 MUST Prefer Autonomous Custom Elements

Autonomous custom elements extending `HTMLElement` MUST be the default.

Customized built-in elements require an explicit compatibility exception for the supported browser set.

Custom element names MUST be stable, kebab-case, include a project or package prefix, and contain a hyphen.

# WC-REG-001 MUST Make Registration Idempotent

Custom element registration MUST be centralized or otherwise discoverable.

Registration code MUST guard repeated execution with `customElements.get(name)` or an equivalent registry check.

Definition side effects beyond registration are forbidden.

# WC-LIFE-001 MUST Own Lifecycle Resources

See:
- [CORE-MEM-001](core.rules.md#core-mem-001-must-explicit-ownership)
- [CORE-WORK-001](core.rules.md#core-work-001-must-bounded-runtime-work)

Constructors MUST only initialize local state, call `super()`, attach owned shadow roots, and bind owner-scoped static listeners.

Constructors MUST NOT read attributes, inspect children, fetch data, start timers, subscribe to external systems, or mutate light DOM children.

Runtime setup MUST happen in `connectedCallback()` and cleanup MUST happen in `disconnectedCallback()`.

Connection work MUST be idempotent. Elements that are frequently moved SHOULD handle state-preserving moves explicitly or make reconnect side effects harmless.

# WC-API-001 MUST Separate Attributes And Properties

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)

Attributes MUST be reserved for primitive, serializable configuration such as strings, booleans, numbers, IDs, and enum-like values.

Rich objects, arrays, callbacks, runtime handles, and domain snapshots MUST be passed through typed properties or methods.

Reflected attributes MUST have one source of truth and a documented serialization rule.

`observedAttributes` MUST list every reactive attribute explicitly.

# WC-TYPE-001 MUST Type Public Element Contracts

See:
- [TS-ANY-001](typescript.rules.md#ts-any-001-must-not-use-unsafe-any)
- [TS-MODULE-001](typescript.rules.md#ts-module-001-must-define-module-boundaries)

Public element classes, properties, methods, attributes, and event details MUST have exported TypeScript contracts.

Modules that create custom elements across boundaries SHOULD augment `HTMLElementTagNameMap` or provide typed factory helpers.

`CustomEvent` payloads MUST use explicit detail types.

Unsafe casts against DOM queries MUST be localized and protected by runtime narrowing when the queried node crosses a trust or module boundary.

# WC-EVENT-001 MUST Use Explicit Event Contracts

See:
- [PAT-EVENT-001](patterns.rules.md#pat-event-001-must-typed-event-boundaries)

Custom elements MUST communicate ownership-changing commands, submissions, and notifications through named events or typed public methods.

Events that intentionally cross a shadow boundary MUST set `composed: true`.

Events that intentionally bubble to an owner MUST set `bubbles: true`.

Event names, detail payloads, cancellation semantics, and ownership of side effects MUST be documented as part of the element contract.

# WC-STATE-001 MUST Keep Element State Minimal

See:
- [CORE-STATE-001](core.rules.md#core-state-001-must-single-source-of-truth)
- [PAT-HSM-001](patterns.rules.md#pat-hsm-001-must-explicit-hierarchical-state-modeling)

Elements MUST store only source state they own.

Derived DOM, CSS classes, ARIA attributes, disabled states, and labels SHOULD be computed from source state during render.

Mutually exclusive UI modes MUST be represented by a single state value, discriminated union, reducer, actor, or state machine.

Contradictory boolean state matrices are forbidden.

# WC-RENDER-001 MUST Render Untrusted Content Safely

See:
- [CORE-SEC-001](core.rules.md#core-sec-001-must-validate-untrusted-input)

Untrusted user, network, storage, Markdown, or connector data MUST NOT be passed to `innerHTML`, `insertAdjacentHTML`, or template string HTML without context-correct escaping or sanitization.

DOM APIs such as `createElement`, `textContent`, `setAttribute`, `append`, and `replaceChildren` SHOULD be preferred for untrusted content.

When HTML templates are used, dynamic interpolations MUST be escaped for the correct HTML, attribute, URL, or script context.

Raw HTML rendering requires an explicit sanitizer contract and tests for forbidden markup.

# WC-SHADOW-001 MUST Choose Encapsulation Intentionally

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Light DOM, open shadow DOM, closed shadow DOM, slots, and parts MUST be chosen as explicit API decisions.

Shadow DOM SHOULD be used when a component must protect internal DOM or CSS from page-level selectors.

Light DOM SHOULD be used when the component is an application shell or must intentionally participate in page-level layout and design tokens.

Closed shadow roots MUST NOT be treated as a security boundary.

# WC-CSS-001 MUST Define Styling Contracts

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)

Every custom element MUST define its host display behavior and sizing assumptions.

Shared design tokens MUST live at an explicit global boundary such as `:root` or a documented theme host.

Shadow DOM components MUST expose styling hooks through custom properties, `::part`, slots, documented attributes, or documented states.

Page CSS MUST NOT depend on private shadow DOM structure.

Component CSS MUST NOT rely on unrelated global selectors for correctness.

# WC-CSS-002 SHOULD Reuse Shadow Styles Efficiently

See:
- [CORE-PERF-001](core.rules.md#core-perf-001-must-measure-performance-claims)

Frequently rendered shadow DOM components SHOULD reuse styles through constructable stylesheets, cached templates, or another documented reuse mechanism.

Creating identical `<style>` nodes on every render SHOULD be avoided unless the component is low-frequency or measurement shows the cost is irrelevant.

Stylesheet sharing MUST NOT introduce mutable global style state that lets one component instance unexpectedly change another.

# WC-ACCESS-001 MUST Preserve Native Semantics

See:
- [CORE-API-001](core.rules.md#core-api-001-must-explicit-api-contracts)
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Custom elements that behave like controls, dialogs, menus, tabs, trees, forms, links, or live regions MUST preserve the expected keyboard, focus, label, disabled, validation, and assistive-technology semantics.

Native controls MUST be preferred when they satisfy the interaction contract.

Elements replacing native form controls SHOULD use `ElementInternals` and form-associated custom element APIs where supported, with documented fallbacks for unsupported target browsers.

# WC-FEATURE-001 MUST Verify Platform Support

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)
- [CORE-EXC-001](core.rules.md#core-exc-001-must-document-rule-exceptions)

Newer platform features such as declarative shadow DOM, scoped registries, custom states, form-associated custom elements, constructable stylesheets, and state-preserving moves MUST have current compatibility proof for the supported browser set.

Unsupported or partially supported features MUST have feature detection, fallback behavior, or a documented rule exception.

Compatibility proof SHOULD be updated when the supported browser matrix changes.

# WC-DEPS-001 SHOULD Minimize Runtime Dependencies

See:
- [CORE-BUILD-001](core.rules.md#core-build-001-must-reproducible-toolchains)

Web Components SHOULD use platform DOM, CSS, and TypeScript APIs before adding browser runtime dependencies.

Frameworks, component registries, CSS frameworks, and generated UI runtimes MUST provide clear value beyond native custom elements before adoption.

Runtime dependency exceptions MUST document bundle impact, transitive dependency risk, alternatives considered, rollback plan, and browser tests.

# WC-TEST-001 MUST Test Real Browser Behavior

See:
- [CORE-TEST-001](core.rules.md#core-test-001-must-deterministic-tests)

Custom elements MUST be tested in a real browser for registration, lifecycle behavior, public properties, observed attributes, event contracts, keyboard interaction, accessibility-critical roles, and responsive layout.

Shadow DOM tests MUST assert through public APIs, user-visible behavior, slots, parts, attributes, events, or documented test hooks.

Tests MUST NOT depend on private shadow internals unless the internals are the unit under test.

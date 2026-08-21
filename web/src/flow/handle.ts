import { applyStyles } from "../elements/styles.ts";

import type { HandleKind, HandlePosition } from "./types.ts";
import { handleStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-handle";
const POSITION_ATTR = "position";
const KIND_ATTR = "kind";
const HANDLE_KIND_SOURCE: HandleKind = "source";
const HANDLE_KIND_TARGET: HandleKind = "target";
const HANDLE_POSITION_TOP: HandlePosition = "top";
const HANDLE_POSITION_RIGHT: HandlePosition = "right";
const HANDLE_POSITION_BOTTOM: HandlePosition = "bottom";
const HANDLE_POSITION_LEFT: HandlePosition = "left";

function coerceKind(value: string | null): HandleKind {
  return value === HANDLE_KIND_TARGET ? HANDLE_KIND_TARGET : HANDLE_KIND_SOURCE;
}

function coercePosition(value: string | null): HandlePosition {
  return value === HANDLE_POSITION_TOP
    || value === HANDLE_POSITION_RIGHT
    || value === HANDLE_POSITION_BOTTOM
    || value === HANDLE_POSITION_LEFT
    ? value
    : HANDLE_POSITION_RIGHT;
}

/**
 * Handle on a flow node.
 *
 * Default-on-connect: missing or invalid `position` serializes to `"right"`;
 * missing or invalid `kind` serializes to `"source"`. The attribute is the
 * source of truth. `observedAttributes` lists `position` and `kind`.
 * `attributeChangedCallback` and `connectedCallback` rewrite illegal values
 * and `removeAttribute` back to those defaults so the property, attribute,
 * and CSS `:host([position=right])` / `:host` default inset cannot desync.
 *
 * Inputs: `kind` and `position` attributes or properties. Outputs: reflected
 * attributes and `handleKind` / `handlePosition`. Ownership: the element owns
 * its attributes. Lifetime: connect until disconnect. Concurrency:
 * runtime-safe. Failure modes: invalid values coerce to the documented
 * defaults. Classification: runtime-safe.
 */
export class FlowHandle extends HTMLElement {
  static get observedAttributes(): string[] {
    return [POSITION_ATTR, KIND_ATTR];
  }

  constructor() {
    super();
    applyStyles(this.attachShadow({ mode: "open" }), handleStyles);
  }

  get handleKind(): HandleKind {
    return coerceKind(this.getAttribute(KIND_ATTR));
  }

  set handleKind(value: HandleKind) {
    this.setAttribute(KIND_ATTR, value);
  }

  get handlePosition(): HandlePosition {
    return coercePosition(this.getAttribute(POSITION_ATTR));
  }

  set handlePosition(value: HandlePosition) {
    this.setAttribute(POSITION_ATTR, value);
  }

  connectedCallback(): void {
    if (this.getAttribute(POSITION_ATTR) === null) this.setAttribute(POSITION_ATTR, HANDLE_POSITION_RIGHT);
    if (this.getAttribute(KIND_ATTR) === null) this.setAttribute(KIND_ATTR, HANDLE_KIND_SOURCE);
  }

  attributeChangedCallback(name: string, _previous: string | null, value: string | null): void {
    if (name === POSITION_ATTR) {
      const next = coercePosition(value);
      if (this.getAttribute(POSITION_ATTR) !== next) this.setAttribute(POSITION_ATTR, next);
      return;
    }
    if (name === KIND_ATTR) {
      const next = coerceKind(value);
      if (this.getAttribute(KIND_ATTR) !== next) this.setAttribute(KIND_ATTR, next);
    }
  }
}

export function registerFlowHandle(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowHandle);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-handle": FlowHandle;
  }
}

import { applyStyles } from "../elements/styles.ts";

import { resizeControlStyles } from "./styles.ts";
import type { ResizeDirection } from "./types.ts";

const ELEMENT_NAME = "flow-node-resize-control";
const DIRECTION_ATTR = "direction";
const DEFAULT_DIRECTION: ResizeDirection = "se";
const BUTTON_TYPE = "button";

export const RESIZE_DIRECTIONS: readonly ResizeDirection[] = ["n", "s", "e", "w", "ne", "nw", "se", "sw"];

const DIRECTION_NAME: Record<ResizeDirection, string> = {
  n: "north",
  s: "south",
  e: "east",
  w: "west",
  ne: "northeast",
  nw: "northwest",
  se: "southeast",
  sw: "southwest",
};

/**
 * True when `value` is one of the eight `ResizeDirection` literals.
 *
 * Inputs: unknown. Outputs: a type predicate. Ownership: none retained.
 * Lifetime: one call. Concurrency: synchronous. Failure modes: non-matching
 * values return false. Units: none. Classification: runtime-safe.
 */
export function isResizeDirection(value: unknown): value is ResizeDirection {
  return value === "n" || value === "s" || value === "e" || value === "w"
    || value === "ne" || value === "nw" || value === "se" || value === "sw";
}

function coerceDirection(value: string | null): ResizeDirection {
  return isResizeDirection(value) ? value : DEFAULT_DIRECTION;
}

/**
 * One resize handle on a flow node.
 *
 * Default-on-connect: missing or invalid `direction` serializes to `"se"`.
 * The attribute is the source of truth. `observedAttributes` lists `direction`.
 * `attributeChangedCallback` and `connectedCallback` rewrite illegal values
 * and `removeAttribute` back to that default so the property, attribute, and
 * CSS `:host([direction=se])` cannot desync.
 *
 * Inputs: `direction` attribute or property. Outputs: reflected attribute and
 * `direction`. Ownership: the element owns its attributes. Lifetime: connect
 * until disconnect. Concurrency: runtime-safe. Failure modes: invalid values
 * coerce to `"se"`. Classification: runtime-safe.
 */
export class FlowNodeResizeControl extends HTMLElement {
  static get observedAttributes(): string[] {
    return [DIRECTION_ATTR];
  }

  readonly #button: HTMLButtonElement;

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    applyStyles(root, resizeControlStyles);
    this.#button = document.createElement("button");
    this.#button.type = BUTTON_TYPE;
    this.#button.part.add("control");
    root.append(this.#button);
    this.#syncLabel();
  }

  get direction(): ResizeDirection {
    return coerceDirection(this.getAttribute(DIRECTION_ATTR));
  }

  set direction(value: ResizeDirection) {
    this.setAttribute(DIRECTION_ATTR, value);
  }

  connectedCallback(): void {
    if (this.getAttribute(DIRECTION_ATTR) === null) this.setAttribute(DIRECTION_ATTR, DEFAULT_DIRECTION);
    this.#syncLabel();
  }

  attributeChangedCallback(name: string, _previous: string | null, value: string | null): void {
    if (name !== DIRECTION_ATTR) return;
    const next = coerceDirection(value);
    if (this.getAttribute(DIRECTION_ATTR) !== next) this.setAttribute(DIRECTION_ATTR, next);
    this.#syncLabel();
  }

  #syncLabel(): void {
    this.#button.setAttribute("aria-label", `Resize ${DIRECTION_NAME[this.direction]}`);
  }
}

export function registerFlowNodeResizeControl(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) {
    customElements.define(ELEMENT_NAME, FlowNodeResizeControl);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-node-resize-control": FlowNodeResizeControl;
  }
}

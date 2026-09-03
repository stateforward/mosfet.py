import { applyStyles } from "../elements/styles.ts";

import { RESIZE_DIRECTIONS } from "./resize-control.ts";
import { nodeResizerStyles } from "./styles.ts";
import {
  MIN_RESIZE_HEIGHT,
  MIN_RESIZE_WIDTH,
  type ResizeConstraints,
} from "./types.ts";

const ELEMENT_NAME = "flow-node-resizer";
const MIN_WIDTH_ATTR = "min-width";
const MIN_HEIGHT_ATTR = "min-height";
const MAX_WIDTH_ATTR = "max-width";
const MAX_HEIGHT_ATTR = "max-height";
const ASPECT_ATTR = "keep-aspect-ratio";
const VISIBLE_ATTR = "visible";

function positiveNumber(args: { value: string | null; fallback: number }): number {
  if (args.value === null || args.value.length === 0) return args.fallback;
  const parsed = Number(args.value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : args.fallback;
}

function optionalPositiveNumber(value: string | null): number | undefined {
  if (value === null || value.length === 0) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

/**
 * Node resize chrome: eight `flow-node-resize-control` children.
 *
 * Inputs: primitive attributes `min-width`, `min-height`, `max-width`,
 * `max-height`, `keep-aspect-ratio`, `visible`. Outputs: eight controls and
 * `constraints()`. `visible` is `true` / `false` / absent (auto). Ownership:
 * this element owns its controls. Lifetime: construct until disconnect.
 * Concurrency: runtime-safe. Failure modes: illegal numeric attributes coerce
 * to documented defaults or are removed. Classification: runtime-safe.
 *
 * Keyboard contract: each of the eight controls is a focusable native button
 * labeled `Resize <direction>`. With a control's button focused, Enter or
 * Space starts a resize in that direction from the node's current bounds,
 * the arrow keys step-resize by 1px in world units per keypress (min/max and
 * aspect constraints apply through the shared clamp logic), and Escape or a
 * second Enter ends the resize. Progression is typed `resize_key` events on
 * the graph; Pointer `resize/keyboard` owns the session exclusively with
 * pointer resize. See `flow-node-resize-control` for the per-key mapping.
 */
export class FlowNodeResizer extends HTMLElement {
  static get observedAttributes(): string[] {
    return [MIN_WIDTH_ATTR, MIN_HEIGHT_ATTR, MAX_WIDTH_ATTR, MAX_HEIGHT_ATTR, ASPECT_ATTR, VISIBLE_ATTR];
  }

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    applyStyles(root, nodeResizerStyles);
    for (const direction of RESIZE_DIRECTIONS) {
      const control = document.createElement("flow-node-resize-control");
      control.direction = direction;
      control.part.add("control", direction);
      root.append(control);
    }
  }

  get minWidth(): number {
    return positiveNumber({ value: this.getAttribute(MIN_WIDTH_ATTR), fallback: MIN_RESIZE_WIDTH });
  }

  set minWidth(value: number) {
    this.setAttribute(MIN_WIDTH_ATTR, String(value));
  }

  get minHeight(): number {
    return positiveNumber({ value: this.getAttribute(MIN_HEIGHT_ATTR), fallback: MIN_RESIZE_HEIGHT });
  }

  set minHeight(value: number) {
    this.setAttribute(MIN_HEIGHT_ATTR, String(value));
  }

  get maxWidth(): number | undefined {
    return optionalPositiveNumber(this.getAttribute(MAX_WIDTH_ATTR));
  }

  set maxWidth(value: number | undefined) {
    if (value === undefined) this.removeAttribute(MAX_WIDTH_ATTR);
    else this.setAttribute(MAX_WIDTH_ATTR, String(value));
  }

  get maxHeight(): number | undefined {
    return optionalPositiveNumber(this.getAttribute(MAX_HEIGHT_ATTR));
  }

  set maxHeight(value: number | undefined) {
    if (value === undefined) this.removeAttribute(MAX_HEIGHT_ATTR);
    else this.setAttribute(MAX_HEIGHT_ATTR, String(value));
  }

  get keepAspectRatio(): boolean {
    return this.getAttribute(ASPECT_ATTR) !== null && this.getAttribute(ASPECT_ATTR) !== "false";
  }

  set keepAspectRatio(value: boolean) {
    if (value) this.setAttribute(ASPECT_ATTR, "");
    else this.removeAttribute(ASPECT_ATTR);
  }

  /**
   * Author override of chrome offer. `true` / `false` force show/hide;
   * `undefined` (attribute absent) lets the host derive from policy+selected.
   */
  get visible(): boolean | undefined {
    const value = this.getAttribute(VISIBLE_ATTR);
    if (value === null) return undefined;
    return value !== "false";
  }

  set visible(value: boolean | undefined) {
    if (value === undefined) this.removeAttribute(VISIBLE_ATTR);
    else this.setAttribute(VISIBLE_ATTR, value ? "true" : "false");
  }

  constraints(): ResizeConstraints {
    return {
      minWidth: this.minWidth,
      minHeight: this.minHeight,
      keepAspectRatio: this.keepAspectRatio,
      ...(this.maxWidth !== undefined ? { maxWidth: this.maxWidth } : {}),
      ...(this.maxHeight !== undefined ? { maxHeight: this.maxHeight } : {}),
    };
  }

  connectedCallback(): void {
    if (this.getAttribute(MIN_WIDTH_ATTR) === null) this.setAttribute(MIN_WIDTH_ATTR, String(MIN_RESIZE_WIDTH));
    if (this.getAttribute(MIN_HEIGHT_ATTR) === null) this.setAttribute(MIN_HEIGHT_ATTR, String(MIN_RESIZE_HEIGHT));
    this.#normalizeVisible();
    this.#notifyChange();
  }

  attributeChangedCallback(name: string, _previous: string | null, value: string | null): void {
    if (name === MIN_WIDTH_ATTR) {
      const next = String(positiveNumber({ value, fallback: MIN_RESIZE_WIDTH }));
      if (this.getAttribute(MIN_WIDTH_ATTR) !== next) this.setAttribute(MIN_WIDTH_ATTR, next);
      return;
    }
    if (name === MIN_HEIGHT_ATTR) {
      const next = String(positiveNumber({ value, fallback: MIN_RESIZE_HEIGHT }));
      if (this.getAttribute(MIN_HEIGHT_ATTR) !== next) this.setAttribute(MIN_HEIGHT_ATTR, next);
      return;
    }
    if (name === MAX_WIDTH_ATTR || name === MAX_HEIGHT_ATTR) {
      if (value === null) {
        this.#notifyChange();
        return;
      }
      if (optionalPositiveNumber(value) === undefined) this.removeAttribute(name);
      this.#notifyChange();
      return;
    }
    if (name === VISIBLE_ATTR) {
      this.#normalizeVisible();
      this.#notifyChange();
    }
  }

  #normalizeVisible(): void {
    const value = this.getAttribute(VISIBLE_ATTR);
    if (value === null) return;
    const next = value === "false" ? "false" : "true";
    if (value !== next) this.setAttribute(VISIBLE_ATTR, next);
  }

  #notifyChange(): void {
    this.dispatchEvent(new Event("flow-resizer-change"));
  }
}

export function registerFlowNodeResizer(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowNodeResizer);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-node-resizer": FlowNodeResizer;
  }
}

import { applyStyles } from "../elements/styles.ts";

import type { HandleKind, HandlePosition } from "./types.ts";
import { handleStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-handle";

export class FlowHandle extends HTMLElement {
  constructor() {
    super();
    applyStyles(this.attachShadow({ mode: "open" }), handleStyles);
  }

  get handleKind(): HandleKind {
    return this.getAttribute("kind") === "target" ? "target" : "source";
  }

  set handleKind(value: HandleKind) {
    this.setAttribute("kind", value);
  }

  get handlePosition(): HandlePosition {
    const value = this.getAttribute("position");
    return value === "top" || value === "right" || value === "bottom" || value === "left" ? value : "right";
  }

  set handlePosition(value: HandlePosition) {
    this.setAttribute("position", value);
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

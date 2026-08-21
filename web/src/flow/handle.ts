import * as hsm from "../hsm.ts";
import { applyStyles } from "../elements/styles.ts";

import type { HandleKind, HandlePosition } from "./types.ts";
import { handleStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-handle";

export class FlowHandle extends hsm.from(HTMLElement) {
  static readonly model = hsm.define(
    "FlowHandle",
    hsm.initial(hsm.target("idle")),
    hsm.state("idle"),
  );

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    applyStyles(root, handleStyles);
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

  connectedCallback(): void {
    hsm.start(this, FlowHandle.model);
  }

  disconnectedCallback(): void {
    void hsm.stop(this).catch(hsm.reportHsmFailure);
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

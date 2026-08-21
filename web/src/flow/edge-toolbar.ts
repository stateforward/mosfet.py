import { applyStyles } from "../elements/styles.ts";

import { toolbarStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-edge-toolbar";

export class FlowEdgeToolbar extends HTMLElement {
  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    applyStyles(root, toolbarStyles);
    root.append(document.createElement("slot"));
  }
}

export function registerFlowEdgeToolbar(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowEdgeToolbar);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-edge-toolbar": FlowEdgeToolbar;
  }
}

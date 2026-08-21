import { applyStyles } from "../elements/styles.ts";

import { toolbarStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-node-toolbar";

export class FlowNodeToolbar extends HTMLElement {
  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    applyStyles(root, toolbarStyles);
    root.append(document.createElement("slot"));
  }
}

export function registerFlowNodeToolbar(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowNodeToolbar);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-node-toolbar": FlowNodeToolbar;
  }
}

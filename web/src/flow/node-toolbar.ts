import * as hsm from "../hsm.ts";
import { applyStyles } from "../elements/styles.ts";

import { toolbarStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-node-toolbar";

export class FlowNodeToolbar extends hsm.from(HTMLElement) {
  static readonly model = hsm.define(
    "FlowNodeToolbar",
    hsm.initial(hsm.target("idle")),
    hsm.state("idle"),
  );

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    applyStyles(root, toolbarStyles);
    root.append(document.createElement("slot"));
  }

  connectedCallback(): void {
    hsm.start(this, FlowNodeToolbar.model);
  }

  disconnectedCallback(): void {
    void hsm.stop(this).catch(hsm.reportHsmFailure);
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

import * as hsm from "../hsm.ts";
import { applyStyles } from "../elements/styles.ts";

import { panelStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-panel";

export class FlowPanel extends hsm.from(HTMLElement) {
  static readonly model = hsm.define(
    "FlowPanel",
    hsm.initial(hsm.target("idle")),
    hsm.state("idle"),
  );

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    applyStyles(root, panelStyles);
    root.append(document.createElement("slot"));
  }

  connectedCallback(): void {
    if (this.getAttribute("position") === null) this.setAttribute("position", "top-left");
    hsm.start(this, FlowPanel.model);
  }

  disconnectedCallback(): void {
    void hsm.stop(this).catch(hsm.reportHsmFailure);
  }
}

export function registerFlowPanel(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowPanel);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-panel": FlowPanel;
  }
}

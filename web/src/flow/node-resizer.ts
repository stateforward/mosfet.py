import * as hsm from "../hsm.ts";
import { applyStyles } from "../elements/styles.ts";

import { resizerStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-node-resizer";

export class FlowNodeResizer extends hsm.from(HTMLElement) {
  static readonly model = hsm.define(
    "FlowNodeResizer",
    hsm.initial(hsm.target("idle")),
    hsm.state("idle"),
  );

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    applyStyles(root, resizerStyles);
    for (const dir of ["right", "bottom", "bottom-right"]) {
      const control = document.createElement("div");
      control.className = "control";
      control.dataset["dir"] = dir;
      root.append(control);
    }
  }

  connectedCallback(): void {
    hsm.start(this, FlowNodeResizer.model);
  }

  disconnectedCallback(): void {
    void hsm.stop(this).catch(hsm.reportHsmFailure);
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

import * as hsm from "../hsm.ts";
import { applyStyles } from "../elements/styles.ts";

import { backgroundStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-background";

export class FlowBackground extends hsm.from(HTMLElement) {
  static readonly model = hsm.define(
    "FlowBackground",
    hsm.initial(hsm.target("idle")),
    hsm.state("idle"),
  );

  static get observedAttributes(): string[] {
    return ["variant"];
  }

  constructor() {
    super();
    applyStyles(this.attachShadow({ mode: "open" }), backgroundStyles);
  }

  get variant(): "dots" | "lines" {
    return this.getAttribute("variant") === "lines" ? "lines" : "dots";
  }

  set variant(value: "dots" | "lines") {
    this.setAttribute("variant", value);
  }

  connectedCallback(): void {
    hsm.start(this, FlowBackground.model);
  }

  disconnectedCallback(): void {
    void hsm.stop(this).catch(hsm.reportHsmFailure);
  }
}

export function registerFlowBackground(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowBackground);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-background": FlowBackground;
  }
}

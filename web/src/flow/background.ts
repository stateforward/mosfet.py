import { applyStyles } from "../elements/styles.ts";

import { backgroundStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-background";

export class FlowBackground extends HTMLElement {
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
}

export function registerFlowBackground(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowBackground);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-background": FlowBackground;
  }
}

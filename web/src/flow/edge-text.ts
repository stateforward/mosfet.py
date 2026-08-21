import * as hsm from "../hsm.ts";

const ELEMENT_NAME = "flow-edge-text";

export class FlowEdgeText extends hsm.from(HTMLElement) {
  static readonly model = hsm.define(
    "FlowEdgeText",
    hsm.initial(hsm.target("idle")),
    hsm.state("idle"),
  );

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).append(document.createElement("slot"));
  }

  connectedCallback(): void {
    hsm.start(this, FlowEdgeText.model);
  }

  disconnectedCallback(): void {
    void hsm.stop(this).catch(hsm.reportHsmFailure);
  }
}

export function registerFlowEdgeText(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowEdgeText);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-edge-text": FlowEdgeText;
  }
}

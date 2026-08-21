import * as hsm from "../hsm.ts";

const ELEMENT_NAME = "flow-viewport-portal";

export class FlowViewportPortal extends hsm.from(HTMLElement) {
  static readonly model = hsm.define(
    "FlowViewportPortal",
    hsm.initial(hsm.target("idle")),
    hsm.state("idle"),
  );

  constructor() {
    super();
    this.style.position = "absolute";
    this.style.inset = "0";
    this.style.pointerEvents = "none";
    this.attachShadow({ mode: "open" }).append(document.createElement("slot"));
  }

  connectedCallback(): void {
    hsm.start(this, FlowViewportPortal.model);
  }

  disconnectedCallback(): void {
    void hsm.stop(this).catch(hsm.reportHsmFailure);
  }
}

export function registerFlowViewportPortal(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowViewportPortal);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-viewport-portal": FlowViewportPortal;
  }
}

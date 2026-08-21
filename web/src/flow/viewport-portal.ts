const ELEMENT_NAME = "flow-viewport-portal";

export class FlowViewportPortal extends HTMLElement {
  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = ":host { display: block; position: absolute; inset: 0; pointer-events: none; }";
    root.append(style, document.createElement("slot"));
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

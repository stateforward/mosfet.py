const ELEMENT_NAME = "flow-edge-text";

export class FlowEdgeText extends HTMLElement {
  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = ":host { display: block; }";
    root.append(style, document.createElement("slot"));
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

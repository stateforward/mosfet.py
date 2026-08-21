import * as hsm from "../hsm.ts";
import { applyStyles } from "../elements/styles.ts";

import { controlsStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-controls";

export class FlowControls extends hsm.from(HTMLElement) {
  static readonly model = hsm.define(
    "FlowControls",
    hsm.initial(hsm.target("idle")),
    hsm.state("idle"),
  );

  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    applyStyles(root, controlsStyles);
    root.append(
      controlButton("+", "zoom-in"),
      controlButton("−", "zoom-out"),
      controlButton("fit", "fit"),
    );
    root.addEventListener("click", this.#onClick);
  }

  connectedCallback(): void {
    if (this.getAttribute("position") === null) this.setAttribute("position", "bottom-left");
    hsm.start(this, FlowControls.model);
  }

  disconnectedCallback(): void {
    void hsm.stop(this).catch(hsm.reportHsmFailure);
  }

  #onClick = (event: Event): void => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    const action = target.dataset["action"];
    const graph = this.closest("flow-graph") ?? (this.getRootNode() as ShadowRoot).host;
    if (!(graph instanceof HTMLElement) || action === undefined) return;
    if (action === "zoom-in" && "zoomIn" in graph && typeof graph.zoomIn === "function") graph.zoomIn();
    if (action === "zoom-out" && "zoomOut" in graph && typeof graph.zoomOut === "function") graph.zoomOut();
    if (action === "fit" && "fitView" in graph && typeof graph.fitView === "function") graph.fitView();
  };
}

function controlButton(label: string, action: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.dataset["action"] = action;
  button.setAttribute("aria-label", action.replace("-", " "));
  return button;
}

export function registerFlowControls(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowControls);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-controls": FlowControls;
  }
}

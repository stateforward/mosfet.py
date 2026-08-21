import { applyStyles } from "../elements/styles.ts";

import { controlsStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-controls";

/**
 * Detail of the `flow-control` CustomEvent.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Postcondition: the control host does NOT zoom or fit. The click is reported
 * here with its `action`; applying zoom-in, zoom-out, or fit is the listener's
 * job, if it chooses to. `preventDefault()` has no effect because the event
 * cannot be canceled.
 */
export type FlowControlDetail = { readonly action: "zoom-in" | "zoom-out" | "fit" };

export class FlowControls extends HTMLElement {
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
  }

  #onClick = (event: Event): void => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    const action = target.dataset["action"];
    if (action !== "zoom-in" && action !== "zoom-out" && action !== "fit") return;
    this.dispatchEvent(new CustomEvent<FlowControlDetail>("flow-control", {
      detail: { action },
      bubbles: true,
      composed: true,
      cancelable: false,
    }));
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

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
      controlButton({ label: "+", action: "zoom-in" }),
      controlButton({ label: "−", action: "zoom-out" }),
      controlButton({ label: "fit", action: "fit" }),
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

const TEST_ID_ATTR = "data-testid";

/**
 * Stable accessible names for flow viewport controls.
 * `fit` is namespaced "Fit flow view" so it cannot collide case-insensitively
 * with the dashboard's "Fit environment map" button, which fits the
 * environment map canvas in the same host tree.
 */
const ACCESSIBLE_NAME: Record<FlowControlDetail["action"], string> = {
  "zoom-in": "zoom in",
  "zoom-out": "zoom out",
  fit: "Fit flow view",
};

function controlButton(args: { label: string; action: FlowControlDetail["action"] }): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = args.label;
  button.dataset["action"] = args.action;
  button.part.add(args.action);
  button.setAttribute(TEST_ID_ATTR, args.action);
  button.setAttribute("aria-label", ACCESSIBLE_NAME[args.action]);
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

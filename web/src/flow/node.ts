import { applyStyles } from "../elements/styles.ts";

import { copyNode, DEFAULT_NODE_HEIGHT, DEFAULT_NODE_WIDTH, type Node } from "./types.ts";
import { nodeStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-node";
const BUTTON_TYPE = "button";

export class FlowNode extends HTMLElement {
  readonly #root: ShadowRoot;
  readonly #button: HTMLButtonElement;
  readonly #label: HTMLSpanElement;
  #node: Node | null = null;

  constructor() {
    super();
    this.#root = this.attachShadow({ mode: "open" });
    applyStyles(this.#root, nodeStyles);
    this.#button = document.createElement("button");
    this.#button.type = BUTTON_TYPE;
    this.#button.part.add("control");
    this.#label = document.createElement("span");
    this.#label.className = "label node-badge";
    this.#label.part.add("badge");
    this.#label.setAttribute("data-testid", "node-badge");
    this.#button.append(this.#label);
    this.#root.append(document.createElement("slot"), this.#button);
  }

  get node(): Node | null {
    return this.#node;
  }

  set node(value: Node | null) {
    this.#node = value === null ? null : copyNode(value);
    this.#sync();
  }

  connectedCallback(): void {
    this.#sync();
  }

  override focus(options?: FocusOptions): void {
    this.#button.focus(options);
  }

  #sync(): void {
    const node = this.#node;
    if (node === null) return;
    this.id = node.id;
    const width = node.width ?? DEFAULT_NODE_WIDTH;
    const height = node.height ?? DEFAULT_NODE_HEIGHT;
    this.style.left = `${node.position.x}px`;
    this.style.top = `${node.position.y}px`;
    this.style.width = `${width}px`;
    this.style.height = `${height}px`;
    this.className = node.className ?? "";
    if (typeof node.data["className"] === "string") this.className = node.data["className"];
    this.part.add("node");
    this.setAttribute("data-testid", "state-node");
    this.toggleAttribute("selected", node.selected === true);
    const selected = node.selected === true;
    this.#button.setAttribute("aria-pressed", selected ? "true" : "false");
    const path = node.data["path"];
    const machineName = node.data["machineName"];
    if (typeof path === "string") this.dataset["path"] = path;
    if (typeof machineName === "string") this.dataset["machineName"] = machineName;
    const label = node.data["label"];
    this.#label.textContent = typeof label === "string" ? label : node.id;
    this.#button.setAttribute("aria-label", this.#label.textContent);
  }
}

export function registerFlowNode(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowNode);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-node": FlowNode;
  }
}

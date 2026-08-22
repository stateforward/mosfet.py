import { applyStyles } from "../elements/styles.ts";

import { FlowNodeResizer } from "./node-resizer.ts";
import {
  copyNode,
  DEFAULT_NODE_HEIGHT,
  DEFAULT_NODE_WIDTH,
  resizeOffered,
  type Node,
  type ResizeConstraints,
} from "./types.ts";
import { nodeStyles } from "./styles.ts";

const ELEMENT_NAME = "flow-node";
const BUTTON_TYPE = "button";

export class FlowNode extends HTMLElement {
  readonly #root: ShadowRoot;
  readonly #button: HTMLButtonElement;
  readonly #label: HTMLSpanElement;
  readonly #resizer: FlowNodeResizer;
  #node: Node | null = null;
  #resizable = true;

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
    this.#resizer = document.createElement("flow-node-resizer");
    this.#resizer.addEventListener("flow-resizer-change", this.#onResizerChange);
    this.#root.append(document.createElement("slot"), this.#button, this.#resizer);
  }

  /**
   * Reflected `FlowGraph.nodesResizable` policy for this node.
   *
   * Inputs: boolean policy. Outputs: the stored policy used by `resizeOffered`
   * with `selected` and `flow-node-resizer.visible`. Ownership: this host owns
   * the field; the graph writes it on paint. Lifetime: until the next set.
   * Concurrency: runtime-safe. Failure modes: none. Classification: runtime-safe.
   */
  get resizable(): boolean {
    return this.#resizable;
  }

  set resizable(value: boolean) {
    this.#resizable = value;
    this.#syncResizer();
  }

  /**
   * Constraints from the hosted `flow-node-resizer`.
   *
   * Inputs: none. Outputs: a new `ResizeConstraints` record. Ownership: caller
   * owns the result. Lifetime: one call. Concurrency: runtime-safe.
   * Failure modes: illegal attributes already coerced on the resizer.
   * Classification: runtime-safe.
   */
  resizeConstraints(): ResizeConstraints {
    return this.#resizer.constraints();
  }

  /**
   * Snapshot of the painted node, or null when none is set.
   *
   * Inputs: none. Outputs: `copyNode` of the host-owned paint record, or null.
   * Ownership: this element owns the stored copy. The getter returns a new
   * copy; mutating `position` or `data` on the result does not change stored
   * paint state or derived DOM. Lifetime: until the next `node` set.
   * Concurrency: runtime-safe. Failure modes: `copyNode` `{ ok: false }`
   * (invalid position, cyclic, over-deep, or non-record `data`) returns null
   * and does not throw.
   * Classification: runtime-safe.
   */
  get node(): Node | null {
    if (this.#node === null) return null;
    const copied = copyNode(this.#node);
    return copied.ok ? copied.value : null;
  }

  /**
   * Store a copy of `value` as paint source state and sync derived DOM.
   *
   * Inputs: `value` is copied with `copyNode` at write time. Ownership: this
   * element owns the stored copy. Later mutation of the caller or of a getter
   * result does not change stored paint state. Lifetime: until the next set.
   * Concurrency: runtime-safe. Failure modes: `copyNode` `{ ok: false }`
   * (invalid position, cyclic, over-deep, or non-record `data`) leaves stored
   * paint unchanged, does not call `#sync` with invalid data, and does not throw.
   * Classification: runtime-safe.
   */
  set node(value: Node | null) {
    if (value === null) {
      this.#node = null;
      this.#sync();
      return;
    }
    const copied = copyNode(value);
    if (!copied.ok) return;
    this.#node = copied.value;
    this.#sync();
  }

  connectedCallback(): void {
    this.#sync();
  }

  override focus(options?: FocusOptions): void {
    this.#button.focus(options);
  }

  readonly #onResizerChange = (): void => {
    this.#syncResizer();
  };

  #sync(): void {
    const node = this.#node;
    if (node === null) {
      this.#syncResizer();
      return;
    }
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
    // Empty labels (machine-graph initial nodes) keep an empty visible badge
    // but still expose a stable accessible name from the node id.
    const accessible = this.#label.textContent;
    this.#button.setAttribute("aria-label", accessible.length > 0 ? accessible : node.id);
    this.#syncResizer();
  }

  #syncResizer(): void {
    const offered = resizeOffered({
      policy: this.#resizable,
      selected: this.#node?.selected === true,
      visible: this.#resizer.visible,
    });
    this.#resizer.hidden = !offered;
    this.#resizer.setAttribute("aria-hidden", offered ? "false" : "true");
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

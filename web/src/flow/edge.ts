import * as hsm from "../hsm.ts";

import { edgePath } from "./path.ts";
import { copyEdge, type Edge, type Node } from "./types.ts";

const ELEMENT_NAME = "flow-edge";
const SVG_NS = "http://www.w3.org/2000/svg";

export class FlowEdge extends HTMLElement {
  readonly path: SVGPathElement;
  readonly hit: SVGPathElement;
  readonly label: SVGTextElement;
  #edge: Edge | null = null;

  constructor() {
    super();
    this.path = document.createElementNS(SVG_NS, "path");
    this.path.classList.add("edge-path");
    this.path.setAttribute("data-testid", "edge-path");
    this.path.setAttribute("part", "path");
    this.hit = document.createElementNS(SVG_NS, "path");
    this.hit.classList.add("edge-hit");
    this.label = document.createElementNS(SVG_NS, "text");
    this.label.classList.add("edge-label");
    this.label.setAttribute("data-testid", "edge-label");
    this.label.setAttribute("part", "label");
  }

  /**
   * Snapshot of the painted edge, or null when none is set.
   *
   * Inputs: none. Outputs: `copyEdge` of the host-owned paint record, or null.
   * Ownership: this element owns the stored copy. The getter returns a new
   * copy; mutating nested `data` on the result does not change stored paint
   * state. Lifetime: until the next `edge` set. Concurrency: runtime-safe.
   * Failure modes: none. Classification: runtime-safe.
   */
  get edge(): Edge | null {
    return this.#edge === null ? null : copyEdge(this.#edge);
  }

  /**
   * Store a copy of `value` as paint source state.
   *
   * Inputs: `value` is copied with `copyEdge` at write time. Ownership: this
   * element owns the stored copy. Later mutation of the caller or of a getter
   * result does not change stored paint state. Lifetime: until the next set.
   * Concurrency: runtime-safe. Failure modes: none. Classification: runtime-safe.
   */
  set edge(value: Edge | null) {
    this.#edge = value === null ? null : copyEdge(value);
  }

  connectedCallback(): void {
    if ((this.style.display ?? "") === "") {
      this.style.display = "contents";
    }
    this.mount();
  }

  disconnectedCallback(): void {
    this.path.remove();
    this.hit.remove();
    this.label.remove();
  }

  mount(layer?: ParentNode | null): void {
    const parent = layer ?? this.parentElement?.querySelector(".edge-layer") ?? this.parentNode;
    if (parent === null || parent === undefined) return;
    parent.append(this.path, this.hit, this.label);
  }

  paint(source: Node, target: Node): void {
    const edge = this.#edge;
    if (edge === null) return;
    const custom = edge.data?.["d"];
    let d: string;
    let labelX: number;
    let labelY: number;
    if (typeof custom === "string") {
      d = custom;
      const label = edge.data?.["labelPosition"];
      if (hsm.isRecord(label) && typeof label["x"] === "number" && typeof label["y"] === "number") {
        labelX = label["x"];
        labelY = label["y"];
      } else {
        labelX = (source.position.x + target.position.x) / 2;
        labelY = (source.position.y + target.position.y) / 2;
      }
    } else {
      const sourceX = source.position.x + (source.width ?? 0) / 2;
      const sourceY = source.position.y + (source.height ?? 0);
      const targetX = target.position.x + (target.width ?? 0) / 2;
      const targetY = target.position.y;
      [d, labelX, labelY] = edgePath(edge.type, { sourceX, sourceY, targetX, targetY });
    }
    this.path.setAttribute("d", d);
    this.hit.setAttribute("d", d);
    this.path.setAttribute("class", `edge-path ${edge.className ?? ""} ${edge.type ?? "bezier"}`);
    if (edge.data?.["lastFired"] === true) this.path.classList.add("last-fired");
    else this.path.classList.remove("last-fired");
    if (edge.label !== undefined && edge.label.length > 0) {
      this.label.textContent = edge.label;
      this.label.setAttribute("x", String(labelX));
      this.label.setAttribute("y", String(labelY));
      this.label.style.display = "";
    } else {
      this.label.textContent = "";
      this.label.style.display = "none";
    }
    if (typeof edge.data?.["eventName"] === "string") this.hit.dataset["eventName"] = edge.data["eventName"];
  }
}

export function registerFlowEdge(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, FlowEdge);
}

declare global {
  interface HTMLElementTagNameMap {
    "flow-edge": FlowEdge;
  }
}

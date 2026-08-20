import { MachineGraphController } from "../../machine-graph-hsm.ts";
import { type MachineGraph } from "../../otel/machines.ts";
import { applyStyles } from "../styles.ts";
import { NativeGraphRenderer } from "./renderer.ts";
import { graphStyles } from "./styles.ts";

const ELEMENT_NAME = "bot-machine-graph";
const STOPPED_CONTROLLER_ERROR = "MachineGraphController is stopped";

function isExpectedControllerStop(error: unknown): boolean {
  return error instanceof Error && error.message === STOPPED_CONTROLLER_ERROR;
}

export type GraphZoomDetail = { zoom: number };
export type GraphEdgeDetail = { eventName: string };

export class BotMachineGraph extends HTMLElement {
  readonly #root: ShadowRoot;
  readonly #frame: HTMLDivElement;
  #controller: MachineGraphController | null = null;
  #renderer: NativeGraphRenderer | null = null;
  #pending: readonly MachineGraph[] | undefined;
  #pendingFocus: string | undefined;
  #connectionGeneration = 0;

  constructor() {
    super();
    this.#root = this.attachShadow({ mode: "open" });
    applyStyles(this.#root, graphStyles);
    this.#frame = document.createElement("div");
    this.#frame.className = "frame";
    this.#frame.part.add("frame");
    this.#frame.setAttribute("data-testid", "frame");
    this.#root.append(this.#frame);
  }

  get graphs(): readonly MachineGraph[] {
    return this.#controller?.snapshot().graphs ?? this.#pending ?? [];
  }

  set graphs(value: readonly MachineGraph[]) {
    this.#pending = value;
    this.setAttribute("data-node-count", String(value.reduce((count, graph) => count + graph.nodes.length, 0)));
    void this.#admit(value, this.#connectionGeneration);
  }

  fit(): void {
    this.#pendingFocus = undefined;
    this.#controller?.fit();
  }

  focusMachine(machineName: string): void {
    this.#pendingFocus = machineName;
    if (this.#controller?.focusMachine(machineName) === true) this.#pendingFocus = undefined;
  }

  connectedCallback(): void {
    const generation = ++this.#connectionGeneration;
    if (this.#renderer === null) {
      this.#renderer = new NativeGraphRenderer(
        this.#frame,
        (zoom) => this.dispatchEvent(new CustomEvent<GraphZoomDetail>("bot-machine-graph-zoom", {
          detail: { zoom },
          bubbles: true,
          composed: true,
        })),
        (eventName) => this.dispatchEvent(new CustomEvent<GraphEdgeDetail>("bot-machine-graph-edge", {
          detail: { eventName },
          bubbles: true,
          composed: true,
        })),
      );
    }
    if (this.#controller === null) {
      this.#controller = new MachineGraphController({ renderer: this.#renderer });
      this.#renderer.setInteractionDispatcher((eventName, data) => {
        const controller = this.#controller;
        if (controller === null) return;
        return controller.dispatch(eventName, data).then(() => undefined).catch((error: unknown) => {
          if (isExpectedControllerStop(error)) return;
          throw error;
        });
      });
    }
    if (this.#pending !== undefined) void this.#admit(this.#pending, generation);
  }

  disconnectedCallback(): void {
    const controller = this.#controller;
    const renderer = this.#renderer;
    this.#connectionGeneration += 1;
    this.#controller = null;
    this.#renderer = null;
    renderer?.dispose();
    if (controller !== null) void controller.stop();
    this.#frame.replaceChildren();
  }

  async #admit(value: readonly MachineGraph[], generation: number): Promise<void> {
    const controller = this.#controller;
    const isCurrent = (): boolean => generation === this.#connectionGeneration && controller === this.#controller;
    if (controller === null || !isCurrent()) return;
    try {
      if (value.length === 0) {
        await controller.dispatch("graph.clear");
      } else {
        await controller.dispatch("graph.set", { graphs: value });
      }
    } catch (error) {
      if (!isCurrent() && isExpectedControllerStop(error)) return;
      throw error;
    }
    if (!isCurrent()) return;
    if (this.#pendingFocus !== undefined && this.#controller?.focusMachine(this.#pendingFocus) === true) {
      this.#pendingFocus = undefined;
    }
  }
}

export function registerBotMachineGraph(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) customElements.define(ELEMENT_NAME, BotMachineGraph);
}

declare global {
  interface HTMLElementTagNameMap {
    "bot-machine-graph": BotMachineGraph;
  }
}

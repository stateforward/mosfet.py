import * as hsm from "@stateforward/hsm.ts";

import { isRecord, namedEvent, startMachine, stopMachine } from "./hsm-runtime.ts";
import { parseMachineGraph, type MachineGraph } from "./otel/machines.ts";

const graphEvents = {
  "graph.set": { name: "graph.set", kind: hsm.Kinds.Event },
  "graph.clear": { name: "graph.clear", kind: hsm.Kinds.Event },
} as const;

export type MachineGraphEventName = keyof typeof graphEvents;

export type MachineGraphPhase = "empty" | "drawing";

export type GraphRenderer = {
  draw(graphs: readonly MachineGraph[]): void;
  destroy(): void;
};

export type MachineGraphSnapshot = {
  readonly phase: MachineGraphPhase;
  readonly statePath: string;
  readonly graphs: readonly MachineGraph[];
};

export type MachineGraphControllerOptions = {
  readonly renderer?: GraphRenderer;
  readonly onSnapshot?: (snapshot: MachineGraphSnapshot) => void;
};

class MachineGraphRuntime extends hsm.Instance {
  controller: MachineGraphController | null = null;
}

function controllerOf(instance: hsm.Instance): MachineGraphController | null {
  return instance instanceof MachineGraphRuntime ? instance.controller : null;
}

function parseGraphs(value: unknown): MachineGraph[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const graphs: MachineGraph[] = [];
  for (const item of value) {
    const graph = parseMachineGraph(item);
    if (graph === null) {
      return null;
    }
    graphs.push(graph);
  }
  return graphs;
}

function graphsFromEvent(event: hsm.Event): MachineGraph[] | null {
  return isRecord(event.data) ? parseGraphs(event.data["graphs"]) : null;
}

function rememberGraph(_ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event): void {
  const graphs = graphsFromEvent(event);
  if (graphs === null) {
    return;
  }
  controllerOf(instance)?.rememberGraphs(graphs);
}

function clearGraph(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.rememberGraphs([]);
}

function drawGraph(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.draw();
}

function destroyGraph(_ctx: hsm.Context, instance: hsm.Instance, _event: hsm.Event): void {
  controllerOf(instance)?.destroyRenderer();
}

const machineGraphModel = hsm.define(
  "MachineGraph",
  hsm.initial(hsm.target("empty")),
  hsm.state(
    "empty",
    hsm.entry(destroyGraph),
    hsm.transition(hsm.on("graph.clear"), hsm.target("."), hsm.effect(clearGraph)),
    hsm.transition(hsm.on("graph.set"), hsm.target("../drawing"), hsm.effect(rememberGraph)),
  ),
  hsm.state(
    "drawing",
    hsm.entry(drawGraph),
    hsm.exit(destroyGraph),
    hsm.transition(hsm.on("graph.set"), hsm.target("."), hsm.effect(rememberGraph)),
    hsm.transition(hsm.on("graph.clear"), hsm.target("../empty"), hsm.effect(clearGraph)),
  ),
);

function phaseFromStatePath(statePath: string): MachineGraphPhase {
  if (statePath.endsWith("/drawing")) {
    return "drawing";
  }
  return "empty";
}

export function isMachineGraphEventName(value: string): value is MachineGraphEventName {
  return Object.hasOwn(graphEvents, value);
}

export class MachineGraphController {
  #runtime = new MachineGraphRuntime();
  #machine: MachineGraphRuntime;
  #graphs: readonly MachineGraph[] = [];
  #renderer: GraphRenderer | null;
  #onSnapshot: ((snapshot: MachineGraphSnapshot) => void) | null;

  constructor(options: MachineGraphControllerOptions = {}) {
    this.#renderer = options.renderer ?? null;
    this.#onSnapshot = options.onSnapshot ?? null;
    this.#runtime.controller = this;
    this.#machine = startMachine(this.#runtime, machineGraphModel);
  }

  snapshot(): MachineGraphSnapshot {
    const statePath = this.#machine.takeSnapshot().state;
    return {
      phase: phaseFromStatePath(statePath),
      statePath,
      graphs: this.#graphs,
    };
  }

  async dispatch(eventName: MachineGraphEventName, data?: unknown): Promise<MachineGraphSnapshot> {
    if (eventName === "graph.set") {
      const graphs = isRecord(data) ? parseGraphs(data["graphs"]) : null;
      if (graphs === null || graphs.length === 0) {
        await this.#machine.dispatch(namedEvent(graphEvents["graph.clear"].name));
        this.#emit();
        return this.snapshot();
      }
    }
    await this.#machine.dispatch(namedEvent(graphEvents[eventName].name, data));
    this.#emit();
    return this.snapshot();
  }

  async stop(): Promise<void> {
    this.destroyRenderer();
    this.#runtime.controller = null;
    await stopMachine(this.#machine);
  }

  rememberGraphs(graphs: readonly MachineGraph[]): void {
    this.#graphs = graphs;
    this.#emit();
  }

  draw(): void {
    if (this.#graphs.length === 0) {
      this.#renderer?.destroy();
      return;
    }
    this.#renderer?.draw(this.#graphs);
  }

  destroyRenderer(): void {
    this.#renderer?.destroy();
  }

  #emit(): void {
    this.#onSnapshot?.(this.snapshot());
  }
}

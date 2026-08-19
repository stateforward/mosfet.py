import { type MachineGraph } from "./otel/machines.ts";

function ownerShell(graph: MachineGraph): MachineGraph {
  const root = graph.nodes.find((node) => node.path === graph.name && node.parent === null);
  const rootNode = {
    path: graph.name,
    parent: null,
    label: root?.label ?? graph.componentName,
  };
  return {
    ...graph,
    currentState: "",
    lastEventName: "",
    nodes: [rootNode],
    edges: [],
    observationCount: 0,
  };
}

function hasRoot(graph: MachineGraph): boolean {
  return graph.nodes.some((node) => node.path === graph.name && node.parent === null);
}

function renderableOwnerChain(
  machine: MachineGraph,
  byName: ReadonlyMap<string, MachineGraph>,
): string[] | null {
  if (!hasRoot(machine)) {
    return null;
  }
  const ownerChain = [machine.name];
  const visitedOwners = new Set<string>(ownerChain);
  let ownerName = machine.owner;
  while (ownerName !== undefined && ownerName !== null) {
    if (visitedOwners.has(ownerName)) {
      return null;
    }
    visitedOwners.add(ownerName);
    const owner = byName.get(ownerName);
    if (owner === undefined || !hasRoot(owner)) {
      return null;
    }
    ownerChain.push(owner.name);
    ownerName = owner.owner;
  }
  return ownerChain;
}

/** Direct, ownerless graphs that can be rendered as environment members. */
export function environmentRootGraphs(machines: readonly MachineGraph[]): MachineGraph[] {
  return machines.filter(
    (machine) => (machine.owner === undefined || machine.owner === null) && hasRoot(machine),
  );
}

/** All renderable graphs in one direct environment member's owned subtree. */
export function machineNamesInOwnedSubtree(
  machines: readonly MachineGraph[],
  rootName: string,
): Set<string> {
  const byName = new Map(machines.map((machine) => [machine.name, machine]));
  const names = new Set<string>();
  for (const machine of machines) {
    const ownerChain = renderableOwnerChain(machine, byName);
    if (ownerChain !== null && ownerChain.includes(rootName)) {
      names.add(machine.name);
    }
  }
  return names;
}

export function graphsForVisibility(
  machines: readonly MachineGraph[],
  visibleMachines: ReadonlyMap<string, boolean>,
): MachineGraph[] {
  const byName = new Map(machines.map((machine) => [machine.name, machine]));
  const included = new Set<string>();
  for (const machine of machines) {
    if (visibleMachines.get(machine.name) !== true) {
      continue;
    }
    const ownerChain = renderableOwnerChain(machine, byName);
    if (ownerChain === null) {
      continue;
    }
    for (const name of ownerChain) {
      included.add(name);
    }
  }
  return machines
    .filter((machine) => included.has(machine.name))
    .map((machine) => (visibleMachines.get(machine.name) === true ? machine : ownerShell(machine)));
}

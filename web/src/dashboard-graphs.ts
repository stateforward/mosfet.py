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
    const ownerChain = [machine.name];
    const visitedOwners = new Set<string>(ownerChain);
    let ownerName = machine.owner;
    let validOwnerChain = true;
    while (ownerName !== undefined && ownerName !== null) {
      if (visitedOwners.has(ownerName)) {
        validOwnerChain = false;
        break;
      }
      visitedOwners.add(ownerName);
      const owner = byName.get(ownerName);
      if (owner === undefined || !hasRoot(owner)) {
        validOwnerChain = false;
        break;
      }
      ownerChain.push(owner.name);
      ownerName = owner.owner;
    }
    if (!validOwnerChain) {
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

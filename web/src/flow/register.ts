import { registerFlowBackground } from "./background.ts";
import { registerFlowControls } from "./controls.ts";
import { registerFlowEdge } from "./edge.ts";
import { registerFlowGraph } from "./graph.ts";
import { registerFlowHandle } from "./handle.ts";
import { registerFlowMinimap } from "./minimap.ts";
import { registerFlowNode } from "./node.ts";
import { registerFlowNodeResizer } from "./node-resizer.ts";
import { registerFlowNodeResizeControl } from "./resize-control.ts";

export function registerFlowElements(): void {
  registerFlowHandle();
  registerFlowNodeResizeControl();
  registerFlowNodeResizer();
  registerFlowNode();
  registerFlowEdge();
  registerFlowBackground();
  registerFlowControls();
  registerFlowMinimap();
  registerFlowGraph();
}

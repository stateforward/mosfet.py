import { registerFlowBackground } from "./background.ts";
import { registerFlowControls } from "./controls.ts";
import { registerFlowEdge } from "./edge.ts";
import { registerFlowEdgeText } from "./edge-text.ts";
import { registerFlowEdgeToolbar } from "./edge-toolbar.ts";
import { registerFlowGraph } from "./graph.ts";
import { registerFlowHandle } from "./handle.ts";
import { registerFlowMinimap } from "./minimap.ts";
import { registerFlowNode } from "./node.ts";
import { registerFlowNodeResizer } from "./node-resizer.ts";
import { registerFlowNodeToolbar } from "./node-toolbar.ts";
import { registerFlowPanel } from "./panel.ts";
import { registerFlowViewportPortal } from "./viewport-portal.ts";

export function registerFlowElements(): void {
  registerFlowHandle();
  registerFlowNode();
  registerFlowEdge();
  registerFlowBackground();
  registerFlowControls();
  registerFlowMinimap();
  registerFlowPanel();
  registerFlowNodeResizer();
  registerFlowNodeToolbar();
  registerFlowEdgeToolbar();
  registerFlowEdgeText();
  registerFlowViewportPortal();
  registerFlowGraph();
}

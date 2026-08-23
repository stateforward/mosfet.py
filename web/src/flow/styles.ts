export const graphStyles = `
:host {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 12rem;
  position: relative;
  overflow: hidden;
  background: #0b0d12;
  touch-action: none;
  user-select: none;
}
.viewport, .world {
  position: absolute;
  inset: 0;
}
.viewport { overflow: hidden; }
.world {
  inset: auto;
  transform-origin: 0 0;
  will-change: transform;
}
.edge-layer {
  position: absolute;
  inset: 0;
  overflow: visible;
  pointer-events: none;
}
.node-layer { position: absolute; inset: 0; }
.connection-line {
  fill: none;
  stroke: #2dd4bf;
  stroke-width: 1.5;
  stroke-dasharray: 4 4;
  pointer-events: none;
}
.selection-box {
  position: absolute;
  border: 1px dashed #2dd4bf;
  background: color-mix(in srgb, #2dd4bf 12%, transparent);
  pointer-events: none;
  z-index: 20;
}
.is-dragging { cursor: grabbing; }
.edge-list {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
`;

export const nodeStyles = `
:host {
  position: absolute;
  box-sizing: border-box;
  display: grid;
  min-width: 2rem;
  min-height: 1.5rem;
  padding: 0;
  border: 0;
  background: transparent;
}
button {
  box-sizing: border-box;
  display: grid;
  place-items: center;
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0.35rem 0.55rem;
  border: 1px solid #3d4a5c;
  border-radius: 8px;
  background: #161b22;
  color: #d5dbe8;
  font: 500 12px/1.2 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  cursor: pointer;
}
:host([selected]) button {
  border-color: #2dd4bf;
  box-shadow: 0 0 0 2px rgba(45, 212, 191, 0.18);
}
button:focus-visible {
  outline: 2px solid #2dd4bf;
  outline-offset: 2px;
}
.label { pointer-events: none; }
`;

export const edgeStyles = `
:host { display: contents; }
`;

export const handleStyles = `
:host {
  display: block;
  position: absolute;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #2dd4bf;
  border: 1px solid #0b0d12;
  z-index: 5;
  right: -4px;
  top: 50%;
  transform: translateY(-50%);
}
:host([position="top"]) { top: -4px; right: auto; left: 50%; transform: translateX(-50%); }
:host([position="right"]) { right: -4px; top: 50%; left: auto; transform: translateY(-50%); }
:host([position="bottom"]) { bottom: -4px; right: auto; left: 50%; top: auto; transform: translateX(-50%); }
:host([position="left"]) { left: -4px; top: 50%; right: auto; transform: translateY(-50%); }
`;

export const nodeResizerStyles = `
:host {
  display: block;
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 6;
}
:host([hidden]) {
  display: none;
  pointer-events: none;
}
`;

export const resizeControlStyles = `
:host {
  display: block;
  position: absolute;
  box-sizing: border-box;
  pointer-events: auto;
  background: #2dd4bf;
  border: 1px solid #0b0d12;
  z-index: 7;
}
button {
  display: block;
  width: 100%;
  height: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: inherit;
}
button:focus-visible {
  outline: 2px solid #e8eaef;
  outline-offset: 2px;
}
:host([direction="n"]), :host([direction="s"]) {
  left: 8px;
  right: 8px;
  height: 6px;
  cursor: ns-resize;
}
:host([direction="n"]) { top: -3px; }
:host([direction="s"]) { bottom: -3px; }
:host([direction="e"]), :host([direction="w"]) {
  top: 8px;
  bottom: 8px;
  width: 6px;
  cursor: ew-resize;
}
:host([direction="e"]) { right: -3px; }
:host([direction="w"]) { left: -3px; }
:host([direction="ne"]), :host([direction="nw"]),
:host([direction="se"]), :host([direction="sw"]) {
  width: 8px;
  height: 8px;
}
:host([direction="ne"]) { top: -4px; right: -4px; cursor: nesw-resize; }
:host([direction="nw"]) { top: -4px; left: -4px; cursor: nwse-resize; }
:host([direction="se"]) { bottom: -4px; right: -4px; cursor: nwse-resize; }
:host([direction="sw"]) { bottom: -4px; left: -4px; cursor: nesw-resize; }
`;

export const backgroundStyles = `
:host {
  display: block;
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background-color: #0b0d12;
  background-image: radial-gradient(rgba(232, 234, 239, 0.07) 1px, transparent 1px);
  background-size: 16px 16px;
}
:host([variant="lines"]) {
  background-image:
    linear-gradient(rgba(232, 234, 239, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(232, 234, 239, 0.05) 1px, transparent 1px);
}
`;

export const controlsStyles = `
:host {
  position: absolute;
  z-index: 6;
  display: grid;
  gap: 0.25rem;
  padding: 0.3rem;
}
:host([position="bottom-left"]) { left: 0.6rem; bottom: 0.6rem; }
:host([position="bottom-right"]) { right: 0.6rem; bottom: 0.6rem; }
:host([position="top-left"]) { left: 0.6rem; top: 0.6rem; }
:host([position="top-right"]) { right: 0.6rem; top: 0.6rem; }
button {
  width: 1.7rem;
  height: 1.7rem;
  border: 1px solid #2a3140;
  border-radius: 0.3rem;
  background: #161922;
  color: #e8eaef;
  cursor: pointer;
}
`;

export const minimapStyles = `
:host {
  display: block;
  --flow-minimap-fill: #1d2430;
  position: absolute;
  right: 0.6rem;
  bottom: 0.6rem;
  z-index: 6;
  width: 8rem;
  height: 5.5rem;
  border: 1px solid #2a3140;
  border-radius: 0.35rem;
  background: #12141a;
  overflow: hidden;
}
canvas { width: 100%; height: 100%; display: block; }
`;

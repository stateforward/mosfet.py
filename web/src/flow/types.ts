export type XYPosition = { readonly x: number; readonly y: number };

export type Viewport = { readonly x: number; readonly y: number; readonly zoom: number };

export type Rect = {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
};

export type HandlePosition = "top" | "right" | "bottom" | "left";

export type HandleKind = "source" | "target";

export type EdgeType = "bezier" | "straight" | "step" | "smoothstep";

export type Node = {
  readonly id: string;
  readonly position: XYPosition;
  readonly data: Record<string, unknown>;
  readonly selected?: boolean;
  readonly parentId?: string;
  readonly type?: string;
  readonly width?: number;
  readonly height?: number;
  readonly className?: string;
};

export type Edge = {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly sourceHandle?: string;
  readonly targetHandle?: string;
  readonly type?: EdgeType | string;
  readonly label?: string;
  readonly selected?: boolean;
  readonly data?: Record<string, unknown>;
  readonly className?: string;
};

export type NodeClickDetail = { readonly node: Node; readonly originalEvent: Event };
export type EdgeClickDetail = { readonly edge: Edge; readonly originalEvent: Event };
export type ViewportChangeDetail = { readonly viewport: Viewport };
export type ConnectDetail = {
  readonly source: string;
  readonly target: string;
  readonly sourceHandle?: string;
  readonly targetHandle?: string;
};
export type SelectionChangeDetail = {
  readonly nodes: readonly Node[];
  readonly edges: readonly Edge[];
};

export const DEFAULT_NODE_WIDTH = 150;
export const DEFAULT_NODE_HEIGHT = 40;

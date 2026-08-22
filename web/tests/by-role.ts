type QueryNode = {
  readonly localName: string;
  readonly id?: string;
  readonly type?: string;
  readonly textContent: string;
  readonly shadowRoot?: QueryNode | null;
  readonly childNodes: ArrayLike<QueryNode>;
  getAttribute(name: string): string | null;
};

function isQueryNode(value: unknown): value is QueryNode {
  return typeof value === "object"
    && value !== null
    && typeof (value as QueryNode).localName === "string"
    && typeof (value as QueryNode).getAttribute === "function";
}

function visit(node: QueryNode, found: QueryNode[]): void {
  found.push(node);
  if (node.shadowRoot !== null && node.shadowRoot !== undefined) visit(node.shadowRoot, found);
  for (const child of Array.from(node.childNodes)) {
    if (isQueryNode(child)) visit(child, found);
  }
}

function collect(root: QueryNode): QueryNode[] {
  const found: QueryNode[] = [];
  if (root.shadowRoot !== null && root.shadowRoot !== undefined) visit(root.shadowRoot, found);
  for (const child of Array.from(root.childNodes)) {
    if (isQueryNode(child)) visit(child, found);
  }
  return found;
}

function nodeById(root: QueryNode, id: string): QueryNode | null {
  if (root.getAttribute("id") === id || root.id === id) return root;
  for (const node of collect(root)) {
    if (node.getAttribute("id") === id || node.id === id) return node;
  }
  return null;
}

function implicitRole(node: QueryNode): string | null {
  const explicit = node.getAttribute("role");
  if (explicit !== null && explicit.length > 0) return explicit;
  if (node.localName === "button") return "button";
  if (node.localName === "select") return "combobox";
  if (node.localName === "textarea") return "textbox";
  if (node.localName === "img" || node.localName === "canvas") return "img";
  if (node.localName !== "input") return null;
  const type = node.getAttribute("type") ?? node.type ?? "text";
  if (type === "checkbox") return "checkbox";
  if (type === "range") return "slider";
  if (type === "button" || type === "submit") return "button";
  return "textbox";
}

function accessibleName(root: QueryNode, node: QueryNode): string {
  const labelledBy = node.getAttribute("aria-labelledby");
  if (labelledBy !== null && labelledBy.length > 0) {
    const parts = labelledBy.split(/\s+/).map((id) => nodeById(root, id)?.textContent ?? "");
    return parts.join(" ").replace(/\s+/g, " ").trim();
  }
  const label = node.getAttribute("aria-label");
  if (label !== null && label.length > 0) return label;
  const id = node.getAttribute("id") ?? node.id ?? "";
  if (id.length > 0) {
    const forLabel = collect(root).find((candidate) => (
      candidate.localName === "label" && candidate.getAttribute("for") === id
    ));
    if (forLabel !== undefined) return forLabel.textContent.replace(/\s+/g, " ").trim();
  }
  return node.textContent.replace(/\s+/g, " ").trim();
}

function nameMatches(actual: string, expected: string | RegExp): boolean {
  if (expected instanceof RegExp) return expected.test(actual);
  return actual.trim().toLowerCase() === expected.trim().toLowerCase();
}

export function getAllByRole(
  root: object,
  role: string,
  name?: string | RegExp,
): object[] {
  if (!isQueryNode(root)) return [];
  return collect(root).filter((node) => {
    if (implicitRole(node) !== role) return false;
    if (name === undefined) return true;
    return nameMatches(accessibleName(root, node), name);
  });
}

export function getByRole(root: object, role: string, name?: string | RegExp): object {
  const matches = getAllByRole(root, role, name);
  if (matches.length !== 1) {
    const label = name === undefined ? role : `${role} ${String(name)}`;
    throw new Error(`expected 1 ${label}, found ${String(matches.length)}`);
  }
  const match = matches[0];
  if (match === undefined) throw new Error(`expected 1 ${role}`);
  return match;
}

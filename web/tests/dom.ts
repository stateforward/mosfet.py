class FakeClassList {
  #values = new Set<string>();
  add(...tokens: string[]): void {
    for (const token of tokens) this.#values.add(token);
  }
  remove(...tokens: string[]): void {
    for (const token of tokens) this.#values.delete(token);
  }
  toggle(token: string, force?: boolean): boolean {
    if (force === true || (force === undefined && !this.#values.has(token))) {
      this.#values.add(token);
      return true;
    }
    this.#values.delete(token);
    return false;
  }
  contains(token: string): boolean {
    return this.#values.has(token);
  }
  toString(): string {
    return [...this.#values].join(" ");
  }
}

class FakeElement {
  readonly tagName: string;
  readonly localName: string;
  readonly childNodes: FakeElement[] = [];
  readonly attributes = new Map<string, string>();
  readonly dataset: Record<string, string> = {};
  readonly style: Record<string, string> & {
    setProperty(name: string, value: string): void;
    getPropertyValue(name: string): string;
  } = Object.assign(Object.create(null) as Record<string, string>, {
    setProperty(this: Record<string, string>, name: string, value: string): void {
      this[name] = value;
    },
    getPropertyValue(this: Record<string, string>, name: string): string {
      const value = this[name];
      return typeof value === "string" ? value : "";
    },
  });
  readonly classList = new FakeClassList();
  readonly part = { add(_token: string): void { return; } };
  parentNode: FakeElement | null = null;
  shadowRoot: FakeShadowRoot | null = null;
  textContent = "";
  hidden = false;
  value = "";
  checked = false;
  disabled = false;
  type = "";
  id = "";
  className = "";
  innerHTML = "";

  constructor(tagName: string) {
    this.tagName = tagName.toUpperCase();
    this.localName = tagName.toLowerCase();
  }

  attachShadow(_init: { mode: string }): FakeShadowRoot {
    this.shadowRoot = new FakeShadowRoot(this);
    return this.shadowRoot;
  }

  append(...nodes: Array<FakeElement | string>): void {
    for (const node of nodes) {
      const child = typeof node === "string" ? new FakeElement("#text") : node;
      if (typeof node === "string") child.textContent = node;
      child.parentNode = this;
      this.childNodes.push(child);
      const connected = (child as FakeElement & { connectedCallback?: () => void }).connectedCallback;
      if (typeof connected === "function") connected.call(child);
    }
  }

  replaceChildren(...nodes: Array<FakeElement | string>): void {
    this.childNodes.length = 0;
    this.append(...nodes);
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
    if (name.startsWith("data-")) {
      const key = name.slice(5).replace(/-([a-z])/g, (_match, letter: string) => letter.toUpperCase());
      this.dataset[key] = value;
    }
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  hasAttribute(name: string): boolean {
    return this.attributes.has(name);
  }

  toggleAttribute(name: string, force?: boolean): void {
    if (force === false) this.attributes.delete(name);
    else this.attributes.set(name, "");
  }

  #listeners = new Map<string, Set<(event: FakeEvent) => void>>();

  addEventListener(type: string, listener: unknown, _options?: unknown): void {
    if (typeof listener !== "function") return;
    const bucket = this.#listeners.get(type) ?? new Set();
    bucket.add(listener as (event: FakeEvent) => void);
    this.#listeners.set(type, bucket);
  }

  removeEventListener(type: string, listener: unknown, _options?: unknown): void {
    if (typeof listener !== "function") return;
    this.#listeners.get(type)?.delete(listener as (event: FakeEvent) => void);
  }

  dispatchEvent(event: FakeEvent): boolean {
    if (event.target === null) event.target = this;
    let current: FakeElement | null = this;
    while (current !== null) {
      for (const listener of current.#listeners.get(event.type) ?? []) listener(event);
      if (!event.bubbles) break;
      if (current instanceof FakeShadowRoot && event.composed) {
        current = current.host;
        continue;
      }
      current = current.parentNode;
    }
    return true;
  }

  composedPath(): FakeElement[] {
    const path: FakeElement[] = [];
    let current: FakeElement | null = this;
    while (current !== null) {
      path.push(current);
      current = current.parentNode;
    }
    return path;
  }

  getRootNode(): FakeElement {
    return this.shadowRoot ?? this;
  }

  setPointerCapture(_pointerId: number): void {
    return;
  }

  hasPointerCapture(_pointerId: number): boolean {
    return false;
  }

  remove(): void {
    const parent = this.parentNode;
    if (parent === null) return;
    const index = parent.childNodes.indexOf(this);
    if (index >= 0) parent.childNodes.splice(index, 1);
    this.parentNode = null;
    disconnectTree(this);
  }

  get isConnected(): boolean {
    return this.parentNode !== null;
  }

  closest(selector: string): FakeElement | null {
    let current: FakeElement | null = this;
    while (current !== null) {
      if (current.localName === selector || current.classList.contains(selector.replace(".", ""))) return current;
      current = current.parentNode;
    }
    return null;
  }

  querySelector(selector: string): FakeElement | null {
    const match = matchSelector(this, selector);
    if (match && this.localName !== selector && !selector.startsWith(".") && !selector.startsWith("[")) {
      // fall through to descendants; host itself is rarely the query target
    }
    for (const child of this.childNodes) {
      if (matchSelector(child, selector)) return child;
      const nested = child.querySelector(selector);
      if (nested !== null) return nested;
    }
    if (this.shadowRoot !== null) {
      if (matchSelector(this.shadowRoot, selector)) return this.shadowRoot;
      const nested = this.shadowRoot.querySelector(selector);
      if (nested !== null) return nested;
    }
    return null;
  }

  querySelectorAll(selector: string): FakeElement[] {
    const found: FakeElement[] = [];
    for (const child of this.childNodes) {
      if (matchSelector(child, selector)) found.push(child);
      found.push(...child.querySelectorAll(selector));
    }
    if (this.shadowRoot !== null) {
      found.push(...this.shadowRoot.querySelectorAll(selector));
    }
    return found;
  }

  getBoundingClientRect(): { left: number; top: number; width: number; height: number; right: number; bottom: number } {
    return { left: 0, top: 0, width: 1000, height: 600, right: 1000, bottom: 600 };
  }

  get clientWidth(): number {
    return 1000;
  }

  get clientHeight(): number {
    return 600;
  }

  tabIndex = 0;

  focus(): void {
    return;
  }
}

function matchSelector(element: FakeElement, selector: string): boolean {
  if (selector.startsWith(".")) return element.classList.contains(selector.slice(1));
  if (selector.startsWith("[data-testid=")) {
    const value = selector.slice("[data-testid=".length).replace(/^["']|["'\]]$]/g, "").replace(/\]$/, "").replace(/^["']|["']$/g, "");
    return element.getAttribute("data-testid") === value;
  }
  return element.localName === selector;
}

function disconnectTree(element: FakeElement): void {
  const disconnected = (element as FakeElement & { disconnectedCallback?: () => void }).disconnectedCallback;
  if (typeof disconnected === "function") disconnected.call(element);
  for (const child of [...element.childNodes]) disconnectTree(child);
  if (element.shadowRoot !== null) disconnectTree(element.shadowRoot);
}

class FakeShadowRoot extends FakeElement {
  adoptedStyleSheets: unknown[] = [];
  host: FakeElement;
  constructor(host: FakeElement) {
    super("shadow");
    this.host = host;
  }
}

class FakeHTMLElement extends FakeElement {
  constructor() {
    const name = new.target.name.replace(/([a-z])([A-Z])/g, "$1-$2").toLowerCase();
    super(name.startsWith("html") ? "div" : name);
  }
}

class FakeDocument {
  readonly body = new FakeElement("body");

  createElement(tag: string): FakeElement {
    const ctor = registry.get(tag);
    if (ctor !== undefined) return new ctor();
    return new FakeElement(tag);
  }

  createElementNS(_ns: string, tag: string): FakeElement {
    return new FakeElement(tag);
  }
}

const registry = new Map<string, new () => FakeHTMLElement>();

class FakeCustomElements {
  get(name: string): (new () => FakeHTMLElement) | undefined {
    return registry.get(name);
  }

  define(name: string, ctor: new () => FakeHTMLElement): void {
    registry.set(name, ctor);
  }
}

class FakeStyleSheet {
  replaceSync(_css: string): void {
    return;
  }
}

class FakeResizeObserver {
  observe(_target: unknown): void {
    return;
  }
  disconnect(): void {
    return;
  }
}

class FakeEvent {
  readonly type: string;
  readonly bubbles: boolean;
  readonly composed: boolean;
  target: unknown = null;
  constructor(type: string, init: { bubbles?: boolean; composed?: boolean } = {}) {
    this.type = type;
    this.bubbles = init.bubbles === true;
    this.composed = init.composed === true;
  }
  composedPath(): unknown[] {
    const target = this.target;
    if (target instanceof FakeElement) return target.composedPath();
    return [];
  }
}

class FakeCustomEvent extends FakeEvent {
  readonly detail: unknown;
  constructor(type: string, init: { detail?: unknown; bubbles?: boolean; composed?: boolean } = {}) {
    super(type, init);
    this.detail = init.detail;
  }
}

class FakePointerEvent extends FakeEvent {
  readonly clientX: number;
  readonly clientY: number;
  readonly pointerId: number;
  readonly button: number;
  readonly buttons: number;
  readonly pointerType: string;
  readonly shiftKey: boolean;
  readonly metaKey: boolean;
  readonly ctrlKey: boolean;
  constructor(type: string, init: {
    clientX?: number;
    clientY?: number;
    pointerId?: number;
    button?: number;
    buttons?: number;
    pointerType?: string;
    shiftKey?: boolean;
    metaKey?: boolean;
    ctrlKey?: boolean;
    bubbles?: boolean;
    composed?: boolean;
  } = {}) {
    super(type, init);
    this.clientX = init.clientX ?? 0;
    this.clientY = init.clientY ?? 0;
    this.pointerId = init.pointerId ?? 1;
    this.button = init.button ?? 0;
    this.buttons = init.buttons ?? (type === "pointerup" || type === "pointercancel" ? 0 : 1);
    this.pointerType = init.pointerType ?? "mouse";
    this.shiftKey = init.shiftKey === true;
    this.metaKey = init.metaKey === true;
    this.ctrlKey = init.ctrlKey === true;
  }
}

class FakeWheelEvent extends FakeEvent {
  readonly clientX: number;
  readonly clientY: number;
  readonly deltaY: number;
  constructor(type: string, init: { clientX?: number; clientY?: number; deltaY?: number; bubbles?: boolean; composed?: boolean } = {}) {
    super(type, init);
    this.clientX = init.clientX ?? 0;
    this.clientY = init.clientY ?? 0;
    this.deltaY = init.deltaY ?? 0;
  }
  preventDefault(): void {
    return;
  }
}

class FakeKeyboardEvent extends FakeEvent {
  readonly key: string;
  constructor(type: string, init: { key?: string; bubbles?: boolean; composed?: boolean } = {}) {
    super(type, init);
    this.key = init.key ?? "";
  }
}

if (typeof (globalThis as { HTMLElement?: unknown }).HTMLElement === "undefined" || !(globalThis as { document?: { createElement?: unknown } }).document?.createElement) {
  Object.assign(globalThis, {
    Element: FakeElement,
    HTMLElement: FakeHTMLElement,
    document: new FakeDocument(),
    customElements: new FakeCustomElements(),
    CSSStyleSheet: FakeStyleSheet,
    ResizeObserver: FakeResizeObserver,
    CustomEvent: FakeCustomEvent,
    Event: FakeEvent,
    PointerEvent: FakePointerEvent,
    WheelEvent: FakeWheelEvent,
    KeyboardEvent: FakeKeyboardEvent,
    getComputedStyle: (element: { style?: { getPropertyValue?: (name: string) => string } }): { getPropertyValue(name: string): string } => ({
      getPropertyValue(name: string): string {
        return element.style?.getPropertyValue?.(name) ?? "";
      },
    }),
    requestAnimationFrame: (callback: (time: number) => void): number => {
      const handle = globalThis.setTimeout(() => callback(0), 0);
      return typeof handle === "number" ? handle : 0;
    },
    cancelAnimationFrame: (id: number): void => {
      globalThis.clearTimeout(id);
    },
  });
}

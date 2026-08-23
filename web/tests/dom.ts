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
  readonly dataset: Record<string, string> = new Proxy({} as Record<string, string>, {
    get: (target, key) => Reflect.get(target, key),
    set: (target, key, value): boolean => {
      if (typeof key !== "string") return false;
      target[key] = String(value);
      const attr = `data-${key.replace(/[A-Z]/g, (ch) => `-${ch.toLowerCase()}`)}`;
      this.attributes.set(attr, String(value));
      return true;
    },
  });
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
  readonly part = {
    add: (token: string): void => {
      const current = this.getAttribute("part");
      const tokens = current === null || current.length === 0 ? [] : current.split(" ");
      if (!tokens.includes(token)) tokens.push(token);
      this.setAttribute("part", tokens.join(" "));
    },
  };
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
      if (this.isConnected) connectTree(child);
    }
  }

  replaceChildren(...nodes: Array<FakeElement | string>): void {
    for (const child of this.childNodes) {
      child.parentNode = null;
    }
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

  removeAttribute(name: string): void {
    this.attributes.delete(name);
    if (name.startsWith("data-")) {
      const key = name.slice(5).replace(/-([a-z])/g, (_match, letter: string) => letter.toUpperCase());
      delete this.dataset[key];
    }
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
      if (current instanceof FakeShadowRoot) {
        current = current.host;
        continue;
      }
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
    let current: FakeElement | null = this;
    while (current !== null) {
      if (current.localName === "body") return true;
      if (current instanceof FakeShadowRoot) {
        current = current.host;
        continue;
      }
      current = current.parentNode;
    }
    return false;
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
    focusedElement = this;
  }
}

let focusedElement: FakeElement | null = null;

function unescapeCss(value: string): string {
  return value.replace(/\\(.)/g, "$1");
}

function matchSelector(element: FakeElement, selector: string): boolean {
  if (selector.startsWith(".")) return element.classList.contains(selector.slice(1));
  const attr = selector.match(/^(?:([a-z][\w-]*)|)\[([^\s=\]]+)="((?:\\.|[^"\\])*)"\]$/i);
  if (attr !== null) {
    const tag = attr[1];
    const name = attr[2];
    const expected = unescapeCss(attr[3] ?? "");
    if (tag !== undefined && element.localName !== tag) return false;
    return element.getAttribute(name ?? "") === expected;
  }
  if (selector.startsWith("[data-testid=")) {
    const value = selector.slice("[data-testid=".length).replace(/^["']|["'\]]$]/g, "").replace(/\]$/, "").replace(/^["']|["']$/g, "");
    return element.getAttribute("data-testid") === value;
  }
  return element.localName === selector;
}

const treeConnected = new WeakSet<FakeElement>();

function connectTree(element: FakeElement): void {
  if (treeConnected.has(element)) return;
  treeConnected.add(element);
  const connected = (element as FakeElement & { connectedCallback?: () => void }).connectedCallback;
  if (typeof connected === "function") connected.call(element);
  if (element.shadowRoot !== null) {
    for (const child of [...element.shadowRoot.childNodes]) connectTree(child);
  }
  for (const child of [...element.childNodes]) connectTree(child);
}

function disconnectTree(element: FakeElement): void {
  if (!treeConnected.has(element)) return;
  const disconnected = (element as FakeElement & { disconnectedCallback?: () => void }).disconnectedCallback;
  if (typeof disconnected === "function") disconnected.call(element);
  treeConnected.delete(element);
  for (const child of [...element.childNodes]) disconnectTree(child);
  if (element.shadowRoot !== null) {
    for (const child of [...element.shadowRoot.childNodes]) disconnectTree(child);
  }
}

class FakeShadowRoot extends FakeElement {
  adoptedStyleSheets: unknown[] = [];
  host: FakeElement;
  constructor(host: FakeElement) {
    super("shadow");
    this.host = host;
  }

  get activeElement(): FakeElement | null {
    let current: FakeElement | null = focusedElement;
    while (current !== null) {
      if (current === this) return focusedElement;
      current = current.parentNode;
    }
    return null;
  }
}

class FakeHTMLElement extends FakeElement {
  constructor(tagName?: string) {
    const fromClass = new.target.name.replace(/([a-z])([A-Z])/g, "$1-$2").toLowerCase();
    const name = tagName ?? fromClass;
    super(name.startsWith("html") ? "div" : name);
  }
}

class FakeHTMLInputElement extends FakeHTMLElement {
  constructor() {
    super("input");
  }
}

class FakeHTMLButtonElement extends FakeHTMLElement {
  constructor() {
    super("button");
  }
}

class FakeDocument {
  readonly body = new FakeElement("body");

  get activeElement(): FakeElement | null {
    return focusedElement;
  }

  createElement(tag: string): FakeElement {
    const ctor = registry.get(tag);
    if (ctor !== undefined) return new ctor();
    if (tag === "input") return new FakeHTMLInputElement();
    if (tag === "button") return new FakeHTMLButtonElement();
    return new FakeHTMLElement(tag);
  }

  createElementNS(_ns: string, tag: string): FakeElement {
    return new FakeElement(tag);
  }

  createTextNode(text: string): FakeElement {
    const node = new FakeElement("#text");
    node.textContent = text;
    return node;
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
  readonly cancelable: boolean;
  defaultPrevented = false;
  target: unknown = null;
  constructor(type: string, init: { bubbles?: boolean; composed?: boolean; cancelable?: boolean } = {}) {
    this.type = type;
    this.bubbles = init.bubbles === true;
    this.composed = init.composed === true;
    this.cancelable = init.cancelable === true;
  }
  preventDefault(): void {
    if (this.cancelable) this.defaultPrevented = true;
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

class FakeMouseEvent extends FakeEvent {
  readonly detail: number;
  readonly clientX: number;
  readonly clientY: number;
  readonly button: number;
  constructor(type: string, init: {
    detail?: number;
    clientX?: number;
    clientY?: number;
    button?: number;
    bubbles?: boolean;
    composed?: boolean;
    cancelable?: boolean;
  } = {}) {
    super(type, init);
    this.detail = init.detail ?? 0;
    this.clientX = init.clientX ?? 0;
    this.clientY = init.clientY ?? 0;
    this.button = init.button ?? 0;
  }
}

class FakeWheelEvent extends FakeEvent {
  readonly clientX: number;
  readonly clientY: number;
  readonly deltaY: number;
  constructor(type: string, init: { clientX?: number; clientY?: number; deltaY?: number; bubbles?: boolean; composed?: boolean; cancelable?: boolean } = {}) {
    super(type, { ...init, cancelable: init.cancelable ?? true });
    this.clientX = init.clientX ?? 0;
    this.clientY = init.clientY ?? 0;
    this.deltaY = init.deltaY ?? 0;
  }
}

class FakeKeyboardEvent extends FakeEvent {
  readonly key: string;
  constructor(type: string, init: { key?: string; bubbles?: boolean; composed?: boolean; cancelable?: boolean } = {}) {
    super(type, init);
    this.key = init.key ?? "";
  }
}

if (typeof (globalThis as { HTMLElement?: unknown }).HTMLElement === "undefined" || !(globalThis as { document?: { createElement?: unknown } }).document?.createElement) {
  Object.assign(globalThis, {
    Element: FakeElement,
    HTMLElement: FakeHTMLElement,
    HTMLInputElement: FakeHTMLInputElement,
    HTMLButtonElement: FakeHTMLButtonElement,
    CSS: {
      escape(value: string): string {
        return [...value].map((ch) => (/[A-Za-z0-9_-]/.test(ch) ? ch : `\\${ch}`)).join("");
      },
    },
    document: new FakeDocument(),
    customElements: new FakeCustomElements(),
    CSSStyleSheet: FakeStyleSheet,
    ResizeObserver: FakeResizeObserver,
    CustomEvent: FakeCustomEvent,
    Event: FakeEvent,
    PointerEvent: FakePointerEvent,
    WheelEvent: FakeWheelEvent,
    MouseEvent: FakeMouseEvent,
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

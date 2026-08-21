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
  readonly style: Record<string, string> = {};
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

  toggleAttribute(name: string, force?: boolean): void {
    if (force === false) this.attributes.delete(name);
    else this.attributes.set(name, "");
  }

  addEventListener(_type: string, _listener: unknown, _options?: unknown): void {
    return;
  }

  removeEventListener(_type: string, _listener: unknown, _options?: unknown): void {
    return;
  }

  dispatchEvent(_event: unknown): boolean {
    return true;
  }

  closest(selector: string): FakeElement | null {
    let current: FakeElement | null = this;
    while (current !== null) {
      if (current.localName === selector || current.classList.contains(selector.replace(".", ""))) return current;
      current = current.parentNode;
    }
    return null;
  }

  querySelector(_selector: string): FakeElement | null {
    return this.childNodes[0] ?? null;
  }

  querySelectorAll(_selector: string): FakeElement[] {
    return [...this.childNodes];
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

if (typeof (globalThis as { HTMLElement?: unknown }).HTMLElement === "undefined" || !(globalThis as { document?: { createElement?: unknown } }).document?.createElement) {
  Object.assign(globalThis, {
    HTMLElement: FakeHTMLElement,
    document: new FakeDocument(),
    customElements: new FakeCustomElements(),
    CSSStyleSheet: FakeStyleSheet,
    ResizeObserver: FakeResizeObserver,
    CustomEvent: FakeCustomEvent,
    Event: FakeEvent,
  });
}

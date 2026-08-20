import {
  isOtelSourceEventName,
  OtelSourceController,
  type OtelSourceSnapshot,
} from "../otel-source-hsm.ts";
import { reportHsmFailure } from "../hsm-runtime.ts";
import { type OtelSource } from "../otel/source.ts";
import { applyStyles } from "./styles.ts";

export type OtelSourceDetail = {
  source: OtelSource;
};

const ELEMENT_NAME = "bot-otel-source";

const cssText = `
:host {
  display: inline-flex;
  align-items: center;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  margin: 0;
  padding: 0.15rem 0.55rem 0.15rem 0.4rem;
  border: 1px solid var(--bot-line, #2a3140);
  border-radius: 999px;
  background: #161922;
  color: var(--bot-muted, #8b93a7);
  font: inherit;
  line-height: 1.2;
  cursor: pointer;
}
.badge::before {
  content: "";
  width: 0.45rem;
  height: 0.45rem;
  border-radius: 50%;
  background: #64748b;
}
.badge[data-phase="connecting"]::before {
  background: #14b8a6;
}
.badge[data-phase="live"]::before {
  background: var(--bot-accent, #2dd4bf);
  box-shadow: 0 0 0.4rem color-mix(in srgb, var(--bot-accent, #2dd4bf) 70%, transparent);
}
.badge[data-phase="error"]::before {
  background: #f87171;
}
.badge[data-phase="idle"]::before {
  background: #64748b;
}
.error {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
}
`;

export class BotOtelSource extends HTMLElement {
  static readonly eventName = "bot-otel-source";

  readonly #root: ShadowRoot;
  readonly #badge: HTMLButtonElement;
  readonly #error: HTMLParagraphElement;
  #controller: OtelSourceController | null = null;
  #abort: AbortController | null = null;

  constructor() {
    super();
    this.#root = this.attachShadow({ mode: "open" });
    applyStyles(this.#root, cssText);
    this.#badge = document.createElement("button");
    this.#badge.type = "button";
    this.#badge.className = "badge";
    this.#badge.part.add("live-badge");
    this.#badge.setAttribute("data-testid", "live-badge");
    this.#badge.dataset["event"] = "source.connect.requested";
    this.#badge.setAttribute("aria-label", "Collector status");
    this.#error = document.createElement("p");
    this.#error.className = "error";
    this.#root.append(this.#badge, this.#error);
  }

  replayReady(): void {
    this.#controller?.emitReady();
  }

  connectedCallback(): void {
    if (this.#controller === null) {
      this.#controller = new OtelSourceController({
        onSnapshot: (snapshot) => {
          this.#render(snapshot);
        },
        onReady: (source) => {
          this.dispatchEvent(
            new CustomEvent<OtelSourceDetail>(BotOtelSource.eventName, {
              detail: { source },
              bubbles: true,
              composed: true,
            }),
          );
        },
      });
    }
    this.#abort = new AbortController();
    this.#root.addEventListener("click", this.#onClick, { signal: this.#abort.signal });
    this.#render(this.#controller.snapshot());
    if (this.#controller.snapshot().phase === "idle") {
      void this.#controller.dispatch("source.connect.requested").catch(reportHsmFailure);
      return;
    }
    if (this.#controller.snapshot().phase === "live") {
      this.replayReady();
    }
  }

  disconnectedCallback(): void {
    this.#abort?.abort();
    this.#abort = null;
    const controller = this.#controller;
    this.#controller = null;
    if (controller !== null) {
      void controller.stop().catch(reportHsmFailure);
    }
  }

  #render(snapshot: OtelSourceSnapshot): void {
    this.#badge.dataset["phase"] = snapshot.phase;
    this.#badge.textContent = snapshot.phase;
    this.#error.textContent = snapshot.errorMessage ?? "";
  }

  readonly #onClick = (event: Event): void => {
    const target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    const button = target.closest("[data-event]");
    if (!(button instanceof HTMLButtonElement)) {
      return;
    }
    const eventName = button.dataset["event"];
    if (eventName === undefined || !isOtelSourceEventName(eventName) || this.#controller === null) {
      return;
    }
    void this.#controller.dispatch(eventName).catch(reportHsmFailure);
  };
}

export function registerBotOtelSource(): void {
  if (customElements.get(ELEMENT_NAME) === undefined) {
    customElements.define(ELEMENT_NAME, BotOtelSource);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "bot-otel-source": BotOtelSource;
  }
}

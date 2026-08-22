import {
  isOtelSourceEventName,
  OtelSource,
  type OtelSourceSnapshot,
} from "../otel-source.ts";
import { catchFailure } from "../hsm.ts";
import { type OtelSource as StreamSource } from "../otel/source.ts";
import { applyStyles } from "./styles.ts";

/**
 * Detail of the `bot-otel-source` CustomEvent emitted when the source is ready.
 * Event contract: `bubbles: true`, `composed: true`, `cancelable: false`.
 * Postcondition: the source is already live and committed on the host; the
 * event only announces readiness. `source` is the host's live shared handle,
 * not a copy: it is the same object `snapshot().source` returns, owned by the
 * host for the host's lifetime. Listeners may read it but must not assume
 * transfer or exclusive mutation, and must not mutate or retain it beyond
 * their own scope in a way that outlives the host's ownership. `preventDefault()`
 * has no effect because the event cannot be canceled.
 */
export type OtelSourceDetail = {
  readonly source: StreamSource;
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

export class BotOtelSource extends OtelSource {
  static readonly eventName = "bot-otel-source";

  readonly #root: ShadowRoot;
  readonly #badge: HTMLButtonElement;
  readonly #error: HTMLParagraphElement;
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
    this.#error = document.createElement("p");
    this.#error.id = "collector-error";
    this.#error.className = "error";
    this.#error.part.add("status");
    this.#error.setAttribute("data-testid", "collector-status");
    this.#error.setAttribute("role", "status");
    this.#error.setAttribute("aria-label", "Collector error");
    this.#error.setAttribute("aria-live", "polite");
    this.#root.append(this.#badge, this.#error);
  }

  replayReady(): void {
    this.emitReady();
  }

  connectedCallback(): void {
    this.onSnapshot = (snapshot) => {
      this.#render(snapshot);
    };
    this.onReady = (source) => {
      this.dispatchEvent(
        new CustomEvent<OtelSourceDetail>(BotOtelSource.eventName, {
          detail: { source },
          bubbles: true,
          composed: true,
          cancelable: false,
        }),
      );
    };
    this.boot();
    this.requestAttach();
    this.#abort = new AbortController();
    this.#root.addEventListener("click", this.#onClick, { signal: this.#abort.signal });
    this.#render(this.snapshot());
    void this.dispatch("source.connect.requested", { origin: documentOrigin(this) }).catch(catchFailure(this));
  }

  disconnectedCallback(): void {
    this.#abort?.abort();
    this.#abort = null;
    this.requestDetach();
  }

  #render(snapshot: OtelSourceSnapshot): void {
    this.#badge.dataset["phase"] = snapshot.phase;
    this.#badge.textContent = snapshot.phase;
    this.#badge.setAttribute("aria-label", `Collector status: ${snapshot.phase}`);
    this.#badge.disabled = snapshot.phase === "connecting";
    if (snapshot.phase === "connecting") {
      this.#badge.setAttribute("aria-busy", "true");
    } else {
      this.#badge.removeAttribute("aria-busy");
    }
    const errorMessage = snapshot.errorMessage ?? "";
    this.#error.textContent = errorMessage;
    if (errorMessage.length > 0) {
      this.#badge.setAttribute("aria-describedby", this.#error.id);
    } else {
      this.#badge.removeAttribute("aria-describedby");
    }
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
    if (eventName === undefined || !isOtelSourceEventName(eventName)) {
      return;
    }
    void this.dispatch(eventName, { origin: documentOrigin(this) }).catch(catchFailure(this));
  };
}

function documentOrigin(host: HTMLElement): string {
  const origin = host.ownerDocument?.defaultView?.location.origin;
  return typeof origin === "string" && origin.length > 0 ? origin : "http://localhost";
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

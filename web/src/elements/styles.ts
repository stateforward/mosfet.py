export function applyStyles(root: ShadowRoot, cssText: string): void {
  if ("adoptedStyleSheets" in root && typeof CSSStyleSheet !== "undefined") {
    const sheet = new CSSStyleSheet();
    sheet.replaceSync(cssText);
    root.adoptedStyleSheets = [...root.adoptedStyleSheets, sheet];
    return;
  }
  const style = document.createElement("style");
  style.textContent = cssText;
  root.append(style);
}

export function replaceStyles(root: ShadowRoot, cssText: string): void {
  if ("adoptedStyleSheets" in root && typeof CSSStyleSheet !== "undefined") {
    const sheet = new CSSStyleSheet();
    sheet.replaceSync(cssText);
    root.adoptedStyleSheets = [sheet];
    return;
  }
  const style = document.createElement("style");
  style.textContent = cssText;
  root.append(style);
}

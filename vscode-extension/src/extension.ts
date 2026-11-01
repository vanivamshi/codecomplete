import * as crypto from "crypto";
import * as vscode from "vscode";

interface CompletionResponse {
  id: string;
  text: string;
  latency_ms: number;
  cached: boolean;
}

let enabled = true;
let shown = 0;
let accepted = 0;
let statusBar: vscode.StatusBarItem;

function config() {
  return vscode.workspace.getConfiguration("codecomplete");
}

// Anonymous per-install id: lets the study count distinct developers without identifying them.
const userId = crypto.createHash("sha256").update(vscode.env.machineId).digest("hex").slice(0, 16);

function sendEvent(event: "shown" | "accepted", id: string, language: string, chars: number) {
  if (!config().get<boolean>("shareUsageStats") || !vscode.env.isTelemetryEnabled) {
    return;
  }
  const url = `${config().get<string>("serverUrl")}/telemetry`;
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event, id, user: userId, language, chars }),
  }).catch(() => undefined);
}

function updateStatus() {
  const rate = shown ? Math.round((100 * accepted) / shown) : 0;
  statusBar.text = enabled ? `$(sparkle) ${accepted}/${shown} (${rate}%)` : "$(circle-slash) CodeComplete";
  statusBar.tooltip = "CodeComplete: accepted / shown suggestions this session. Click to toggle.";
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Multi-line ("block") completion right after opening a block; otherwise finish the current line.
function chooseMode(document: vscode.TextDocument, position: vscode.Position): "line" | "block" {
  const before = document.lineAt(position.line).text.slice(0, position.character);
  if (before.trim() !== "" || position.line === 0) {
    return "line";
  }
  const prev = document.lineAt(position.line - 1).text.trimEnd();
  return prev.endsWith(":") ? "block" : "line";
}

class Provider implements vscode.InlineCompletionItemProvider {
  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken,
  ): Promise<vscode.InlineCompletionItem[] | undefined> {
    if (!enabled) {
      return;
    }
    // Debounce: VS Code cancels this request when the user types again.
    await sleep(config().get<number>("debounceMs", 150));
    if (token.isCancellationRequested) {
      return;
    }

    const offset = document.offsetAt(position);
    const text = document.getText();
    const prefix = text.slice(Math.max(0, offset - config().get<number>("maxPrefixChars", 4000)), offset);
    const suffix = text.slice(offset, offset + config().get<number>("maxSuffixChars", 1000));

    // Don't suggest in the middle of a word.
    const after = document.lineAt(position.line).text.slice(position.character);
    if (/^\w/.test(after)) {
      return;
    }

    const abort = new AbortController();
    const sub = token.onCancellationRequested(() => abort.abort());
    let result: CompletionResponse;
    try {
      const res = await fetch(`${config().get<string>("serverUrl")}/complete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prefix, suffix, mode: chooseMode(document, position) }),
        signal: abort.signal,
      });
      if (!res.ok) {
        return;
      }
      result = (await res.json()) as CompletionResponse;
    } catch {
      return; // cancelled or server unreachable
    } finally {
      sub.dispose();
    }

    if (token.isCancellationRequested || !result.text.trim()) {
      return;
    }

    shown++;
    updateStatus();
    sendEvent("shown", result.id, document.languageId, result.text.length);

    const item = new vscode.InlineCompletionItem(result.text, new vscode.Range(position, position));
    // VS Code runs this command only when the user accepts the suggestion.
    item.command = {
      title: "accepted",
      command: "codecomplete.accepted",
      arguments: [result.id, document.languageId, result.text.length],
    };
    return [item];
  }
}

export function activate(context: vscode.ExtensionContext) {
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.command = "codecomplete.toggle";
  updateStatus();
  statusBar.show();

  context.subscriptions.push(
    statusBar,
    vscode.languages.registerInlineCompletionItemProvider({ language: "python" }, new Provider()),
    vscode.commands.registerCommand("codecomplete.accepted", (id: string, language: string, chars: number) => {
      accepted++;
      updateStatus();
      sendEvent("accepted", id, language, chars);
    }),
    vscode.commands.registerCommand("codecomplete.toggle", () => {
      enabled = !enabled;
      updateStatus();
    }),
    vscode.commands.registerCommand("codecomplete.showStats", () => {
      const rate = shown ? ((100 * accepted) / shown).toFixed(1) : "0";
      vscode.window.showInformationMessage(`CodeComplete: ${accepted} of ${shown} suggestions accepted (${rate}%) this session.`);
    }),
  );
}

export function deactivate() {}

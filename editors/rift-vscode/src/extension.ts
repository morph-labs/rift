import * as vscode from "vscode";
import { MorphLanguageClient } from "./client";
import { WebviewProvider } from "./elements/WebviewProvider";
import {
  ensureRiftHook,
  checkExtensionVersion,
} from "./activation/environmentSetup";
export let chatProvider: WebviewProvider;
export let logProvider: WebviewProvider;

export function activate(context: vscode.ExtensionContext) {
  const autostart: boolean | undefined = vscode.workspace
    .getConfiguration("rift")
    .get("autostart");

  checkExtensionVersion();

  if (autostart) {
    ensureRiftHook();
  }

  const morph_language_client = new MorphLanguageClient(context);

  context.subscriptions.push(
    vscode.commands.registerCommand("rift.restart", () => {
      morph_language_client.restart().then(() => console.log("restarted"));
    }),
  );

  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider("*", morph_language_client),
  );

  chatProvider = new WebviewProvider(
    "Chat",
    context.extensionUri,
    morph_language_client,
  );
  logProvider = new WebviewProvider(
    "Logs",
    context.extensionUri,
    morph_language_client,
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("RiftChat", chatProvider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
  );
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("RiftLogs", logProvider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
  );

  console.log('Congratulations, your extension "rift" is now active!');

  const disposablefocusOmnibar = vscode.commands.registerCommand(
    "rift.focus_omnibar",
    async () => {
      // vscode.window.createTreeView("RiftChat", chatProvider)
      vscode.commands.executeCommand("RiftChat.focus");

      morph_language_client.focusOmnibar();
    },
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("rift.reset_chat", () => {
      morph_language_client.restartActiveAgent();
    }),
  );

  context.subscriptions.push(disposablefocusOmnibar);
  context.subscriptions.push(morph_language_client);
}

// This method is called when your extension is deactivated
export function deactivate() {}

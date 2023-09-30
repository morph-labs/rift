import * as vscode from "vscode";
import { port, MorphLanguageClient } from "./client";
import { WebviewProvider } from "./elements/WebviewProvider";
import {
  forceResolveServerOptions,
  onDeactivate,
  tryResolveServerOptions,
} from "./activation/downloadBuild";
import { autoBuild, upgradeLocalBuildAsNeeded } from "./activation/localBuild";
import { ServerOptions } from "vscode-languageclient/node";
export let chatProvider: WebviewProvider;
export let logProvider: WebviewProvider;

let morph_language_client: MorphLanguageClient;

export async function activate(context: vscode.ExtensionContext) {
  const buildServer = async (
    progress: vscode.Progress<{ message?: string; increment?: number }>,
  ) => {
    try {
      await autoBuild(progress);
      const options = await tryResolveServerOptions(progress, port);
      if (options) {
        onServerOptionsResolved(options);
      } else {
        throw Error("Build failed or not started. Is AutoStart enabled?");
      }
    } catch (e) {
      vscode.window.showErrorMessage(
        `${
          (e as any).message
        }\nEnsure that python3.10 is available and try installing Rift manually: https://www.github.com/morph-labs/rift`,
        "Close",
      );
      throw e;
    }
  };

  vscode.commands.registerCommand("rift.build-server", () => {
    vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification },
      async (progress) => {
        return buildServer(progress);
      },
    );
  });

  vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification },
    async (progress) => {
      try {
        await upgradeLocalBuildAsNeeded(progress);
        const options = await forceResolveServerOptions(progress, port);
        onServerOptionsResolved(options);
      } catch (e) {
        console.error("Error Downloading Server Build", e);
        const resp = await vscode.window.showErrorMessage(
          "Error Downloading Server Build: " + e,
          "Try Building Locally",
        );
        if (resp === "Try Building Locally") {
          buildServer(progress);
        }
      }
    },
  );

  const onServerOptionsResolved = (serverOptions: ServerOptions) => {
    if (morph_language_client) {
      throw Error("Invalid state - client already exists");
    }

    morph_language_client = new MorphLanguageClient(context, serverOptions);

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

    morph_language_client.focusOmnibar();

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

    context.subscriptions.push(
      vscode.commands.registerCommand("rift.reset_chat", () => {
        morph_language_client.restartActiveAgent();
      }),
    );

    context.subscriptions.push(morph_language_client);
  };

  const disposablefocusOmnibar = vscode.commands.registerCommand(
    "rift.focus_omnibar",
    async () => {
      // vscode.window.createTreeView("RiftChat", chatProvider)
      vscode.commands.executeCommand("RiftChat.focus");
      morph_language_client?.focusOmnibar();
    },
  );
  context.subscriptions.push(disposablefocusOmnibar);
}

// This method is called when your extension is deactivated
export function deactivate() {
  onDeactivate.map((d) => d());
}

import * as vscode from "vscode";
import { port, MorphLanguageClient } from "./client";
import { WebviewProvider } from "./elements/WebviewProvider";
import {
  downloadAndStartServer,
  onDeactivate,
  startServerIfAvailable,
} from "./activation/downloadBuild";
import { autoBuild, upgradeLocalBuildAsNeeded } from "./activation/localBuild";
import { ServerOptions } from "vscode-languageclient/node";
export let chatProvider: WebviewProvider;
export let logProvider: WebviewProvider;

let morph_language_client: MorphLanguageClient;

export async function activate(context: vscode.ExtensionContext) {
  console.log('Congratulations, your extension "rift" is now active!');

  vscode.window
    .withProgress<ServerOptions>(
      { location: vscode.ProgressLocation.Notification },
      async (progress): Promise<ServerOptions> => {
        try {
          await upgradeLocalBuildAsNeeded(progress);

          return await downloadAndStartServer(progress, port);
        } catch (e) {
          console.error("Error Downloading Server Build", e);
          const resp = await vscode.window.showErrorMessage(
            "Error Downloading Server Build: " + e,
            "Try Building Locally",
          );
          if (resp === "Try Building Locally") {
            try {
              await autoBuild(progress);
              const started = await startServerIfAvailable(progress, port);
              if (started) return started;
            } catch (e) {
              vscode.window.showErrorMessage(
                `${
                  (e as any).message
                }\nEnsure that python3.10 is available and try installing Rift manually: https://www.github.com/morph-labs/rift`,
                "Close",
              );
            }
          }
          throw Error("No Server Available");
        }
      },
    )
    .then((serverOptions) => {
      onServerAvailable(serverOptions);
    });

  const onServerAvailable = (serverOptions: ServerOptions) => {
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
  };
}

// This method is called when your extension is deactivated
export function deactivate() {
  onDeactivate.map((d) => d());
}

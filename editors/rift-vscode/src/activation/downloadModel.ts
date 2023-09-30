import * as path from "path";
import { morphDir } from "./localBuild";
import { downloadFile, exists } from "./downloadBuild";
import * as vscode from "vscode";
import { rm } from "fs/promises";

const repoName = "morph-labs";

export const metadataForModelName = (configName: string) => {
  const modelName = configName.split(":")[1].split("@")[0].trim();
  const modelPathConfig = configName.split(":")[1].split("@")[1];
  const remoteURL = `https://huggingface.co/${repoName}/${modelName}/resolve/main/ggml-model-Q8_0.gguf`;
  const baseDir = path.join(morphDir, "models");
  const modelPath = modelPathConfig ?? path.join(baseDir, modelName);

  return { remoteURL, modelPath };
};

export async function modelExists(configName: string) {
  const { modelPath } = metadataForModelName(configName);
  return await exists(modelPath);
}

export async function downloadModel(configName: string) {
  if (await modelExists(configName)) {
    return;
  }

  const { remoteURL, modelPath } = metadataForModelName(configName);
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, cancellable: true },
    async (p, t) => {
      p.report({
        message:
          "Downloading Rift Model. You can use remote hosted models while this downloads.",
      });
      await downloadFile(remoteURL, modelPath, p, t);
    },
  );
}

export async function deleteModel(configName: string) {
  const { modelPath } = metadataForModelName(configName);
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification },
    async (p) => {
      p.report({ message: "Deleting Rift Model..." });
      await rm(modelPath);
    },
  );
}

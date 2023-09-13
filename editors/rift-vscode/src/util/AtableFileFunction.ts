import * as vscode from "vscode";
import { AtableFile } from "../types";
export const AtableFileFromUri = (Uri: vscode.Uri): AtableFile => {
  return {
    fileName: Uri.path.split("/").pop() ?? Uri.path,
    fullPath: Uri.fsPath + (Uri.fragment ? "#" + Uri.fragment : ""),
    fromWorkspacePath: vscode.workspace.asRelativePath(Uri),
    symbolName: Uri.fragment,
  };
};
export const AtableFileFromFsPath = (fsPath: string): AtableFile => {
  const uri = vscode.Uri.file(fsPath);
  return AtableFileFromUri(uri);
};

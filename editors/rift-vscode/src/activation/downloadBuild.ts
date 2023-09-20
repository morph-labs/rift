import * as tcpPortUsed from "tcp-port-used";
import * as vscode from "vscode";
import { getExtensionVersion, morphDir } from "./localBuild";
import * as path from "path";
import * as fs from "fs";
import * as util from "util";
import fetch from "node-fetch";
import { mkdir, readdir, rm, stat } from "fs/promises";
import { finished } from "stream/promises";
import * as decompress from "decompress";
import { exec as _exec, spawn } from "child_process";
import {
  Executable,
  ServerOptions,
  StreamInfo,
  TransportKind,
} from "vscode-languageclient/node";
import * as net from "net";

const exec = util.promisify(_exec);

export const onDeactivate: (() => void)[] = [];

// ref: https://stackoverflow.com/questions/40284523/connect-external-language-server-to-vscode-extension

// https://nodejs.org/api/child_process.html#child_processspawncommand-args-options

/** Creates the ServerOptions for a system in the case that a language server is already running on the given port. */
function tcpServerOptions(port: number): ServerOptions {
  const socket = net.connect({
    port: port,
    host: "127.0.0.1",
  });
  const si: StreamInfo = {
    reader: socket,
    writer: socket,
  };
  return () => {
    return Promise.resolve(si);
  };
}

const exists = async (p?: string) => {
  try {
    if (p) {
      await stat(p);
      return true;
    }
  } catch (e) {}
  return false;
};

const platformSpecific = <T>(options: {
  windows?: T;
  mac?: T;
  linux?: T;
}): T | undefined => {
  if (process.platform === "darwin") {
    return options.mac;
  }
  if (process.platform === "linux") {
    return options.linux;
  }
  if (process.platform === "win32") {
    return options.windows;
  }
  throw Error("Unsupported Platform: " + process.platform);
};

const downloadFile = async (
  url: string,
  savePath: string,
  progress: vscode.Progress<{ message?: string; increment?: number }>,
): Promise<boolean> => {
  const res = await fetch(url);
  if (!res.ok || !res.body) {
    console.error("Error getting file", { res, url });
    return false;
  }
  const total = res.headers.get("content-length");
  console.log(total);

  await mkdir(path.dirname(savePath), { recursive: true });
  const fileStream = fs.createWriteStream(savePath);

  if (total) {
    res.body.on("data", (d: Uint8Array) => {
      progress.report({ increment: (d.length / +total) * 100 });
    });
  }

  await finished(res.body.pipe(fileStream));
  return true;
};

export const startServer = async () => {};

export const isServerRunning = async (port: number) => {
  if (await tcpPortUsed.check(port)) {
    // todo: check if the version/service is valid, kill and reinstall if not
    return true;
  }
  return false;
};

export const tryResolveServerOptions = async (
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  port: number,
): Promise<ServerOptions | undefined> => {
  if (await isServerRunning(port)) {
    return tcpServerOptions(port);
  }

  const customRiftPath = vscode.workspace
    .getConfiguration("rift")
    .get<string>("riftPath");

  const { binLocation } = getDefaultInstallProps();

  let binPath;
  if (await exists(customRiftPath)) {
    binPath = customRiftPath;
  } else if (await exists(binLocation)) {
    binPath = binLocation;
  }

  if (binPath && vscode.workspace.getConfiguration("rift").get("autostart")) {
    const args = ["--port", `${port}`];
    const transport = { kind: TransportKind.socket, port: port } as const;

    const e: Executable = {
      command: binPath,
      transport: transport,
      args: args,
    };

    return e;
  }

  return undefined;
};

export const forceResolveServerOptions = async (
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  port: number,
): Promise<ServerOptions> => {
  const available = await tryResolveServerOptions(progress, port);
  if (available) {
    return available;
  }

  if (vscode.workspace.getConfiguration("rift").get("autostart")) {
    const { bundleLocation, zipLocation, version, bundleID } =
      getDefaultInstallProps();

    if (!(await exists(bundleLocation))) {
      await rm(zipLocation).catch(() => {});

      const url = `https://github.com/morph-labs/rift/releases/download/v${version}/${bundleID}.zip`;
      console.log(new Date().toISOString(), "downloading...");
      progress.report({ message: "Rift: Downloading server..." });
      const gotFile = await downloadFile(url, zipLocation, progress);
      if (!gotFile) {
        throw Error("Remote build not available: " + bundleID);
      }

      console.log(new Date().toISOString(), "decompressing...");
      progress.report({
        message: "Rift: Extracting server...",
        increment: -100,
      });
      await decompress(zipLocation, bundleLocation, { strip: 1 });

      readdir(morphDir).then((entries) => {
        for (const entry of entries) {
          if (entry !== bundleID && entry.startsWith("rift-")) {
            console.log("Cleaning up", path.join(morphDir, entry));
            rm(path.join(morphDir, entry), { force: true, recursive: true });
          }
        }
      });
      console.log(new Date().toISOString(), "done!");

      await platformSpecific({
        mac: async () => {
          // Not needed? Maybe only assigned to browser-downloaded archives.
          // await exec("xattr -d com.apple.quarantine **/*", {
          //   cwd: bundleLocation,
          // }).catch((e) => console.error(e));
          await exec("chmod +x core", {
            cwd: bundleLocation,
          });
        },
        linux: async () => {
          await exec("chmod +x core", {
            cwd: bundleLocation,
          });
        },
      })?.();
    }
    const started = await tryResolveServerOptions(progress, port);
    if (started) {
      return started;
    }
  } else {
    progress.report({
      message:
        "Rift: AutoStart disabled, waiting for server to be manually started",
    });
    await waitForPort(port);
    return tcpServerOptions(port);
  }

  throw Error("Could not download and start prebuilt server.");
};

async function waitForPort(port: number) {
  while (!(await tcpPortUsed.check(port))) {
    console.log("waiting for server to come online...");
    try {
      await tcpPortUsed.waitUntilUsed(port, 500, 1000000);
    } catch (e) {
      console.error(e);
    }
  }
}

function getDefaultInstallProps() {
  const version = getExtensionVersion();
  let osName =
    process.platform === "darwin"
      ? "macOS"
      : process.platform === "linux"
      ? "Linux"
      : process.platform === "win32"
      ? "Windows"
      : "unknown";
  if (osName === "unknown") {
    throw Error("bad os.");
  }

  if (osName === "macOS" && process.arch === "arm64") {
    osName += "-arm64";
  }

  const bundleID = "rift-" + osName + "-v" + version;
  const bundleLocation = path.join(morphDir, bundleID);
  const zipLocation = bundleLocation + ".zip";
  const name =
    platformSpecific({
      windows: "core.exe",
    }) ?? "core";

  const binLocation = path.join(bundleLocation, name);

  return { bundleLocation, zipLocation, version, bundleID, binLocation };
}

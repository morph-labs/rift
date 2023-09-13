import * as path from "path";
import * as fs from "fs";
// import {getRiftServerUrl} from "../bridge";
import * as semver from "semver";
import * as toml from "@iarna/toml";
import * as vscode from "vscode";
import * as os from "os";
import * as tcpPortUsed from "tcp-port-used";

import * as util from "util";
import fetch from "node-fetch";
import { exec } from "child_process";
const aexec = util.promisify(exec);

const PACKAGE_JSON_RAW_GITHUB_URL =
  "https://raw.githubusercontent.com/morph-labs/rift/main/editors/rift-vscode/package.json";

const WINDOWS_REMOTE_SIGNED_SCRIPTS_ERROR =
  "A Python virtual enviroment cannot be activated because running scripts is disabled for this user. In order to use Rift, please enable signed scripts to run with this command in PowerShell: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, reload VS Code, and then try again.";

const RIFT_COMMIT = "main";
const PIP_INSTALL_ARGS = `install --upgrade "git+https://github.com/morph-labs/rift.git@${RIFT_COMMIT}#subdirectory=rift-engine&egg=pyrift"`;

const morphDir = path.join(os.homedir(), ".morph");

export function getExtensionUri(): vscode.Uri {
  return vscode.extensions.getExtension("morph.rift-vscode")!.extensionUri;
}

function getExtensionVersion() {
  const extension = vscode.extensions.getExtension("morph.rift-vscode");
  return extension?.packageJSON.version || "";
}

export function checkExtensionVersion() {
  fetch(PACKAGE_JSON_RAW_GITHUB_URL)
    .then(async (res) => res.json())
    .then((packageJson: any) => {
      const localVersion = semver.coerce(getExtensionVersion());
      const remoteVersion = semver.coerce(packageJson.version);
      if (!localVersion || !remoteVersion) {
        return console.log("unable to compare extension versions");
      }
      if (localVersion.compare(remoteVersion) === -1) {
        vscode.window.showInformationMessage(
          `You are using an out-of-date version (${getExtensionVersion()}) of the Rift VSCode Extension (latest ${
            packageJson.version
          }). Please update to the latest version from the VSCode Marketplace, or from source at https://www.github.com/morph-labs/rift.`,
        );
      }
    })
    .catch((e) => console.log("Error checking for extension updates: ", e));
}

// URL for the raw GitHub content of pyproject.toml in the rift engine
const PYPROJECT_TOML_RAW_GITHUB_URL =
  "https://raw.githubusercontent.com/morph-labs/rift/main/rift-engine/pyproject.toml";

// Function to get the Rift URI
export function getRiftUri(): string {
  return PYPROJECT_TOML_RAW_GITHUB_URL;
}

export function getLocalRiftVersion(): Promise<any> {
  console.log("checking rift version");
  return new Promise((resolve, reject) => {
    // run `python -m rift.__about__` & stores the output
    exec(
      `${morphBinPath(
        process.platform === "win32" ? "py" : "python",
      )} -m rift.__about__`,
      (error, stdout, stderr) => {
        if (error) {
          console.log(`error: ${error.message}`);
          resolve(null);
          return;
        }
        if (stderr) {
          console.log(`stderr: ${stderr}`);
          resolve(null);
          return;
        }
        if (stdout) {
          console.log(`stdout=${stdout}`);
          const riftVersion = semver.coerce(stdout);
          console.log(`riftVersion=${riftVersion}`);
          resolve(riftVersion);
          return;
        }
        resolve(null);
      },
    );
  });
}

async function checkRiftVersion() {
  const localRiftVersion = await getLocalRiftVersion();

  console.log(`localRiftVersion: ${localRiftVersion}`);
  await fetch(PYPROJECT_TOML_RAW_GITHUB_URL)
    .then((res) => res.text())
    .then((data) => {
      // parse toml and get the version field
      console.log(`data=${data}`);
      const pyprojectToml: toml.JsonMap = toml.parse(data);
      console.log(`pyprojectToml=${pyprojectToml}`);
      const pyprojectTomlVersion: any = pyprojectToml.project["version"];
      console.log(`tomlVersion=${pyprojectTomlVersion}`);
      const pyprojectVersion = semver.coerce(pyprojectTomlVersion);
      console.log(`pyprojectVersion=${pyprojectVersion}`);
      // If riftVersion is strictly less than pyprojectVersion
      if (localRiftVersion.compare(pyprojectVersion) === -1) {
        vscode.window
          .showInformationMessage(
            `You are using an out-of-date version ${localRiftVersion} < ${pyprojectVersion} of the Rift language server. Please update to the latest version.`,
            "Try auto update",
          )
          .then((selection) => {
            if (selection === "Try auto update") {
              autoInstallHook()
                .then((_) => {
                  vscode.window.showInformationMessage(
                    "Rift installation successful.",
                  );
                })
                .catch((e) => {
                  vscode.window.showErrorMessage(
                    `unexpected error during auto install: ` +
                      e.message +
                      `\n Try installing Rift manually: https://www.github.com/morph-labs/rift`,
                  );
                });
            }
          });
      }
    })
    .catch((e) => {
      console.log("Error while checking the Rift version: ", e);
    });
}

export function morphBinPath(executable: string): string {
  const executablePath: string =
    process.platform === "win32"
      ? `${morphDir}\\env\\Scripts\\${executable}`
      : `${morphDir}/env/bin/${executable}`;
  return executablePath;
}

export function ensureRift(): void {
  if (!vscode.workspace.getConfiguration("rift").get("autostart", true)) {
    return;
  }
  console.log("Start - Checking if `rift` is in PATH.");

  console.log("Command set for 'which'/'where' based on platform.");

  const riftExecutablePath: string = morphBinPath("rift");

  const riftInMorphDir = fs.existsSync(riftExecutablePath);

  if (!riftInMorphDir) {
    console.error(`rift executable not found in ${riftExecutablePath}`);
    throw new Error(
      `rift executable is not found in ${riftExecutablePath}. Please make sure it is correctly installed and try again.`,
    );
  }
  console.log(`End - rift found in PATH or ${riftExecutablePath}`);

  checkRiftVersion();
}

// invoke this optionally via popup in case `ensureRift` fails
async function autoInstall() {
  console.log("Executing: const morphDir = path.join(os.homedir(), '.morph');");
  const morphDir = path.join(os.homedir(), ".morph");
  console.log("Executing: if (!fs.existsSync(morphDir))...");
  if (!fs.existsSync(morphDir)) {
    console.log("Executing: fs.mkdirSync(morphDir);");
    fs.mkdirSync(morphDir);
  }
  console.log(
    'Executing: const pythonCommands = process.platform === "win32" ?...',
  );
  const pythonCommands =
    process.platform === "win32"
      ? ["py -3.10", "py -3", "py"]
      : ["python3.10", "python3", "python"];
  console.log("Executing: let pythonCommand: string | null = null;");
  let pythonCommand: string | null = null;
  console.log("Executing: for... loop over pythonCommands");
  for (const command of pythonCommands) {
    console.log(
      "Executing: const { stdout } = await exec(`${command} --version`);",
    );
    console.log(`Command: ${command}`);
    try {
      const { stdout } = await aexec(`${command} --version`);
      console.log(
        "Executing: const versionMatch = stdout.match(/Python (d+.d+)(?:.d+)?/);",
      );
      const versionMatch = stdout.match(/Python (\d+\.\d+)(?:\.\d+)?/);
      if (versionMatch) {
        // Coerce the matched version to a semver object
        const version = semver.coerce(versionMatch[1]);
        // Compare the coerced version with the desired version
        if (
          semver.gte(
            version || new semver.SemVer("3.10.0"),
            new semver.SemVer("3.10.0"),
          )
        ) {
          pythonCommand = command;
          break;
        }
      }
    } catch (error) {
      continue;
    }
  }
  console.log("Executing: if (pythonCommand === null)...");
  if (pythonCommand === null) {
    console.log("Throwing new Error('Python 3.10 or above is not found...');");
    throw new Error(
      "Python 3.10 or above is not found on your system. Please install it and try again.",
    );
  }
  console.log("Executing: const createVenvCommand = `${pythonCommand}...`");
  const createVenvCommand = `${pythonCommand} -m venv ${morphDir}/env`;
  console.log("Executing: await exec(createVenvCommand);");
  await aexec(createVenvCommand);

  console.log(
    "Executing: const activateVenvAndInstallRiftCommand = `source...`",
  );
  const activateVenvAndInstallRiftCommand =
    process.platform === "win32"
      ? `${morphDir}\\env\\Scripts\\pip ${PIP_INSTALL_ARGS}`
      : `${morphDir}/env/bin/pip ${PIP_INSTALL_ARGS}`;

  console.log("Executing: await exec(activateVenvAndInstallRiftCommand);");
  await aexec(activateVenvAndInstallRiftCommand);
  console.log("autoInstall finished");
}

async function autoInstallHook() {
  const autoInstallPromise = autoInstall();
  vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification },
    async (progress) => {
      progress.report({ message: `installing Rift...` });
      await autoInstallPromise.catch((error: any) => {
        vscode.window.showErrorMessage(
          `${error.message}\nEnsure that python3.10 is available and try installing Rift manually: https://www.github.com/morph-labs/rift`,
          "Close",
        );
      });
    },
  );
  await autoInstallPromise;
}

export function ensureRiftHook() {
  /**
   * The ensureRiftHook function handles errors during ensureRift execution. If an error is encountered,
   * an error message is shown in the vscode window with an option "Try auto install". Choosing this
   * option initiates autoInstall. If autoInstall runs successfully, ensureRift is executed again.
   * If new errors appear during these operations, an error message instructs the user on how to install Rift manually.
   */
  try {
    ensureRift(); // ; vscode.commands.executeCommand("rift.start_server");
  } catch (e: any) {
    console.log("ensure rift failed");
    vscode.window
      .showErrorMessage(e.message, "Try auto install")
      .then((selection) => {
        if (selection === "Try auto install") {
          autoInstallHook()
            .then((_) => {
              vscode.window.showInformationMessage(
                "Rift installation successful.",
              );
            })
            .catch((e) =>
              vscode.window.showErrorMessage(
                `unexpected error during auto install: ` +
                  e.message +
                  `\n Try installing Rift manually: https://www.github.com/morph-labs/rift`,
              ),
            )
            .then((_) => {
              // console.log("executeCommand rift.start_server");
              // vscode.commands.executeCommand("rift.start_server");
              vscode.commands.executeCommand("rift.restart");
            });
        }
      });
  }
}

export async function runRiftCodeEngine() {
  // check if port 7797 is already being used, if so clear it
  await tcpPortUsed.check(7797).then((flag) => {
    console.log(`tcpPortUsed=${flag}`);
    if (flag) {
      // const { exec } = require("child_process");
      // exec("fuser -k 7797/tcp"); // execute kill command
      vscode.window
        .showErrorMessage(
          "Error: port 7797 is already in use.",
          "Kill rift processes",
          "Kill processes bound to port 7797",
        )
        .then((selection) => {
          if (selection === "Kill rift processes") {
            if (process.platform === "win32") {
              exec("taskkill /IM rift.exe /F", (err, stdout, stderr) => {
                if (err) {
                  vscode.window.showErrorMessage(
                    "Could not kill the rift processes. Error - " + err.message,
                  );
                }
              });
            } else if (process.platform === "linux") {
              exec("pkill -f rift", (err, stdout, stderr) => {
                if (err) {
                  vscode.window.showErrorMessage(
                    "Could not kill the rift processes. Error - " + err.message,
                  );
                }
              });
            } else if (process.platform === "darwin") {
              exec("pkill -f rift", (err, stdout, stderr) => {
                if (err) {
                  vscode.window.showErrorMessage(
                    "Could not kill the rift processes. Error - " + err.message,
                  );
                }
              });
            } else {
              vscode.window.showErrorMessage(
                "Sorry, this feature is not supported on your platform.",
              );
            }
          }
          if (selection === "Kill processes bound to port 7797") {
            if (process.platform === "win32") {
              exec(
                'FOR /F "tokens=5" %a IN (\'netstat -aon ^| find "7797" ^| find "LISTENING"\') DO taskkill /F /PID %a',
                (err, stdout, stderr) => {
                  if (err) {
                    vscode.window.showErrorMessage(
                      "Could not kill the port 7797 processes. Error - " +
                        err.message,
                    );
                  }
                },
              );
            } else if (
              process.platform === "linux" ||
              process.platform === "darwin"
            ) {
              exec("fuser -k 7797/tcp", (err, stdout, stderr) => {
                if (err) {
                  vscode.window.showErrorMessage(
                    "Could not kill the port 7797 processes. Error - " +
                      err.message,
                  );
                }
              });
            } else {
              vscode.window.showErrorMessage(
                "Sorry, this feature is not supported on your platform.",
              );
            }
          }
        });
    }
  });

  aexec(`${morphDir}/env/bin/rift`)
    .then((_) => {
      vscode.window.showInformationMessage(
        "Rift Code Engine started successfully.",
      );
    })
    .catch((e) => {
      vscode.window.showErrorMessage(
        "unexpected error: " +
          e.message +
          "\nTry installing Rift manually: https://www.github.com/morph-labs/rift",
      );
    });
  // });
}

async function runRiftCodeEngineWithProgress() {
  vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification },
    async (progress) => {
      progress.report({ message: `Starting Rift Code Engine...` });
      // await runRiftCodeEngine();
    },
  );
}

// vscode.commands.registerCommand("rift.start_server", runRiftCodeEngineWithProgress);

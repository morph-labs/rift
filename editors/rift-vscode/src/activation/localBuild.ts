import * as path from "path";
import * as fs from "fs";
import * as semver from "semver";
import * as toml from "@iarna/toml";
import * as vscode from "vscode";
import * as os from "os";

import * as util from "util";
import fetch from "node-fetch";
import { exec } from "child_process";
const aexec = util.promisify(exec);

// const PACKAGE_JSON_RAW_GITHUB_URL =
//   "https://raw.githubusercontent.com/morph-labs/rift/main/editors/rift-vscode/package.json";

// const WINDOWS_REMOTE_SIGNED_SCRIPTS_ERROR =
//   "A Python virtual enviroment cannot be activated because running scripts is disabled for this user. In order to use Rift, please enable signed scripts to run with this command in PowerShell: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, reload VS Code, and then try again.";

const RIFT_COMMIT = "main";
const PIP_INSTALL_ARGS = `install --upgrade "git+https://github.com/morph-labs/rift.git@${RIFT_COMMIT}#subdirectory=rift-engine&egg=pyrift"`;

export const morphDir = path.join(os.homedir(), ".morph");

export function getExtensionVersion() {
  const extension = vscode.extensions.getExtension("morph.rift-vscode");
  return extension?.packageJSON.version || "";
}

// export function checkExtensionVersion() {
//   fetch(PACKAGE_JSON_RAW_GITHUB_URL)
//     .then(async (res) => res.json())
//     .then((packageJson: any) => {
//       const localVersion = semver.coerce(getExtensionVersion());
//       const remoteVersion = semver.coerce(packageJson.version);
//       if (!localVersion || !remoteVersion) {
//         return console.log("unable to compare extension versions");
//       }
//       if (localVersion.compare(remoteVersion) === -1) {
//         vscode.window.showInformationMessage(
//           `You are using an out-of-date version (${getExtensionVersion()}) of the Rift VSCode Extension (latest ${
//             packageJson.version
//           }). Please update to the latest version from the VSCode Marketplace, or from source at https://www.github.com/morph-labs/rift.`,
//         );
//       }
//     })
//     .catch((e) => console.log("Error checking for extension updates: ", e));
// }

// URL for the raw GitHub content of pyproject.toml in the rift engine
const PYPROJECT_TOML_RAW_GITHUB_URL =
  "https://raw.githubusercontent.com/morph-labs/rift/main/rift-engine/pyproject.toml";

function getLocalRiftVersion(binPath: string): Promise<any> {
  console.log("checking rift version");
  return new Promise((resolve, reject) => {
    // run `python -m rift.__about__` & stores the output
    exec(`${binPath} -v`, (error, stdout, stderr) => {
      if (stdout) {
        console.log(`stdout=${stdout}`);
        const riftVersion = semver.coerce(stdout);
        console.log(`riftVersion=${riftVersion}`);
        resolve(riftVersion);
        return;
      }
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
      resolve(null);
    });
  });
}

export async function upgradeLocalBuildAsNeeded(
  progress: vscode.Progress<{ message?: string; increment?: number }>,
) {
  const localBinPath = vscode.workspace
    .getConfiguration("rift")
    .get<string>("riftPath");

  if (!localBinPath || !fs.existsSync(localBinPath)) {
    return;
  }

  const localRiftVersion = await getLocalRiftVersion(localBinPath);

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
          .then(async (selection) => {
            if (selection === "Try auto update") {
              try {
                const _ = await autoBuild(progress);
                vscode.window.showInformationMessage(
                  "Rift installation successful.",
                );
              } catch (e: any) {
                vscode.window.showErrorMessage(
                  `unexpected error during auto install: ` +
                    e.message +
                    `\n Try installing Rift manually: https://www.github.com/morph-labs/rift`,
                );
              }
            }
          });
      }
    })
    .catch((e) => {
      console.log("Error while checking the Rift version: ", e);
    });
}

// invoke this optionally via popup in case `ensureRift` fails
export async function autoBuild(
  progress: vscode.Progress<{ message?: string; increment?: number }>,
) {
  console.log("Executing: const morphDir = path.join(os.homedir(), '.morph');");
  const morphDir = path.join(os.homedir(), ".morph");
  console.log("Executing: if (!fs.existsSync(morphDir))...");
  if (!fs.existsSync(morphDir)) {
    console.log("Executing: fs.mkdirSync(morphDir);");
    fs.mkdirSync(morphDir);
  }
  progress.report({ message: "Detecting Python version..." });
  const pythonCommands =
    process.platform === "win32"
      ? ["py -3.10", "py -3", "py"]
      : ["python3.10", "python3", "python"];
  console.log("Executing: let pythonCommand: string | null = null;");
  let pythonCommand: string | null = null;
  console.log("Executing: for... loop over pythonCommands");
  for (const command of pythonCommands) {
    console.log(`Command: ${command}`);
    try {
      const { stdout } = await aexec(`${command} --version`);
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

  if (pythonCommand === null) {
    throw new Error(
      "Python 3.10 or above is not found on your system. Please install it and try again.",
    );
  }
  progress.report({ message: "Creating Python environment..." });
  const createVenvCommand = `${pythonCommand} -m venv ${morphDir}/env`;
  console.log("Executing: await exec(createVenvCommand);");
  await aexec(createVenvCommand);

  progress.report({ message: "Rift: Building Server..." });
  const envPath =
    process.platform === "win32"
      ? `${morphDir}\\env\\Scripts\\`
      : `${morphDir}/env/bin/`;

  const activateVenvAndInstallRiftCommand = envPath + `pip ${PIP_INSTALL_ARGS}`;
  await aexec(activateVenvAndInstallRiftCommand);
  await vscode.workspace
    .getConfiguration("rift")
    .update("riftPath", envPath + "rift", true);
}

// TODO: integrate process killing into new startup flow somehow?

// export async function runRiftCodeEngine() {
//   // check if port 7797 is already being used, if so clear it
//   await tcpPortUsed.check(7797).then((flag) => {
//     console.log(`tcpPortUsed=${flag}`);
//     if (flag) {
//       // const { exec } = require("child_process");
//       // exec("fuser -k 7797/tcp"); // execute kill command
//       vscode.window
//         .showErrorMessage(
//           "Error: port 7797 is already in use.",
//           "Kill rift processes",
//           "Kill processes bound to port 7797",
//         )
//         .then((selection) => {
//           if (selection === "Kill rift processes") {
//             if (process.platform === "win32") {
//               exec("taskkill /IM rift.exe /F", (err, stdout, stderr) => {
//                 if (err) {
//                   vscode.window.showErrorMessage(
//                     "Could not kill the rift processes. Error - " + err.message,
//                   );
//                 }
//               });
//             } else if (process.platform === "linux") {
//               exec("pkill -f rift", (err, stdout, stderr) => {
//                 if (err) {
//                   vscode.window.showErrorMessage(
//                     "Could not kill the rift processes. Error - " + err.message,
//                   );
//                 }
//               });
//             } else if (process.platform === "darwin") {
//               exec("pkill -f rift", (err, stdout, stderr) => {
//                 if (err) {
//                   vscode.window.showErrorMessage(
//                     "Could not kill the rift processes. Error - " + err.message,
//                   );
//                 }
//               });
//             } else {
//               vscode.window.showErrorMessage(
//                 "Sorry, this feature is not supported on your platform.",
//               );
//             }
//           }
//           if (selection === "Kill processes bound to port 7797") {
//             if (process.platform === "win32") {
//               exec(
//                 'FOR /F "tokens=5" %a IN (\'netstat -aon ^| find "7797" ^| find "LISTENING"\') DO taskkill /F /PID %a',
//                 (err, stdout, stderr) => {
//                   if (err) {
//                     vscode.window.showErrorMessage(
//                       "Could not kill the port 7797 processes. Error - " +
//                         err.message,
//                     );
//                   }
//                 },
//               );
//             } else if (
//               process.platform === "linux" ||
//               process.platform === "darwin"
//             ) {
//               exec("fuser -k 7797/tcp", (err, stdout, stderr) => {
//                 if (err) {
//                   vscode.window.showErrorMessage(
//                     "Could not kill the port 7797 processes. Error - " +
//                       err.message,
//                   );
//                 }
//               });
//             } else {
//               vscode.window.showErrorMessage(
//                 "Sorry, this feature is not supported on your platform.",
//               );
//             }
//           }
//         });
//     }
//   });
// }

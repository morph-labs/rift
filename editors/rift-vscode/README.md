Rift is an AI-native language server and extension that lets you deploy a personal AI software engineer — locally hosted, private, secure, and free.

Rift and this extension are fully [open-source](https://github.com/morph-labs/rift/tree/main/editors/rift-vscode).

## About
The future of AI code assistants is open-source, private, secure, and on-device. Rift understands, explains, and writes code with language models that run entirely on your device using the open source [Rift Code Engine](https://github.com/morph-labs/rift/tree/main/rift-engine).

## Installation
Install the VSCode extension from the VSCode Marketplace or by building and installing from the VSIX bundle produced by the following steps:

- Increment the semver number (e.g. 0.0.8 to 0.0.9) in the `package.json`
- run `vsce package`
- Install from the VSIX by searching "VSIX" from the VSCode command palette.

## Usage 
1. Ensure the [Rift Code Engine](https://github.com/morph-labs/rift/tree/main/rift-engine) is installed and running on port 7797:

```bash
git clone https://www.github.com/morph-labs/rift
cd rift

# set up a virtual environment with Python (>=3.9), then install the `rift` Python package
pip install -e ./rift-engine

# launch the language server
python -m rift.server.core --host 127.0.0.1 --port 7797
```

This requires a working Python (>=3.9) installation.
2. Access the chat interface by clicking on the sidebar icon.
3. Trigger code completions in the editor window using the keyboard shortcut (`Ctrl + M`) or by running the `Rift: Code Completion` command (`Ctrl + Shift + P`  +  type "Rift"). If the extension is unable to connect to the server, try running the command `Developer: Reload Window`

## Development
See [here](https://github.com/morph-labs/rift/blob/main/editors/rift-vscode/CONTRIBUTING.md) for instructions on how to develop this extension.

## Community
Join our [community]([https://discord.gg/wa5sgWMfqv](https://discord.gg/wa5sgWMfqv)) to share your feedback, get help, and engage with the [Morph](https://morph.so) team. Help us shape the future of software.

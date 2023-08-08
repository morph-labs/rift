# Rift
Rift is an open-source AI-native [language server](https://microsoft.github.io/language-server-protocol/) for the development environments of the future. Rift makes your IDE *agentic*. Software will soon be written mostly by AI software engineers that work alongside you. Codebases will soon be living, spatial artifacts that *maintain context*, *listen to*, *anticipate*, *react to*, and *execute* your every intent. Rift is infrastructure for that future. The [Rift Code Engine](./rift-engine/) provides infrastructure for this by implementing an AI-native extension of the language server protocol. The [Rift VSCode extension](./editors/rift-vscode) implements an client and end-user interface which is the first step into that future.

![rift screencast](assets/rift-screencast.gif) <!-- TODO: pranav -->

- [Discord](https://discord.gg/wa5sgWMfqv)
- [Getting started](#getting-started)
- [Installation](#manual-installation)
- [Features](#features)
- [Usage](#usage)
- [Tips](#tips)
- [The road ahead](#the-road-ahead)
- [FAQ](#faq)

## Features
<!-- TODO(jesse): talk about features available with Rift 2.0 -->

## Usage
<!-- TODO(jesse): write -->

## Tips
<!-- TODO(jesse): write -->

## Getting started
Install the VSCode extension from the VSCode Marketplace. By default, the extension will attempt to automatically start the Rift Code Engine every time the extension is activated. During this process, if the `rift` executable is not found, the extension will ask you to attempt an automatic installation of a Python environment and the Rift Code Engine. To disable this behavior, such as for development, go to the VSCode settings, search for "rift", and set `rift.autostart` to `false`.

If the automatic installation of the Rift Code Engine fails, follow the below instructions for manual installation.

### Manual nstallation
**Rift Code Engine**:
- Set up a Python virtual environment for Python 3.10 or higher.
  - On Mac OSX:
    - Install [homebrew](https://brew.sh).
    - `brew install python@3.10`
    - `mkdir ~/.morph/ && cd ~/.morph/ && python3.10 -m venv env`
    - `source ./env/bin/activate/`
  - On Linux:
    - On Ubuntu:
      - `sudo apt install software-properties-common -y`
      - `sudo add-apt-repository ppa:deadsnakes/ppa`
      - `sudo apt install python3.10 && sudo apt install python3.10-venv`
      - `mkdir ~/.morph/ && cd ~/.morph/ && python3.10 -m venv env`
      - `source ./env/bin/activate/`
    - On Arch:
      - `yay -S python310`
      - `mkdir ~/.morph/ && cd ~/.morph/ && python3.10 -m venv env`
      - `source ./env/bin/activate/`
- Install Rift.
  - Make sure that `which pip` returns a path whose prefix matches the location of a virtual environment, such as the one installed above.
  - Using `pip` and PyPI:
    - `pip install --upgrade pyrift`
  - Using `pip` from GitHub:
    - `pip install "git+https://github.com/morph-labs/rift.git@ea0ee39bd86c331616bdaf3e8c02ed7c913b0933#egg=pyrift&subdirectory=rift-engine"`
  - From source:
    - `cd ~/.morph/ && git clone git@github.com:morph-labs/rift && cd ./rift/rift-engine/ && pip install -e .`
      
**Rift VSCode Extension** (via `code --install-extension`, change the executable as needed):
- `cd ./editors/rift-vscode && npm i && bash reinstall.sh`

## FAQ 
<!-- TODO(jesse): write -->

## The road ahead
<!-- TODO(jesse): rephrase / polish in light of Rift 2.0 -->
Existing code generation tooling is presently mostly code-agnostic, operating at the level of tokens in / tokens out of code LMs. The [language server protocol](https://microsoft.github.io/language-server-protocol/) (LSP) defines a standard for *language servers*, objects which index a codebase and provide structure- and runtime-aware interfaces to external development tools like IDEs.

The Rift Code Engine is an AI-native language server which will expose interfaces for code transformations and code understanding in a uniform, model- and language-agnostic way --- e.g. `rift.summarize_callsites` or `rift.launch_ai_swe_async` should work on a Python codebase with [StarCoder](https://huggingface.co/blog/starcoder) as well as it works on a Rust codebase using [CodeGen](https://github.com/salesforce/CodeGen). Within the language server, models will have full programatic access to language-specific tooling like compilers, unit and integration test frameworks, and static analyzers to produce correct code with minimal user intervention. We will develop UX idioms as needed to support this functionality in the Rift IDE extensions.

## Contributing
We welcome contributions to Rift at all levels of the stack, for example:
- adding support for new open-source models in the Rift Code Engine
- implementing the Rift API for your favorite programming language
- UX polish in the VSCode extension
- adding support for your favorite editor.

See our [contribution guide](/CONTRIBUTORS.md) for details and guidelines.

Programming is evolving. Join the [community](https://discord.gg/wa5sgWMfqv), contribute to our roadmap, and help shape the future of software.

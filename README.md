# Rift

At [Morph](https://morph.so), we believe that:
- Future software will be mostly written by AI, and augmented/supervised by humans.
- Keeping personal data safe and computations secure will become increasingly important.
- The future of software belongs to everyone.

That is why we are releasing Rift, an [open-source language server](./rift-engine) and [IDE extension](./editors/rift-vscode) that lets everyone deploy a personal AI software engineer — locally hosted, private, secure, open-source, and free.

## The road ahead
Existing code generation tooling is presently mostly code-agnostic, operating at the level of tokens in / tokens out of code LMs. The [language server protocol](https://microsoft.github.io/language-server-protocol/) (LSP) defines a standard for *language servers*, objects which index a codebase and provide structure- and runtime-aware interfaces to external development tools like IDEs.

The Rift Code Engine is an AI-native language server which will expose interfaces for code transformations and code understanding in a uniform, model- and language-agnostic way --- e.g. `rift.summarize_callsites` or `rift.launch_ai_swe_async` should work on a Python codebase with [StarCoder](https://huggingface.co/blog/starcoder) as well as it works on a Rust codebase using [CodeGen](https://github.com/salesforce/CodeGen). Within the language server, models will have full programatic access to language-specific tooling like compilers, unit and integration test frameworks, and static analyzers to produce correct code with minimal user intervention. We will develop UX idioms as needed to support this functionality in the Rift IDE extensions.

## Getting started
### VSCode extension
Install the VSCode extension from the VSCode Marketplace or by building and installing from the VSIX bundle produced by the following steps:

- `cd ./editors/rift-vscode`
- Increment the semver number (e.g. 0.0.8 to 0.0.9) in the `package.json`
- run `vsce package`

If you also want to develop the extension itself, you should open the extension in an extension development host using the following steps:
- `cd ./editors/rift-vscode`
- Run `code .` to open a VSCode process on the `rift-vscode` workspace.
- Press `Ctrl + F5` to compile the extension and create a new VSCode window with the extension loaded.

### Rift Code Engine
Run the Rift server by following the instructions in the [readme](./rift-engine/README.md).

## Contributing
We welcome contributions to Rift at all levels of the stack, from adding support for new open-source models in the Rift Code Engine, to implementing the Rift API for your favorite programming language, to UX polish in the VSCode extension, to adding support for your [favorite](./editors/emacs) [editor](./editors/vim).

Programming is evolving. Join the [community](https://discord.gg/wa5sgWMfqv), contribute to our roadmap, and help shape the future of software.

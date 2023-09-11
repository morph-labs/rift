# Rift VSCode Extension

Rift turns your IDE into an intelligent development environment and connects code language models to your codebase and editor. Work 10X faster with your personal team of AI software engineers.

## About

Rift (https://www.github.com/morph-labs/rift) is an open-source AI-native language server that turns VSCode into an IDE from the future. Create and iterate on entire projects by communicating and collaborating with coding agents that can anticipate, maintain context on, and execute on your intentions expressed in natural language.

Today, with Rift, you can do all of the following without ever leaving VSCode or copy-pasting from a browser chat window:

- Generate an entire workspace, or a module based on other parts of your codebase
  ![smol screencast](https://github.com/morph-labs/rift/blob/main/assets/smol.gif?raw=true)
- Conversationally iterate on code edits over selected regions which are streamed directly into your editor window
  ![code edit screencast](https://github.com/morph-labs/rift/blob/main/assets/code-edit.gif?raw=true)
- Request, review, and iterate on PRs: codebase-wide, multi-file diffs
  ![aider screencast](https://github.com/morph-labs/rift/blob/main/assets/aider.gif?raw=true)

## Installation

From the VSCode Marketplace:

- Click on the extension icon in the sidebar
- Search for "Rift"
- Click the install button.

For development / testing:

Run the following steps in a terminal:

```bash
# clone latest version of extension and rift language server
git clone https://www.github.com/morph-labs/rift

# reinstall the extension
cd editors/rift-vscode
bash reinstall.sh # installs the extension to `code`, change the executable as needed
```

Then open VSCode.

## Usage

- Press Command+M to focus the Rift Omnibar.
  - Once focused, you can either engage with the current chat or use a slash-command (e.g. `/aider`) to spawn a new agent.
- Each instance of a Rift Chat or Code Edit agent will remain attached to the open file / selection you used to spawn it.
  - To switch to a new file or request a code edit on a new selection, spawn a new agent by pressing Command+M and running a slash-command (e.g. `/edit`)
  - Both Rift Chat and Code Edit see a window around your cursor or selection in the currently active editor window. To tell them about other resources in your codebase, mention them with `@`.
  - Code Edit
- You can `@`-mention files and directories to tell your agents about other parts of the codebase.
- Currently, Rift works best when the active workspace directory is the same as the root directory of the `git` project.

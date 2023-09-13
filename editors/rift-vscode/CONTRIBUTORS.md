# Contributing to the Rift VSCode Extension

Love Rift and want to get involved? You're in the right place! We appreciate your interest in contributing. This guide will help you understand how you can contribute to the project effectively.

## Table of Contents

- [How to Contribute](#how-to-contribute)
  - [Report Bugs](#report-bugs)
  - [Suggest Features](#suggest-features)
  - [Update Docs](#update-docs)
  - [Write Code](#write-code)
- [VSCode Extension Setup](#vscode-extension-setup)
- [Architecture](#architecture)

## How to Contribute

### Report Bugs

If you encounter any bugs or issues while using the Rift VSCode Extension, please help us improve by reporting them. To report a bug, follow these steps:

1. Go to the [Issues](https://github.com/morph-labs/rift/issues) section of the project on GitHub.
1. Click on the "New Issue" button.
1. Provide a clear and detailed description of the bug you encountered, along with steps to reproduce it if possible.

### Suggest Features

Have a great idea for a new feature? We'd love to hear it! To suggest a new feature for the Rift VSCode Extension, follow these steps:

1. Go to the [Issues](https://github.com/morph-labs/rift/issues) section of the project on GitHub.
1. Click on the "New Issue" button.
1. Describe the feature you'd like to see implemented, providing as much context and use cases as possible.

### Update Docs

Documentation is vital for a successful open-source project. If you find any areas in the documentation that need improvement or want to add missing information, we appreciate your efforts. To contribute to the documentation

### Write Code

We welcome contributions to the Rift VSCode Extension's codebase. If you want to contribute code, please follow these steps:

1. Browse through the Issues section to find an existing issue you'd like to work on or identify a new improvement to make.
1. Comment on the issue to express your interest in working on it and wait for approval from project maintainers.
1. Fork the repository to your GitHub account.
1. Create a new branch for your contribution, with a descriptive name related to the issue or task you're addressing.
1. Make the necessary code changes and improvements.
1. Write tests if applicable to ensure the code behaves as expected.
1. Submit a Pull Request (PR) to the main repository, clearly explaining the changes you made and referencing the related issue.
1. We'll review your PR as soon as possible and provide feedback or merge it into the project.

## VSCode Extension Setup

For development:

1. Open VSCode in this directory.
1. Select "Run and Debug" or press "Ctrl + F5" to load the extension development host.

For testing:

```bash
# clone latest version of extension and rift language server
git clone https://www.github.com/morph-labs/rift

# reinstall the extension
cd editors/rift-vscode
bash reinstall.sh # installs the extension to `code`, change the executable as needed
```

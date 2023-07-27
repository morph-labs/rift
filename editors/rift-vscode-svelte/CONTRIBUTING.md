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

TODO: Provide instructions on setting up the Rift VSCode Extension locally for development and testing purposes.

## Architecture

The following output is the directory structure and organization of rift-vscode:

```sh
tree -I 'node_modules|out|dist'
```

```sh
.
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── build
│   └── node-extension.webpack.config.cjs
├── media
│   ├── icon.svg
│   ├── icons
│   │   ├── resetDark.svg
│   │   └── resetLight.svg
│   ├── reset.css
│   ├── scripts
│   │   ├── microlight.min.js
│   │   ├── showdown.min.js
│   │   └── tailwind.min.js
│   └── vscode.css
├── package-lock.json
├── package.json
├── resources
│   ├── icon.png
│   ├── icon.svg
│   ├── iconold.png
│   └── vsce-icon.png
├── rollup.config.js
├── src
│   ├── client.ts
│   ├── elements
│   │   └── WebviewProvider.ts
│   ├── extension.ts
│   ├── getNonce.ts
│   ├── lib
│   │   ├── PubSub.ts
│   │   └── Store.ts
│   ├── test
│   │   ├── runTest.ts
│   │   └── suite
│   │       ├── extension.test.ts
│   │       └── index.ts
│   └── types.ts
├── tsconfig.json
├── vsc-extension-quickstart.md
├── webpack.config.js
└── webviews
    ├── components
    │   ├── AcceptRejectBar.svelte
    │   ├── ChatWebview.svelte
    │   ├── LogsWebview.svelte
    │   ├── OmniBar.svelte
    │   ├── chat
    │   │   ├── Chat.svelte
    │   │   ├── Response.svelte
    │   │   ├── UserInput.svelte
    │   │   └── dropdown
    │   │       ├── Dropdown.svelte
    │   │       └── DropdownCard.svelte
    │   ├── icons
    │   │   ├── AcceptRejectCheck.svelte
    │   │   ├── AcceptRejectClose.svelte
    │   │   ├── ArrowDownSvg.svelte
    │   │   ├── ArrowRightSvg.svelte
    │   │   ├── ChatSvg.svelte
    │   │   ├── CopySvg.svelte
    │   │   ├── EllipsisDarkSvg.svelte
    │   │   ├── EllipsisSvg.svelte
    │   │   ├── LogGreenSvg.svelte
    │   │   ├── LogRedSvg.svelte
    │   │   ├── LogYellowSvg.svelte
    │   │   ├── ResetSvg.svelte
    │   │   ├── RiftSvg.svelte
    │   │   ├── SendSvg.svelte
    │   │   ├── UserSvg.svelte
    │   │   ├── copySvg.js
    │   │   ├── oldUserSvg.svelte
    │   │   └── vscode-icon.svg
    │   ├── logs
    │   │   ├── Agent.svelte
    │   │   ├── Log.svelte
    │   │   └── Logs.svelte
    │   └── stores.ts
    ├── globals.d.ts
    ├── pages
    │   ├── Chat.ts
    │   └── Logs.ts
    └── tsconfig.json

18 directories, 68 files

```

Thank you for your interest in contributing to Rift VSCode Extension! Happy coding!

#!/bin/bash

# Step 1: Removes files matching *.vsix
find . -name "*.vsix" -type f -delete

# Step 2: Run the following commands

echo "Uninstalling extension..."
code --uninstall-extension morph.rift-vscode

echo "Running 'npm run clean'..."
npm run clean

echo "Running npm i"
npm i

echo "Creating VSIX package..."
npx @vscode/vsce package

# You may need to replace the '*.vsix' wildcard with the actual VSIX filename if it's different.
echo "Installing extension..."
code --install-extension *.vsix

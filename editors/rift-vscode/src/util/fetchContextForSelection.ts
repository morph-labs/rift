import * as vscode from "vscode";

export const fetchContextForSelection = async (
  uri: vscode.Uri,
  selection: vscode.Selection
) => {
  const diagnostics = vscode.languages.getDiagnostics(uri);

  const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
    "vscode.executeDocumentSymbolProvider",
    uri
  );
  if (!symbols) {
    vscode.window.showErrorMessage(
      "Error: no document symbols available, results may suffer."
    );
  }

  const activeSymbol = (symbols ?? []).find(
    (s) =>
      s.range.start.isBeforeOrEqual(selection.start) &&
      s.range.end.isAfterOrEqual(selection.end)
  );

  let activeRange = new vscode.Range(selection.start, selection.end);
  if (activeSymbol) {
    activeRange = activeSymbol.range;
  }

  const childrenHovers: {
    name: string;
    hover: string;
    range: vscode.Range;
  }[] = [];
  const collectChildrenHovers = async (n?: vscode.DocumentSymbol) => {
    await Promise.all(
      (n?.children ?? []).map((child) =>
        vscode.commands
          .executeCommand<vscode.Hover[]>(
            "vscode.executeHoverProvider",
            uri,
            child.range.start
          )
          .then((hovers) => {
            childrenHovers.push(
              ...hovers
                .filter((h) => h.range?.intersection(activeRange))
                .map((h) => ({
                  name: child.name,
                  range: child.range,
                  hover: h.contents
                    .map((c) => (typeof c === "string" ? c : c.value))
                    .join("\n"),
                }))
            );
          })
          .then(() => collectChildrenHovers(child))
      )
    );
  };

  for (const symbol of symbols) {
    if (symbol.range.intersection(activeRange)) {
      await collectChildrenHovers(symbol);
    }
  }

  const filteredDiagnostics = diagnostics.filter((d) =>
    d.range.intersection(activeRange)
  );

  console.log({
    selection,
    diagnostics,
    filteredDiagnostics,
    symbols,
    activeSymbol,
    activeRange,
    childrenHovers,
  });

  const stringRange = (r: vscode.Range) =>
    `${r.start.line}:${r.start.character}-${r.end.line}:${r.end.character}`;

  const oldText = await vscode.env.clipboard.readText();
  await vscode.commands.executeCommand(
    "workbench.action.terminal.copyLastCommandOutput"
  );
  // warning: this will erroneously be set to the old paste contents when no terminal is open
  const terminalResponse = await vscode.env.clipboard.readText();
  await vscode.env.clipboard.writeText(oldText);

  return {
    // Text content is shared via other means
    // text: editor.document
    //   .getText(activeRange)
    //   .split("\n")
    //   .map((line, num) => `${num + activeRange.start.line}: ${line}`)
    //   .join("\n"),

    diagnostics: filteredDiagnostics
      .map((d) => `${stringRange(d.range)}: ${d.message}`)
      .join("\n"),

    hovers: childrenHovers
      .map((c) =>
        `${stringRange(c.range)}: ${c.name} => ${c.hover.replace(
          /\s*\n\s*/g,
          "\n"
        )}`.trim()
      )
      .join("\n\n"),

    terminalResponse,
  };
};

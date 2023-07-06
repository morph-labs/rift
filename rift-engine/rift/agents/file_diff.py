from dataclasses import dataclass
from diff_match_patch import diff_match_patch
from rift.lsp import (
    CreateFile,
    TextEdit,
    Range,
    TextDocumentEdit,
    TextDocumentIdentifier,
    WorkspaceEdit,
)
from rift.lsp.types import ChangeAnnotation
import os
from typing import List, Tuple

@dataclass
class FileChange:
    uri: TextDocumentIdentifier
    old_content: str
    new_content: str
    is_new_file: bool = False

def get_file_change(path: str, new_content: str) -> FileChange:
    uri = TextDocumentIdentifier(uri="file://" + path, version=None)
    if os.path.isfile(path):
        with open(path, "r") as f:
            old_content = f.read()
            return FileChange(uri=uri, old_content=old_content, new_content=new_content)
    else:
        return FileChange(uri=uri, old_content="", new_content=new_content, is_new_file=True)

def edits_from_file_change(
    file_change: FileChange
) -> WorkspaceEdit:
    dmp = diff_match_patch()
    diff = dmp.diff_main(file_change.old_content, file_change.new_content)

    line = 0  # current line number
    char = 0  # current character position within the line
    edits = []  # list of TextEdit objects

    for op, text in diff:
        # count the number of lines in `text` and the number of characters in the last line
        lines = text.split("\n")
        last_line_chars = len(lines[-1])
        line_count = len(lines) - 1  # don't count the current line

        end_line = line + line_count
        end_char = (
            char + last_line_chars if line_count == 0 else last_line_chars
        )  # if we moved to a new line, start at char 0

        if op == -1:
            # text was deleted
            edits.append(TextEdit(Range.mk(line, char, end_line, end_char), ""))
        elif op == 1:
            # text was added
            edits.append(
                TextEdit(Range.mk(line, char, line, char), text)
            )  # new text starts at the current position

        # update position
        line = end_line
        char = end_char

    documentChanges = []
    if file_change.is_new_file:
        documentChanges.append(CreateFile(kind="create", uri=file_change.uri.uri))
    documentChanges.append(TextDocumentEdit(textDocument=file_change.uri, edits=edits))
    return WorkspaceEdit(documentChanges=documentChanges)

def edits_from_file_changes(
        file_changes: List[FileChange]  
) -> WorkspaceEdit:
    documentChanges = []
    for file_change in file_changes:
        documentChanges.append(edits_from_file_change(file_change=file_change).documentChanges)
    return WorkspaceEdit(documentChanges=documentChanges)

if __name__ == "__main__":
    file1 = "tests/diff/file1.txt"
    file2 = "tests/diff/file2.txt"
    with open(file1, "r") as f1, open(file2, "r") as f2:
        uri = TextDocumentIdentifier(uri="file://" + file1, version=None)
        file_change = get_file_change(path=file1, new_content=f2.read())
        workspace_edit = edits_from_file_change(file_change=file_change)
        print(f"\nworkspace_edit: {workspace_edit}\n")
        dummy_path = "dummy.txt"
        dummy_content = "dummy content"
        file_change = get_file_change(path=dummy_path, new_content=dummy_content)
        workspace_edit = edits_from_file_change(file_change=file_change)
        print(f"\ntest_new_file: {workspace_edit}\n")

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

def apply_diff_edits_many(
    data: List[Tuple[TextDocumentIdentifier, str, str]]
) -> List[WorkspaceEdit]:
    
    workspace_edits = []
    
    for uri, old_content, new_content in data:
        dmp = diff_match_patch()
        diff = dmp.diff_main(old_content, new_content)

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

        workspace_edits.append(WorkspaceEdit(
            documentChanges=[TextDocumentEdit(uri, edits)],
            changeAnnotations=['hi', ChangeAnnotation(
                label='smol developer says hi',
                needsConfirmation=True,
                description='smol developer says wahats up'
            )
            ]
            ))

    return workspace_edits

def apply_diff_edits(
    uri: TextDocumentIdentifier, old_content: str, new_content: str
) -> WorkspaceEdit:
    dmp = diff_match_patch()
    diff = dmp.diff_main(old_content, new_content)

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

    return WorkspaceEdit(documentChanges=[TextDocumentEdit(uri, edits)])


def on_file_change(uri: TextDocumentIdentifier, new_content: str) -> WorkspaceEdit:
    """
    Handles changes to a file identified by the given URI by comparing the new content
    with the previous content and generating a workspace edit that reflects the changes.

    Args:
        uri (TextDocumentIdentifier): The identifier for the file that has changed.
        new_content (str): The new content of the file.

    Returns:
        WorkspaceEdit: A workspace edit that contains the necessary changes to be applied to the file.

    Note:
        This does not perform any changes to the file itself. It only generates the workspace edit
    """
    if os.path.isfile(uri.uri):
        with open(uri.uri, "r") as f:
            old_content = f.read()
            return apply_diff_edits(uri=uri, old_content=old_content, new_content=new_content)
    else:
        workspace_edits = apply_diff_edits(uri, "", new_content)
        if not workspace_edits.documentChanges:
            workspace_edits.documentChanges = []
        workspace_edits.documentChanges.insert(0, CreateFile(kind="create", uri=uri.uri))
        return workspace_edits


if __name__ == "__main__":
    file1 = "tests/diff/file1.txt"
    file2 = "tests/diff/file2.txt"
    with open(file1, "r") as f1, open(file2, "r") as f2:
        uri = TextDocumentIdentifier(uri="file://" + file1, version=None)
        workspace_edit = on_file_change(uri=uri, new_content=f2.read())
        print(f"\nworkspace_edit: {workspace_edit}\n")
        dummy_uri = TextDocumentIdentifier(uri="file://" + "dummy.txt", version=None)
        dummy_content = "dummy content"
        test_new_file = on_file_change(uri=dummy_uri, new_content=dummy_content)
        print(f"\ntest_new_file: {test_new_file}\n")

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import rift.lsp.types as lsp
from diff_match_patch import diff_match_patch
from rift.lsp import (
    CreateFile,
    Range,
    TextDocumentEdit,
    TextDocumentIdentifier,
    TextEdit,
    WorkspaceEdit,
)
from rift.lsp.types import ChangeAnnotation


@dataclass
class FileChange:
    uri: TextDocumentIdentifier
    old_content: str
    new_content: str
    description: Optional[str] = None
    is_new_file: bool = False
    annotation_label: Optional[str] = None


def get_file_change(path: str, new_content: str) -> FileChange:
    """
    This function is used to generate a FileChange instance from a given file path and string of new content.
    If the file at the specified path doesn't exist, an empty string would be assigned as old_content, thus indicating a new file creation.

    Parameters:
    path: A string representing the path of the file to be changed.
    new_content: A string representing the new content to be written into the file.

    Returns:
    A FileChange instance that represents the changes to be made in the source file.
    """
    uri = TextDocumentIdentifier(uri="file://" + path, version=0)
    if os.path.isfile(path):
        with open(path, "r") as f:
            old_content = f.read()
            return FileChange(uri=uri, old_content=old_content, new_content=new_content)
    else:
        return FileChange(uri=uri, old_content="", new_content=new_content, is_new_file=True)


def edits_from_file_change(
    file_change: FileChange, user_confirmation: bool = False
) -> WorkspaceEdit:
    dmp = diff_match_patch()
    diff = dmp.diff_lineMode(file_change.old_content, file_change.new_content, None)
    dmp.diff_cleanupSemantic(diff)

    line = 0  # current line number
    char = 0  # current character position within the line
    edits = []  # list of TextEdit objects
    annotation_label = file_change.annotation_label or "rift"

    new_text = ""

    for op, text in diff:
        if op == -1:  # remove
            pass
        elif op == 0:  # keep
            new_text += text
        elif op == 1:  # add
            new_text += text

    lines = file_change.old_content.split("\n")
    edits = [
        TextEdit(
            Range.mk(0, 0, len(lines) - 1, len(lines[-1])), new_text, annotationId=annotation_label
        )
    ]

    documentChanges = []

    changeAnnotations: dict[lsp.ChangeAnnotationIdentifier, lsp.ChangeAnnotation] = dict()
    if file_change.is_new_file:
        documentChanges.append(
            CreateFile(kind="create", uri=file_change.uri.uri, annotationId=annotation_label)
        )
    documentChanges.append(TextDocumentEdit(textDocument=file_change.uri, edits=edits))
    changeAnnotations[annotation_label] = lsp.ChangeAnnotation(
        label=annotation_label, needsConfirmation=user_confirmation, description=None
    )
    return WorkspaceEdit(documentChanges=documentChanges, changeAnnotations=changeAnnotations)


def edits_from_file_changes(
    file_changes: List[FileChange], user_confirmation: bool = False
) -> WorkspaceEdit:
    """
    Generate a WorkspaceEdit by aggregating the edits from multiple FileChanges.

    Parameters:
    file_changes: A list of FileChange, each representing a change to a file.
    user_confirmation: Whether the user should confirm each modification manually. False by default.

    Returns:
    WorkspaceEdit containing the aggregated documentChanges and changeAnnotations.
    """

    # List to store all document changes.
    documentChanges: List[
        Union[lsp.TextDocumentEdit, lsp.CreateFile, lsp.RenameFile, lsp.DeleteFile]
    ] = []

    # Dictionary to store all change annotations.
    changeAnnotations: Dict[ChangeAnnotationIdentifier, ChangeAnnotation] = dict()

    # Iterate through each file change.
    for file_change in file_changes:
        # Generate a WorkspaceEdit for this file change.
        edit = edits_from_file_change(file_change=file_change, user_confirmation=user_confirmation)

        # Add the document changes for this workspace edit to our list.
        documentChanges += edit.documentChanges

        # If any changeAnnotations were made in our workspace edit, update our dictionary with them.
        if edit.changeAnnotations is not None:
            changeAnnotations.update(edit.changeAnnotations)

    # Return a new WorkspaceEdit that aggregates all the edits made to each file.
    return WorkspaceEdit(documentChanges=documentChanges, changeAnnotations=changeAnnotations)


if __name__ == "__main__":
    file1 = "tests/diff/file1.txt"
    file2 = "tests/diff/file2.txt"
    with open(file1, "r") as f1, open(file2, "r") as f2:
        uri = TextDocumentIdentifier(uri="file://" + file1, version=0)
        file_change = get_file_change(path=file1, new_content=f2.read())
        workspace_edit = edits_from_file_change(file_change=file_change)
        print(f"\nworkspace_edit: {workspace_edit}\n")
        dummy_path = "dummy.txt"
        dummy_content = "dummy content"
        file_change = get_file_change(path=dummy_path, new_content=dummy_content)
        workspace_edit = edits_from_file_change(file_change=file_change)
        print(f"\ntest_new_file: {workspace_edit}\n")

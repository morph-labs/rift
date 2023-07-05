"""
Author: E.W.Ayers <contact@edayers.com>
This file is adapted from  https://github.com/EdAyers/sss
"""
from dataclasses import dataclass, replace
import logging
from typing import Any, Awaitable, Callable, Optional, Union
import asyncio
from rift.util.misc import set_ctx
from .types import (
    InitializeParams,
    InitializeResult,
    PeerInfo,
    ServerCapabilities,
    ApplyWorkspaceEditParams,
    ApplyWorkspaceEditResponse,
)
import rift.lsp.types as lsp
from collections import defaultdict
from ..rpc import InitializationMode, rpc_method
from ..rpc.extrarpc import ExtraRpc
from rift.util.ofdict import ofdict

""" Implementation of an LSP server """

logger = logging.getLogger("LSP")


class LspServer(ExtraRpc):
    capabilities: ServerCapabilities
    position_encoding = "utf-16"
    # [todo] consider using io.StringIO for the documents because they are mutating.
    documents: dict[lsp.DocumentUri, lsp.TextDocumentItem]
    change_callbacks: defaultdict[lsp.DocumentUri, set[Callable]]
    fts: dict[str, asyncio.Future]
    """ set of open documents, the server will keep these synced with the client editor automatically. """

    def __init__(self, transport):
        self.change_callbacks = defaultdict(set)
        self.capabilities = ServerCapabilities()
        self.documents = dict()
        self.fts = dict()
        super().__init__(transport, init_mode=InitializationMode.ExpectInit)

    @rpc_method("initialize")
    async def on_initialize(self, params: InitializeParams) -> InitializeResult:
        # [todo] inject lsp capabilities here.
        logger.info(f"initializing LSP server {self.name}")
        return InitializeResult(
            serverInfo=PeerInfo(name=self.name, version=None),
            capabilities=self.capabilities,
        )

    @rpc_method("initialized")
    async def on_client_initialized(self, params):
        logger.info("client initialized.")

    async def apply_insert_text(
        self, uri: lsp.DocumentUri, position: lsp.Position, text: str, version: int = 0
    ):
        assert version is not None, "version must be given, or we get no edit."
        textDocument = lsp.TextDocumentIdentifier(uri=uri, version=version)  # [todo] version
        newText = text
        pos = position
        params = lsp.ApplyWorkspaceEditParams(
            edit=lsp.WorkspaceEdit(
                documentChanges=[
                    lsp.TextDocumentEdit(
                        textDocument=textDocument,
                        edits=[
                            lsp.TextEdit(
                                range=lsp.Range(start=pos, end=pos),
                                newText=newText,
                            )
                        ],
                    )
                ]
            )
        )
        return await self.apply_workspace_edit(params)

    async def apply_workspace_edit(
        self, params: ApplyWorkspaceEditParams
    ) -> ApplyWorkspaceEditResponse:
        if isinstance(params, ApplyWorkspaceEditParams):
            response = await self.request("workspace/applyEdit", params)
            response = ofdict(ApplyWorkspaceEditResponse, response)
            return response
        else:
            raise TypeError("expected ApplyWorkspaceEditParams or InsertionEdit")

    @rpc_method("textDocument/didOpen")
    def on_did_open(self, params: lsp.DidOpenTextDocumentParams):
        item = params.textDocument
        logger.debug(f"editor opened {item.uri}")
        self.documents[item.uri] = item

    @rpc_method("textDocument/didChange")
    async def _on_did_change(self, params: lsp.DidChangeTextDocumentParams):
        item_id = params.textDocument
        document = self.documents.get(item_id.uri, None)
        if document is None:
            logger.error(f"document {item_id.uri} not opened")
            return
        text = document.text
        for change in params.contentChanges:
            text = change.apply(text)
        document_after = replace(document, version=item_id.version, text=text)
        self.documents[item_id.uri] = document_after

        kwargs: Any = dict(before=document, after=document_after, changes=params)

        callbacks: list[Any] = [
            callback(**kwargs) for callback in self.change_callbacks[item_id.uri]
        ]
        callbacks.append(self.on_change(**kwargs))
        await asyncio.wait(map(asyncio.ensure_future, callbacks))

    def register_change_callback(self, callback, uri: lsp.DocumentUri):
        """Register a callback to be called when the document changes"""
        self.change_callbacks[uri].add(callback)

    async def on_change(
        self,
        *,
        before: lsp.TextDocumentItem,
        after: lsp.TextDocumentItem,
        changes: lsp.DidChangeTextDocumentParams,
    ):
        """Override this method to handle document changes"""
        pass

    @rpc_method("textDocument/didSave")
    def on_did_save(self, params: lsp.DidSaveTextDocumentParams):
        pass

    @rpc_method("textDocument/didClose")
    def on_did_close(self, params: lsp.DidCloseTextDocumentParams):
        pass

    @rpc_method("$/setTrace")
    def on_set_trace(self, params: lsp.SetTraceParams):
        # [todo] logging stuff.
        pass

    async def get_workspace_configuration(
        self, section: Optional[str] = None, scopeUri: Optional[lsp.DocumentUri] = None
    ) -> list[dict[str, Any]]:
        """Runs the workspace/configuration request."""
        params = lsp.ConfigurationParams(
            items=[lsp.ConfigurationItem(section=section, scopeUri=scopeUri)]
        )
        return await self.request("workspace/configuration", params)

from rift.server.core import CodeCapabilitiesServer
import asyncio
from rift.rpc.jsonrpc import RpcServer, rpc_request, rpc_method
from rift.rpc.io_transport import AsyncStreamTransport
from typing import Any
from rift.server.core import splash
import logging
from rift.lsp.types import InitializeParams
from rift.util.ofdict import todict
import rift.lsp.types as lsp

if __name__ == "__main__":
    class MockLspClient(RpcServer):
        @rpc_request('morph/run')
        async def run(self, params: Any) -> Any:
            ...

        @rpc_request('initialize')
        async def initialize(self, params: InitializeParams) -> Any:
            ...

        @rpc_method('morph/chat_progress')
        async def chat_progress(self, params: Any):
            print("PROGRESS: ", params)

        @rpc_method('window/logMessage')
        async def logmessage(self, params: Any):
            ...

        @rpc_method('morph/code_completion_1_send_progress')
        async def chat_progress(self, params: Any):
            print("PROGRESS: ", params)            
            
        @rpc_method('morph/smol_dev_1_request_chat')
        async def smol_agent_chat(self, params: Any):
            print("SMOL CHAT: ", params)
            return {"message": "print hello world in a python file"}
        
        @rpc_method('morph/smol_dev_1_send_progress')
        async def smol_agent_progress(self, params: Any):
            print("SMOL PROGRESS: ", params)            

        @rpc_method('window/logMessage')
        async def logmessage(self, params: Any):
            ...

        @rpc_method('workspace/applyEdit')
        async def applyEdit(self, params: Any) -> lsp.ApplyWorkspaceEditResponse:
            print("**********************")
            print("**********************")
            print("**********************")
            print("**********************")
            print("**********************")
            print("EDIT: ", params)
            return {"applied": True}

        @rpc_request('textDocument/didOpen')
        async def on_did_open(self, params: lsp.DidOpenTextDocumentParams):
            ...
            
    async def main():
        reader, writer = await asyncio.open_connection("127.0.0.1", 7797)
        transport = AsyncStreamTransport(reader, writer)
        client = MockLspClient(transport=transport)
        t = asyncio.create_task(client.listen_forever())
        print("CAPABILITIES: ", await client.initialize(params=InitializeParams()))
        from pydantic import BaseModel
        from rift.server.chat_agent import RunChatParams

        # register a file
        on_did_open_params = lsp.DidOpenTextDocumentParams(textDocument=lsp.TextDocumentItem(text="yeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeehaw", uri="file:///home/pv/Downloads/yeehaw-dev/yeehaw.py", languageId="python", version=0))
        print("REGISTER FILE: ", await client.on_did_open(params=on_did_open_params))
        
        from rift.agents.smol import SmolAgentParams
        from rift.server.lsp import RunAgentParams
        class RunParams(BaseModel):
            agent_type: str = "chat"
        params = todict(
            RunAgentParams(agent_type="smol_dev", agent_params=SmolAgentParams(instructionPrompt="write hello world in Python", position=lsp.Position(0,0), textDocument=lsp.TextDocumentIdentifier(uri="file:///home/pv/Downloads/yeehaw-dev/yeehaw.py", version=0)))
        )
        print("RUN RESULT: ", await client.run(params=params))
        print("initialized")
        await t

    asyncio.run(main())

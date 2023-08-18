import asyncio
import json
import logging
import os
import pickle as pkl
from dataclasses import dataclass
from typing import Any, List

import rift.lsp.types as lsp
import rift.server.core as core
import rift.server.lsp as server
import rift.util.file_diff as file_diff
import smol_dev
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rift.agents.abstract import AgentRegistryResult
from rift.lsp.types import InitializeParams
from rift.rpc.io_transport import AsyncStreamTransport
from rift.rpc.jsonrpc import RpcServer, rpc_method, rpc_request
from rift.server.core import CodeCapabilitiesServer, rift_splash
from rift.util.ofdict import todict

logger = logging.getLogger(__name__)
import time


@dataclass
class SendUpdateParams:
    msg: str


def smol_splash():
    _splash = """

   __                                 __  
  / /   but make it...           __   \ \ 
 | |             ___ __ _  ___  / /    | |
 | |            (_-</  ' \\/ _ \\/ /     | |
 | |           /___/_/_/_/\\___/_/      | |
 | |                                   | |
  \_\                                 /_/
    
                    

    """

    def stream_string(string):
        for char in string:
            print(char, end="", flush=True)
            time.sleep(0.0015)
            # print('\r', end='')

    stream_string(_splash)


# class RiftClient(RpcServer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         time.sleep(1)
#         smol_splash()

#     @rpc_request("morph/send_update")
#     async def run(self, params: SendUpdateParams) -> None:
#         ...

#     @rpc_request("initialize")
#     async def initialize(self, params: InitializeParams) -> lsp.InitializeResult:
#         ...

#     @rpc_request("morph/applyWorkspaceEdit")
#     async def apply_workspace_edit(params: lsp.ApplyWorkspaceEditParams) -> lsp.ApplyWorkspaceEditResponse:
#         ...

#     @rpc_request("morph/loadFiles")
#     async def load_files(params: server.LoadFilesParams) -> server.LoadFilesResult:
#         ...


async def main():
    reader, writer = await asyncio.open_connection("127.0.0.1", 7797)
    transport = AsyncStreamTransport(reader, writer)
    client = RiftClient(transport=transport)


import asyncio
from concurrent.futures import ThreadPoolExecutor


async def ainput(prompt: str = "") -> str:
    with ThreadPoolExecutor(1, "AsyncInput") as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, input, prompt)


async def main(params):
    FORMAT = "%(message)s"
    console = Console(stderr=True)
    logging.basicConfig(
        level=logging.DEBUG if params.debug else logging.INFO,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(console=console)],
    )
    client: core.CodeCapabilitiesServer = core.create_metaserver(port=params.port)
    logger.info(f"started Rift server on port {params.port}")
    t = asyncio.create_task(client.run_forever())
    await asyncio.sleep(2)
    smol_splash()

    console.print("\n> Press any key to continue.\n")
    await ainput()

    with open(params.prompt_file, "r") as f:
        prompt = f.read()

    logger.info("Starting smol-dev with prompt:")
    console.print(prompt, markup=True, highlight=True)
    console.print("\n> Press any key to continue.\n")
    await ainput()

    def stream_handler(chunk):
        def stream_string(string):
            for char in string:
                print(char, end="", flush=True)
                time.sleep(0.0012)

        stream_string(chunk.decode("utf-8"))

    plan = smol_dev.plan(prompt, streamHandler=stream_handler)

    logger.info("Running with plan:")

    console.print(plan, emoji=True, markup=True)

    console.print("\n> Press any key to continue.\n")
    await ainput()

    file_paths = smol_dev.specify_filePaths(prompt, plan)

    logger.info("Got file paths:")
    console.print(json.dumps(file_paths), markup=True)

    file_changes = []

    console.print("\n> Press any key to continue.\n")
    await ainput()

    for file_path in file_paths:
        logger.info(f"Generating code for {file_path}")
        code = smol_dev.generate_code(prompt, plan, file_path, streamHandler=stream_handler)
        console.print(
            f"""```\
            {code}
            ```
            """,
            markup=True,
        )
        absolute_file_path = os.path.join(os.getcwd(), file_path)
        logger.info(f"Generating a diff for {absolute_file_path}")
        file_change = file_diff.get_file_change(path=absolute_file_path, new_content=code)
        await client.server.apply_workspace_edit(
            lsp.ApplyWorkspaceEditParams(
                file_diff.edits_from_file_changes([file_change], user_confirmation=True),
                label="rift",
            )
        )
        # file_changes.append(file_change)

    # finalWorkspaceEdit = file_diff.edits_from_file_changes(file_changes, user_confirmation=True)
    # logger.info(f"DOCUMENT CHANGES: {finalWorkspaceEdit.documentChanges}")
    # document_change = finalWorkspaceEdit.documentChanges[0]
    # document_change.textDocument.uri = "file://generated/manifest.json"
    # finalWorkspaceEdit = lsp.WorkspaceEdit(documentChanges=finalWorkspaceEdit.documentChanges[:1], changeAnnotations=finalWorkspaceEdit.changeAnnotations)
    # logger.info(f"Applying workspace edit")
    # with open("edit.pkl", "wb") as f:
    #     pkl.dump(finalWorkspaceEdit, f)
    # response = await client.server.apply_workspace_edit(lsp.ApplyWorkspaceEditParams(edit=finalWorkspaceEdit, label="rift"))
    # logger.info(f"RESPONSE: {response}")
    await t


def _main(prompt_file: str, port: int = 7797, debug: bool = False):
    @dataclass
    class ClientParams:
        prompt_file: str
        port: int
        debug: bool = False

    params = ClientParams(prompt_file=prompt_file, port=port)
    asyncio.run(main(params=params), debug=debug)


if __name__ == "__main__":
    import fire

    fire.Fire(_main)

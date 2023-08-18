import asyncio
import dataclasses
import inspect
import json
import logging
import os
import pickle as pkl
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, ClassVar, Dict, List, Literal, Optional, Type

import rift.lsp.types as lsp
import rift.server.core as core
import rift.server.lsp as server
import rift.util.file_diff as file_diff
import tqdm.asyncio
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rift.lsp.types import InitializeParams
from rift.rpc.io_transport import AsyncStreamTransport
from rift.rpc.jsonrpc import RpcServer, rpc_method, rpc_request
from rift.server.core import CodeCapabilitiesServer, rift_splash
from rift.util.ofdict import todict

logger = logging.getLogger(__name__)
import time
import types

import art
import fire

try:
    import smol_dev
except ImportError:
    raise Exception(
        "`smol_dev` not found. Try `pip install -e rift-engine[smol-dev]` from the Rift root directory."
    )
from rift.agents.cli.agent import Agent, ClientParams, launcher
from rift.agents.cli.util import ainput


@dataclass
class SmolAgentClientParams(ClientParams):
    """
    This class is used to specify the parameters for the SmolAgent.
    It inherits from the ClientParams class.

    Attributes:
    prompt_file: Optional[str] - The path to the prompt file. If not provided, the user will be asked to input a prompt.
    debug: bool - A flag to indicate whether the application is in debug mode. Default is False.
    model: Literal["gpt-3.5-turbo-0613", "gpt-4-0613"] - The model to be used. Default is "gpt-3.5-turbo-0613".
    """

    prompt_file: Optional[str] = None  # path to prompt file
    debug: bool = False
    model: Literal["gpt-3.5-turbo-0613", "gpt-4-0613"] = "gpt-3.5-turbo-0613"


@dataclass
class SmolAgent(Agent):
    """
    This class represents the SmolAgent, which is a CLI agent that generates code based on a given prompt.
    It inherits from the Agent class.

    Attributes:
    name: ClassVar[str] - The name of the agent. For SmolAgent, it is "smol".
    run_params: Type[SmolAgentClientParams] - The parameters for running the agent. It uses the SmolAgentClientParams class.
    splash: Optional[str] - The splash screen for the agent. It is a string of ASCII art.
    """

    name: ClassVar[str] = "smol"
    run_params: Type[SmolAgentClientParams] = SmolAgentClientParams
    splash: Optional[
        str
    ] = """\


   __                                 __
  / /   but make it...           __   \ \      
 | |             ___ __ _  ___  / /    | |      
 | |            (_-</  ' \\/ _ \\/ /     | |      
 | |           /___/_/_/_/\\___/_/      | |      
 | |                                   | |      
  \_\                                 /_/       



    """

    async def run(self) -> AsyncIterable[List[file_diff.FileChange]]:
        params = self.run_params
        await ainput("\n> Press any key to continue.\n")

        if params.prompt_file is None:
            prompt = await ainput("\n> Prompt file not found. Please input a prompt.\n")
        else:
            with open(params.prompt_file, "r") as f:
                prompt = f.read()

        logger.info("Starting smol-dev with prompt:")
        self.console.print(prompt, markup=True, highlight=True)

        await ainput("\n> Press any key to continue.\n")

        def stream_handler(chunk):
            def stream_string(string):
                for char in string:
                    print(char, end="", flush=True)
                    time.sleep(0.0012)

            stream_string(chunk.decode("utf-8"))

        plan = smol_dev.plan(prompt, stream_handler=stream_handler, model=params.model)

        logger.info("Running with plan:")
        self.console.print(plan, emoji=True, markup=True)

        await ainput("\n> Press any key to continue.\n")

        file_paths = smol_dev.specify_file_paths(prompt, plan, model=params.model)

        logger.info("Got file paths:")
        self.console.print(json.dumps(file_paths, indent=2), markup=True)

        file_changes = []

        await ainput("\n> Press any key to continue.\n")

        @dataclass
        class PBarUpdater:
            pbars: Dict[int, Any] = field(default_factory=dict)
            dones: Dict[int, Any] = field(default_factory=dict)
            messages: Dict[int, Optional[str]] = field(default_factory=dict)
            lock: asyncio.Lock = asyncio.Lock()

            def update(self):
                for position, pbar in self.pbars.items():
                    if self.dones[position]:
                        pbar.display(self.messages[position])
                    else:
                        pbar.update()

        updater = PBarUpdater()

        async def generate_code_for_filepath(file_path: str, position: int) -> file_diff.FileChange:
            stream_handler = lambda chunk: pbar.update(n=len(chunk))
            code_future = asyncio.ensure_future(
                smol_dev.generate_code(
                    prompt, plan, file_path, stream_handler=stream_handler, model=params.model
                )
            )
            with tqdm.asyncio.tqdm(position=position, unit=" chars", unit_scale=True) as pbar:
                async with updater.lock:
                    updater.pbars[position] = pbar
                    updater.dones[position] = False
                done = False
                waiter = asyncio.get_running_loop().create_future()

                def cb(fut):
                    waiter.cancel()

                code_future.add_done_callback(cb)

                async def spinner():
                    spinner_index: int = 0
                    steps = ["[⢿]", "[⣻]", "[⣽]", "[⣾]", "[⣷]", "[⣯]", "[⣟]", "[⡿]"]
                    while True:
                        c = steps[spinner_index % len(steps)]
                        pbar.set_description(f"{c} Generating code for {file_path}")
                        async with updater.lock:
                            updater.update()
                        spinner_index += 1
                        await asyncio.sleep(0.05)
                        if waiter.done():
                            # pbar.display(f"[✔️] Generated code for {file_path}")
                            async with updater.lock:
                                updater.dones[position] = True
                                updater.messages[position] = f"[✔️] Generated code for {file_path}"
                                pbar.set_description(f"[✔️] Generated code for {file_path}")
                                updater.update()
                            return

                t = asyncio.create_task(spinner())
                code = await code_future
                absolute_file_path = os.path.join(os.getcwd(), file_path)
                file_change = file_diff.get_file_change(path=absolute_file_path, new_content=code)
                return file_change

        fs = [
            asyncio.create_task(generate_code_for_filepath(fp, position=i))
            for i, fp in enumerate(file_paths)
        ]

        yield await asyncio.gather(*fs)


if __name__ == "__main__":
    launcher(SmolAgent, SmolAgentClientParams)

import asyncio
import dataclasses
import inspect
import json
import logging
import os
import pickle as pkl
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, ClassVar, Dict, List, Optional, Type

import smol_dev
import tqdm.asyncio
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

import rift.lsp.types as lsp
import rift.server.core as core
import rift.server.lsp as server
import rift.util.file_diff as file_diff
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

from rift.agents.cli.util import ainput, stream_string, stream_string_ascii


@dataclass
class ClientParams:
    """
    Base class for special parameters for instances of `Agent`.
    Subclass to add agent-specific attributes.

    Attributes:
    port: int - The port number where the server will be running. Default is 7797.
    debug: bool - A flag to indicate whether the application is in debug mode. Default is False.
    """

    port: int = 7797
    debug: bool = False


@dataclass
class Agent:
    """
    Abstract base class for agents that can be interacted with through a CLI
    and which produce code diffs that can be sent to an LSP server.

    To implement your own agent:
    - create another file in this directory
    - subclass `Agent`
    - run it as a Python script with `launcher` as the entrypoint from inside a VSCode terminal (if using the Rift VSCode extension). see `smol.py` for an example.

    Attributes:
    name: str - The name of the agent.
    run_params: ClientParams - The parameters for running the agent.
    splash: Optional[str] - The splash screen for the agent. Default is None.
    console: Console - The console for displaying output. Default is a new Console instance.
    """

    name: str
    run_params: ClientParams
    splash: Optional[str] = None
    console: Console = field(default_factory=Console)

    async def run(self, *args, **kwargs) -> AsyncIterable[List[file_diff.FileChange]]:
        """
        Async generator which emits batches of file changes.

        This method should be overridden by subclasses to provide the specific implementation.
        """
        ...


def get_dataclass_function(cls):
    """Returns a function whose signature is set to be a list of arguments
    which are precisely the dataclass's attributes.

    Args:
        dataclass: The dataclass to get the function for.

    Returns:
        The function whose signature is set to be a list of arguments
        which are precisely the dataclass's attributes.
    """

    def get_attributes(cls):
        attributes = []
        for field in dataclasses.fields(cls):
            if isinstance(field.type, cls):
                attributes.extend(dataclasses.fields(field.type).keys())
            else:
                attributes.append(field)

        attributes = [
            inspect.Parameter(
                name=field.name,
                kind=inspect.Parameter.POSITIONAL_ONLY,
                default=None,
                annotation=field.type,
            )
            for field in attributes
        ]

        return attributes

    attributes = get_attributes(cls)

    def function(*args):
        """A function whose signature is set to be the dataclass's attributes."""
        return cls(*args)

    function.__signature__ = inspect.Signature(parameters=attributes)
    return function


async def main(agent_cls, params):
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

    while True:
        await asyncio.sleep(1)
        if agent_cls.splash is not None:
            stream_string(agent_cls.splash)
        else:
            stream_string_ascii(agent_cls.name)

        agent = agent_cls(run_params=params, console=console)

        @dataclass
        class AgentRunStats:
            _start: float = field(default_factory=time.time)
            stats: Dict[Any, Any] = field(default_factory=dict)

            def __post_init__(self):
                self.stats["elapsed_time"] = None
                self.stats["changed_files"] = list()

            def elapsed(self):
                return time.time() - self._start

            def report_stats(self):
                console.print(Panel("[AgentRunStats] report:\n" + json.dumps(self.stats, indent=2)))

        agent_stats = AgentRunStats()

        async for file_changes in agent.run():
            label = "rift"
            if len(file_changes) > 0:
                label = file_changes[0].description or label
            for file_change in file_changes:
                agent_stats.stats["changed_files"].append(file_change.uri.uri)
            resp = await client.server.apply_workspace_edit(
                lsp.ApplyWorkspaceEditParams(
                    file_diff.edits_from_file_changes(file_changes, user_confirmation=True),
                    label=label,
                )
            )

        agent_stats.stats["elapsed_time"] = agent_stats.elapsed()

        console.print("\n")

        agent_stats.report_stats()

        await ainput("\n> Press any key to restart.\n")

    await t


def launcher(agent_cls: Type[Agent], param_cls: Type[ClientParams], loop=None):
    import fire

    params = fire.Fire(param_cls)
    asyncio.run(main(agent_cls=agent_cls, params=params), debug=params.debug)

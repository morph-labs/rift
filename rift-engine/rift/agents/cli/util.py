import asyncio
import dataclasses
import inspect
import json
import logging
import os
import pickle as pkl
from concurrent.futures import ThreadPoolExecutor
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
from rift.lsp.types import InitializeParams
from rift.rpc.io_transport import AsyncStreamTransport
from rift.rpc.jsonrpc import RpcServer, rpc_method, rpc_request
from rift.server.core import CodeCapabilitiesServer, rift_splash
from rift.util.ofdict import todict

logger = logging.getLogger(__name__)
import time
import types

import fire
import fire.core


def _PrintResult(component_trace, verbose=False, serialize=None):
    """
    Prints the result of a Fire command.

    Args:
        component_trace: The component trace of the Fire command.
        verbose: Whether to print verbose output. Default is False.
        serialize: A function to serialize the result. If not None, it must be callable. Default is None.
    """
    result = component_trace.GetResult()
    if serialize:
        if not callable(serialize):
            raise fire.core.FireError(
                "The argument `serialize` must be empty or callable:", serialize
            )
        result = serialize(result)

    if fire.value_types.HasCustomStr(result):
        print(str(result))
        return

    if isinstance(result, (list, set, frozenset, types.GeneratorType)):
        for i in result:
            print(fire.core._OneLineResult(i))
    elif inspect.isgeneratorfunction(result):
        raise NotImplementedError
    elif isinstance(result, dict) and value_types.IsSimpleGroup(result):
        print(fire.core._DictAsString(result, verbose))
    elif isinstance(result, tuple):
        print(fire.core._OneLineResult(result))
    elif dataclasses._is_dataclass_instance(result):
        print(fire.core._OneLineResult(result))
    elif isinstance(result, value_types.VALUE_TYPES):
        if result is not None:
            print(result)
    else:
        help_text = fire.helptext.HelpText(result, trace=component_trace, verbose=verbose)
        output = [help_text]
        Display(output, out=sys.stdout)


fire.core._PrintResult = _PrintResult


def stream_string(string):
    """
    Prints a string character by character with a delay between each character.

    Args:
        string: The string to print.
    """
    for char in string:
        print(char, end="", flush=True)
        time.sleep(0.0015)


def stream_string_ascii(name: str):
    """
    Prints an ASCII art representation of a string character by character with a delay between each character.

    Args:
        name: The string to convert to ASCII art and print.
    """
    _splash = art.text2art(name, font="smslant")

    stream_string(_splash)


async def ainput(prompt: str = "") -> str:
    """
    Asynchronously waits for user input.

    Args:
        prompt: The prompt to display to the user. Default is an empty string.

    Returns:
        The user's input as a string.
    """
    with ThreadPoolExecutor(1, "AsyncInput") as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, input, prompt)

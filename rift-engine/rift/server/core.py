import asyncio
import logging
import os
import sys
import time
from typing import Literal, Optional, Union

import certifi

from rift.__about__ import __version__
from rift.rpc.io_transport import AsyncStreamTransport, create_pipe_streams
from rift.server.lsp import LspServer

try:
    from rift.llm.gpt4all_model import Gpt4AllModel, Gpt4AllSettings
except ImportError as e:
    Gpt4AllModel = e

logger = logging.getLogger(__name__)


def rift_splash():
    _splash = """


   ██████╗ ██╗███████╗████████╗
   ██╔══██╗██║██╔════╝╚══██╔══╝
   ██████╔╝██║█████╗     ██║
   ██╔══██╗██║██╔══╝     ██║
   ██║  ██║██║██║        ██║
   ╚═╝  ╚═╝╚═╝╚═╝        ╚═╝
                                   __
   by  ____ ___  ____  _________  / /_
      / __ `__ \/ __ \/ ___/ __ \/ __ \\
     / / / / / / /_/ / /  / /_/ / / / /
    /_/ /_/ /_/\____/_/  / .___/_/ /_/
                        /_/


    """

    def stream_string(string):
        for char in string:
            print(char, end="", flush=True)
            time.sleep(0.0012)
            # print('\r', end='')

    stream_string(_splash)


# ref: https://stackoverflow.com/questions/64303607/python-asyncio-how-to-read-stdin-and-write-to-stdout

LspHost = Union[Literal["stdio"], str]
LspPort = Union[Literal["stdio"], int]

ModelType = Literal["openai", "hf", "gpt4all"]


class CodeCapabilitiesServer:
    server: Optional[LspServer] = None

    def __init__(
        self,
        lsp_host: LspHost = "127.0.0.1",
        lsp_port: LspPort = 7797,
    ):
        self.lsp_host = lsp_host
        self.lsp_port = lsp_port

    async def on_lsp_connection(self, reader, writer):
        transport = AsyncStreamTransport(reader, writer)
        await self.run_lsp(transport)

    async def run_lsp(self, transport):
        server = LspServer(transport)
        self.server = server
        try:
            logger.info("[CodeCapabilitiesServer] listening...")
            await server.listen_forever()
        except Exception as e:
            logger.error("caught: " + str(e))
            logger.info(
                "connection closed, but Rift is still running and accepting new connections."
            )

    async def run_lsp_tcp_client_mode(self):
        assert isinstance(self.lsp_host, str)
        assert isinstance(self.lsp_port, int)
        reader, writer = await asyncio.open_connection(self.lsp_host, self.lsp_port)
        transport = AsyncStreamTransport(reader, writer)
        await self.run_lsp(transport)

    async def run_lsp_tcp(self):
        assert isinstance(self.lsp_host, str)
        assert isinstance(self.lsp_port, int)
        try:
            server = await asyncio.start_server(
                self.on_lsp_connection, self.lsp_host, self.lsp_port
            )
        except OSError as e:
            logger.error(str(e))
            logger.info(f"try connecting to {self.lsp_host}:{self.lsp_port}")
            return await self.run_lsp_tcp_client_mode()
        else:
            async with server:
                addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
                logger.info(f"listening with LSP protocol on {addrs}")
                await server.serve_forever()

    async def run_lsp_stdio(self):
        reader, writer = await create_pipe_streams(in_pipe=sys.stdin, out_pipe=sys.stdout)
        transport = AsyncStreamTransport(reader, writer)
        await self.run_lsp(transport)

    async def _run_forever_fut(self):
        """Runs the language server.

        If lsp_port = 'stdio', then the LSP listens on stdin and stdout.
        There is also a web server at 7787 that the webview can connect to.
        """
        lsp_task = asyncio.create_task(
            self.run_lsp_stdio() if self.lsp_port == "stdio" else self.run_lsp_tcp()
        )
        return lsp_task

    async def run_forever(self):
        """Runs the language server.

        If lsp_port = 'stdio', then the LSP listens on stdin and stdout.
        There is also a web server at 7787 that the webview can connect to.
        """
        loop = asyncio.get_event_loop()
        lsp_task = await self._run_forever_fut()
        await lsp_task
        logger.debug(f"exiting {type(self).__name__}.listen_forever")


def create_metaserver(
    host: LspHost = "127.0.0.1",
    port: LspPort = 7797,
    version=False,
    debug=False,
) -> CodeCapabilitiesServer:
    """
    Main entry point for the rift server
    Args:
        - host: host to listen on. If 'stdio', then listen on stdin and stdout.
        - port: port number to listen on. If 'stdio', then listen on stdin and stdout. Note that this doesn't work with gpt4all.
        - model_type: one of 'openai', 'hf', 'gpt4all'.
        - chat_model_type: optional, defaults to same as model_type
        - version: if true, print version and exit.
        - debug: if true, print debug messages.
    """
    if version:
        print(__version__)
        return
    from rich.console import Console
    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    console = Console(stderr=True)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(console=console)],
    )

    rift_splash()

    logger.info(f"starting Rift server on {host}:{port}")
    metaserver = CodeCapabilitiesServer(lsp_host=host, lsp_port=port)
    return metaserver


def main(
    host: LspHost = "127.0.0.1",
    port: LspPort = 7797,
    version=False,
    debug=False,
    socket: LspPort = 7797,
):
    if socket != port:
        port = int(socket)
        logger.info(f"setting port={int(socket)}")
    metaserver = create_metaserver(host, port, version, debug)
    if metaserver:
        asyncio.run(metaserver.run_forever(), debug=debug)


def entrypoint():
    import fire

    fire.Fire(main)


if __name__ == "__main__":
    os.environ["SSL_CERT_FILE"] = certifi.where()
    sys.stdout.reconfigure(encoding="utf-8")
    entrypoint()

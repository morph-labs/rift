"""
Author: E.W.Ayers <contact@edayers.com>
This file is adapted from  https://github.com/EdAyers/sss
"""
import asyncio
import sys
from .transport import (
    Transport,
    TransportClosedOK,
    TransportClosedError,
    TransportError,
)


async def create_pipe_streams(in_pipe, out_pipe):
    """Converts a pair of pipes into a reader/writer async pair."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, in_pipe)
    w_transport, w_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, out_pipe
    )
    writer = asyncio.StreamWriter(w_transport, w_protocol, reader, loop)
    return reader, writer


class AsyncStreamTransport(Transport):
    """Create a transport from a asyncio StreamReader, StreamWriter pair.

    We assume the message protocol is that described in LSP
    https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#baseProtocol

    That is, a sequence of newline delimited http-like header strings, terminated by a double newline.
    One of the headers needs to be "content-length" integer, and then that number of bytes is read from the stream.
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer

    async def recv(self):
        """Recieves data from the stream. If EOF is reached, raises TransportClosedOK error."""
        # read the header
        header = {}
        while True:
            line = await self.reader.readline()
            if line == b"":
                assert self.reader.at_eof()
                if len(header) == 0:
                    raise TransportClosedOK("end of stream")
                else:
                    raise TransportClosedError(f"unexpected end of stream")
            try:
                # [todo] should be 'latin-1' encoding?
                line = line.decode().strip()
            except UnicodeDecodeError as e:
                raise TransportError(f"invalid utf-8 in header:\n{line}'")
            if line == "":
                break
            if ":" not in line:
                if "HTTP/" in line:
                    # [todo] gracefully return a valid http 400 error and a message about
                    # how to get started with rift.
                    raise TransportError(
                        f"Looks like you're trying to use Rift with a web browser. Please read the getting started docs to learn how to use Rift."
                    )
                raise TransportError(f"invalid header, expecting a colon:\n{line}")
            k, v = line.split(":", 1)
            header[k.lower()] = v
        content_length = header.get("content-length")
        if content_length is None:
            raise TransportError("invalid datagram: no content-length in header")
        content_length = int(content_length)
        # read the body
        try:
            data = await self.reader.readexactly(content_length)
            return data
        except asyncio.IncompleteReadError as e:
            raise TransportClosedError("unexpected end of stream") from e

    async def send(self, data: bytes, header={}):
        header["Content-Length"] = len(data)
        header = "".join(f"{k}:{v}\r\n" for k, v in header.items())
        header += "\r\n"
        self.writer.write(header.encode())
        self.writer.write(data)
        await self.writer.drain()

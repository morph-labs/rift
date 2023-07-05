"""
Author: E.W.Ayers <contact@edayers.com>
This file is adapted from  https://github.com/EdAyers/sss
"""
from asyncio import Future, Task
import asyncio
from functools import singledispatch, partial
from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import logging
import sys
from typing import (
    Any,
    Optional,
    Union,
)
import inspect
import warnings

from rift.util.ofdict import MyJsonEncoder, ofdict, todict, todict_dataclass
import json
from .transport import (
    Transport,
    TransportClosedError,
    TransportClosedOK,
    TransportError,
)

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Error codes for JSON-RPC.

    Reference:
    - https://www.jsonrpc.org/specification#error_object
    - https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#errorCodes
    """

    ### JSON-RPC codes

    parse_error = -32700
    """ It doesn't parse as JSON """
    invalid_request = -32600
    """ The JSON sent is not a valid Request object. """
    method_not_found = -32601
    """ The method does not exist / is not available. """
    invalid_params = -32602
    """ Your parameters are not valid. (eg fields missing, bad types etc) """
    internal_error = -32603
    """ The internal JSON-RPC server code messed up. """

    ### Codes specific to LSP

    server_error = -32000
    """ There was a problem in the method handler. """

    server_not_initialized = -32002
    """ The server has not been initialized. """

    request_failed = -32803
    """A request failed but it was syntactically correct, e.g the
	 * method name was known and the parameters were valid. The error
	 * message should contain human readable information about why
	 * the request failed.
	 *"""

    server_cancelled = -32802
    """The server cancelled the request. This error code should
	 * only be used for requests that explicitly support being
	 * server cancellable."""

    content_modified = -32801
    """ Content got modified outside of normal conditions. """
    request_cancelled = -32800
    """ The client cancelled a request and the server has detected the cancel. """


encoder = MyJsonEncoder()


@dataclass
class Request:
    """A request object for JSON-RPC.

    reference: https://www.jsonrpc.org/specification#request_object
    """

    method: str
    """A string identifying the RPC method to call."""
    id: Optional[Union[str, int]] = field(default=None)
    """Unique identifier of the request, the response object should use the same request id."""
    params: Optional[Any] = field(default=None)
    """A structured value that holds the parameter values to be used during the invocation of the method."""

    @property
    def is_notification(self):
        """True if this is a notification request. Notifications do not have an id and so can't be responded to."""
        return self.id is None

    def to_bytes(self):
        """Encode the request as bytes. Note that this will automatically convert Python objects to JSON using MyJsonEncoder."""
        return encoder.encode(self).encode()

    def __str__(self):
        if self.id is None:
            return f"notify {self.method}"
        else:
            return f"request {self.method}:{self.id}"


@dataclass
class ResponseError(Exception):
    """A response error object in JSON-RPC.

    Note that you can raise these as you would an exception.
    If you raise this in an RPC method handler, the error will be sent to the client.

    ref: https://www.jsonrpc.org/specification#error_object"""

    code: ErrorCode
    message: str
    data: Optional[Any] = field(default=None)

    def __str__(self):
        return f"{self.code.name}: {self.message}"


def invalid_request(message: str) -> ResponseError:
    """Raise this to tell JSON-RPC that the request was invalid.

    You should use this if the request does not parse as JSON or is not a valid Request or Response.
    If it is a valid JSON-RPC object but invalid parameters then use `invalid_params` instead.
    """
    return ResponseError(ErrorCode.invalid_request, message)


def method_not_found(method_name: str) -> ResponseError:
    """Raise this if the given method name (Request.method) is not found."""
    return ResponseError(
        ErrorCode.method_not_found,
        f"no method found for {method_name}",
        data=method_name,
    )


def invalid_params(message: str = "invalid params") -> ResponseError:
    """Raise this if the parameters of the request are invalid."""
    return ResponseError(ErrorCode.invalid_params, message)


def internal_error(message: str) -> ResponseError:
    """This should be raised if the server code has an internal error that is not the fault of the client.

    Any exceptions raised in a method handler will be caught and converted to this error.
    """
    return ResponseError(ErrorCode.internal_error, message)


def server_not_initialized(message: str) -> ResponseError:
    return ResponseError(ErrorCode.server_not_initialized, message)


@dataclass
class Response:
    """JSON-RPC response.

    https://www.jsonrpc.org/specification#response_object
    """

    id: Optional[Union[str, int]] = field(default=None)
    result: Optional[Any] = field(default=None)
    error: Optional[ResponseError] = field(default=None)
    jsonrpc: str = field(default="2.0")

    def __todict__(self):
        d = todict_dataclass(self)
        if "error" not in d and "result" not in d:
            d["result"] = None
        return d

    def to_bytes(self):
        return encoder.encode(self).encode()


class Dispatcher:
    """Dispatcher for JSON-RPC requests.

    It is a dictionary mapping method names to python functions to handle these methods.
    The Python function should have a single argument.
    If the python function's argument and return type are annotated, then the dispatcher will use
    `todict` and `fromdict` to convert the arguments to and from JSON.

    """

    def __init__(self, methods=None, extra_kwargs={}):
        self.methods = methods or {}
        self.extra_kwargs = extra_kwargs

    def __contains__(self, method):
        return method in self.methods

    def __getitem__(self, method):
        return partial(self.methods[method], **self.extra_kwargs)

    def param_type(self, method):
        fn = self.methods[method]
        sig = inspect.signature(fn)
        if len(sig.parameters) == 0:
            T = Any
        else:
            P = next(iter(sig.parameters.values()))
            T = P.annotation
            if T is inspect.Parameter.empty:
                T = Any
        return T

    def return_type(self, method):
        fn = self.methods[method]
        sig = inspect.signature(fn)
        a = sig.return_annotation
        if a is inspect.Signature.empty:
            return Any
        else:
            return a

    def register(self, name=None):
        def core(fn):
            funcname = name or fn.__name__
            if funcname in self.methods:
                warnings.warn(f"method with name {funcname} already registered, overwriting")
            self.methods[funcname] = fn
            return fn

        return core

    def with_kwargs(self, **kwargs):
        return Dispatcher(self.methods, {**self.extra_kwargs, **kwargs})

    async def dispatch(self, method: str, params: Any):
        fn = self[method]
        result = fn(params)
        if asyncio.iscoroutine(result):
            result = await result
        return result


server_count = 0
"""Counter for labelling servers with a unique id."""

RequestId = Union[str, int]


class InitializationMode(Enum):
    NoInit = 0
    """ No initialization required. """
    ExpectInit = 1
    """ We expect to receive an initialize request from the peer. """
    SendInit = 2
    """ We should send an initialize request to the peer. """


class RpcServerStatus(Enum):
    preinit = 0
    """Initialize has not been called yet."""
    running = 1
    """The server is running."""
    shutdown = 2
    """The server has been shutdown."""


class ExitNotification(Exception):
    """Thrown when the server recieved an exit notifaction from its peer."""


def rpc_method(name: Optional[str] = None):
    """Decorate your method with this to say that you are implementing a JSON-RPC method.

    Example:
    ```
    class MyServer(RpcServer):
        @rpc_method('foo')
        def foo(self, params : int) -> str:
            return f"foo {params}"
    ```

    Methods should have a single argument.
    Methods can be async.
    If the method's argument is annotated with type T, then the argument will be converted from JSON to T using `fromdict(T, ·)`.
    Most builtin types are supported, as well as dataclasses and pydantic models.
    """

    def decorator(fn):
        setattr(fn, "rpc_method", name or fn.__name__)
        return fn

    return decorator


def rpc_request(name: Optional[str] = None):
    """Decorate your _stub_ method with this to have a client RPC.

    The resulting class method will call the server's method with the given name.
    If a return annotation is given, the JSON result will be converted to that type using `fromdict(T, ·)`.

    # Example:
    ```
    class MyClient(RpcServer):
        @rpc_request('foo')
        def foo(self, params : int) -> str:
            # this is never called, it is just a stub
            ...


    client = MyClient()
    x = await client.foo(42)
    assert x == "foo 42"
    ```
    """

    def decorator(fn):
        assert asyncio.iscoroutinefunction(fn)
        fn_name = name or fn.__name__

        async def method(self, params):
            return await self.request(fn_name, params)

        return method

    return decorator


class RpcServer:
    """Implementation of a JSON-RPC server.

    To create your own server, subclass this class and implement the methods you want to support.

    ```
    class MyServer(RpcServer):

        @rpc_method('foo')
        def foo(self, params : int) -> str:
            return f"foo {params}"

    transport = ...
    server = MyServer(transport)

    asyncio.run(server.serve_forever())

    ```

    # Extra functionality

    Following the conventions of LSP for extra functionality.

    ## Cancellation

    The server will listen for the "$/cancelRequest" notification.
    Upon recieving this, the in-flight `@rpc_method` task for the request will be cancelled.
    More details can be found [here](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#cancelRequest).

    ## Initialisation

    When you create a server, you can specify an `initialization_mode` as an argument to `__init__()`.
    If this is set to `ExpectInit`, then the server will wait for an `initialize` request from the client before accepting any other requests.
    You can write custom code to be performed on initialization by making an `@rpc_method('initialize')` method handler (LSP _servers_ are `ExpectInit`).
    If this is set to `SendInit`, then the server will send an `initialize` request to the client before accepting any other requests (LSP _clients_ (ie editors) are `SendInit`).
    A `NoInit` server will accept requests immediately.

    ## Shutdown

    A "shutdown" request will put the server into a shutdown state.
    All in-flight requests will be cancelled and no more notifications or requests will be accepted or sent.
    You can write custom code to be performed on shutdown by making an `@rpc_method('shutdown')` method handler.
    Once in the shutdown state, the server will stay alive until the client sends an `exit` notification.
    An `exit` notification will kill the server immediately.

    [todo] rename to RpcConnection, then RpcServer and RpcClient handle the different Init conventions for
    lifecycle.
    [todo] add warnings if requests go unanswered for too long.
    """

    dispatcher: Dispatcher
    status: RpcServerStatus
    transport: Transport
    init_mode: InitializationMode
    name: str
    """ This is a human-readable name that will be used in log messages."""
    request_counter: int
    """ Unique id for each request I make to my peer. """
    my_requests: dict[RequestId, Future[Any]]
    """ Requests that I have made to my peer. """
    their_requests: dict[RequestId, Task]
    """ Requests that my peer has made to me. """
    notification_tasks: set[asyncio.Task]
    """ Tasks running from notifications that my peer has sent to me. """

    def __init__(
        self,
        transport: Transport,
        dispatcher=None,
        name=None,
        init_mode: InitializationMode = InitializationMode.NoInit,
    ):
        if not isinstance(transport, Transport):
            raise TypeError(
                f"transport must be an instance of {Transport.__module__}.Transport, not {type(transport)}"
            )
        global server_count
        server_count += 1
        if name is None:
            self.name = f"<{type(self).__name__} {server_count}>"
        else:
            assert isinstance(name, str)
            self.name = name
        self.init_mode = init_mode
        if init_mode == InitializationMode.NoInit:
            self.status = RpcServerStatus.running
        else:
            self.status = RpcServerStatus.preinit
        self.transport = transport
        self.dispatcher = dispatcher or Dispatcher()
        self.my_requests = {}
        self.their_requests = {}
        self.request_counter = 1000 * server_count
        self.notification_tasks = set()

        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            rpc_method = getattr(method, "rpc_method", None)
            if rpc_method is not None:
                # [todo] assert that the signature is correct
                logger.debug(f"registering RPC method '{rpc_method}' to {method.__qualname__}")
                self.dispatcher.register(rpc_method)(method)

    def __str__(self):
        return self.name

    async def _send(self, r: Union[Response, Request]):
        await self.transport.send(r.to_bytes())

    async def notify(self, method: str, params: Optional[Any]):
        """Send a notification to the peer."""
        if self.status != RpcServerStatus.running:
            raise RuntimeError(
                f"can't send notifications while server is in {self.status.name} state"
            )
        req = Request(method=method, params=params)
        await self._send(req)

    async def request(self, method: str, params: Optional[Any]) -> asyncio.Future[Any]:
        """Send a request to the peer and wait for a response.

        Args:
            - method: the name of the method to call.
            - params: the parameters to pass to the method, it can be any python object that will be converted to JSON using `MyJsonEncoder`.

        Returns:
            An awaitable that yields a json-like python object (ie something that you would get from `json.loads()`) representing the response from the peer.

        Raises:
            - RuntimeError: if the server is not in the running state.
            - ResponseError: if the peer responds with an error, you can use ResponseError.code to determine the cause of the error.

        Todo:
           - Cancellation support: rather than a coro it should return a cancellable future.
           - timeout?
        """
        if self.status != RpcServerStatus.running:
            if self.init_mode != InitializationMode.SendInit or method != "initialize":
                raise RuntimeError(
                    f"can't make new requests while server is in {self.status.name} state"
                )
        self.request_counter += 1
        id = self.request_counter
        req = Request(method=method, id=id, params=params)
        # print("REQUEST: ", req)
        fut = asyncio.get_running_loop().create_future()
        # [todo] I think the pythonic way to do this is to have this dict be a weakref, and the
        # caller is responsible for holding the request object.
        # If the request future is disposed then we send a cancel request to client.
        if id in self.my_requests:
            raise RuntimeError(f"non-unique request id {id} found")
        self.my_requests[id] = fut
        await self._send(req)
        result = await fut
        return result

    async def _send_init(self, init_param):
        """Send an initialization request to the peer."""
        await self.request("initialize", init_param)
        self.status = RpcServerStatus.running
        # [todo] allow inheriting classes to do things here.
        await self.notify("initialized", None)

    async def listen_forever(self, init_param=None):
        """Runs forever. Serves your client.

        It will return when:
        - the transport closes gracefully
        - the exit notification is received.
        - a keyboard interrupt is received (ie, the user presses ctrl-c)

        Args:
            - init_param is a parameter that will be passed to the peer's `initialize` method,
              assuming that this server is in `InitializationMode.SendInit` mode.

        Raises:
            - TransportClosedError:the transport closes with an error
            - TransportError: some other error at the transport level occurred
        """
        # [todo] add a lock to prevent multiple server loops from running at the same time.

        if self.init_mode == InitializationMode.SendInit:
            if self.status != RpcServerStatus.preinit:
                raise RuntimeError(
                    f"can't start server while server is in {self.status.name} state"
                )
            if init_param is None:
                raise ValueError(f"init_param must be provided in {self.init_mode.name} mode")
            task = asyncio.create_task(self._send_init(init_param))
            self.notification_tasks.add(task)
            task.add_done_callback(self.notification_tasks.discard)
        try:
            while True:
                try:
                    data = await self.transport.recv()
                    messages = json.loads(data)
                    if isinstance(messages, dict):
                        # datagram contains a single message
                        messages = [messages]
                    elif not isinstance(messages, list):
                        raise TypeError(f"expected list or dict, got {type(messages)}")
                    for message in messages:
                        self._handle_message(message)
                except TransportClosedOK as e:
                    logger.info(f"{self.name} transport closed gracefully: {e}")
                    return
                except (json.JSONDecodeError, TypeError) as e:
                    logger.exception("invalid json")
                    response = Response(
                        error=ResponseError(message=str(e), code=ErrorCode.parse_error)
                    )
                    await self._send(response)
                    continue
                except ExitNotification as e:
                    logger.info(f"{self.name} received exit notification")
                    return
                except KeyboardInterrupt as e:
                    logger.info(f"{self.name} recieved kb interrupt")
                    return
                except TransportClosedError as e:
                    logger.error(f"{self.name} transport closed in error:\n{e}")
                    raise e
                except TransportError as e:
                    logger.error(f"{self.name} transport error:\n{e}")
                    raise e
                except Exception as e:
                    # we should only reach here if there is a bug in the RPC code.
                    logger.exception(
                        f"{self.name} unhandled {type(e).__name__}:\n{e}\nThis likely caused by a bug in the RPC server code."
                    )
                    raise e
        finally:
            logger.info(f"exiting serve_forever loop")
            (_, e, _) = sys.exc_info()  # sys.exception() is 3.11 only
            if e is None:
                e = ConnectionError(f"{self} shutdown")
            for fut in self.my_requests.values():
                fut.set_exception(e)
            self._shutdown()

    def _shutdown(self):
        # [todo] also send cancel notifications to all our pending request futures.
        for t in self.their_requests.values():
            t.cancel("shutdown")
        for t in self.notification_tasks:
            t.cancel("shutdown")
        self.status = RpcServerStatus.shutdown
        logger.info(f"{self} entered shutdown state")

    def _handle_message(self, message: Any):
        if "result" in message or "error" in message:
            # message is a Response
            res = ofdict(Response, message)
            if res.id not in self.my_requests:
                logger.error(f"received response for unknown request: {res}")
                return
            fut = self.my_requests.pop(res.id)
            if fut.done():
                logger.error(f"received response for already completed request: {res} {fut}")
            else:
                if res.error is not None:
                    fut.set_exception(res.error)
                else:
                    fut.set_result(res.result)
        else:
            # message is a Request.
            req = ofdict(Request, message)
            if req.method == "exit":
                # exit notification should kill the server immediately.
                if self.status != RpcServerStatus.shutdown:
                    logger.warning("exit notification received before shutdown request")
                # https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#exit
                raise ExitNotification()
            if req.method == "shutdown":
                self._shutdown()
            task = asyncio.create_task(
                self._on_request(req),
                name=f"{self.name} handle {req}",
            )
            id = req.id
            if id is not None:
                # Request expects a reponse
                if id in self.their_requests:
                    raise invalid_request(f"request id {id} is already in use")
                self.their_requests[id] = task
                task.add_done_callback(lambda _: self.their_requests.pop(id))
            else:
                # Request is a notification, no response expected.
                self.notification_tasks.add(task)
                task.add_done_callback(self.notification_tasks.discard)

    async def _on_request(self, req: Request) -> None:
        """Handles a request from the peer."""
        try:
            result = await self._on_request_core(req)
        except asyncio.CancelledError as e:
            if not req.is_notification:
                await self._send(
                    Response(
                        id=req.id,
                        error=ResponseError(code=ErrorCode.request_cancelled, message=str(e)),
                    )
                )
        except ResponseError as e:
            await self._send(Response(id=req.id, error=e))
        except Exception as e:
            # if we get here, the method handler code has a bug.
            logger.exception(
                f"{self} {req} unhandled {type(e).__name__}:\n{e}\nThis is likely caused by a bug in the {req.method} method handler."
            )
            await self._send(
                Response(
                    id=req.id,
                    error=ResponseError(code=ErrorCode.server_error, message=str(e)),
                )
            )
        else:
            if not req.is_notification:
                await self._send(Response(id=req.id, result=result))
            else:
                if result is not None:
                    logger.warning(
                        f"notification handler {req.method} returned a value, this will be ignored"
                    )

    async def _on_request_core(self, req: Request):
        """Inner part of self._on_request, without error handling."""
        if self.status == RpcServerStatus.preinit:
            INIT_METHOD = "initialize"
            if self.init_mode == InitializationMode.ExpectInit:
                if req.method == INIT_METHOD:
                    self.status = RpcServerStatus.running
                else:
                    raise server_not_initialized(
                        f"please request method {INIT_METHOD} before requesting anything else"
                    )
            elif self.init_mode == InitializationMode.SendInit:
                raise server_not_initialized(f"please wait for me to send a {INIT_METHOD} request")
            else:
                raise internal_error("invalid server state")
        if self.status == RpcServerStatus.shutdown:
            if req.method == "shutdown":
                if "shutdown" in self.dispatcher:
                    return await self.dispatcher.dispatch("shutdown", None)
                else:
                    return None
            raise invalid_request("server has shut down")

        if req.method == "$/cancelRequest":
            if not req.is_notification:
                raise invalid_request("cancel request must be a notification")
            if not isinstance(req.params, dict) or not "id" in req.params:
                raise invalid_params('params must be a dict with "id" key')
            id = req.params["id"]
            t = self.their_requests.get(id, None)
            if t is not None:
                t.cancel("requested by peer")
            # if t is None then the request has already completed and removed itself from self.their_requests
            return None

        if req.method not in self.dispatcher:
            raise method_not_found(req.method)

        T = self.dispatcher.param_type(req.method)
        try:
            params = ofdict(T, req.params)
        except TypeError as e:
            message = f"{req.method} {type(e).__name__} failed to decode params to {T}: {e}"
            logger.exception(message)
            raise invalid_params(message)
        result = await self.dispatcher.dispatch(req.method, params)
        return result

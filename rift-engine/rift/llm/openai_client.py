import asyncio
import json
import logging
import random
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache, cached_property
from threading import Lock
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Coroutine,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    overload,
)
from urllib.parse import parse_qs, urlparse

import aiohttp
import rift.lsp.types as lsp
import rift.util.asyncgen as asg
from pydantic import BaseModel, BaseSettings, SecretStr
from rift.llm.abstract import (
    AbstractChatCompletionProvider,
    AbstractCodeCompletionProvider,
    AbstractCodeEditProvider,
    ChatResult,
    EditCodeResult,
    InsertCodeResult,
)
from rift.llm.openai_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
)
from rift.util.TextStream import TextStream

logger = logging.getLogger(__name__)

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)

from tiktoken import get_encoding

ENCODER = get_encoding("cl100k_base")
ENCODER_LOCK = Lock()

class MissingKeyError(Exception):
    ...


@dataclass
class OpenAIError(Exception):
    """Error raised by calling the OpenAI API"""

    message: str
    status: int

    def __str__(self):
        return self.message


@cache
def get_num_tokens(content: str):
    return len(ENCODER.encode(content))


def message_size(msg: Message):
    with ENCODER_LOCK:
        length = get_num_tokens(msg.content)
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        # see https://platform.openai.com/docs/guides/gpt/managing-tokens
        length += 6
        return length


def messages_size(messages: List[Message]) -> int:
    return sum([len(msg.content) for msg in messages])


def split_sizes(size1: int, size2: int, max_size: int) -> tuple[int, int]:
    """
    Adjusts and returns the input sizes so that their sum does not exceed
    a specified maximum size, ensuring a balance between the two if necessary.
    """
    if size1 + size2 <= max_size:
        return size1, size2
    share = int(max_size / 2)
    size1_bound = min(size1, share)
    size2_bound = min(size2, share)
    if size1 > share:
        available1 = max_size - size2_bound
        size1 = max(size1_bound, available1)
    available2 = max_size - size1
    size2 = max(size2_bound, available2)
    return size1, size2


def split_lists(list1: list, list2: list, max_size: int) -> tuple[list, list]:
    size1, size2 = split_sizes(len(list1), len(list2), max_size)
    return list1[-size1:], list2[:size2]


"""
Contents Order in the Context:

1) System Message: This includes an introduction and the current file content.
2) Non-System Messages: These are the previous dialogue turns in the chat, both from the user and the system.
3) Model's Responses Buffer: This is a reserved space for the response that the model will generate.

Truncation Strategy for Sizes:

1) System Message Size: Limited to the maximum of either MAX_SYSTEM_MESSAGE_SIZE tokens or the remaining tokens available after accounting for non-system messages and the model's responses buffer.
2) Non-System Messages Size: Limited to the number of tokens available after considering the size of the system message and the model's responses buffer.
3) Model's Responses Buffer Size: Always reserved to MAX_LEN_SAMPLED_COMPLETION tokens.

The system message size can dynamically increase beyond MAX_SYSTEM_MESSAGE_SIZE if there is remaining space within the MAX_CONTEXT_SIZE after accounting for non-system messages and the model's responses.
"""

MAX_CONTEXT_SIZE = 4096  # Total token limit for GPT models
MAX_LEN_SAMPLED_COMPLETION = 768  # Reserved tokens for model's responses
MAX_SYSTEM_MESSAGE_SIZE = 1024  # Token limit for system message


def calc_max_non_system_msgs_size(
    system_message_size: int,
    max_context_size: int = MAX_CONTEXT_SIZE,
    max_len_sampled_completion: int = MAX_LEN_SAMPLED_COMPLETION,
) -> int:
    """Maximum size of the non-system messages"""
    return max_context_size - max_len_sampled_completion - system_message_size


def calc_max_system_message_size(
    non_system_messages_size: int,
    max_system_message_size: int = MAX_SYSTEM_MESSAGE_SIZE,
    max_context_size: int = MAX_CONTEXT_SIZE,
    max_len_sampled_completion: int = MAX_LEN_SAMPLED_COMPLETION,
) -> int:
    """Maximum size of the system message"""

    # Calculate the maximum size for the system message. It's either the maximum defined limit
    # or the remaining tokens in the context size after accounting for model responses and non-system messages,
    # whichever is larger. This ensures that the system message can take advantage of spare space, if available.
    return max(
        max_system_message_size,
        max_context_size - max_len_sampled_completion - non_system_messages_size,
    )


def format_visible_files(documents: Optional[List[lsp.Document]] = None) -> str:
    if documents is None:
        return ""
    message = ""
    message += "Visible files:\n"
    for doc in documents:
        message += f"{doc.uri}```\n{doc.document.text}\n```\n"
    return message


def create_system_message_chat(
    before_cursor: str,
    region: str,
    after_cursor: str,
    documents: Optional[List[lsp.Document]] = None,
) -> Message:
    """
    Create system message wiht up to MAX_SYSTEM_MESSAGE_SIZE tokens
    """

    message = f"""
You are an expert software engineer and world-class systems architect with deep technical and design knowledge. Answer the user's questions about the code as helpfully as possible, quoting verbatim from the visible files if possible to support your claims.

The current file is split into a prefix, region, and suffix. Unless if the region is empty, assume that the user's question is about the region.

==== PREFIX ====
{before_cursor}
==== REGION ====
{region}
==== SUFFIX ====
{after_cursor}
"""
    if documents:
        message += "Additional files:\n"
        for doc in documents:
            message += f"{doc.uri}```\n{doc.document.text}\n```\n"
    message += """Answer the user's question."""
    # logger.info(f"{message=}")
    return Message.system(message)


def truncate_around_region(
    document: str,
    document_tokens: List[int],
    region_start,
    region_end: Optional[int] = None,
    max_size: Optional[int] = None,
):
    if region_end is None:
        region_end = region_start
    if region_start:
        before_cursor: str = document[:region_start]
        region: str = document[region_start:region_end]
        after_cursor: str = document[region_end:]
        tokens_before_cursor: List[int] = ENCODER.encode(before_cursor)
        tokens_after_cursor: List[int] = ENCODER.encode(after_cursor)
        region_tokens: List[int] = ENCODER.encode(region)
        (tokens_before_cursor, tokens_after_cursor) = split_lists(
            tokens_before_cursor, tokens_after_cursor, max_size
        )
        logger.debug(
            f"Truncating document to ({len(tokens_before_cursor)}, {len(tokens_after_cursor)}) tokens around cursor"
        )
        tokens: List[int] = tokens_before_cursor + region_tokens + tokens_after_cursor
    else:
        # if there is no cursor offset provided, simply take the last max_size tokens
        tokens = document_tokens[-max_size:]
        logger.debug(f"Truncating document to last {len(tokens)} tokens")
    return tokens


def create_system_message_chat_truncated(
    document: str,
    max_size: int,
    cursor_offset_start: Optional[int] = None,
    cursor_offset_end: Optional[int] = None,
    documents: Optional[List[lsp.Document]] = None,
    current_file_weight: float = 0.5,
    encoder=ENCODER,
) -> Message:
    """
    Create system message with up to max_size tokens
    """
    # logging.getLogger().info(f"{max_size=}")
    hardcoded_message = create_system_message_chat("", "", "")
    hardcoded_message_size = message_size(hardcoded_message)
    max_size = max_size - hardcoded_message_size

    # if document_list:
    #     # truncate the main document as necessary
    #     max_document_size = int(current_file_weight * max_size)
    # else:
    #     max_document_size = max_size

    # rescale `max_size_document` if we need to make room for the other documents
    max_size_document = int(max_size * (current_file_weight if documents else 1.0))

    before_cursor = document[:cursor_offset_start]
    region = document[cursor_offset_start:cursor_offset_end]
    after_cursor = document[cursor_offset_end:]

    # TODO: handle case when region is too large
    # calculate truncation for the ur-document
    if get_num_tokens(document) > max_size_document:
        tokens_before_cursor = ENCODER.encode(before_cursor)
        tokens_after_cursor = ENCODER.encode(after_cursor)
        (tokens_before_cursor, tokens_after_cursor) = split_lists(
            tokens_before_cursor,
            tokens_after_cursor,
            max_size_document - len(ENCODER.encode(region)),
        )
        logger.debug(
            f"Truncating document to ({len(tokens_before_cursor)}, {len(tokens_after_cursor)}) tokens around cursor"
        )
        before_cursor = ENCODER.decode(tokens_before_cursor)
        after_cursor = ENCODER.decode(tokens_after_cursor)

    # document_tokens = encoder.encode(document)
    # if len(document_tokens) > max_size_document:
    #     document_tokens: List[int] = truncate_around_region(
    #         document, document_tokens, cursor_offset_start, cursor_offset_end, max_size_document
    #     )
    # truncated_document = encoder.decode(document_tokens)

    truncated_document_list = []
    logger.info(f"document list = {documents}")
    if documents:
        max_document_list_size = ((1.0 - current_file_weight) * max_size) // len(documents)
        max_document_list_size = int(max_document_list_size)
        for doc in documents:
            # TODO: Need a check for using up our limit
            document_contents = doc.document.text
            # logger.info(f"{document_contents=}")
            tokens = encoder.encode(document_contents)
            logger.info("got tokens")
            if len(tokens) > max_document_list_size:
                tokens = tokens[:max_document_list_size]
                logger.info("truncated tokens")
                logger.debug(f"Truncating document to first {len(tokens)} tokens")
            logger.info("creating new doc")
            new_doc = lsp.Document(doc.uri, document=lsp.DocumentContext(encoder.decode(tokens)))
            logger.info("created new doc")
            truncated_document_list.append(new_doc)

    return create_system_message_chat(before_cursor, region, after_cursor, truncated_document_list)


def truncate_messages(
    messages: List[Message],
    max_context_size: int = MAX_CONTEXT_SIZE,
    max_len_sampled_completion=MAX_LEN_SAMPLED_COMPLETION,
):
    system_message_size = message_size(messages[0])
    max_size = calc_max_non_system_msgs_size(
        system_message_size,
        max_context_size=max_context_size,
        max_len_sampled_completion=max_len_sampled_completion,
    )
    # logger.info(f"{max_size=}")
    tail_messages: List[Message] = []
    running_length = 0
    for msg in reversed(messages[1:]):
        # logger.info(f"{running_length=}")
        running_length += message_size(msg)
        if running_length > max_size:
            break
        tail_messages.insert(0, msg)
    return [messages[0]] + tail_messages


class OpenAIClient(BaseSettings, AbstractCodeCompletionProvider, AbstractChatCompletionProvider):
    api_key: Optional[SecretStr] = None
    api_url: str = "https://api.openai.com/v1"
    default_model: Optional[str] = None

    class Config:
        env_prefix = "OPENAI_"
        env_file = ".env"
        keep_untouched = (cached_property,)

    def __str__(self):
        k = self.api_key.get_secret_value()
        k = f"{k[:3]}...{k[-4:]}"
        return f"{self.__class__.__name__} {self.api_url} {k}"

    @property
    def base_url(self) -> str:
        return urlparse(self.api_url)._replace(path="", query="", params="", fragment="").geturl()

    @property
    def url_path(self) -> str:
        return urlparse(self.api_url).path

    @property
    def url_query(self) -> dict:
        q = urlparse(self.api_url).query
        return parse_qs(q)

    @property
    def headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "User-Agent": __name__,
        }

    @cached_property
    def session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(
            base_url=self.base_url,
            headers=self.headers,
        )

    async def handle_error(self, resp: aiohttp.ClientResponse):
        status_code = resp.status
        message = await self.get_error_message(resp)
        message = f"{status_code} error from {self.base_url}: {message}"
        logger.error(message)
        if status_code == 404 and self.default_model == "gpt-4":
            logger.error(
                "Please double check you have access to GPT-4 API: https://openai.com/waitlist/gpt-4-api"
            )
        raise OpenAIError(message=message, status=status_code)

    async def get_error_message(self, resp):
        if resp.content_type == "application/json":
            j = await resp.json()
        else:
            t = await resp.text()
            try:
                j = json.loads(t)
            except json.JSONDecodeError:
                return t
        if isinstance(j, str):
            return j
        for k in ["error", "message", "detail"]:
            if k in j:
                e = j[k]
                if isinstance(e, str):
                    return e
                if "message" in e:
                    m = e["message"]
                    if isinstance(m, str):
                        return m
        raise ValueError(f"Could not parse error message from {j}")

    def _make_path(self, endpoint: str) -> str:
        return self.url_path + endpoint

    async def _post_streaming(
        self,
        endpoint: str,
        params: I,
        input_type: Type[I],
        stream_data_type: Type[O],
    ) -> AsyncGenerator[O, None]:
        if not self.api_key:
            logger.error("Missing API key")
            raise MissingKeyError
        if not getattr(params, "stream", True):
            raise ValueError("To not use streaming please use the _post_endpoint method")
        if not isinstance(params, input_type):
            raise TypeError(f"expected {input_type}, got {type(params)}")
        payload = params.dict(exclude_none=True)
        payload["stream"] = True
        path = self._make_path(endpoint)
        # logger.info(f"{payload=}")
        async with self.session.post(path, params=self.url_query, json=payload) as resp:
            if not resp.ok:
                await self.handle_error(resp)
            while True:
                line_ = await resp.content.readline()
                if line_ == b"":
                    break
                if line_ == b"\n":
                    continue
                line = line_.decode("utf-8")  # [todo] where to get encoding from?
                if line.startswith("data:"):
                    line = line.split("data:")[1]
                    line = line.strip()
                    if line == "[DONE]":
                        break
                    data = stream_data_type.parse_raw(line)
                    yield data
                else:
                    raise ValueError(f"unrecognised stream line: {line}")

    async def _post_endpoint(
        self,
        endpoint: str,
        params: I,
        input_type: Type[I],
        output_type: Type[O],
    ) -> O:
        if not self.api_key:
            logger.error("Missing API key")
            raise Exception("[OpenAIClient] missing API key")
        if not isinstance(params, input_type):
            raise TypeError(f"expected {input_type}, got {type(params)}")
        if getattr(params, "stream", False):
            raise ValueError("To use streaming please use the _post_streaming method")
        payload = params.dict(exclude_none=True)
        path = self._make_path(endpoint)
        async with self.session.post(path, params=self.url_query, json=payload) as resp:
            if not resp.ok:
                await self.handle_error(resp)
            assert resp.content_type == "application/json"
            j = await resp.json()
        r = output_type.parse_obj(j)  # type: ignore
        return r  # type: ignore

    @overload
    def chat_completions(
        self, messages: List[Message], *, stream: Literal[True], **kwargs
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        ...

    @overload
    def chat_completions(
        self, messages: List[Message], *, stream: Literal[False], **kwargs
    ) -> Coroutine[Any, Any, ChatCompletionResponse]:
        ...

    def chat_completions(self, messages: List[Message], *, stream: bool = False, **kwargs) -> Any:
        # logger.info(f"{messages=}")
        endpoint = "/chat/completions"
        input_type = ChatCompletionRequest
        # TODO: don't hardcode
        logit_bias = {99750: -100}  # forbid repetition of the cursor sentinel
        params = ChatCompletionRequest(
            messages=messages,
            stream=stream,
            logit_bias=logit_bias,
            max_tokens=MAX_LEN_SAMPLED_COMPLETION,
            **kwargs,
        )
        if self.default_model:
            params.model = self.default_model
        output_type = ChatCompletionResponse

        if stream:
            return self._post_streaming(
                endpoint,
                params=params,
                input_type=input_type,
                stream_data_type=ChatCompletionChunk,
            )
        else:
            return self._post_endpoint(
                endpoint, params=params, input_type=input_type, output_type=output_type
            )

    async def run_chat(
        self,
        document: Optional[str],
        messages: List[Message],
        message: str,
        cursor_offset_start: Optional[int] = None,
        cursor_offset_end: Optional[int] = None,
        documents: Optional[List[lsp.Document]] = None,
    ) -> ChatResult:
        chatstream = TextStream()
        non_system_messages = []
        for msg in messages:
            logger.debug(str(msg))
            non_system_messages.append(Message.mk(role=msg.role, content=msg.content))
        non_system_messages += [Message.user(content=message)]
        non_system_messages_size = messages_size(non_system_messages)

        max_system_msg_size = calc_max_system_message_size(non_system_messages_size)
        # logger.info(f"{max_system_msg_size=}")

        # logger.info(f"{documents=}")
        system_message = create_system_message_chat_truncated(
            document or "", max_system_msg_size, cursor_offset_start, cursor_offset_end, documents
        )

        messages = [system_message] + non_system_messages

        num_old_messages = len(messages)
        # Truncate the messages to ensure that the total set of messages (system and non-system) fit within MAX_CONTEXT_SIZE
        messages = truncate_messages(messages)
        logger.info(
            f"Truncated {num_old_messages - len(messages)} non-system messages due to context length overflow."
        )

        stream = TextStream.from_aiter(
            asg.map(lambda c: c.text, self.chat_completions(messages, stream=True))
        )

        event = asyncio.Event()

        async def worker():
            try:
                async for delta in stream:
                    chatstream.feed_data(delta)
                chatstream.feed_eof()
            except MissingKeyError as e:
                event.set()
                raise e
            finally:
                chatstream.feed_eof()

        t = asyncio.create_task(worker())
        chatstream._feed_task = t
        # logger.info("Created chat stream, awaiting results.")
        return ChatResult(text=chatstream, event=event)

    async def edit_code(
        self,
        document: str,
        cursor_offset_start: int,
        cursor_offset_end: int,
        goal=None,
        latest_region: Optional[str] = None,
        documents: Optional[List[lsp.Document]] = None,
        current_file_weight: float = 0.5,
    ) -> EditCodeResult:
        # logger.info(f"[edit_code] entered {latest_region=}")
        if goal is None:
            goal = f"""
            Generate code to replace the given `region`. Write a partial code snippet without imports if needed.
            """

        def create_messages(
            before_cursor: str,
            region: str,
            after_cursor: str,
            documents: Optional[List[lsp.Document]] = None,
        ) -> List[Message]:
            user_message = (
                f"Please generate code completing the task which will replace the below region: {goal}\n"
                "==== PREFIX ====\n"
                f"{before_cursor}"
                "==== REGION ====\n"
                f"{latest_region or region}\n"
                "==== SUFFIX ====\n"
                f"{after_cursor}\n"
            )
            user_message = format_visible_files(documents) + user_message

            return [
                Message.system(
                    "You are a brilliant coder and an expert software engineer and world-class systems architect with deep technical and design knowledge. You value:\n"
                    "- Conciseness\n"
                    "- DRY principle\n"
                    "- Self-documenting code with plenty of comments\n"
                    "- Modularity\n"
                    "- Deduplicated code\n"
                    "- Readable code\n"
                    "- Abstracting things away to functions for reusability\n"
                    "- Logical thinking\n"
                    "\n\n"
                    "You will be presented with a *task* and a source code file split into three parts: a *prefix*, *region*, and *suffix*. "
                    "The task will specify a change or new code that will replace the given region.\n You will receive the source code in the following format:\n"
                    "==== PREFIX ====\n"
                    "${source code file before the region}\n"
                    "==== REGION ====\n"
                    "${region}\n"
                    "==== SUFFIX ====\n"
                    "{source code file after the region}\n\n"
                    "When presented with a task, you will:\n(1) write a detailed and elegant plan to solve this task,\n(2) write your solution for it surrounded by triple backticks, and\n(3) write a 1-2 sentence summary of your solution.\n"
                    f"Your solution will be added verbatim to replace the given region. Do *not* repeat the prefix or suffix in any way.\n"
                    "The solution should directly replaces the given region. If the region is empty, just write something that will replace the empty string. *Do not repeat the prefix or suffix in any way*. If the region is in the middle of a function definition or class declaration, do not repeat the function signature or class declaration. Write a partial code snippet without imports if needed. Preserve indentation.\n"
                    f"For example, if the source code looks like this:\n"
                    "==== PREFIX ====\n"
                    "def hello_world():\n    \n"
                    "==== REGION ====\n"
                    "\n"
                    "==== SUFFIX ====\n"
                    "if __name__ == '__main__':\n    hello_world()\n\n"
                    "And the task is 'implement this function and return 0', then a good response would be\n"
                    "We will implement hello world by first using the Python `print` statement and then returning the integer literal 0.\n"
                    "```\n"
                    "# print hello world\n"
                    "    print('hello world!')\n"
                    "    # return the integer 0\n"
                    "    return 0\n"
                    "```\n"
                    "I added an implementation of the rest of the `hello_world` function which uses the Python `print` statement to print 'hello_world' before returning the integer literal 0.\n"
                ),
                Message.assistant("Hello! How can I help you today?"),
                Message.user(user_message),
            ]

        messages_skeleton = create_messages("", "", "")
        max_size = MAX_CONTEXT_SIZE - MAX_LEN_SAMPLED_COMPLETION - messages_size(messages_skeleton)

        # rescale `max_size_document` if we need to make room for the other documents
        max_size_document = int(max_size * (current_file_weight if documents else 1.0))

        before_cursor = document[:cursor_offset_start]
        region = document[cursor_offset_start:cursor_offset_end]
        after_cursor = document[cursor_offset_end:]

        # TODO: handle case when region is too large
        # calculate truncation for the ur-document
        if get_num_tokens(document) > max_size_document:
            tokens_before_cursor = ENCODER.encode(before_cursor)
            tokens_after_cursor = ENCODER.encode(after_cursor)
            (tokens_before_cursor, tokens_after_cursor) = split_lists(
                tokens_before_cursor,
                tokens_after_cursor,
                max_size_document - len(ENCODER.encode(region)),
            )
            logger.debug(
                f"Truncating document to ({len(tokens_before_cursor)}, {len(tokens_after_cursor)}) tokens around cursor"
            )
            before_cursor = ENCODER.decode(tokens_before_cursor)
            after_cursor = ENCODER.decode(tokens_after_cursor)

        # calculate truncation for the other context documents, if necessary
        truncated_documents = []
        if documents:
            max_document_list_size = ((1.0 - current_file_weight) * max_size) // len(documents)
            max_document_list_size = int(max_document_list_size)
            for doc in documents:
                tokens = ENCODER.encode(doc.document.text)
                if len(tokens) > max_document_list_size:
                    tokens = tokens[:max_document_list_size]
                    logger.debug(f"Truncating document to first {len(tokens)} tokens")
                doc = lsp.Document(
                    uri=doc.uri, document=lsp.DocumentContext(ENCODER.decode(tokens))
                )
                truncated_documents.append(doc)

        messages = create_messages(
            before_cursor=before_cursor,
            region=region,
            after_cursor=after_cursor,
            documents=truncated_documents,
        )
        # logger.info(f"{messages=}")

        event = asyncio.Event()        
        def error_callback(e):
            event.set()

        stream = TextStream.from_aiter(
            asg.map(lambda c: c.text, self.chat_completions(messages, stream=True), error_callback=error_callback)
        )

        logger.info("constructed stream")
        # logger.info(f"{stream=}")
        thoughtstream = TextStream()
        codestream = TextStream()
        planstream = TextStream()

        async def worker():
            logger.info("[edit_code:worker]")
            try:
                prelude, stream2 = stream.split_once("```")
                # logger.info(f"{prelude=}")
                async for delta in prelude:
                    # logger.info(f"plan {delta=}")
                    planstream.feed_data(delta)
                planstream.feed_eof()
                lang_tag = await stream2.readuntil("\n")
                before, after = stream2.split_once("\n```")
                # logger.info(f"{before=}")
                logger.info("reading codestream")
                async for delta in before:
                    # logger.info(f"code {delta=}")
                    codestream.feed_data(delta)
                codestream.feed_eof()
                # thoughtstream.feed_data("\n")
                logger.info("reading thoughtstream")
                async for delta in after:
                    thoughtstream.feed_data(delta)
                thoughtstream.feed_eof()
            except Exception as e:
                event.set()
                raise e
            finally:
                planstream.feed_eof()
                thoughtstream.feed_eof()
                codestream.feed_eof()
                # logger.info("FED EOF TO ALL")

        t = asyncio.create_task(worker())
        thoughtstream._feed_task = t
        codestream._feed_task = t
        planstream._feed_task = t
        # logger.info("[edit_code] about to return")
        return EditCodeResult(thoughts=thoughtstream, code=codestream, plan=planstream, event=event)

    async def insert_code(self, document: str, cursor_offset: int, goal=None) -> InsertCodeResult:
        raise Exception("unreachable code")


async def _main():
    client = OpenAIClient()  # type: ignore
    print(client)
    messages = [
        Message.system("you are a friendly and witty chatbot."),
        Message.user("please tell me a joke involving a lemon and a rubiks cube."),
        Message.assistant("i won't unless if you ask nicely"),
    ]

    stream = await client.run_chat("fee fi fo fum", messages=messages, message="pretty please?")
    async for delta in stream.text:
        print(delta)
    # print("\n\n")
    # async for x in client.chat_completions(messages, stream=True):
    #     text = x.choices[0].delta.content or ""
    #     print(text, end="")

    # print("\n\n")


if __name__ == "__main__":
    asyncio.run(_main())

from typing import List
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property, cache
import json
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Coroutine,
    Literal,
    Optional,
    Type,
    TypeVar,
    overload,
)
from urllib.parse import parse_qs, urlparse
import aiohttp
from pydantic import BaseModel, BaseSettings, SecretStr
from rift.llm.abstract import (
    AbstractCodeCompletionProvider,
    AbstractChatCompletionProvider,
    InsertCodeResult,
    ChatResult,
)

from rift.util.TextStream import TextStream
from rift.llm.openai_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
)
import rift.util.asyncgen as asg
import logging
from threading import Lock

logger = logging.getLogger(__name__)

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)

from tiktoken import get_encoding

ENCODER = get_encoding("cl100k_base")
ENCODER_LOCK = Lock()

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
MAX_LEN_SAMPLED_COMPLETION = 512  # Reserved tokens for model's responses
MAX_SYSTEM_MESSAGE_SIZE = 1024 # Token limit for system message

def calc_max_non_system_msgs_size(system_message_size: int) -> int:
    """ Maximum size of the non-system messages """
    return MAX_CONTEXT_SIZE - MAX_LEN_SAMPLED_COMPLETION - system_message_size

def calc_max_system_message_size(non_system_messages_size: int) -> int:
    """ Maximum size of the system message """

    # Calculate the maximum size for the system message. It's either the maximum defined limit 
    # or the remaining tokens in the context size after accounting for model responses and non-system messages,
    # whichever is larger. This ensures that the system message can take advantage of spare space, if available.
    return max(
            MAX_SYSTEM_MESSAGE_SIZE,
            MAX_CONTEXT_SIZE - MAX_LEN_SAMPLED_COMPLETION - non_system_messages_size)

def create_system_message(document: str) -> Message:
    """
    Create system message wiht up to MAX_SYSTEM_MESSAGE_SIZE tokens
    """
    return Message.system(
                    f"""
You are an expert software engineer and world-class systems architect with deep technical and design knowledge. Answer the user's questions about the code as helpfully as possible, quoting verbatim from the current file to support your claims.

Current file:
```
{document}
```

Answer the user's question."""
    )

def create_system_message_truncated(document: str, max_size: int, cursor_offset: Optional[int]) -> Message:
    """
    Create system message with up to max_size tokens
    """

    hardcoded_message = create_system_message("")
    hardcoded_message_size = message_size(hardcoded_message)
    max_size = max_size - hardcoded_message_size

    doc_tokens = ENCODER.encode(document)
    if len(doc_tokens) > max_size:
        if cursor_offset:
            before_cursor = document[:cursor_offset]
            after_cursor = document[cursor_offset:]
            tokens_before_cursor = ENCODER.encode(before_cursor)
            tokens_after_cursor = ENCODER.encode(after_cursor)
            (tokens_before_cursor, tokens_after_cursor) = split_lists(
                tokens_before_cursor, tokens_after_cursor, max_size)
            logger.debug(
                f"Truncating document to ({len(tokens_before_cursor)}, {len(tokens_after_cursor)}) tokens around cursor")
            tokens = tokens_before_cursor + tokens_after_cursor
        else:
            # if there is no cursor offset provided, simply take the last max_size tokens
            tokens = doc_tokens[-max_size:]
            logger.debug(f"Truncating document to last {len(tokens)} tokens")

        document = ENCODER.decode(tokens)

    return create_system_message(document)

def truncate_messages(messages: List[Message]):
    system_message_size = message_size(messages[0])
    max_size = calc_max_non_system_msgs_size(system_message_size)
    tail_messages: List[Message] = []
    running_length = 0
    for msg in reversed(messages[1:]):
        running_length += message_size(msg)
        if running_length > max_size:
            break
        tail_messages.insert(0, msg)

    return [messages[0]] + tail_messages


class OpenAIClient(
    BaseSettings, AbstractCodeCompletionProvider, AbstractChatCompletionProvider
):
    api_key: SecretStr
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
        return (
            urlparse(self.api_url)
            ._replace(path="", query="", params="", fragment="")
            .geturl()
        )

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
        logging.error(message)
        if status_code == 404 and self.default_model == "gpt-4":
            logging.info("Please double check you have access to GPT-4 API: https://openai.com/waitlist/gpt-4-api")
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
        if not getattr(params, "stream", True):
            raise ValueError(
                "To not use streaming please use the _post_endpoint method"
            )
        if not isinstance(params, input_type):
            raise TypeError(f"expected {input_type}, got {type(params)}")
        payload = params.dict(exclude_none=True)
        payload["stream"] = True
        path = self._make_path(endpoint)
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

    def chat_completions(
        self, messages: List[Message], *, stream: bool = False, **kwargs
    ) -> Any:
        endpoint = "/chat/completions"
        input_type = ChatCompletionRequest
        # TODO: don't hardcode
        logit_bias = {99750: -100}  # forbid repetition of the cursor sentinel
        params = ChatCompletionRequest(
            messages=messages, stream=stream, logit_bias=logit_bias,
            max_tokens=MAX_LEN_SAMPLED_COMPLETION, **kwargs
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
        self, document: str, messages: List[Message], message: str, cursor_offset: Optional[int] = None
    ) -> ChatResult:
        chatstream = TextStream()

        non_system_messages = (
            [Message.mk(role=msg.role, content=msg.content) for msg in messages]
            +
            [Message.user(content=message)]
            )
        non_system_messages_size = messages_size(non_system_messages)

        max_system_msg_size = calc_max_system_message_size(non_system_messages_size)
        system_message = create_system_message_truncated(document, max_system_msg_size, cursor_offset)

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

        async def worker():
            try:
                async for delta in stream:
                    chatstream.feed_data(delta)
                chatstream.feed_eof()
            finally:
                chatstream.feed_eof()

        t = asyncio.create_task(worker())
        chatstream._feed_task = t
        logger.info("Created chat stream, awaiting results.")
        return ChatResult(text=chatstream)

    async def insert_code(
        self, document: str, cursor_offset: int, goal=None
    ) -> InsertCodeResult:
        CURSOR_SENTINEL = "感"
        if goal is None:
            goal = f"""
            Generate code to be inserted at the cursor location, marked by {CURSOR_SENTINEL}.
            """

        def create_messages(before_cursor: str, after_cursor: str) -> List[Message]:
            doc_text_with_cursor = before_cursor + CURSOR_SENTINEL + after_cursor
            return [
                Message.system(
                    f"""You are an expert software engineer and world-class systems architect with deep technical and design knowledge.
                    When presented with a task, first write a detailed and elegant plan to solve this task and then
                    write code to do it surrounded by triple backticks.
                    The code will be added verbatim to the cursor location, marked by {CURSOR_SENTINEL}.
                    Add comments in the code to explain your reasoning."""
                ),
                Message.user(
                    f"Here is the code:\n```\n{doc_text_with_cursor}\n```\n\nYour task is:\n{goal}\nInsert code at the {CURSOR_SENTINEL} which completes the task. The code will be added verbatim to the cursor location, marked by {CURSOR_SENTINEL}. Do not include code that is already there."
                ),
            ]

        before_cursor = document[:cursor_offset]
        after_cursor = document[cursor_offset:]
        messages_skeleton = create_messages("", "")
        max_size_document = MAX_CONTEXT_SIZE - MAX_LEN_SAMPLED_COMPLETION - messages_size(messages_skeleton)

        if get_num_tokens(document) > max_size_document:
            tokens_before_cursor = ENCODER.encode(before_cursor)
            tokens_after_cursor = ENCODER.encode(after_cursor)
            (tokens_before_cursor, tokens_after_cursor) = split_lists(
                tokens_before_cursor, tokens_after_cursor, max_size_document)
            logger.debug(
                f"Truncating document to ({len(tokens_before_cursor)}, {len(tokens_after_cursor)}) tokens around cursor")
            before_cursor = ENCODER.decode(tokens_before_cursor)
            after_cursor = ENCODER.decode(tokens_after_cursor)

        messages = create_messages(before_cursor, after_cursor)

        stream = TextStream.from_aiter(
            asg.map(lambda c: c.text, self.chat_completions(messages, stream=True))
        )
        thoughts = TextStream()
        codestream = TextStream()

        async def worker():
            try:
                try:
                    prelude = await stream.readuntil("```")
                    logger.debug(f"prelude: {prelude}")
                    lang_tag = await stream.readuntil("\n")
                    if lang_tag:
                        logger.debug(f"lang_tag: {lang_tag}")
                except EOFError:
                    logger.error("never found a code block")
                    return
                before, after = stream.split_once("\n```")
                async for delta in before:
                    codestream.feed_data(delta)
                codestream.feed_eof()
                async for delta in after:
                    thoughts.feed_data("\n")
                    thoughts.feed_data(delta)
            finally:
                thoughts.feed_eof()
                codestream.feed_eof()

        t = asyncio.create_task(worker())
        thoughts._feed_task = t
        codestream._feed_task = t
        return InsertCodeResult(thoughts=thoughts, code=codestream)


async def _main():
    client = OpenAIClient()  # type: ignore
    print(client)
    messages = [
        Message.system("you are a friendly and witty chatbot."),
        Message.user("please tell me a joke involving a lemon and a rubiks cube."),
        Message.assistant("i won't unless if you ask nicely"),
    ]

    stream = await client.run_chat(
        "fee fi fo fum", messages=messages, message="pretty please?"
    )
    async for delta in stream.text:
        print(delta)
    # print("\n\n")
    # async for x in client.chat_completions(messages, stream=True):
    #     text = x.choices[0].delta.content or ""
    #     print(text, end="")

    # print("\n\n")


if __name__ == "__main__":
    asyncio.run(_main())

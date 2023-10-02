import asyncio
import ctypes
import logging
import threading
from functools import cache
from pathlib import Path
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

from pydantic import BaseModel, BaseSettings, SecretStr

from gpt4all import GPT4All
from gpt4all.pyllmodel import (
    LLModel,
    LLModelPromptContext,
    PromptCallback,
    RecalculateCallback,
    ResponseCallback,
    llmodel,
)
from pydantic import BaseSettings

from rift.llm.abstract import (
    AbstractChatCompletionProvider,
    AbstractCodeCompletionProvider,
    ChatResult,
    InsertCodeResult,
    EditCodeResult
)
from rift.llm.openai_client import (
    calc_max_system_message_size,
    create_system_message_chat_truncated,
    messages_size,
    truncate_messages,
)
from rift.llm.openai_types import Message
from rift.util.TextStream import TextStream
from rift.llm.prompt import SourceCodeFileWithRegion

import rift.lsp.types as lsp
import rift.util.asyncgen as asg
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

from rift.util.logging import configure_logger

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)

import transformers

from threading import Lock

ENCODER = transformers.AutoTokenizer.from_pretrained("TheBloke/CodeLlama-7B-Instruct-fp16")
ENCODER_LOCK = Lock()

logger = logging.getLogger(__name__)

from threading import Lock

# ENCODER = get_encoding("cl100k_base")
# from transformers import LlamaTokenizer
import transformers

ENCODER_LOCK = Lock()


generate_lock = asyncio.Lock()

default_args = dict(
    logits_size=0,
    tokens_size=0,
    n_past=0,
    n_ctx=1024,
    n_predict=256,
    top_k=40,
    top_p=0.9,
    temp=0.1,
    n_batch=8,
    repeat_penalty=1.2,
    repeat_last_n=10,
    context_erase=0.5,
)


def generate_stream(self: LLModel, prompt: str, **kwargs) -> TextStream:
    loop = asyncio.get_event_loop()
    cancelled_flag = threading.Event()
    output = TextStream(on_cancel=cancelled_flag.set)
    prompt_chars = ctypes.c_char_p(prompt.encode("utf-8"))
    kwargs = {**default_args, **kwargs}
    keys = [x for x, _ in LLModelPromptContext._fields_]
    context_args = {k: kwargs[k] for k in keys if k in kwargs}
    rest_kwargs = {k: kwargs[k] for k in kwargs if k not in keys}
    if len(rest_kwargs) > 0:
        logger.warning(f"Unrecognized kwargs: {rest_kwargs}")
    context = LLModelPromptContext(**context_args)

    def prompt_callback(token_id, response: Optional[bytes] = None):
        return not cancelled_flag.is_set()

    def response_callback(token_id, response: bytes):
        if cancelled_flag.is_set():
            logger.debug("response_callback cancelled")
            return False
        text = response.decode("utf-8")
        loop.call_soon_threadsafe(output.feed_data, text)
        return True

    def recalc_callback(is_recalculating):
        return is_recalculating

    def run():
        return llmodel.llmodel_prompt(
            self.model,
            prompt_chars,
            PromptCallback(prompt_callback),
            ResponseCallback(response_callback),
            RecalculateCallback(recalc_callback),
            context,
        )

    async def run_async():
        async with generate_lock:
            await loop.run_in_executor(None, run)
            output.feed_eof()

    output._feed_task = asyncio.create_task(run_async())
    return output


def model_name_to_tokenizer(name: str):
    if name == "ggml-gpt4all-j-v1.3-groovy":
        return transformers.AutoTokenizer.from_pretrained("nomic-ai/gpt4all-j")
    elif name == "ggml-mpt-7b-chat":
        return transformers.AutoTokenizer.from_pretrained("nomic-ai/gpt4all-mpt")
    elif name == "ggml-replit-code-v1-3b":
        return transformers.AutoTokenizer.from_pretrained("nomic-ai/ggml-replit-code-v1-3b")
    elif name == "ggml-model-gpt4all-falcon-q4_0":
        return transformers.AutoTokenizer.from_pretrained("nomic-ai/gpt4all-falcon")
    elif name == "orca-mini-3b.ggmlv3.q4_0":
        try:
            return transformers.AutoTokenizer.from_pretrained("psmathur/orca_mini_3b")
        except ImportError as e:
            logging.getLogger().error(
                f"ImportError: {e} - you may need to install the protobuf package"
            )
            raise e
    else:
        error_msg = f"WARNING: No tokenizer found for model={name}. Defaulting to llama tokenizer."
        logger.error(error_msg)
        # raise Exception(error_msg)
        # return transformers.AutoTokenizer.from_pretrained("nomic-ai/gpt4all-mpt")
        return ENCODER


# DEFAULT_MODEL_NAME = "ggml-gpt4all-j-v1.3-groovy"
# DEFAULT_MODEL_NAME = "ggml-mpt-7b-chat"
DEFAULT_MODEL_NAME = "ggml-rift-coder-v0-7b"
# DEFAULT_MODEL_NAME = "ggml-replit-code-v1-3b"

create_model_lock = threading.Lock()


class Gpt4AllSettings(BaseSettings):
    model_name: str = DEFAULT_MODEL_NAME
    model_path: Optional[Path] = None
    model_type: Optional[str] = None

    class Config:
        env_prefix = "GPT4ALL_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __str__(self):
        s = self.model_name
        if self.model_path is not None:
            s += f" at ({self.model_path})"
        return s

    def create_model(self):
        with create_model_lock:
            kwargs = {"model_name": self.model_name}
            if self.model_path is not None:
                kwargs["model_path"] = str(self.model_path)
            if self.model_type is not None:
                kwargs["model_type"] = self.model_type
            model = GPT4All(**kwargs)
            return model



@cache
def get_num_tokens(content: str, encoder=ENCODER):
    return len(encoder.encode(content))


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

        
class Gpt4AllModel(AbstractCodeCompletionProvider, AbstractChatCompletionProvider):
    def __init__(self, config: Optional[Gpt4AllSettings] = None):
        if config is None:
            config = Gpt4AllSettings()
        self.config = config
        logger.info(f"creating gpt4all model {self.config}")
        self.name = config.model_name
        self._model_future = None
        self.ENCODER = model_name_to_tokenizer(self.config.model_name)

    async def load(self):
        await self._get_model()

    @cache
    def get_num_tokens(self, content):
        return len(self.ENCODER.encode(content))

    @property
    async def model(self):
        return await self._get_model()

    async def _get_model(self):
        if self._model_future is None:
            self._model_future = asyncio.get_running_loop().run_in_executor(
                None, self.config.create_model
            )
        return await self._model_future

    async def insert_code(
        self, code: str, cursor_offset: int, goal: Optional[str] = None
    ) -> InsertCodeResult:
        model = await self._get_model()
        before_cursor = code[:cursor_offset]
        after_cursor = code[cursor_offset:]
        prompt = before_cursor
        if goal is not None:
            # [todo] prompt engineering here
            # goal is a string that the user writes saying what they want the edit to achieve.
            prompt = goal + "\n\n" + prompt
        inner_model = model.model
        assert inner_model is not None
        output = generate_stream(inner_model, prompt)
        return InsertCodeResult(code=output, thoughts=None)

    async def run_chat(
        self,
        document: str,
        messages: List[Message],
        message: str,
        cursor_offset_start: Optional[int] = None,
        cursor_offset_end: Optional[int] = None,
        documents: Optional[List[Any]] = None,
    ) -> ChatResult:
        logger.debug("run_chat called")
        model = await self._get_model()
        chatstream = TextStream()
        non_system_messages = []
        for msg in messages:
            logger.debug(str(msg))
            non_system_messages.append(Message.mk(role=msg.role, content=msg.content))
        non_system_messages += [Message.user(content=message)]
        non_system_messages_size = messages_size(non_system_messages)
        max_system_msg_size = calc_max_system_message_size(
            non_system_messages_size,
            max_system_message_size=768,
            max_context_size=2048,
            max_len_sampled_completion=256,
        )
        # logger.info(f"{max_system_msg_size=}")
        system_message = create_system_message_chat_truncated(
            document or "",
            max_system_msg_size,
            cursor_offset_start,
            cursor_offset_end,
            documents,
            encoder=self.ENCODER,
        )
        # logger.info(f"{system_message=}")
        messages = (
            [
                #                 Message.system(
                #                     f"""
                # You are an expert software engineer and world-class systems architect with deep technical and design knowledge. Answer the user's questions about the code as helpfully as possible, quoting verbatim from the current file to support your claims.
                # Current file:
                # ```
                # {document}
                # ```
                # Answer the user's question."""
                #                 )
                system_message
            ]
            + [Message.mk(role=msg.role, content=msg.content) for msg in messages]
            + [Message.user(content=message)]
        )

        num_old_messages = len(messages)
        # messages = auto_truncate(messages)
        messages = truncate_messages(
            messages, max_context_size=2048, max_len_sampled_completion=256
        )

        logger.info(f"Truncated {num_old_messages - len(messages)} due to context length overflow.")

        def build_prompt(msgs: List[Message]) -> str:
            result = """### Instruction:
            The prompt below is a conversation to respond to. Write an appropriate and helpful response.
            \n### Prompt: """

            for msg in msgs:
                result += f"[{msg.role}]\n{msg.content}" + "\n"

            return result + "[assistant]\n" + "### Response\n"

        inner_model = model.model
        prompt = build_prompt(messages)

        logger.info(f"Created chat prompt with {len(prompt)} characters.")

        stream = generate_stream(inner_model, prompt)

        async def worker():
            try:
                async for delta in stream:
                    chatstream.feed_data(delta)
                chatstream.feed_eof()
            finally:
                chatstream.feed_eof()

        t = asyncio.create_task(worker())
        chatstream._feed_task = t
        # logger.info("Created chat stream, awaiting results.")
        return ChatResult(text=chatstream)

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
        model = await self._get_model()
        # logger.info(f"[edit_code] entered {latest_region=}")
        if goal is None:
            goal = f"""
            Generate code to replace the given `region`. Write a partial code snippet without imports if needed.
            """

        # messages_skeleton = create_messages("", "", "", goal=goal, latest_region=latest_region)
        # messages_skeleton = create_messages("", "", "", )
        messages_skeleton = [Message.user(content=SourceCodeFileWithRegion(before_region="", region=latest_region or "", after_region="", instruction=goal).get_prompt(train=True))]
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

        # messages = create_messages(
        #     before_cursor=before_cursor,
        #     region=region,
        #     after_cursor=after_cursor,
        #     documents=truncated_documents,
        #     goal=goal,
        # )
        # logger.info(f"{messages=}")

        event = asyncio.Event()

        def error_callback(e):
            event.set()

        def postprocess(chunk):
            if chunk["choices"]:
                choice = chunk["choices"][0]
                if choice["finish_reason"]:
                    return ""
                if "content" in choice["delta"]:
                    return choice["delta"]["content"]
                else:
                    return ""
            return ""

        def postprocess2(chunk: CompletionChunk) -> str:
            return chunk["choices"][0]["text"]

        pre_prompt: SourceCodeFileWithRegion = SourceCodeFileWithRegion(
            region=region, before_region=before_cursor, after_region=after_cursor, instruction=goal
        )

        prompt = pre_prompt.get_prompt()
        # logger.info(f"{prompt=}")
        # stream = TextStream.from_aiter(self.completion(prompt, stream=True))

        inner_model = model.model
        # prompt = build_prompt(messages)

        logger.info(f"Created chat prompt with {len(prompt)} characters.")

        stream = TextStream.from_aiter(generate_stream(inner_model, prompt))

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

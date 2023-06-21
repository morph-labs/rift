import asyncio
import ctypes
import logging
from pathlib import Path
import threading
from typing import Optional, List

from pydantic import BaseSettings, Field
from rift.llm.abstract import (
    AbstractCodeCompletionProvider,
    InsertCodeResult,
    AbstractChatCompletionProvider,
    ChatResult,
)
from rift.llm.openai_types import Message
from gpt4all.pyllmodel import (
    LLModel,
    LLModelPromptContext,
    llmodel,
    PromptCallback,
    ResponseCallback,
    RecalculateCallback,
)

from gpt4all import GPT4All

from functools import cache

from rift.util.TextStream import TextStream

logger = logging.getLogger(__name__)

from threading import Lock

# ENCODER = get_encoding("cl100k_base")
from transformers import LlamaTokenizer

ENCODER = LlamaTokenizer.from_pretrained("oobabooga/llama-tokenizer")
ENCODER_LOCK = Lock()


@cache
def get_num_tokens(content):
    return len(ENCODER.encode(content))


def message_length(msg: Message):
    with ENCODER_LOCK:
        return get_num_tokens(msg.content)


def auto_truncate(messages: List[Message]):
    tail_messages = []
    running_length = 0
    for msg in reversed(messages[1:]):
        running_length += message_length(msg)
        if running_length > 1536:
            break
        tail_messages.insert(0, msg)
    return [messages[0]] + tail_messages


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
        logger.debug("starting gpt4all model")
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

DEFAULT_MODEL_NAME = "ggml-gpt4all-j-v1.3-groovy"
# DEFAULT_MODEL_NAME = "ggml-mpt-7b-chat"
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


class Gpt4AllModel(AbstractCodeCompletionProvider, AbstractChatCompletionProvider):
    def __init__(self, config: Optional[Gpt4AllSettings] = None):
        if config is None:
            config = Gpt4AllSettings()
        self.config = config
        logger.info(f"creating gpt4all model {self.config}")
        self.name = config.model_name
        self._model_future = None

    async def load(self):
        await self._get_model()

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
        self, document: str, messages: List[Message], message: str
    ) -> ChatResult:
        logger.debug("run_chat called")
        model = await self._get_model()
        chatstream = TextStream()
        messages = (
            [
                Message.system(
                    f"""
You are an expert software engineer and world-class systems architect with deep technical and design knowledge. Answer the user's questions about the code as helpfully as possible, quoting verbatim from the current file to support your claims.

Current file:
```
{document}
```

Answer the user's question."""
                )
            ]
            + [Message.mk(role=msg.role, content=msg.content) for msg in messages]
            + [Message.user(content=message)]
        )

        num_old_messages = len(messages)
        messages = auto_truncate(messages)

        logger.info(
            f"Truncated {num_old_messages - len(messages)} due to context length overflow."
        )

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

        logger.info("Created chat stream, awaiting results.")

        return ChatResult(text=chatstream)

import torch
import asyncio
import datetime
import json
import logging
import re
import time
from typing import Any, Optional, List, Dict, Callable
import rift.lsp.types as lsp
import rift.util.asyncgen as asg

from pydantic import BaseModel, BaseSettings

from transformers import AutoModelForCausalLM, AutoTokenizer

from rift.llm.abstract import (
    AbstractCodeCompletionProvider,
    EditCodeResult,
    ChatResult,
    AbstractCodeEditProvider,
)
from rift.util.TextStream import TextStream
from rift.llm.openai_client import format_visible_files, messages_size, get_num_tokens, split_lists
import numpy as np

from rift.llm.openai_types import Message


logger = logging.getLogger(__name__)

_mock_string = """```
def mock():
    endpoint = "/chat/completions"
    input_type = ChatCompletionRequest
    params = ChatCompletionRequest(messages=messages, stream=stream, **kwargs)
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
```
"""


class DataOutput(BaseModel):
    id: str
    created: str
    text: str


PastKv = tuple[tuple[np.array, ...], ...]


class HuggingFaceClient(AbstractCodeEditProvider):
    temperature: int = 1
    max_len: int = (
        128  # todo: short length for testing purpose. Maybe need to make it to 2048 in production?
    )

    def __init__(self, model_name=None):
        model_name = model_name or "Salesforce/codegen-350m-mono"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = self.tokenizer.model_max_length

    def infer(self, kwargs):
        result = self.model(**kwargs)
        return result

    async def generate(self, prompt: str, max_tokens: Optional[int] = None):
        """
        This method generates and yields completions from a HF model based on given prompt text and cursor offset.

        - prompt: The text you want to generate completion for. This should be of str data type.

        This method generates and yields code completion word by word, and adjusts it's behaviour to also support line-by-line yield if a newline character is encountered in the new generated word.
        """
        # Tokenize the prompt text
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
        buffer = []
        generated = ""
        past_kv: Optional[PastKv] = None

        # We aim to continuously generate code completion until we've completed a statement.
        while True:
            # For the model's first inference, provide the tokenized prompt
            if past_kv is None:
                kwargs = prompt_tokens  # type: ignore
                buffer.append(prompt_tokens["input_ids"].squeeze().detach().numpy())
            else:  # For subsequent inferences, provide the last generated token and previously calculated key values
                input_ids = torch.tensor(np.reshape(buffer[-1][:, None], (1, -1)), dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logger.info(f"{input_ids=}")
                logger.info(f"{attention_mask=}")
            logger.info(f"{kwargs['input_ids']=}")

            kwargs["use_cache"] = True
            kwargs["past_key_values"] = past_kv

            logger.info("get model output")
            # Perform inference in a non-blocking manner
            # model_output = await asyncio.get_event_loop().run_in_executor(None, self.infer, kwargs)
            model_output = self.infer(kwargs)
            await asyncio.sleep(0.0001)

            outputs: np.array = model_output.logits.detach().numpy()
            past_kv = model_output.past_key_values

            # Extract the most probable next token from the output logits (scaled inversely by temperature for smoother gradients)
            logits = outputs[:, -1] * 1.0 / (self.temperature + 1e-8)
            out_tk = np.argmax(logits, -1)
            logger.info(f"{out_tk=}")

            # Add the token to buffer to be used in the next iteration
            buffer.append(out_tk)

            # logger.info(f"{buffer=}")
            # import itertools
            # for x in itertools.chain.from_iterable(buffer):
            #     logger.info(x)
            # logger.info(f"{self.tokenizer.decode([x for x in itertools.chain(*buffer)])}")

            # If the token is an end of sentence token, break the loop.
            if int(np.squeeze(out_tk)) == self.tokenizer.eos_token_id:
                break

            # If we exceed the max length, break the loop.
            if max_tokens and len(buffer) >= max_tokens:
                break

            # Decode the token into a word and add it to our string of generated text
            new_word = self.tokenizer.decode(out_tk.squeeze())
            yield new_word
            logger.info(f"{new_word=}")

            # # If the word contains a newline character.
            # if "\n" in new_word:
            #     # Split the word by the new line character
            #     # Append the first part to the last part of the generated string and yield it as a generated line
            #     result = generated.split("\n")[-1] + new_word.split("\n")[0]
            #     yield result

            #     # Yield each of the remaining parts as a new line with a trailing newline character.
            #     for line in new_word.split("\n")[1:]:
            #         yield line + "\n"

            # Add the generated word
            generated += new_word

    async def chat_completions(self, messages: List[Message], stream: bool = True):
        MAX_LEN_SAMPLED_COMPLETION = 768

        def prepare_prompt():  # TODO: truncation here?
            prompt = ""

            def format_message(msg: Message) -> str:
                return f"===\n[{msg.role}]\n{msg.content}\n\n"

            for message in messages:
                prompt += format_message(message)

            prompt += "===\n[assistant]\n"
            return prompt

        async for delta in self.generate(
            prompt=prepare_prompt(), max_tokens=MAX_LEN_SAMPLED_COMPLETION
        ):
            logger.info(f"{delta=}")
            yield delta

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
        ENCODER = self.tokenizer
        MAX_CONTEXT_SIZE = self.tokenizer.model_max_length
        MAX_LEN_SAMPLED_COMPLETION = 768

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
                    "    logger.info('hello world!')\n"
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
        max_size_document = int(max_size * (current_file_weight if documents else 1.0))
        before_cursor = document[:cursor_offset_start]
        region = document[cursor_offset_start:cursor_offset_end]
        after_cursor = document[cursor_offset_end:]
        if get_num_tokens(document, ENCODER) > max_size_document:
            tokens_before_cursor = ENCODER.encode(before_cursor)
            tokens_after_cursor = ENCODER.encode(after_cursor)
            (tokens_before_cursor, tokens_after_cursor) = split_lists(
                tokens_before_cursor,
                tokens_after_cursor,
                max_size_document - len(ENCODER.encode(region)),
            )
            before_cursor = ENCODER.decode(tokens_before_cursor)
            after_cursor = ENCODER.decode(tokens_after_cursor)
        truncated_documents = []
        if documents:
            max_document_list_size = ((1.0 - current_file_weight) * max_size) // len(documents)
            max_document_list_size = int(max_document_list_size)
            for doc in documents:
                tokens = ENCODER.encode(doc.document.text)
                if len(tokens) > max_document_list_size:
                    tokens = tokens[:max_document_list_size]
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
        event = asyncio.Event()

        def error_callback(e):
            event.set()

        stream = TextStream.from_aiter(self.chat_completions(messages, stream=True))

        logger.info("constructed stream")
        # thoughtstream = TextStream()
        # codestream = TextStream()
        # planstream = TextStream()
        # async def worker():
        #     try:
        #         prelude, stream2 = stream.split_once("```")
        #         async for delta in prelude:
        #             planstream.feed_data(delta)
        #         planstream.feed_eof()
        #         lang_tag = await stream2.readuntil("\n")
        #         before, after = stream2.split_once("\n```")
        #         async for delta in before:
        #             codestream.feed_data(delta)
        #         codestream.feed_eof()
        #         async for delta in after:
        #             thoughtstream.feed_data(delta)
        #         thoughtstream.feed_eof()
        #     except Exception as e:
        #         event.set()
        #         raise e
        #     finally:
        #         planstream.feed_eof()
        #         thoughtstream.feed_eof()
        #         codestream.feed_eof()
        # t = asyncio.create_task(worker())
        # thoughtstream._feed_task = t
        # codestream._feed_task = t
        # planstream._feed_task = t
        return EditCodeResult(thoughts=None, code=stream, plan=None, event=event)


if __name__ == "__main__":
    PROMPT = """\
def hello_world():
    # TODO
"""

    async def main():
        client = HuggingFaceClient(model_name="Salesforce/codegen-350m-mono")
        async for delta in client.generate(prompt=PROMPT, max_tokens=32):
            logger.info(delta)

    asyncio.run(main())

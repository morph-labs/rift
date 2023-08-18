from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from rift.llm.openai_types import Message, MessageRole
from tiktoken import get_encoding

ENCODER = get_encoding("cl100k_base")


def token_length(string: str) -> int:
    return len(ENCODER.encode(string))


class Prompt(ABC):
    def __init__(self, size: int) -> None:
        self.size = size

    @abstractmethod
    def fit(self, max_size: int) -> Optional[Tuple[str, int]]:
        raise NotImplementedError

    @abstractproperty
    def min_size(self) -> int:
        raise NotImplementedError

    def __add__(self, other) -> "ConcatPrompt":
        return ConcatPrompt(self, other)

    def __or__(self, other) -> "EitherPrompt":
        return EitherPrompt(self, other)

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class StringPrompt(Prompt):
    def __init__(self, string: str) -> None:
        super().__init__(token_length(string))
        self.string = string

    def fit(self, max_size: int) -> Optional[Tuple[str, int]]:
        if self.size <= max_size:
            return self.string, self.size
        return None

    @property
    def min_size(self) -> int:
        return self.size

    def __str__(self) -> str:
        return self.string


class SplitStringPrompt(Prompt):
    def __init__(self, lhs: str, separator: str, rhs: str, min_size: Optional[int] = None) -> None:
        super().__init__(token_length(lhs) + token_length(rhs) + token_length(separator))
        self.string1 = lhs
        self.string2 = rhs
        self.separator = separator
        self.min_size_ = min_size if min_size else token_length(self.separator)

    def fit(self, max_size: int) -> Optional[Tuple[str, int]]:
        if self.min_size <= max_size:
            separator_size = token_length(self.separator)
            remaining_size = max_size - separator_size
            tokens_lhs = ENCODER.encode(self.string1)
            tokens_rhs = ENCODER.encode(self.string2)
            size_lhs = remaining_size // 2
            size_lhs = max(size_lhs, remaining_size - len(tokens_rhs))
            # cut tokens_lhs to the rightmost size_lhs tokens
            tokens_lhs = tokens_lhs[-size_lhs:] if size_lhs > 0 else []
            size_rhs = remaining_size - len(tokens_lhs)
            # cut tokens_rhs to the leftmost size_rhs tokens
            tokens_rhs = tokens_rhs[:size_rhs] if size_rhs > 0 else []
            combined_string = (
                ENCODER.decode(tokens_lhs) + self.separator + ENCODER.decode(tokens_rhs)
            )
            return combined_string, len(tokens_lhs) + separator_size + len(tokens_rhs)
        return None

    @property
    def min_size(self) -> int:
        return self.min_size_

    def __str__(self) -> str:
        return self.string1 + self.separator + self.string2


class ConcatPrompt(Prompt):
    def __init__(self, prompt1: Prompt, prompt2: Prompt) -> None:
        super().__init__(prompt1.size + prompt2.size)
        self.prompt1 = prompt1
        self.prompt2 = prompt2

    def fit(self, max_size: int) -> Optional[Tuple[str, int]]:
        max_size1 = max_size - self.prompt2.min_size
        first = self.prompt1.fit(max_size1)
        if first is None:
            return None

        string1, size1 = first
        remaining_size = max_size - size1
        second = self.prompt2.fit(remaining_size)
        if second is None:
            return None

        string2, size2 = second
        return string1 + string2, size1 + size2

    @property
    def min_size(self) -> int:
        return self.prompt1.min_size + self.prompt2.min_size

    def __str__(self) -> str:
        return str(self.prompt1) + str(self.prompt2)


class EitherPrompt(Prompt):
    def __init__(self, prompt1: Prompt, prompt2: Prompt) -> None:
        super().__init__(max(prompt1.size, prompt2.size))
        self.prompt1 = prompt1
        self.prompt2 = prompt2

    def fit(self, max_size: int) -> Optional[Tuple[str, int]]:
        first = self.prompt1.fit(max_size)
        if first is not None:
            return first
        return self.prompt2.fit(max_size)

    @property
    def min_size(self) -> int:
        return min(self.prompt1.min_size, self.prompt2.min_size)

    def __str__(self) -> str:
        return "(" + str(self.prompt1) + " | " + str(self.prompt2) + ")"


def generate_list_prompts(
    prompt_func: Callable[[List[str]], Prompt], elements: List[str], max_size: int
) -> List[Prompt]:
    """
    Generates a list of prompts using a given prompt function, a list of elements, and a maximum size.
    Split up the list into smaller lists until the prompt fits into the maximum size.

    Args:
        prompt_func (Callable[[List[str]], Prompt]): The prompt function used to create prompts.
        elements (List[str]): The list of elements to be used as input to the prompt function.
        max_size (int): The maximum size allowed for each prompt.

    Returns:
        List[Prompt]: The list of generated prompts.
    """
    prompts = []
    prompt = prompt_func(elements)
    if prompt.fit(max_size) is not None:
        return [prompt]
    else:
        middle = len(elements) // 2
        left_elements = elements[:middle]
        right_elements = elements[middle:]
        left_prompts = generate_list_prompts(prompt_func, left_elements, max_size)
        right_prompts = generate_list_prompts(prompt_func, right_elements, max_size)
        prompts.extend(left_prompts)
        prompts.extend(right_prompts)
        return prompts


# every message follows <im_start>{role/name}\n{content}<im_end>\n
# see https://platform.openai.com/docs/guides/gpt/managing-tokens
EXTRA_TOKENS_PER_MESSAGE = 6


@dataclass
class PromptMessage:
    role: MessageRole
    prompt: Prompt

    @property
    def min_size(self) -> int:
        return self.prompt.min_size + EXTRA_TOKENS_PER_MESSAGE

    @property
    def size(self) -> int:
        return self.prompt.size + EXTRA_TOKENS_PER_MESSAGE


# Class PromptMessages represents a collection of PromptMessage objects and provides a method to fit them into a given maximum size.
class PromptMessages:
    def __init__(self, messages: List[PromptMessage] = []) -> None:
        self.messages = messages

    def add_prompt_message(self, role: MessageRole, prompt: Prompt) -> None:
        new_message = PromptMessage(role, prompt)
        self.messages.append(new_message)

    def fit(self, max_size: int) -> Optional[List[Message]]:
        min_size = sum(message.min_size for message in self.messages)
        fitted_messages: List[Message] = []
        for message in self.messages:
            message_max_size = message.min_size
            if message_max_size > max_size:
                return fitted_messages
            min_size_rest = min_size - message.min_size
            message_max_size = max(message_max_size, max_size - min_size_rest)
            fitted_prompt = message.prompt.fit(message_max_size - EXTRA_TOKENS_PER_MESSAGE)
            if fitted_prompt is None:
                return fitted_messages
            fitted_string, fitted_size = fitted_prompt
            fitted_messages.append(Message.mk(message.role, fitted_string))
            max_size -= fitted_size + EXTRA_TOKENS_PER_MESSAGE
            min_size = min_size_rest
        return fitted_messages

    def __str__(self) -> str:
        return "\n".join(str(message) for message in self.messages)


from unittest import TestCase


class Tests(TestCase):
    def test_string_prompt(self):
        prompt = StringPrompt("Hello, World!")
        self.assertEqual(prompt.fit(20), ("Hello, World!", 4))
        self.assertEqual(prompt.fit(3), None)
        self.assertEqual(prompt.fit(0), None)
        self.assertEqual(prompt.min_size, 4)
        self.assertEqual(str(prompt), "Hello, World!")

    def test_split_string_prompt(self):
        prompt = SplitStringPrompt(
            lhs="Text Before The", rhs="This is after.", separator="<cursor>"
        )
        self.assertEqual(prompt.fit(2), None)
        self.assertEqual(prompt.min_size, 3)
        self.assertEqual(prompt.fit(3), ("<cursor>", 3))
        self.assertEqual(prompt.fit(4), ("<cursor>This", 4))
        self.assertEqual(prompt.fit(5), (" The<cursor>This", 5))
        self.assertEqual(prompt.fit(6), (" The<cursor>This is", 6))
        self.assertEqual(prompt.fit(7), (" Before The<cursor>This is", 7))
        self.assertEqual(prompt.fit(10), ("Text Before The<cursor>This is after.", 10))
        self.assertEqual(prompt.fit(11), ("Text Before The<cursor>This is after.", 10))
        self.assertEqual(str(prompt), "Text Before The<cursor>This is after.")

    def test_concat_prompt(self):
        prompt1 = StringPrompt("Hello")
        prompt2 = SplitStringPrompt(lhs="", rhs=", World!", separator="", min_size=0)
        concat_prompt = prompt1 + prompt2
        self.assertEqual(concat_prompt.fit(1), ("Hello", 1))
        self.assertEqual(concat_prompt.fit(2), ("Hello,", 2))
        self.assertEqual(concat_prompt.fit(3), ("Hello, World", 3))
        self.assertEqual(concat_prompt.fit(4), ("Hello, World!", 4))
        self.assertEqual(concat_prompt.min_size, 1)
        self.assertEqual(concat_prompt.size, 4)

        prompt2 = SplitStringPrompt(lhs="", rhs=", World!", separator="", min_size=1)
        concat_prompt = prompt1 + prompt2
        self.assertEqual(concat_prompt.fit(1), None)
        self.assertEqual(concat_prompt.fit(2), ("Hello,", 2))
        self.assertEqual(concat_prompt.min_size, 2)
        self.assertEqual(concat_prompt.size, 4)

    def test_concat_prompt2(self):
        prompt1 = StringPrompt("Make some comments on the following program:\n")
        prompt2 = SplitStringPrompt(
            lhs="def f1(): return 1\ndef f2(): return 2\ndef f3(): return 3\n",
            rhs="def f4(): return 4\ndef f5(): return 5\ndef f6(): return 6\n",
            separator="",
        )
        prompt = prompt1 + prompt2
        self.assertNotEqual(prompt.fit(prompt.min_size), None)
        self.assertEqual(prompt.fit(prompt.min_size - 1), None)
        self.assertNotEqual(prompt.fit(prompt.size - 1), prompt.fit(prompt.size))
        self.assertEqual(prompt.fit(prompt.size + 1), prompt.fit(prompt.size))
        fit = prompt.fit(prompt.min_size + 16)
        self.assertEqual(
            fit,
            (
                "Make some comments on the following program:\ndef f3(): return 3\ndef f4(): return 4\n",
                24,
            ),
        )

    # Tests EitherPrompt's behavior of choosing the longest fitting prompt, falling back to the shorter one if necessary.
    def test_either_prompt(self):
        prompt1 = StringPrompt("This is a much longer prompt that exceeds the maximum size.")
        prompt2 = StringPrompt("Short prompt.")
        prompt = prompt1 | prompt2
        assert prompt.min_size == prompt2.min_size
        assert prompt.size == prompt.size
        assert prompt.fit(prompt2.size) == prompt2.fit(prompt2.size)
        assert prompt.fit(prompt1.size) == prompt1.fit(prompt1.size)
        assert prompt.min_size == prompt2.min_size
        assert prompt.size == prompt1.size

    def test_generate_list_prompts(self):
        elements = ["Element 1", "Element 2", "Element 3", "Element 4", "Element 5"]

        def list_prompt_func(elements):
            separator = ", "
            prompt_string = separator.join(elements)
            return StringPrompt(prompt_string)

        max_size = list_prompt_func(elements).size // 2
        prompts = generate_list_prompts(list_prompt_func, elements, max_size)

        assert len(prompts) == 3  # The list should be split into 3 prompts
        assert (v := prompts[0].fit(max_size)) and v[0] == "Element 1, Element 2"
        assert (v := prompts[1].fit(max_size)) and v[0] == "Element 3"
        assert (v := prompts[2].fit(max_size)) and v[0] == "Element 4, Element 5"

    def test_prompt_messages(self):
        prompt1 = StringPrompt("Hello")
        prompt2 = SplitStringPrompt(lhs="", rhs=", World!", separator="")
        prompt_message1 = PromptMessage("system", prompt1)
        prompt_message2 = PromptMessage("user", prompt2)

        prompt_messages = PromptMessages([prompt_message1])
        prompt_messages.add_prompt_message("user", prompt2)

        self.assertEqual(len(prompt_messages.messages), 2)
        self.assertEqual(prompt_messages.messages[0], prompt_message1)
        self.assertEqual(prompt_messages.messages[1], prompt_message2)

        self.assertEqual(prompt_messages.fit(prompt_message1.size), [Message.mk("system", "Hello")])
        self.assertEqual(prompt_messages.fit(prompt_message1.size - 1), [])
        fit0 = prompt_messages.fit(prompt_message1.size + prompt_message2.min_size)
        self.assertEqual(fit0, [Message.mk("system", "Hello"), Message.mk("user", "")])
        fit1 = prompt_messages.fit(prompt_message1.size + prompt_message2.min_size + 1)
        self.assertEqual(fit1, [Message.mk("system", "Hello"), Message.mk("user", ",")])
        fit2 = prompt_messages.fit(prompt_message1.size + prompt_message2.size)
        self.assertEqual(fit2, [Message.mk("system", "Hello"), Message.mk("user", ", World!")])
        fit3 = prompt_messages.fit(prompt_message1.size + prompt_message2.size - 1)
        self.assertNotEqual(fit3, fit2)
        fit4 = prompt_messages.fit(prompt_message1.size + prompt_message2.size + 1)
        self.assertEqual(fit4, fit2)

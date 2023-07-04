from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from rift.llm.openai_types import (
    MessageRole,
    Message,
)
from tiktoken import get_encoding
from typing import (
    List,
    Optional,
    Tuple,
)

ENCODER = get_encoding("cl100k_base")

def token_length(string):
    return len(ENCODER.encode(string))

class Prompt(ABC):
    def __init__(self, size) -> None:
        self.size = size

    @abstractmethod
    def fit(self, max_size) -> Optional[Tuple[str, int]]:
        raise NotImplementedError

    @abstractproperty
    def min_size(self) -> int:
        raise NotImplementedError

    def __add__(self, other) -> "ConcatPrompt":
        return ConcatPrompt(self, other)

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
            tokens1 = ENCODER.encode(self.string1)
            tokens2 = ENCODER.encode(self.string2)
            size1 = remaining_size // 2
            size1 = max(size1, remaining_size - len(tokens2))
            # cut tokens1 to the rightmost size1 tokens
            tokens1 = tokens1[-size1:] if size1 > 0 else []
            size2 = remaining_size - len(tokens1)
            # cut tokens2 to the leftmost size2 tokens
            tokens2 = tokens2[:size2] if size2 > 0 else []
            combined_string = (
                ENCODER.decode(tokens1) 
                + self.separator 
                + ENCODER.decode(tokens2)
            )
            return combined_string, len(tokens1) + separator_size + len(tokens2)
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

# Class PromptMessages represents a collection of PromptMessage objects and provides a method to fit them into a given maximum size.
class PromptMessages:
    def __init__(self, messages: List[PromptMessage] = []) -> None:
        self.messages = messages

    def add_prompt_message(self, role: MessageRole, prompt: Prompt) -> None:
        new_message = PromptMessage(role, prompt)
        self.messages.append(new_message)

    def fit(self, max_size: int) -> Optional[List[Message]]:
        min_size = sum(message.min_size  for message in self.messages)
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



def test_string_prompt():
    prompt = StringPrompt("Hello, World!")
    assert(prompt.fit(20) == ("Hello, World!", 4))
    assert(prompt.fit(3) == None)
    assert(prompt.fit(0) == None)
    assert(prompt.min_size == 4)
    assert(str(prompt) == "Hello, World!")

def test_split_string_prompt():
    prompt = SplitStringPrompt(lhs="Text Before The", rhs="This is after.", separator="<cursor>")
    assert prompt.fit(2) is None
    assert prompt.min_size == 3
    assert prompt.fit(3) == ("<cursor>", 3)
    assert prompt.fit(4) == ('<cursor>This', 4)
    assert prompt.fit(5) == (' The<cursor>This', 5)
    assert prompt.fit(6) == (' The<cursor>This is', 6)
    assert prompt.fit(7) == (' Before The<cursor>This is', 7)
    assert prompt.fit(10) == ('Text Before The<cursor>This is after.', 10)
    assert prompt.fit(11) == ('Text Before The<cursor>This is after.', 10)
    assert str(prompt) == "Text Before The<cursor>This is after."

def test_concat_prompt():
    prompt1 = StringPrompt("Hello")
    prompt2 = SplitStringPrompt(lhs="", rhs=", World!", separator="", min_size=0)
    concat_prompt = prompt1 + prompt2
    assert concat_prompt.fit(1) == ('Hello', 1)
    assert concat_prompt.fit(2) == ('Hello,', 2)
    assert concat_prompt.fit(3) == ('Hello, World', 3)
    assert concat_prompt.fit(4) == ('Hello, World!', 4)
    assert concat_prompt.min_size == 1
    assert concat_prompt.size == 4

    prompt2 = SplitStringPrompt(lhs="", rhs=", World!", separator="", min_size=1)
    concat_prompt = prompt1 + prompt2
    assert concat_prompt.fit(1) == None
    assert concat_prompt.fit(2) == ('Hello,', 2)
    assert concat_prompt.min_size == 2
    assert concat_prompt.size == 4


def test_concat_prompt2():
    prompt1 = StringPrompt("Make some comments on the following program:\n")
    prompt2 = SplitStringPrompt(lhs="def f1(): return 1\ndef f2(): return 2\ndef f3(): return 3\n", rhs="def f4(): return 4\ndef f5(): return 5\ndef f6(): return 6\n", separator="")
    prompt = prompt1 + prompt2
    assert prompt.fit(prompt.min_size) != None
    assert prompt.fit(prompt.min_size - 1) == None
    assert prompt.fit(prompt.size - 1) != prompt.fit(prompt.size)
    assert prompt.fit(prompt.size + 1) == prompt.fit(prompt.size)
    fit = prompt.fit(prompt.min_size + 16)
    assert fit == ('Make some comments on the following program:\ndef f3(): return 3\ndef f4(): return 4\n', 24)


def test_prompt_messages():
    prompt1 = StringPrompt("Hello")
    prompt2 = SplitStringPrompt(lhs="", rhs=", World!", separator="")
    prompt_message1 = PromptMessage("system", prompt1)
    prompt_message2 = PromptMessage("user", prompt2)

    prompt_messages = PromptMessages([prompt_message1])
    prompt_messages.add_prompt_message("user", prompt2)

    assert len(prompt_messages.messages) == 2
    assert prompt_messages.messages[0] == prompt_message1
    assert prompt_messages.messages[1] == prompt_message2

    assert prompt_messages.fit(prompt_message1.min_size) == [Message.mk("system", "Hello")]
    assert prompt_messages.fit(prompt_message1.min_size - 1) == []
    fit0 = prompt_messages.fit(prompt_message1.min_size + prompt_message2.min_size)
    assert fit0 == [Message.mk("system", "Hello"), Message.mk("user", "")]
    fit1 = prompt_messages.fit(prompt_message1.min_size + prompt_message2.min_size + 1)
    assert fit1 == [Message.mk("system", "Hello"), Message.mk("user", ",")]
    fit2 = prompt_messages.fit(prompt_message1.min_size + prompt2.size + EXTRA_TOKENS_PER_MESSAGE)
    assert fit2 == [Message.mk("system", "Hello"), Message.mk("user", ", World!")]
    fit3 = prompt_messages.fit(prompt_message1.min_size + prompt2.size + EXTRA_TOKENS_PER_MESSAGE - 1)
    assert fit3 != fit2
    fit4 = prompt_messages.fit(prompt_message1.min_size + prompt2.size + EXTRA_TOKENS_PER_MESSAGE + 1)
    assert fit4 == fit2

if __name__ == "__main__":
    test_string_prompt()
    test_split_string_prompt()
    test_concat_prompt()
    test_concat_prompt2()
    test_prompt_messages()


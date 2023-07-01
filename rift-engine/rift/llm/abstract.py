from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List
from rift.util.TextStream import TextStream
from rift.llm.openai_types import Message


@dataclass
class InsertCodeResult:
    code: TextStream
    thoughts: Optional[TextStream] = field(default=None)


@dataclass
class ChatResult:
    text: TextStream

class AbstractCodeCompletionProvider(ABC):
    @abstractmethod
    async def insert_code(
        self, document: str, cursor_offset: int, goal: Optional[str] = None
    ) -> InsertCodeResult:
        """Perform code completion on the given document at the given cursor offset.

        Args:
            - document: The document to perform code completion on.
            - cursor_offset: The offset of the cursor in the document, the output should be code that can be inserted at this point.
            - goal: A natural language statement with the goal of the code completion. If None, then we should just perform a code completion.
        """
        raise NotImplementedError()

    async def load(self):
        """Do any side activities needed to load the model."""
        pass


class AbstractChatCompletionProvider(ABC):
    @abstractmethod
    async def run_chat(
        self, document: str, messages: List[Message], message: str, cursor_offset: Optional[int] = None
    ) -> ChatResult:
        """
        Process the chat messages and return the completion results.

        Parameters:
        -----------
        document: str
            The document context where the chat is taking place.
        messages: list[Message]
            A list of messages exchanged in the chat.
        message: str
            The latest message exchanged in the chat.
        cursor_offset: Optional[int]
            The offset of the cursor in the document.

        Returns:
        --------
        ChatResult
            The completion results of the chat.

        Raises:
        -------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError()

    async def load(self):
        """Do any side activities needed to load the model."""
        pass

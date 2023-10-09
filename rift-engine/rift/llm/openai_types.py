import textwrap
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

""" Type definitions for interacting with the OpenAI API """

MessageRole = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "content_filter"]


class Message(BaseModel):
    """OpenAI chat message.

    ref: https://platform.openai.com/docs/guides/chat
    """

    role: MessageRole
    content: str
    name: Optional[str] = None
    """System messages can come with a 'name' parameter. """

    @classmethod
    def mk(cls, role: str, content: str):
        if role in ["system", "user", "assistant"]:
            return cls(role=role, content=content, name=None)  # type: ignore
        else:
            return cls(role="system", content=content, name=role)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)

    def __str__(self):
        content = textwrap.indent(self.content, "  ")
        return f"{self.name or self.role}:\n{content}"


class MessageDelta(BaseModel):
    role: Optional[MessageRole] = None
    content: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionChoiceDelta(BaseModel):
    delta: MessageDelta
    index: int
    finish_reason: Optional[FinishReason] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str
    created: datetime
    model: str
    choices: list[ChatCompletionChoiceDelta]

    @property
    def text(self):
        assert len(self.choices) == 1
        return self.choices[0].delta.content or ""


class ChatCompletionRequest(BaseModel):
    """Request body for OpenAI completion API.

    ref: https://platform.openai.com/docs/api-reference/chat/create
    """

    model: str = Field(default="gpt-3.5-turbo")
    """ ID of the model to use.
    See [model compatibility page](https://platform.openai.com/docs/models/model-endpoint-compatibility).
    As of 2022-04-06 these are
    OpenAI: gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
    """
    messages: list[Message]
    """ A list of messages describing the conversation so far. """

    max_tokens: Optional[int] = Field(default=None)
    """ The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length. """
    stream: Optional[bool] = Field(default=None)
    """
    If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a data: [DONE] message.

    Capabilities note: don't set this manually, instead use the stream=True argument to OpenAIClient requests. We will automatically wrap in a generator etc.
    """
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    """ What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both. """
    top_p: Optional[int] = Field(default=None)
    """ An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both. """
    n: Optional[int] = Field(default=None)
    """How many chat completion choices to generate for each input message."""
    stop: Optional[Union[str, list[str]]] = Field(default=None)
    """ Up to 4 sequences where the API will stop generating further tokens. """
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    """ Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. """
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    """ Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. """
    logit_bias: Optional[dict] = Field(default=None)
    """ Modify the likelihood of specified tokens appearing in the completion. Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token. """
    user: Optional[str] = Field(default=None)
    """A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids)."""


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: Optional[int] = Field(default=None)
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[FinishReason] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: datetime
    choices: list[Choice]
    usage: Optional[Usage] = None  # [todo] not optional?


class Logprob(BaseModel):
    index: int
    token: str
    logprobs: float
    top_logprobs: dict
    text_offset: int


class TextCompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = Field(default=None)
    """ The maximum number of tokens to generate in the completion. The token count of your prompt plus max_tokens cannot exceed the model's context length. Most models have a context length of 2048 tokens (except for the newest models, which support 4096). """
    temperature: Optional[float] = Field(default=None)
    stream: Optional[bool] = Field(default=None)
    logprobs: Optional[int] = Field(default=None)
    """ Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response. """
    echo: Optional[bool] = Field(default=None)
    """ Echo back the prompt in addition to the completion """
    suffix: Optional[str] = Field(default=None)
    """ The suffix that comes after a completion of inserted text. """

    n: Optional[int] = Field(default=None)
    """ How many completions to generate for each prompt. """


class TextChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = Field(default=None)
    finish_reason: Optional[str] = Field(default=None)


class TextCompletionResponse(BaseModel):
    id: str
    object: str
    created: datetime
    model: str
    choices: list[TextChoice]
    usage: Optional[Usage] = None


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: str = Field(default="text-embedding-ada-002")
    user: Optional[str] = Field(default=None)


class EmbeddingObject(BaseModel):
    object: Literal["embedding"]
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: Literal["list"]
    data: list[EmbeddingObject]
    model: str = Field(default="text-embedding-ada-002")
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    owned_by: str
    permission: Any = None


class ModelList(BaseModel):
    data: list[ModelInfo]

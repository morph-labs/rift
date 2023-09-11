import functools
import logging
import os
import weakref
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, SecretStr

from rift.llm.abstract import AbstractChatCompletionProvider, AbstractCodeCompletionProvider


class ModelConfig(BaseModel):
    chatModel: str
    codeEditModel: str
    openaiKey: Optional[SecretStr] = None

    def __hash__(self):
        return hash((self.chatModel, self.codeEditModel))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def create_chat(self) -> AbstractChatCompletionProvider:
        c = create_client(self.chatModel, self.openaiKey)
        assert isinstance(c, AbstractChatCompletionProvider)
        return c

    def create_completions(self) -> AbstractCodeCompletionProvider:
        return create_client(self.codeEditModel, self.openaiKey)

    @classmethod
    def default(cls):
        return ModelConfig(
            codeEditModel="openai:gpt-4",
            chatModel="openai:gpt-3.5-turbo",
        )


CLIENTS = weakref.WeakValueDictionary()


def create_client(
    config: str, openai_api_key: Optional[SecretStr] = None
) -> AbstractCodeCompletionProvider:
    """Create a client for the given config. If the client has already been created, then it will return a cached one.

    Note that it uses a WeakValueDictionary, so if the client is no longer referenced, it will be garbage collected.
    This is useful because it means you can call create_client multiple times without allocating the same model, but
    if you need to dispose a model this won't keep a reference that prevents it from being garbage collected.
    """
    global CLIENTS

    if config in CLIENTS:
        return CLIENTS[config]
    else:
        client = create_client_core(config, openai_api_key)
        CLIENTS[config] = client
        return client


def parse_type_name_path(config: str) -> Tuple[str, str, str]:
    assert ":" in config, f"Invalid config: {config}"
    type, rest = config.split(":", 1)
    type = type.strip()
    if "@" in rest:
        name, path = rest.split("@", 1)
    else:
        name = rest
        path = ""
    name = name.strip()
    path = path.strip()
    return (type, name, path)


def create_client_core(
    config: str, openai_api_key: Optional[SecretStr]
) -> AbstractCodeCompletionProvider:
    """
    The function parses the `config` string to extract the `type` and the rest of the configuration. It then checks the `type` and based on that, returns different instances of code completion providers.

    For example, if the `type` is `"hf"`, it imports and returns an instance of `HuggingFaceClient` from `rift.llm.hf_client`. If the `type` is `"openai"`, it imports and returns an instance of `OpenAIClient` from `rift.llm.openai_client` with some additional keyword arguments. If the `type` is `"gpt4all"`, it imports and returns an instance of `Gpt4AllModel` from `rift.llm.gpt4all_model` with some additional settings and keyword arguments.

    If the `type` is none of the above, it raises a `ValueError` with a message indicating that the model is unknown.
    """
    type, name, path = parse_type_name_path(config)
    if type == "hf":
        from rift.llm.hf_client import HuggingFaceClient

        return HuggingFaceClient(name)
    elif type == "openai":
        from rift.llm.openai_client import OpenAIClient

        kwargs = {}
        if name:
            kwargs["default_model"] = name
        if openai_api_key:
            kwargs["api_key"] = openai_api_key
        else:
            if not os.environ.get("OPENAI_API_KEY"):
                logging.getLogger().error(
                    "Trying to create an OpenAIClient without an OpenAI key set in Rift settings or set as the OPENAI_API_KEY environment variable."
                )
        if path:
            kwargs["api_url"] = path
        return OpenAIClient.parse_obj(kwargs)

    elif type == "gpt4all":
        from rift.llm.gpt4all_model import Gpt4AllModel, Gpt4AllSettings

        kwargs = {}
        if name:
            kwargs["model_name"] = name
        if path:
            kwargs["model_path"] = path
        settings = Gpt4AllSettings.parse_obj(kwargs)
        return Gpt4AllModel(settings)

    else:
        raise ValueError(f"Unknown model: {config}")

"""
Reference implementation of a minimal agent.

This module provides a minimal implementation of the Agent API defined in rift.agents.abstract.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import aiohttp

import rift.agents.registry as registry
import rift.llm.openai_types as openai
from rift.agents.abstract import Agent, AgentParams, AgentState, RequestChatRequest
from rift.lsp.types import TextDocumentIdentifier


@dataclass
class CurlAgentParams(AgentParams):
    textDocument: TextDocumentIdentifier
    instructionPrompt: Optional[str] = None


@dataclass
class CurlAgentState(AgentState):
    params: CurlAgentParams
    messages: list[openai.Message]


@dataclass
class CurlAgent(Agent):
    """
    CurlAgent is a minimal implementation of the Agent API.
    It asks the user for a URL and prints the output of CURLing that url.
    """

    state: Optional[CurlAgentState] = None
    agent_type: str = "curl_agent"
    params_cls: ClassVar[Any] = CurlAgentParams

    async def run(self):
        # Send an initial update
        await self.send_update("Please enter a URL")

        # Enter a loop to continuously interact with the user
        while True:
            # Request a URL from the user
            user_response_t = self.add_task(
                "get user response", self.request_chat, [RequestChatRequest(self.state.messages)]
            )

            # Send a progress update
            await self.send_progress()

            # Wait for the user's response
            user_response = await user_response_t.run()

            # Append the user's response to the state's messages
            self.state.messages.append(openai.Message.user(user_response))

            # Make a GET request to the user's URL and append the response to the state's messages
            async with aiohttp.ClientSession() as session:
                async with session.get(user_response) as response:
                    response_text = await response.text()
                    self.state.messages.append(openai.Message.assistant(response_text))

    @classmethod
    async def create(cls, params: CurlAgentParams, server):
        # Convert the parameters to a CurlAgentParams object

        # Create the initial state
        state = CurlAgentState(
            params=params,
            messages=[openai.Message.assistant("Please enter a URL")],
        )

        # Create the CurlAgent object
        obj = cls(
            state=state,
            agent_id=params.agent_id,
            server=server,
        )

        # Return the CurlAgent object
        return obj

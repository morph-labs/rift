from dataclasses import dataclass, field
from typing import Dict, Optional, ClassVar
import asyncio
import smol_dev
from rift.lsp import LspServer as BaseLspServer
from rift.server.selection import RangeSet

from rift.llm.openai_types import Message as ChatMessage
from rift.agents.abstract import (
    Agent,
    AgentTask,
    AgentState,
    AgentProgress,
    AgentRunParams,
    AgentRunResult,
    RequestInputResponse,
    RequestChatResponse,
    RequestChatRequest
)
import rift.lsp.types as lsp
from logging import getLogger
from rift.llm.abstract import AbstractChatCompletionProvider
from .file_diff import FileChange, get_file_change, edits_from_file_changes 
from typing import List

logger = getLogger(__name__)


@dataclass
class SmolAgentRunResult(AgentRunResult):
    ...
    
@dataclass
class SmolAgentProgress(AgentProgress):
    response: Optional[str] = None
    thoughts: Optional[str] = None

@dataclass
class SmolAgentParams:
    instructionPrompt: str
    textDocument: lsp.TextDocumentIdentifier
    position: lsp.Position

@dataclass
class SmolAgentState(AgentState):
    messages: List[ChatMessage]
    model: AbstractChatCompletionProvider
    document: lsp.TextDocumentItem
    cursor: lsp.Position
    params: SmolAgentParams
    ranges: RangeSet = field(default_factory=RangeSet)
    smol_dev: smol_dev = smol_dev  # lets you access smol_dev methods

@dataclass
class SmolAgent(Agent):
    agent_type: str = "smol_dev"
    state: SmolAgentState
    tasks: Dict[str, AgentTask]
    server: BaseLspServer
    count: ClassVar[int] = 0
    id: int

    @classmethod
    def create(cls, params: SmolAgentParams, model, server):
        cls.count += 1
        obj = cls(
            status="running",
            state=SmolAgentState(
                model=model,
                messages=[
                    ChatMessage.system("""
        You are an AI agent that generates code based on a prompt. 
        When you are given the prompt, ask 3 more questions about the most important implementation details that the user might want to modify or correct. 
        Then, generate code based on the prompt and the answers to the questions. """)
                ],
                params=params,
                document=server.documents[params.textDocument.uri],
                cursor=params.position,
                ranges=RangeSet(),
            ), tasks=dict(), server=server, id=SmolAgent.count
        )
        return obj
    
    def add_task(self, task: AgentTask):
        self.tasks[task.id] = task
        return task.id
    
    async def run(self) -> AgentRunResult:
        prompt_task = AgentTask("Getting Prompt", "running", [], None)
        self.add_task(prompt_task)
        self.state.messages.append(ChatMessage.assistant("""What do you want me to code?"""))
        user_response = await self.request_chat(RequestChatRequest(self.state.messages))
        # print('user_response', user_response) # right now we assume 'morph/smol_dev_1_request_chat' returns a dict with a key 'message'
        response = ""
        stream = await self.state.model.run_chat(
            "", self.state.messages, str(user_response["message"])
        )
        async for delta in stream.text:
            response += delta
        
        # # in future, loop 3 times.. right now commented out because test doesnt support
        # for i in range(3):
        #     response = ""
        #     user_response = await self.request_chat(RequestChatRequest(self.state.messages))
        #     stream = await self.state.model.run_chat(
        #          "", self.state.messages, str(user_response["message"])
        #     )
        #     async for delta in stream.text:
        #         response += delta
        #         from asyncio import Lock
        #         response_lock = Lock()
        #         async with response_lock:
        #             await self.send_progress(ChatProgress(response=response))
        # prompt_task.status = "done"
        
        # This is just an example. You should create a run function based on your needs.
        task_id = self.add_task(AgentTask("Generate code", "running", [], None))
        task = self.tasks[task_id]

        try:
            prompt = ''.join([message.content for message in self.state.messages])
            # planning
            plan_task = self.add_task(AgentTask("running", "Planning...", [], None))
            plan = self.state.smol_dev.plan(prompt)
            # temporarily commented out # plan_task.status = "done"
            await self.send_progress(SmolAgentProgress(tasks=self.tasks, thoughts=plan))
            
            # specify file paths
            filepath_task = self.add_task(AgentTask("running", "Determining Filepath...", [], None))
            file_paths = self.state.smol_dev.specify_filePaths(prompt, plan)
            # filepath_task.status = "done"
            await self.send_progress(SmolAgentProgress(tasks=self.tasks, thoughts=file_paths))
            
            self.add_task(AgentTask("Reticulating splines...", "done", [], None))
            
            # generate code
            generated_code : List[FileChange] = []
            import os
            for file_path in file_paths:
                codegen_task = self.add_task(AgentTask("running", "Codegen for: " + file_path, [], None))
                code = self.state.smol_dev.generate_code(file_path, self.state.params.instructionPrompt, plan)
                absolute_file_path = os.getcwd() + '/' + file_path
                uri = 'file://' + absolute_file_path
                file_change = get_file_change(path=absolute_file_path, new_content=code)
                
                generated_code.append(file_change)
                await self.send_progress(SmolAgentProgress(tasks=self.tasks, thoughts=code))
                # temporarily commented out # codegen_task.status = "done"
                self.send_result(code) # todo: check what send_result actually wants
            finalWorkspaceEdit = edits_from_file_changes(generated_code)
            x = await self.server.request("workspace/applyEdit", finalWorkspaceEdit)
            print("X: ", x)
            # temporarily commented out # task.status = "done"
            return SmolAgentRunResult()

        except Exception as e:
            task.status = "error"
            logger.error(f"{self} failed to run: {e}")
            return SmolAgentRunResult()

        finally:
            await self.send_progress(SmolAgentProgress(tasks=self.tasks, response=None))

    async def request_input(self, request_input_request):
        return await self.server.request(
            f"morph/{self.agent_type}_{self.id}_request_input", request_input_request
        )

    async def request_chat(self, request_chat_request):
        return await self.server.request(
            f"morph/{self.agent_type}_{self.id}_request_chat", request_chat_request
        )

    async def send_progress(self, progress):
        await self.server.notify(f"morph/{self.agent_type}_{self.id}_send_progress", progress)

    async def send_result(self, result):
        ...  # unreachable

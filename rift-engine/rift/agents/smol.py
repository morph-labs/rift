from dataclasses import dataclass
from typing import Dict, Optional
import asyncio
import smol_dev
from rift.lsp import LspServer as BaseLspServer, lsp
from rift.llm.openai_types import Message as ChatMessage
from rift.agents.abstract import (
    Agent,
    AgentTask,
    AgentState,
    AgentProgress,
    AgentRunParams,
    AgentRunResult,
)
from logging import getLogger
from file_diff import apply_diff_edits

logger = getLogger(__name__)


@dataclass
class ChatProgress(
    AgentProgress
):  # reports what tasks are active and responsible for reporting new tasks
    response: Optional[str] = None
    done_streaming: bool = False
    
@dataclass
class SmolAgentState(AgentState):
    smol_dev: smol_dev.SmolDeveloper = smol_dev  # lets you access smol_dev methods
    messages: List[ChatMessage]
    model: AbstractChatCompletionProvider

@dataclass
class SmolAgent(Agent):
    state: DevAgentState
    tasks: Dict[str, AgentTask]
    server: BaseLspServer
    count: ClassVar[int] = 0
    id: int
    active_task_id: Optional[str] = None

    @classmethod
    def create(cls, messages, server):
        SmolAgent.count += 1
        obj = SmolAgent(
            state=SmolAgentState(messages=[
                ChatMessage.system("""
    You are an AI agent that generates code based on a prompt. 
    When you are given the prompt, ask 3 more questions about the most important implementation details that the user might want to modify or correct. 
    Then, generate code based on the prompt and the answers to the questions. """)
            ]), tasks=dict(), server=server, id=SmolAgent.count
        )
        obj.active_task_id = obj.add_task(AgentTask("running", "Get user response", [], None))
        obj.wait_user_response_task = obj.tasks[active_task_id]
        obj.generate_response_task = obj.tasks[
            obj.add_task(AgentTask("done", "Generate response", [], None))
        ]
        return obj
    
    async def run(self, params: AgentRunParams) -> AgentRunResult:
        prompt_task = self.add_task(AgentTask("running", "Getting Prompt", [], None))
        self.state.messages.append(ChatMessage.assistant("""What do you want me to code?"""))
        user_response = await self.request_chat(self.state.messages)

        response = ""
        stream = await self.model.run_chat(
            self.document.text, self.state.messages, user_response, offset
        )
        async for delta in stream.text:
            response += delta
        
        # loop 3 times
        for i in range(3):
            response = ""
            user_response = await self.request_chat(self.state.messages)
            stream = await self.model.run_chat(
                self.document.text, self.state.messages, user_response, None
            )
            async for delta in stream.text:
                response += delta
                from asyncio import Lock
                response_lock = Lock()
                async with response_lock:
                    await self.send_progress(ChatProgress(response=response))
        prompt_task.status = "done"
        
        # This is just an example. You should create a run function based on your needs.
        task_id = self.add_task(AgentTask("running", "Generate code", [], None))
        self.active_task_id = task_id
        task = self.tasks[task_id]

        try:
            prompt = ''.join([m.get('content', '') for m in self.state.messages])
            # planning
            plan_task = self.add_task(AgentTask("running", "Planning...", [], None))
            plan = self.state.smol_dev.plan(prompt)
            plan_task.status = "done"

            # specify file paths
            filepath_task = self.add_task(AgentTask("running", "Determining Filepath...", [], None))
            file_paths = self.state.smol_dev.specify_filePaths(prompt, plan)
            filepath_task.status = "done"
            
            self.add_task(AgentTask("done", "Reticulating splines...", [], None))
            
            # generate code
            generated_code = dict()
            for file_path in file_paths:
                codegen_task = self.add_task(AgentTask("running", "Codegen for: " + file_path, [], None))
                code = self.state.smol_dev.generate_code(file_path, params.prompt, plan)
                generated_code[file_path] = apply_diff_edits(
                    TextDocumentIdentifier(uri='file://' + file_path, version=None),
                    "", # todo - read in existing file content
                    code
                )
                codegen_task.status = "done"
                send_result(generated_code[file_path]) # todo: check what send_result actually wants
            task.status = "done"
            return AgentRunResult(success=True, result=generated_code)

        except Exception as e:
            task.status = "error"
            logger.error(f"{self} failed to run: {e}")
            return AgentRunResult(success=False, error=str(e))

    async def request_input(self) -> RequestInputResponse:
        response_fut = await self.server.request(
            f"morph/{self.agent_type}_{self.id}_request_input", request_input_request
        )
        return await response_fut

    async def request_chat(
        self, request_chat_request: RequestChatRequest
    ) -> RequestChatResponse:
        response_fut = await self.server.request(
            f"morph/{self.agent_type}_{self.id}_request_chat", request_chat_request
        )
        return await response_fut

    async def send_progress(self, progress: AgentProgress) -> None:
        await self.notify("morph/{self.agent_type}_{self.id}_send_progress", progress)

    async def send_result(self):
        # Implementation depends on the specifics of your project.
        pass

    def __str__(self):
        return f"<{type(self).__name__}> {self.id}"

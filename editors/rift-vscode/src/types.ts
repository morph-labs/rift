import type * as vscode from "vscode";
import type { TextDocumentIdentifier } from "vscode-languageclient/node";


export interface Task {
  description: string;
  status: AgentStatus;
}

export interface TaskWithSubtasks {
  task: Task;
  subtasks: Task[];
}

export type InputRequest = {
  msg: string;
  place_holder: string;
};

export type ChatMessageType = {
  role: "user" | "assistant";
  content: string;
};

export type AgentRegistryItem = {
  agent_type: string;
  agent_description: string;
  display_name: string;
  agent_icon: string | null;
};

export class WebviewAgent {
  type: string;
  hasNotification: boolean;
  isDeleted: boolean = false;
  chatHistory: ChatMessage[];
  inputRequest?: InputRequest | null;
  taskWithSubtasks?: TaskWithSubtasks;
  isStreaming: boolean = false;
  streamingText: string = "";
  doesShowAcceptRejectBar: boolean = false;

  constructor(
    type: string,
    hasNotification?: boolean,
    chatHistory?: ChatMessage[],
    inputRequest?: InputRequest | null,
    tasks?: TaskWithSubtasks
  ) {
    this.type = type;
    this.hasNotification = hasNotification ?? false;
    this.chatHistory = chatHistory ?? [];
    this.inputRequest = inputRequest;
    this.taskWithSubtasks = tasks;
  }
}

export type WebviewState = {
  selectedAgentId: string;
  isOmnibarFocused: boolean;
  agents: {
    [id: string]: WebviewAgent;
  };
  availableAgents: AgentRegistryItem[];
  files: {
    recentlyOpenedFiles: AtableFile[];
    nonGitIgnoredFiles: AtableFile[];
  };
  symbols: AtableFile[];
};

export const DEFAULT_STATE: WebviewState = {
  selectedAgentId: "",
  isOmnibarFocused: false,
  agents: {},
  availableAgents: [
    {
      agent_type: "rift_chat",
      agent_description: "",
      agent_icon: "",
      display_name: "Rift Chat",
    },
  ],
  files: {
    recentlyOpenedFiles: [],
    nonGitIgnoredFiles: [],
  },
  symbols: [],
};

export type OptionalTextDocument = {
  uri: string;
  version: number;
} | null;

export type EditorMetadata = { selection: vscode.Selection | null, position: vscode.Position | null, textDocument: OptionalTextDocument };

export interface AgentParams {
  agent_type: string;
  agent_id: string | null;
  position: vscode.Position | null;
  selection: vscode.Selection | null;
  textDocument: OptionalTextDocument;
  workspaceFolderPath: string | null;
  visibleEditorMetadata: EditorMetadata[];
}

export interface RunChatParams {
  message: string;
  messages: {
    role: string;
    content: string;
  }[];
}

export interface RunAgentResult {
  id: string;
}

export type AgentStatus = "running" | "done" | "error";

export type CodeLensStatus =
  | "running"
  | "ready"
  | "accepted"
  | "rejected"
  | "error"
  | "done";

export interface RunAgentProgress {
  id: number;
  textDocument: TextDocumentIdentifier;
  log?: {
    severity: string;
    message: string;
  };
  cursor?: vscode.Position;
  /** This is the set of ranges that the agent has added so far. */
  ranges?: vscode.Range[];
  status: AgentStatus;
}

export type ChatAgentPayload =
  | {
      response?: string;
      done_streaming?: boolean;
    }
  | undefined;

export type CodeEditPayload = any;
export type AnyPayload = ChatAgentPayload | CodeEditPayload | any;
export interface AgentProgress<T = AnyPayload> {
  agent_id: string;
  agent_type: string;
  tasks?: TaskWithSubtasks;
  payload: T | undefined;
}

export type ChatAgentProgress = AgentProgress<ChatAgentPayload>;

export interface AgentIdParams {
  id: string;
}

export type ChatMessage =
  | {
      role: "assistant";
      content: string;
      name?: null | string | undefined;
    }
  | ChatMessageUser;

export type ChatMessageUser = {
  role: "user";
  content: string;
  name?: null | string | undefined;
};

export interface AgentChatRequest {
  messages: ChatMessage[];
}

export interface AgentInputRequest {
  msg: string;
  place_holder: string;
}

export interface AgentInputResponse {
  response: string;
}

export interface AgentUpdate {
  msg: string;
}
export type AgentResult = {
  id: string;
  type: string;
}; //is just an ID rn
export type AgentSymbols = {
  path: string;
  symbols: {
    name: string;
    scope: string;
    kind: string;
    range: [[number, number], [number, number]];
  }[];
}[];

export interface AtableFile {
  fileName: string; //example.ts
  fullPath: string; //Users/brent/dev/project/src/example.ts
  fromWorkspacePath: string; //project/src/example.ts
  symbolName?: string; // MainNamespace.SomeClass.someMethod
}

export interface AtableFileWithCommand extends AtableFile {
  onEnter: (...args: any) => void;
}

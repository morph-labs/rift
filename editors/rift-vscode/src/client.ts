import { join } from "path";
import type { ExtensionContext, TextEditor } from "vscode";
import * as vscode from "vscode";
import {
    Executable,
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    State,
    StreamInfo,
    TextDocumentIdentifier,
    TextDocumentPositionParams,
    TransportKind,
} from "vscode-languageclient/node";
import * as net from "net";
import * as tcpPortUsed from "tcp-port-used";
import { chatProvider, logProvider } from "./extension";
import PubSub from "./lib/PubSub";
import {
    AgentChatRequest,
    AgentIdParams,
    AgentInputRequest,
    AgentParams,
    AgentProgress,
    AgentRegistryItem,
    AgentResult,
    AgentStatus,
    AgentUpdate,
    ChatAgentProgress,
    ChatMessage,
    CodeEditPayload,
    CodeLensStatus,
    DEFAULT_STATE,
    OptionalTextDocument,
    RunAgentResult,
    WebviewAgent,
    WebviewState,
} from "./types";
import { Store } from "./lib/Store";
import {
    AtableFileFromFsPath,
    AtableFileFromUri,
} from "./util/AtableFileFunction";

let client: LanguageClient; //LanguageClient

const DEFAULT_PORT = 7797;

// ref: https://stackoverflow.com/questions/40284523/connect-external-language-server-to-vscode-extension

// https://nodejs.org/api/child_process.html#child_processspawncommand-args-options

/** Creates the ServerOptions for a system in the case that a language server is already running on the given port. */
function tcpServerOptions(
    context: ExtensionContext,
    port = DEFAULT_PORT
): ServerOptions {
    let socket = net.connect({
        port: port,
        host: "127.0.0.1",
    });
    const si: StreamInfo = {
        reader: socket,
        writer: socket,
    };
    return () => {
        return Promise.resolve(si);
    };
}

/** Creates the server options for spinning up our own server.*/
function createServerOptions(
    context: vscode.ExtensionContext,
    port = DEFAULT_PORT
): ServerOptions {
    if (!vscode.workspace.workspaceFolders)
        throw new Error("workspace folder does not exist");
    let cwd = vscode.workspace.workspaceFolders[0].uri.path;
    // [todo]: we will supply different bundles for the 3 main platforms; windows, linux, osx.
    // there needs to be a decision point here where we decide which platform we are on and
    // then choose the appropriate bundle.
    let command = join(context.extensionPath, "resources", "rift");
    let args: string[] = [];
    args = [...args, "--port", port.toString()];
    let e: Executable = {
        command,
        args,
        transport: { kind: TransportKind.socket, port },
        options: { cwd },
    };
    return {
        run: e,
        debug: e,
    };
}

const GREEN = vscode.window.createTextEditorDecorationType({
    backgroundColor: "rgba(0,255,0,0.1)",
});

const RED = vscode.window.createTextEditorDecorationType({
    backgroundColor: "rgba(255,0,0,0.1)",
});

type CodeCompletionPayload =
    | {
        additive_ranges?: vscode.Range[];
        cursor?: vscode.Position;
        negative_ranges?: vscode.Range[];
        response?: string;
        textDocument?: TextDocumentIdentifier;
    }
    | "accepted"
    | "rejected";

async function code_edit_send_progress_handler(
    params: AgentProgress<CodeEditPayload>,
    agent: Agent
): Promise<void> {
    console.log(`code_edit_send_progress PARAMS:`, params);
    if (params.tasks?.task.status) {
        agent.onStatusChangeEmitter.fire(params.tasks.task.status);
    }
    if (
        params.tasks?.task.status === "done" ||
        params.tasks?.task.status === "error"
    ) {
        agent.onCodeLensStatusChangeEmitter.fire(params.tasks?.task.status);
    }

    if (
        params.payload !== "accepted" &&
        params.payload !== "rejected" &&
        params.payload?.ready
    ) {
        console.log("READY, FIRING");
        agent.onCodeLensStatusChangeEmitter.fire("ready");
        agent.onStatusChangeEmitter.fire("running");
    }

    console.log(`URI: ${agent?.textDocument?.uri?.toString()}`);
    const editors: TextEditor[] = vscode.window.visibleTextEditors.filter(
        (e) => e.document.uri.toString() == agent?.textDocument?.uri?.toString()
    );

    if (editors.length === 0) {
        return;
    }
    // multiple editors can be pointing to the same resource

    for (const editor of editors) {
        console.log(`EDITOR: ${editor}`);
        // [todo] check editor is visible
        const version = editor.document.version;

        if (params.payload == "accepted" || params.payload == "rejected") {
            agent.onCodeLensStatusChangeEmitter.fire(params.payload);
            editor.setDecorations(GREEN, []);
            editor.setDecorations(RED, []);
            agent.morph_language_client.sendDoesShowAcceptRejectBarChange(
                agent.id,
                false
            );
            agent.morph_language_client.changeLensEmitter.fire(); // this causes the code lenses to rerender or un-render
            console.log("SET DECORATIONS TO NONE");
            // agent.morph_language_client.delete({ id: agent.id })
            continue;
        }

        if (params.payload?.additive_ranges) {
            console.log(`ADDITIVE RANGES: ${params.payload.additive_ranges}`);
            editor.setDecorations(
                GREEN,
                params.payload.additive_ranges.map((r) => {
                    const result = new vscode.Range(
                        r.start.line,
                        r.start.character,
                        r.end.line,
                        r.end.character
                    );
                    console.log(
                        `RESULT: ${r.start.line} ${r.start.character} ${r.end.line} ${r.end.character}`
                    );
                    return result;
                })
            );
        }
        if (params.payload?.negative_ranges) {
            editor.setDecorations(
                RED,
                params.payload.negative_ranges.map(
                    (r) =>
                        new vscode.Range(
                            r.start.line,
                            r.start.character,
                            r.end.line,
                            r.end.character
                        )
                )
            );
        }
    }
}

export class AgentStateLens extends vscode.CodeLens {
    id: string;
    constructor(range: vscode.Range, agent: any, command?: vscode.Command) {
        super(range, command);
        this.id = agent.id;
    }
}

interface ModelConfig {
    chatModel: string;
    completionsModel: string;
    /** The API key for OpenAI, you can also set OPENAI_API_KEY. */
    openai_api_key?: string;
}

export type AgentIdentifier = string;

export class MorphLanguageClient
    implements vscode.CodeLensProvider<AgentStateLens>
{
    client: LanguageClient | undefined = undefined;
    red: vscode.TextEditorDecorationType;
    green: vscode.TextEditorDecorationType;
    context: vscode.ExtensionContext;
    changeLensEmitter: vscode.EventEmitter<void>;
    onDidChangeCodeLenses: vscode.Event<void>; // call this event to rerender
    agents: { [id: string]: Agent } = {};
    private webviewState = new Store<WebviewState>(DEFAULT_STATE);

    constructor(context: vscode.ExtensionContext) {
        this.red = { key: "TEMP_VALUE", dispose: () => {} };
        this.green = { key: "TEMP_VALUE", dispose: () => {} };
        this.context = context;
        this.webviewState.subscribe((state) => {
            // console.log('webview state:')
            // console.log(state)
            chatProvider.stateUpdate(state);
            logProvider.stateUpdate(state);
        });

        this.create_client().then(() => {
            this.context.subscriptions.push(
                vscode.commands.registerCommand("extension.listAgents", async () => {
                    if (client) {
                        return await this.list_agents();
                    }
                }),
                vscode.commands.registerCommand("rift.cancel", (id: string) =>
                    this.client?.sendNotification("morph/cancel", { id })
                ),
                vscode.commands.registerCommand("rift.accept", (id: string) =>
                    this.client?.sendNotification("morph/accept", { id })
                ),
                vscode.commands.registerCommand("rift.reject", (id: string) =>
                    this.client?.sendNotification("morph/reject", { id })
                ),
                vscode.workspace.onDidChangeConfiguration(
                    this.on_config_change.bind(this)
                )
            );

            // the below 3 lines populate the webview state with initial state needed for @URI chips
            const activeUri = vscode.window.activeTextEditor?.document.uri;
            if (activeUri)
                this.webviewState.update((pS) => ({
                    ...pS,
                    files: {
                        ...pS.files,
                        recentlyOpenedFiles: [AtableFileFromUri(activeUri)],
                    },
                }));
            this.refreshNonGitIgnoredFiles();

            this.create("rift_chat");
            this.refreshAvailableAgents();
        });

        this.changeLensEmitter = new vscode.EventEmitter<void>();
        this.onDidChangeCodeLenses = (
            listener: (e: void) => any,
            thisArgs?: any,
            disposables?: vscode.Disposable[] | undefined
        ) => {
            return this.changeLensEmitter.event(listener, thisArgs, disposables);
        };
    }

    public getWebviewState() {
        return this.webviewState.value;
    }

    public provideCodeLenses(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): AgentStateLens[] {
        // this returns all of the lenses for the document.
        let items: AgentStateLens[] = [];
        // console.log("AGENTS: ", this.agents);

        for (const [id, agent] of Object.entries(this.agents)) {
            if (!["code_edit"].includes(agent.agent_type)) {
                continue;
            }
            console.log("provideCodeLens called. agent code lens status:");
            console.log(agent.codeLensStatus);

            if (agent?.selection) {
                if (agent?.textDocument?.uri?.toString() == document.uri.toString()) {
                    const line = agent?.selection.isReversed
                        ? agent?.selection.active
                        : agent?.selection.anchor;
                    const linetext = document.lineAt(line);

                    //####### HARDCODED REMOVE THIS #################
                    // agent.status = "done";
                    //##############################################

                    if (agent.codeLensStatus === "running") {
                        const running = new AgentStateLens(linetext.range, agent, {
                            title: "running",
                            command: "rift.cancel",
                            tooltip: "click to stop this agent",
                            arguments: [agent.id],
                        });
                        items.push(running);
                    } else if (agent.codeLensStatus === "ready") {
                        this.sendDoesShowAcceptRejectBarChange(
                            agent.id,
                            agent.codeLensStatus === "ready"
                        );
                        const accept = new AgentStateLens(linetext.range, agent, {
                            title: "Accept ✅ ",
                            command: "rift.accept",
                            tooltip: "Accept the edits below",
                            arguments: [agent.id],
                        });

                        const reject = new AgentStateLens(linetext.range, agent, {
                            title: " Reject ❌",
                            command: "rift.reject",
                            tooltip: "Reject the edits below and restore the original text",
                            arguments: [agent.id],
                        });
                        items.push(accept, reject);
                    } else {
                        this.sendDoesShowAcceptRejectBarChange(agent.id, false);
                    }
                }
            }
        }
        return items;
    }

    is_running() {
        return this.client && this.client.state == State.Running;
    }

    private async list_agents() {
        if (!this.client) throw new Error();
        const result: AgentRegistryItem[] = await this.client.sendRequest(
            "morph/listAgents",
            {}
        );

        return result;
    }

    public refreshWebviewState() {
        chatProvider.stateUpdate(this.webviewState.value);
        logProvider.stateUpdate(this.webviewState.value);
    }

    public async refreshAvailableAgents() {
        console.log("refreshing webview agents");
        const availableAgents = (await this.list_agents()).reverse();
        this.webviewState.update((state) => ({ ...state, availableAgents }));
    }

    async create_client() {
        if (this.client && this.client.state != State.Stopped) {
            console.log(
                `client already exists and is in state ${this.client.state} `
            );
            return;
        }
        const port = DEFAULT_PORT;
        let serverOptions: ServerOptions;
        while (!(await tcpPortUsed.check(port))) {
            console.log("waiting for server to come online");
            try {
                await tcpPortUsed.waitUntilUsed(port, 500, 1000000);
            } catch (e) {
                console.error(e);
            }
        }
        console.log(`server detected on port ${port} `);
        serverOptions = tcpServerOptions(this.context, port);
        const clientOptions: LanguageClientOptions = {
            documentSelector: [{ language: "*" }],
        };
        this.client = new LanguageClient(
            "morph-server",
            "Morph Server",
            serverOptions,
            clientOptions
        );
        this.client.onDidChangeState(async (e) => {
            console.log(`client state changed: ${e.oldState} ▸ ${e.newState} `);
            if (e.newState === State.Stopped) {
                console.log("morph server stopped, restarting...");
                await this.client?.dispose();
                console.log("morph server disposed");
                await this.create_client();
            }
        });
        this.client.onNotification("morph/error", async (params: any) => {vscode.window.showErrorMessage(params.msg)});
        await this.client.start();
        console.log("rift-engine started");
    }

    async on_config_change(_args: any) {
        if (!this.client) throw new Error("no client");
        const x = await this.client.sendRequest(
            "workspace/didChangeConfiguration",
            {}
        );
    }

    async notify_focus(tdpp: TextDocumentPositionParams | { symbol: string }) {
        // [todo] unused
        console.log(tdpp);
        await this.client?.sendNotification("morph/focus", tdpp);
    }

    async hello_world() {
        const result = await this.client?.sendRequest("hello_world");
        return result;
    }

    morphNotifyChatCallback: (progress: ChatAgentProgress) => any =
        async function (progress) {
            throw new Error("no callback set");
        };

    async create(agent_type: string) {
        if (!this.client) throw new Error();

        const editor: TextEditor | undefined = vscode.window.activeTextEditor;

        const folders = vscode.workspace.workspaceFolders;
        if (!folders) throw new Error("no current workspace");
        const workspaceFolderPath = folders[0].uri.fsPath;
        let document = editor?.document;
        let textDocument: OptionalTextDocument = null;
        if (document != undefined) {
            textDocument = { uri: document.uri.toString(), version: 0 };
        }
        let position = editor?.selection?.active ?? null;

        const agentParams: AgentParams = {
            agent_type: agent_type,
            agent_id: "", // agent ID has not been assigned yet
            selection: editor?.selection ?? null,
            position,
            textDocument,
            workspaceFolderPath,
        };

        const result: RunAgentResult = await this.client.sendRequest(
            "morph/create_agent",
            agentParams
        );
        console.log("run agent result");
        console.log(result);
        const agent_id = result.id;

        const agent = new Agent(
            this,
            agent_id,
            agent_type,
            editor?.selection || null,
            textDocument
        );

        this.webviewState.update((state) => ({
            ...state,
            selectedAgentId: agent_id,
            agents: {
                [agent_id]: new WebviewAgent(agent_type),
                ...state.agents,
            },
        }));

        this.agents[agent_id] = agent;
        console.log(`REGISTERED NEW AGENT of type ${agent_type}`);
        this.changeLensEmitter.fire();

        this.client.onRequest(
            `morph/${agent_type}_${agent_id}_request_input`,
            agent.handleInputRequest.bind(agent)
        );
        this.client.onRequest(
            `morph/${agent_type}_${agent_id}_request_chat`,
            agent.handleChatRequest.bind(agent)
        );
        this.client.onNotification(
            `morph/${agent_type}_${agent_id}_send_update`,
            agent.handleUpdate.bind(agent)
        ); // this should post a message to the rift logs webview if `tasks` have been updated
        this.client.onNotification(
            `morph/${agent_type}_${agent_id}_send_progress`,
            agent.handleProgress.bind(agent)
        ); // this should post a message to the rift logs webview if `tasks` have been updated
        this.client.onNotification(
            `morph/${agent_type}_${agent_id}_send_result`,
            agent.handleResult.bind(agent)
        ); // this should be custom
    }

    async cancel(params: AgentIdParams) {
        if (!this.client) throw new Error();
        let response = await this.client.sendRequest("morph/cancel", params);
        return response;
    }

    async delete(params: AgentIdParams) {
        if (!this.client) throw new Error();
        // let response = await this.client.sendRequest("morph/delete", params);
        let response = await this.client.sendRequest("morph/cancel", params);

        this.webviewState.update((state) => {
            const updatedAgents = { ...state.agents };
            const anotherAvailableAgent = Object.keys(updatedAgents).find(
                (key) => key != params.id && updatedAgents[key].isDeleted === false
            );
            if (anotherAvailableAgent) {
                updatedAgents[params.id].isDeleted = true;
            }

            // update selected agent if you deleted your selected agent
            const updatedSelectedAgentId =
                params.id == state.selectedAgentId
                    ? anotherAvailableAgent!
                    : state.selectedAgentId;

            return {
                ...state,
                selectedAgentId: updatedSelectedAgentId,
                agents: updatedAgents,
            };
        });

        return response;
    }

    async restart_agent(agentId: string) {
        if (!this.client) throw new Error();
        if (!(agentId in this.webviewState.value.agents))
            throw new Error(
                `tried to restart agent ${agentId} but couldn't find it in agents object`
            );
        const agent_type = this.webviewState.value.agents[agentId].type;
        let result: RunAgentResult = await this.client.sendRequest(
            "morph/restart_agent",
            { id: agentId }
        );
        this.webviewState.update((state) => ({
            ...state,
            agents: {
                ...state.agents,
                [agentId]: new WebviewAgent(agent_type),
            },
        }));
    }

    public restartActiveAgent() {
        this.restart_agent(this.webviewState.value.selectedAgentId);
    }

    async refreshNonGitIgnoredFiles() {
        // another day we will implement this logic. That day is not today :( which is sad bc I like coding

        // async function getGlobPatternsFromGitIgnores() {
        //   const gitignores = await vscode.workspace.findFiles("**/.gitignore")
        //   // const ignoreMatcher = IgnoreMatcher.fromLines(await vscode.workspace.fs.readFile(gitignore[0]))
        //   const fullGitIgnoreFolderPathToGlobArray:{[fullGitIgnorePath: string]: string[]} = {}
        //   //TODO: make work for nested .gitignores. I think we can just do this by prepending filepaths to the globs. Not sure though
        //   for(let gitignore of gitignores) {
        //     const globPatterns: string[] = []
        //     const gitignoreUint8Array = await vscode.workspace.fs.readFile(gitignore)
        //     const gitignoreString = gitignoreUint8Array.toString()
        //     globPatterns.push(...gitignoreString.split(/\n/).filter(pattern => (pattern.trim() === '' || pattern.trim().startsWith('#'))))
        //     fullGitIgnoreFolderPathToGlobArray[gitignore.path] = globPatterns
        //   }
        //   return fullGitIgnoreFolderPathToGlobArray
        // }
        const time = Date.now();
        // const gitIgnoreToGlobsMap = await getGlobPatternsFromGitIgnores()

        const latency = Date.now() - time;
        console.log(
            `latency in regetting gitignore globs is ${latency}ms. If too high, consider adding event listeners to when the gitignores change instead of refetching them every time`
        );

        let vsCodeFiles: vscode.Uri[] = await vscode.workspace.findFiles(
            "**/*",
            "**/node_modules/*"
        );
        let allPaths: Set<vscode.Uri> = new Set();

        for (let file of vsCodeFiles) {
            let parentDir = file.fsPath.split("/").slice(0, -1).join("/");
            // console.log(`parentDir=${parentDir}`)
            allPaths.add(vscode.Uri.parse(parentDir));
            allPaths.add(file);
        }

        let allFiles: vscode.Uri[] = [...allPaths];

        this.webviewState.update((pS) => ({
            ...pS,
            files: {
                ...pS.files,
                nonGitIgnoredFiles: allFiles.map(AtableFileFromUri),
            },
        }));
    }

    sendChatHistoryChange(agentId: string, newChatHistory: ChatMessage[]) {
        console.log(`updating chat history for ${agentId}`, newChatHistory);
        if (
            this.webviewState.value.agents[agentId].chatHistory.length ==
            newChatHistory.length
        ) {
            return;
        }

        if (this.webviewState.value.agents[agentId].chatHistory.length > 0)
            console.warn(
                "discrepancy between server agent chat history and client agent chathistory. taking server as truth"
            );
        console.log("newChatHistory:", newChatHistory);
        this.webviewState.update((state) => {
            if (!(agentId in state.agents))
                throw new Error("changing chatHistory for nonexistent agent");
            return {
                ...state,
                agents: {
                    ...state.agents,
                    [agentId]: {
                        ...state.agents[agentId],
                        chatHistory: [...newChatHistory],
                    },
                },
            };
        });
    }

    sendProgressChange(params: AgentProgress) {
        const { agent_id, tasks } = params;
        const payload = params.payload;
        if (
            payload &&
            {}.hasOwnProperty.call(payload, "messages") &&
            payload.messages
        ) {
            this.sendChatHistoryChange(agent_id, params.payload.messages);
        }
        if (
            payload == "accepted" ||
            payload == "rejected" ||
            typeof payload == "string"
        )
            return;

        if (!(agent_id in this.webviewState.value.agents)) {
            console.log(params);
            throw new Error(`progress for nonexistent agent: ${agent_id}`);
        }

        const response =
            payload && {}.hasOwnProperty.call(payload, "response")
                ? payload.response
                : undefined;

        if (response)
            this.webviewState.update((state) => ({
                ...state,
                agents: {
                    ...state.agents,
                    [agent_id]: {
                        ...state.agents[agent_id],
                        streamingText: response,
                        isStreaming: true,
                    },
                },
            }));

        if (tasks) {
            this.webviewState.update((state) => ({
                ...state,
                agents: {
                    ...state.agents,
                    [agent_id]: {
                        ...state.agents[agent_id],
                        type: params.agent_type,
                        taskWithSubtasks: {
                            ...state.agents[agent_id].taskWithSubtasks,
                            ...tasks,
                        },
                    },
                },
            }));
        }
        if (payload && "done_streaming" in payload && payload.done_streaming) {
            this.webviewState.update((prevState) => {
                return {
                    ...prevState,
                    agents: {
                        ...prevState.agents,
                        [agent_id]: {
                            ...prevState.agents[agent_id],
                            agent_id: agent_id,
                            agent_type: params.agent_type,
                            isStreaming: false,
                            streamingText: "",
                        },
                    },
                };
            });
        }
    }

    sendDoesShowAcceptRejectBarChange(
        agentId: string,
        doesShowAcceptRejectBar: boolean
    ) {
        this.webviewState.update((state) => ({
            ...state,
            agents: {
                ...state.agents,
                [agentId]: { ...state.agents[agentId], doesShowAcceptRejectBar },
            },
        }));
    }

    sendHasNotificationChange(agentId: string, hasNotification: boolean) {
        if (!(agentId in this.webviewState.value.agents))
            throw new Error(`cant update nonexistent agent: ${agentId}`);
        this.webviewState.update((state) => ({
            ...state,
            agents: {
                ...state.agents,
                [agentId]: {
                    ...state.agents[agentId],
                    hasNotification:
                        agentId == state.selectedAgentId ? false : hasNotification, //this ternary operatory will make sure we don't set currently selected agents as having notifications
                },
            },
        }));
    }

    sendSelectedAgentChange(agentId: string) {
        this.webviewState.update((state) => {
            if (!(agentId in state.agents))
                throw new Error(
                    `tried to change selectedAgentId to an unavailable agent. tried to change to ${agentId} but available agents are: ${Object.keys(
                        state.agents
                    )}`
                );

            return { ...state, selectedAgentId: agentId };
        });
    }

    sendRecentlyOpenedFilesChange(recentlyOpenedFiles: string[]) {
        const atableFiles = recentlyOpenedFiles.map((fspath) =>
            AtableFileFromFsPath(fspath)
        );
        this.webviewState.update((pS) => ({
            ...pS,
            files: { ...pS.files, recentlyOpenedFiles: atableFiles },
        }));
        this.refreshNonGitIgnoredFiles();
    }

    focusOmnibar() {
        this.webviewState.update((state) => {
            return {
                ...state,
                isOmnibarFocused: true,
            };
        });
    }

    blurOmnibar() {
        this.webviewState.update((state) => {
            return {
                ...state,
                isOmnibarFocused: false,
            };
        });
    }

    dispose() {
        this.client?.dispose();
    }
}

class Agent {
    status: AgentStatus;
    private _codeLensStatus: CodeLensStatus; // instead call the onCodeLensStatusChangeEmitter.fire() which will rerender the code lenses
    green: vscode.TextEditorDecorationType;
    ranges: vscode.Range[] = [];
    onStatusChangeEmitter: vscode.EventEmitter<AgentStatus>;
    onCodeLensStatusChangeEmitter: vscode.EventEmitter<CodeLensStatus>;
    morph_language_client: MorphLanguageClient;
    get codeLensStatus() {
        return this._codeLensStatus;
    }

    constructor(
        morph_language_client: MorphLanguageClient,
        public readonly id: string,
        public readonly agent_type: string,
        public readonly selection: vscode.Selection | null,
        public textDocument: OptionalTextDocument
    ) {
        this.morph_language_client = morph_language_client;
        this.id = id;
        this.status = "running";
        this._codeLensStatus = "running";
        this.agent_type = agent_type;
        this.selection = selection;
        this.textDocument = textDocument;
        this.green = vscode.window.createTextEditorDecorationType({
            backgroundColor: "rgba(0,255,0,0.1)",
        });
        this.onStatusChangeEmitter = new vscode.EventEmitter<AgentStatus>();
        this.onCodeLensStatusChangeEmitter =
            new vscode.EventEmitter<CodeLensStatus>();
        this.onStatusChangeEmitter.event(() =>
            morph_language_client.changeLensEmitter.fire()
        );
        this.onCodeLensStatusChangeEmitter.event((e: CodeLensStatus) => {
            this._codeLensStatus = e;
            morph_language_client.changeLensEmitter.fire();
        });
    }

    async handleInputRequest(params: AgentInputRequest) {
        if (!(this.id in this.morph_language_client.agents))
            throw Error("Agent does not exist");

        let response = await vscode.window.showInputBox({
            ignoreFocusOut: true,
            placeHolder: params.place_holder,
            prompt: params.msg,
        });
        return { response: response };
    }

    async handleChatRequest(params: AgentChatRequest) {
        if (!(this.id in this.morph_language_client.agents))
            throw Error("Agent does not exist");
        console.log("handleChatRequest");
        console.log(params);
        console.log("agentType:", this.agent_type);
        // if(!params.id) throw new Error('no params')

        this.morph_language_client.sendChatHistoryChange(this.id, params.messages);
        this.morph_language_client.sendHasNotificationChange(this.id, true);

        const agentType = this.agent_type;
        const agentId = this.id;

        console.log("agentId:", this.id);

        // return "BLAH BLAH"
        async function getUserInput() {
            console.log("getUserInput");

            return new Promise((res, rej) => {
                console.log("subscribing to changes");
                PubSub.sub(`${agentType}_${agentId}_chat_request`, (message) => {
                    console.log("resolving promise");
                    res(message);
                });
            });
        }

        let chatRequest = await getUserInput();
        console.log("received user input and returning to server");
        console.log(chatRequest);
        return chatRequest;
    }
    async handleUpdate(params: AgentUpdate) {
        if (!(this.id in this.morph_language_client.agents))
            throw Error("Agent does not exist");
        console.log("handleUpdate");
        console.log(params);

        vscode.window.showInformationMessage(params.msg);
    }
    async handleProgress(params: AgentProgress) {
        if (!(this.id in this.morph_language_client.agents))
            throw Error("Agent does not exist");

        console.log("handle Progress:");
        console.log(params);
        this.morph_language_client.sendProgressChange(params);

        if (this.agent_type === "code_edit") {
            const params2 = params as AgentProgress<CodeEditPayload>; // dont ask me how I know -b
            console.log("code edit progress");
            console.log(params);

            code_edit_send_progress_handler(params2, this);
        }
    }
    async handleResult(params: AgentResult) {
        if (!(this.id in this.morph_language_client.agents))
            throw Error("Agent does not exist");
        console.log("handleResult");
        console.log(params);

        throw new Error("no logic written for handle result yet");
    }
}

import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode'
import * as vscode from 'vscode'
import { ChildProcessWithoutNullStreams, spawn } from 'child_process'
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    Executable,
    TransportKind,
    StreamInfo,
    TextDocumentPositionParams,
    NotificationType,
    TextDocumentIdentifier,
    State,
} from 'vscode-languageclient/node'
import * as net from 'net'
import { join } from 'path';
import { ChatAgentProgress } from './types';
import delay from 'delay'
import * as tcpPortUsed from 'tcp-port-used'

let client: LanguageClient

const DEFAULT_PORT = 7797

// ref: https://stackoverflow.com/questions/40284523/connect-external-language-server-to-vscode-extension

// https://nodejs.org/api/child_process.html#child_processspawncommand-args-options

/** Creates the ServerOptions for a system in the case that a language server is already running on the given port. */
function tcpServerOptions(context: ExtensionContext, port = DEFAULT_PORT): ServerOptions {
    let socket = net.connect({
        port: port, host: "127.0.0.1"
    })
    const si: StreamInfo = {
        reader: socket, writer: socket
    }
    return () => {
        return Promise.resolve(si)
    }
}

/** Creates the server options for spinning up our own server.*/
function createServerOptions(context: vscode.ExtensionContext, port = DEFAULT_PORT): ServerOptions {
    let cwd = vscode.workspace.workspaceFolders![0].uri.path
    // [todo]: we will supply different bundles for the 3 main platforms; windows, linux, osx.
    // there needs to be a decision point here where we decide which platform we are on and
    // then choose the appropriate bundle.
    let command = join(context.extensionPath, 'resources', 'lspai')
    let args: string[] = []
    args = [...args, '--port', port.toString()]
    let e: Executable = {
        command,
        args,
        transport: { kind: TransportKind.socket, port },
        options: { cwd },
    }
    return {
        run: e, debug: e
    }
}

interface RunCodeHelperParams {
    task: string
    position: vscode.Position
    textDocument: TextDocumentIdentifier
}

interface RunAgentParams {
    agent_type: string
    agent_params: any
}

interface RunAgentResult {
    id: number
    agentId: string | null
}


interface RunChatParams {
    message: string
    messages: { // does not include latest message
        role: string,
        content: string
    }[],
    position: vscode.Position,
    textDocument: TextDocumentIdentifier,
}


interface RunAgentResult {
    id: number
}

interface RunAgentSyncResult {
    id: number
    text: string
}

export type AgentStatus = 'running' | 'done' | 'error' | 'accepted' | 'rejected'

export interface RunAgentProgress {
    id: number
    textDocument: TextDocumentIdentifier
    log?: {
        severity: string;
        message: string;
    }
    cursor?: vscode.Position
    /** This is the set of ranges that the agent has added so far. */
    ranges?: vscode.Range[]
    status: AgentStatus
}

/** Represents an agent */
class Agent {
    status: AgentStatus;
    green: vscode.TextEditorDecorationType;
    ranges: vscode.Range[] = []
    onStatusChangeEmitter: vscode.EventEmitter<AgentStatus>
    onStatusChange: vscode.Event<AgentStatus>
    constructor(public readonly id: number, public readonly startPosition: vscode.Position, public textDocument: TextDocumentIdentifier) {
        this.status = 'running'
        this.green = vscode.window.createTextEditorDecorationType({ backgroundColor: 'rgba(0,255,0,0.1)' })
        this.onStatusChangeEmitter = new vscode.EventEmitter<AgentStatus>()
        this.onStatusChange = this.onStatusChangeEmitter.event
    }
    handleProgress(params: RunAgentProgress) {
        if (params.status) {
            if (this.status !== params.status) {
                this.status = params.status
                this.onStatusChangeEmitter.fire(params.status)
            }
        }
        if (params.ranges) {
            this.ranges = params.ranges
        }
        const editors = vscode.window.visibleTextEditors.filter(e => e.document.uri.toString() == params.textDocument.uri)
        for (const editor of editors) {
            // [todo] check editor is visible
            const version = editor.document.version
            if (params.status == 'accepted' || params.status == 'rejected') {
                editor.setDecorations(this.green, [])
                continue
            }
            if (params.ranges) {
                editor.setDecorations(this.green, params.ranges.map(r => new vscode.Range(r.start.line, r.start.character, r.end.line, r.end.character)))
            }
        }
    }
}

export class AgentStateLens extends vscode.CodeLens {
    id: number
    constructor(range: vscode.Range, agentState: any, command?: vscode.Command) {
        super(range, command)
        this.id = agentState.agent_id
    }
}

interface ModelConfig {
    chatModel: string
    completionsModel: string
    /** The API key for OpenAI, you can also set OPENAI_API_KEY. */
    openai_api_key?: string
}

type AgentType = "chat" | "code-completion"
export type AgentIdentifier = string

export class MorphLanguageClient implements vscode.CodeLensProvider<AgentStateLens> {
    client: LanguageClient
    red: vscode.TextEditorDecorationType
    green: vscode.TextEditorDecorationType
    context: vscode.ExtensionContext
    changeLensEmitter: vscode.EventEmitter<void>
    onDidChangeCodeLenses: vscode.Event<void>
    agents = new Map<AgentIdentifier, Agent>()
    agentStates = new Map<AgentIdentifier, any>()

    constructor(context: vscode.ExtensionContext) {
        this.context = context
        this.create_client()
        this.changeLensEmitter = new vscode.EventEmitter<void>()
        this.onDidChangeCodeLenses = this.changeLensEmitter.event
        this.context.subscriptions.push(
            vscode.commands.registerCommand('rift.cancel', (id: number) => this.client.sendNotification('morph/cancel', { id })),
            vscode.commands.registerCommand('rift.accept', (id: number) => this.client.sendNotification('morph/accept', { id })),
            vscode.commands.registerCommand('rift.reject', (id: number) => this.client.sendNotification('morph/reject', { id })),
            vscode.workspace.onDidChangeConfiguration(this.on_config_change.bind(this)),
        )

    }

    // TODO: needs to be modified to account for whether or not an agent has an active cursor in the document whatsoever
    public provideCodeLenses(document: vscode.TextDocument, token: vscode.CancellationToken): AgentStateLens[] {
        // this returns all of the lenses for the document.
        const items: AgentStateLens[] = []
        for (const agentState of this.agentStates.values()) {
            if (agentState.agent_type == "code_completion" && agentState.params.textDocument.uri.toString() == document.uri.toString()) {
                const line = agentState.params.position.line
                const linetext = document.lineAt(line)
                if (agentState.status === 'running') {
                    const running = new AgentStateLens(linetext.range, agentState, {
                        title: 'running',
                        command: 'rift.cancel',
                        tooltip: 'click to stop this agent',
                        arguments: [agentState.agent_id],
                    })
                    items.push(running)
                }
                else if (agentState.status === 'done' || agentState.status === 'error') {
                    const accept = new AgentStateLens(linetext.range, agentState, {
                        title: 'Accept ✅ ',
                        command: 'rift.accept',
                        tooltip: 'Accept the edits below',
                        arguments: [agentState.agent_id],
                    })
                    const reject = new AgentStateLens(linetext.range, agentState, {
                        title: ' Reject ❌',
                        command: 'rift.reject',
                        tooltip: 'Reject the edits below and restore the original text',
                        arguments: [agentState.agent_id]
                    })
                    items.push(accept, reject)
                }
            }
        }
        return items
    }    

    // // TODO: needs to be modified to account for whether or not an agent has an active cursor in the document whatsoever
    // public provideCodeLenses(document: vscode.TextDocument, token: vscode.CancellationToken): AgentLens[] {
    //     // this returns all of the lenses for the document.
    //     const items: AgentLens[] = []
    //     for (const agent of this.agents.values()) {
    //         if (agent.textDocument.uri === document.uri.toString()) {
    //             const line = agent.startPosition.line
    //             const linetext = document.lineAt(line)
    //             if (agent.status === 'running') {
    //                 const running = new AgentLens(linetext.range, agent, {
    //                     title: 'running',
    //                     command: 'rift.cancel',
    //                     tooltip: 'click to stop this agent',
    //                     arguments: [agent.id],
    //                 })
    //                 items.push(running)
    //             }
    //             else if (agent.status === 'done' || agent.status === 'error') {
    //                 const accept = new AgentLens(linetext.range, agent, {
    //                     title: 'Accept ✅ ',
    //                     command: 'rift.accept',
    //                     tooltip: 'Accept the edits below',
    //                     arguments: [agent.id],
    //                 })
    //                 const reject = new AgentLens(linetext.range, agent, {
    //                     title: ' Reject ❌',
    //                     command: 'rift.reject',
    //                     tooltip: 'Reject the edits below and restore the original text',
    //                     arguments: [agent.id]
    //                 })
    //                 items.push(accept, reject)
    //             }
    //         }
    //     }
    //     return items
    // }

    public resolveCodeLens(codeLens: AgentStateLens, token: vscode.CancellationToken) {
        // you use this to resolve the commands for the code lens if
        // it would be too slow to compute the commands for the entire document.
        return null
    }

    is_running() {
        return this.client && this.client.state == State.Running
    }

    async create_client() {
        if (this.client && this.client.state != State.Stopped) {
            console.log(`client already exists and is in state ${this.client.state}`)
            return
        }
        const port = DEFAULT_PORT
        let serverOptions: ServerOptions
        while (!(await tcpPortUsed.check(port))) {
            console.log('waiting for server to come online')
            try {
                await tcpPortUsed.waitUntilUsed(port, 500, 1000000)
            }
            catch (e) {
                console.error(e)
            }
        }
        console.log(`server detected on port ${port}`)
        serverOptions = tcpServerOptions(this.context, port)
        const clientOptions: LanguageClientOptions = {
            documentSelector: [{ language: '*' }]
        }
        this.client = new LanguageClient(
            'morph-server', 'Morph Server',
            serverOptions, clientOptions,
        )
        this.client.onDidChangeState(async e => {
            console.log(`client state changed: ${e.oldState} ▸ ${e.newState}`)
            if (e.newState === State.Stopped) {
                console.log('morph server stopped, restarting...')
                await this.client.dispose()
                console.log('morph server disposed')
                await this.create_client()
            }
        })
        await this.client.start()
        this.client.onNotification('morph/progress', this.morph_notify.bind(this))
        console.log('rift-engine started')
    }


    async on_config_change(args) {
        const x = await this.client.sendRequest('workspace/didChangeConfiguration', {})
    }


    async morph_notify(params: RunAgentProgress) {
        if (!this.is_running()) {
            throw new Error('client not running, please wait...') // [todo] better ux here.
        }
        const agent = this.agents.get(params.id)
        if (!agent) {
            throw new Error('agent not found')
        }
        agent.handleProgress(params)
    }

    async notify_focus(tdpp: TextDocumentPositionParams | { symbol: string }) {
        // [todo] unused
        console.log(tdpp)
        await this.client.sendNotification('morph/focus', tdpp)
    }

    async hello_world() {
        const result = await this.client.sendRequest('hello_world')
        return result
    }

    async run_agent(params: RunCodeHelperParams) {
        if (!this.client) {
            throw new Error(`waiting for a connection to rift-engine, please make sure the rift-engine is running on port ${DEFAULT_PORT}`) // [todo] better ux here.
        }
        const result: RunAgentResult = await this.client.sendRequest('morph/run_agent', params)
        const agent = new Agent(result.id, params.position, params.textDocument)
        agent.onStatusChange(e => this.changeLensEmitter.fire())
        this.agents.set(result.id.toString(), agent)
        // note this returns fast and then the updates are sent via notifications
        this.changeLensEmitter.fire()
        return `starting agent ${result.id}...`
    }

    async run_agent_sync(params: RunCodeHelperParams) {
        console.log("run_agent_sync")
        const result: RunAgentSyncResult = await this.client.sendRequest('morph/run_agent_sync', params)
        const agent = new Agent(result.id, params.position, params.textDocument)
        // agent.onStatusChange(e => this.changeLensEmitter.fire())
        this.agents.set(result.id, agent)
        // this.changeLensEmitter.fire()
        return result.text
    }

    morphNotifyChatCallback: (progress: ChatAgentProgress) => any = async function (progress) {
        throw new Error('no callback set')
    }

    async run_chat(params: RunChatParams, callback: (progress: ChatAgentProgress) => any) {
        console.log('run chat')
        this.morphNotifyChatCallback = callback
        this.client.onNotification('morph/chat_progress', this.morphNotifyChatCallback.bind(this))

        const result = await this.client.sendRequest('morph/run_chat', params)
        // note this returns fast and then the updates are sent via notifications
        return 'starting...'
    }

    // create a SpawnAgentResult that contains a server-computed agentId


    // note(jesse): for CodeCompletionAgent, we'll want to:
    // feed in params = {
    //   agent_type: "code_completion"
    //   agentParams: RunCodeHelperParams
    // }
    // with no-ops for all callbacks except for `send_progress`

    async run(
        params: RunAgentParams,
        request_input_callback: (request_input_request: any) => any,
        request_chat_callback: (request_chat_request: any) => any,
        send_progress_callback: (send_progress_request: any) => any,
        send_result_callback: (send_result_request: any) => any,
    ) {
        const result: RunAgentResult = await this.client.sendRequest('morph/run', params);
        const agent_id = result.id; // TODO(jesse): does this create a race condition? // TODO: better identifiers than ints returned by server
        const agent_type = params.agent_type;
        console.log(`running ${agent_type}`)
        const agentIdentifier = `${agent_type}_${agent_id}`
        console.log(`agentIdentifier: ${agentIdentifier}`)
        this.agentStates.set(agentIdentifier, {agent_id: agent_id, agent_type: agent_type, status: "running", ranges: [], tasks: [], emitter: new vscode.EventEmitter<AgentStatus>, params: params.agent_params})
        this.client.onNotification(`morph/${agent_type}_${agent_id}_request_input`, request_input_callback.bind(this))
        this.client.onNotification(`morph/${agent_type}_${agent_id}_request_chat`, request_chat_callback.bind(this))
        // note(jesse): for the chat agent, the request_chat callback should register another callback for handling user responses --- it should unpack the future identifier from the request_chat_request and re-pass it to the language server
        this.client.onNotification(`morph/${agent_type}_${agent_id}_send_progress`, send_progress_callback.bind(this)) // this should post a message to the rift logs webview if `tasks` have been updated
        // actually, i wonder if the server should just be generally responsible for sending notifications to the client about active tasks
        this.client.onNotification(`morph/${agent_type}_${agent_id}_send_result`, send_result_callback.bind(this)) // this should be custom
    }

    // run should spawn an agent
    // run should specify:
    // - agent type
    // - callbacks for request_input, request_chat, send_progress, and send_result

    // async run_chat(params: RunChatParams, callback: (progress: ChatAgentProgress) => any) {
    //     console.log('run chat')
    //     this.morphNotifyChatCallback = callback
    //     this.client.onNotification('morph/chat_progress', this.morphNotifyChatCallback.bind(this))

    //     const result = await this.client.sendRequest('morph/run_chat', params)
    //     // note this returns fast and then the updates are sent via notifications
    //     return 'starting...'
    // }


    dispose() {
        this.client.dispose()
    }

    async provideInlineCompletionItems(doc: vscode.TextDocument, position: vscode.Position, context: vscode.InlineCompletionContext, token: vscode.CancellationToken) {
        const params: RunCodeHelperParams = { task: "complete the code", position: position, textDocument: TextDocumentIdentifier.create(doc.uri.toString()) };
        const snippet = new vscode.SnippetString(await this.run_agent_sync(params));
        // return new vscode.InlineCompletionList([{insertText: snippet}]);
        return snippet;
    }
}

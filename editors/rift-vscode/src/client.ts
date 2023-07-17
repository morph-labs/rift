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
import { ChatHelperProgress } from './types';
import delay from 'delay'
import * as tcpPortUsed from 'tcp-port-used'

let client: LanguageClient

// ref: https://stackoverflow.com/questions/40284523/connect-external-language-server-to-vscode-extension

// https://nodejs.org/api/child_process.html#child_processspawncommand-args-options

/** Creates the ServerOptions for a system in the case that a language server is already running on the given port. */
function tcpServerOptions(context: ExtensionContext, host, port): ServerOptions {
    let socket = net.connect({
        port: port, host: host
    })
    const si: StreamInfo = {
        reader: socket, writer: socket
    }
    return () => {
        return Promise.resolve(si)
    }
}

/** Creates the server options for spinning up our own server.*/
function createServerOptions(context: vscode.ExtensionContext, port): ServerOptions {
    let cwd = vscode.workspace.workspaceFolders![0].uri.path
    // [todo]: we will supply different bundles for the 3 main platforms; windows, linux, mac.
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

interface RunHelperParams {
    task: string
    position: vscode.Position
    textDocument: TextDocumentIdentifier
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


interface RunHelperResult {
    id: number
}

interface RunHelperSyncResult {
    id: number
    text: string
}

type HelperStatus = 'running' | 'done' | 'error' | 'accepted' | 'rejected'

interface RunHelperProgress {
    id: number
    textDocument: TextDocumentIdentifier
    log?: {
        severity: string;
        message: string;
    }
    cursor?: vscode.Position
    /** This is the set of ranges that the helper has added so far. */
    ranges?: vscode.Range[]
    status: HelperStatus
}

/** Represents a helper */
class Helper {
    status: HelperStatus;
    green: vscode.TextEditorDecorationType;
    ranges: vscode.Range[] = []
    onStatusChangeEmitter: vscode.EventEmitter<HelperStatus>
    onStatusChange: vscode.Event<HelperStatus>
    constructor(public readonly id: number, public readonly startPosition: vscode.Position, public textDocument: TextDocumentIdentifier) {
        this.status = 'running'
        this.green = vscode.window.createTextEditorDecorationType({ backgroundColor: 'rgba(0,255,0,0.1)' })
        this.onStatusChangeEmitter = new vscode.EventEmitter<HelperStatus>()
        this.onStatusChange = this.onStatusChangeEmitter.event
    }
    handleProgress(params: RunHelperProgress) {
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

export class HelperLens extends vscode.CodeLens {
    id: number
    constructor(range: vscode.Range, helper: Helper, command?: vscode.Command) {
        super(range, command)
        this.id = helper.id
    }
}

interface ModelConfig {
    chatModel: string
    completionsModel: string
    /** The API key for OpenAI, you can also set OPENAI_API_KEY. */
    openai_api_key?: string
}

export class MorphLanguageClient implements vscode.CodeLensProvider<HelperLens> {
    client: LanguageClient
    red: vscode.TextEditorDecorationType
    green: vscode.TextEditorDecorationType
    context: vscode.ExtensionContext
    changeLensEmitter: vscode.EventEmitter<void>
    onDidChangeCodeLenses: vscode.Event<void>
    helpers = new Map<number, Helper>()

    constructor(context: vscode.ExtensionContext) {
        this.context = context
        this.create_client()
        this.changeLensEmitter = new vscode.EventEmitter<void>()
        this.onDidChangeCodeLenses = this.changeLensEmitter.event
        // [todo] rename rift and morph/ to release name
        this.context.subscriptions.push(
            vscode.commands.registerCommand('rift.cancel', (id: number) => this.client.sendNotification('morph/cancel', { id })),
            vscode.commands.registerCommand('rift.accept', (id: number) => this.client.sendNotification('morph/accept', { id })),
            vscode.commands.registerCommand('rift.reject', (id: number) => this.client.sendNotification('morph/reject', { id })),
            vscode.workspace.onDidChangeConfiguration(this.on_config_change.bind(this)),
        )

    }

    public provideCodeLenses(document: vscode.TextDocument, token: vscode.CancellationToken): HelperLens[] {
        // this returns all of the lenses for the document.
        const items: HelperLens[] = []
        for (const helper of this.helpers.values()) {
            if (helper.textDocument.uri === document.uri.toString()) {
                const line = helper.startPosition.line
                const linetext = document.lineAt(line)
                if (helper.status === 'running') {
                    const running = new HelperLens(linetext.range, helper, {
                        title: 'running',
                        command: 'rift.cancel',
                        tooltip: 'click to stop this helper',
                        arguments: [helper.id],
                    })
                    items.push(running)
                }
                else if (helper.status === 'done' || helper.status === 'error') {
                    const accept = new HelperLens(linetext.range, helper, {
                        title: 'Accept ✅ ',
                        command: 'rift.accept',
                        tooltip: 'Accept the edits below',
                        arguments: [helper.id],
                    })
                    const reject = new HelperLens(linetext.range, helper, {
                        title: ' Reject ❌',
                        command: 'rift.reject',
                        tooltip: 'Reject the edits below and restore the original text',
                        arguments: [helper.id]
                    })
                    items.push(accept, reject)
                }
            }
        }
        return items
    }

    public resolveCodeLens(codeLens: HelperLens, token: vscode.CancellationToken) {
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
        const config = workspace.getConfiguration('rift')

        const port = config.get('engineServer.port')
        const host = config.get('engineServer.host')

        let serverOptions: ServerOptions
        while (!(await tcpPortUsed.check(port, host))) {
            console.log('waiting for server to come online')
            try {
                await tcpPortUsed.waitUntilUsedOnHost(port, host, 500, 1000000)
            }
            catch (e) {
                console.error(e)
            }
        }
        console.log(`server detected on port ${host}:${port}`)
        serverOptions = tcpServerOptions(this.context, host, port)
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


    async morph_notify(params: RunHelperProgress) {
        if (!this.is_running()) {
            throw new Error('client not running, please wait...') // [todo] better ux here.
        }
        const helper = this.helpers.get(params.id)
        if (!helper) {
            throw new Error('helper not found')
        }
        helper.handleProgress(params)
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

    async run_helper(params: RunHelperParams) {
        if (!this.client) {
            const config = workspace.getConfiguration('rift')

            const port = config.get('engineServer.port')
            const host = config.get('engineServer.host')

            throw new Error(`waiting for a connection to rift-engine, please make sure the rift-engine is running on ${host}:${port}`) // [todo] better ux here.
        }
        const result: RunHelperResult = await this.client.sendRequest('morph/run_helper', params)
        const helper = new Helper(result.id, params.position, params.textDocument)
        helper.onStatusChange(e => this.changeLensEmitter.fire())
        this.helpers.set(result.id, helper)
        // note this returns fast and then the updates are sent via notifications
        this.changeLensEmitter.fire()
        return `starting helper ${result.id}...`
    }

    async run_helper_sync(params: RunHelperParams) {
        console.log("run_helper_sync")
        const result: RunHelperSyncResult = await this.client.sendRequest('morph/run_helper_sync', params)
        const helper = new Helper(result.id, params.position, params.textDocument)
        // helper.onStatusChange(e => this.changeLensEmitter.fire())
        this.helpers.set(result.id, helper)
        // this.changeLensEmitter.fire()
        return result.text
    }

    morphNotifyChatCallback: (progress: ChatHelperProgress) => any = async function (progress) {
        throw new Error('no callback set')
    }

    async run_chat(params: RunChatParams, callback: (progress: ChatHelperProgress) => any) {
        console.log('run chat')
        this.morphNotifyChatCallback = callback
        this.client.onNotification('morph/chat_progress', this.morphNotifyChatCallback.bind(this))

        const result = await this.client.sendRequest('morph/run_chat', params)
        // note this returns fast and then the updates are sent via notifications
        return 'starting...'
    }


    dispose() {
        this.client.dispose()
    }

    async provideInlineCompletionItems(doc: vscode.TextDocument, position: vscode.Position, context: vscode.InlineCompletionContext, token: vscode.CancellationToken) {
        const params: RunHelperParams = { task: "complete the code", position: position, textDocument: TextDocumentIdentifier.create(doc.uri.toString()) };
        const snippet = new vscode.SnippetString(await this.run_helper_sync(params));
        // return new vscode.InlineCompletionList([{insertText: snippet}]);
        return snippet;
    }
}

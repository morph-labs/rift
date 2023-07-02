import * as vscode from 'vscode';
import { HelloWorldProvider } from './elements/HelloWorldProvider';
import { AgentsProvider } from './elements/AgentsProvider';
import { LogsProvider } from './elements/LogsProvider';
import { ChatProvider } from './elements/ChatProvider';


export function activate(context: vscode.ExtensionContext) {
    console.log('Congratulations, your extension "rift-vscode-svelte" is now active!');

    const chatProvider = new ChatProvider(context.extensionUri);
    const agentsProvider = new AgentsProvider(context.extensionUri);
    const logsProvider = new LogsProvider(context.extensionUri);
    const helloWorldProvider = new HelloWorldProvider(context.extensionUri);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider("chat-sidebar", chatProvider)
    );
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider("agents-sidebar", agentsProvider)
    );
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider("logs-sidebar", logsProvider)
    );
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider("hello-world-sidebar", helloWorldProvider)
    );
}

export function deactivate() { }

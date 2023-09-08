<script lang="ts">
  import SendSvg from "./icons/SendSvg.svelte"
  import { dropdownStatus, filteredAgents, filteredFiles, focusedFileIndex, starterKitConfig, state } from "./stores"
  import SlashDropdown from "./chat/dropdown/SlashDropdown.svelte"
  import {onDestroy, onMount, tick} from "svelte"
  import AtDropdown from "./chat/dropdown/AtDropdown.svelte"
  import type {AgentRegistryItem, AtableFile} from "../../src/types"
  import {CommandProps, Editor} from "@tiptap/core"
  import StarterKit from "@tiptap/starter-kit"
  import { Placeholder } from "@tiptap/extension-placeholder"
  import type { Transaction } from "@tiptap/pm/state"
  import type { SuggestionOptions } from "@tiptap/suggestion"
  import { FileChip } from "./FileChip"
    import AtIconSvg from "./icons/AtIconSvg.svelte";


  const suggestion:Omit<SuggestionOptions<AtableFile>, 'editor'> = {
  items: ({ query }) => {
            const allFiles = Array.from(
                new Set([
                    ...$state.symbols,
                    ...$state.files.recentlyOpenedFiles,
                    ...$state.files.nonGitIgnoredFiles,
                ])
            );
            const seen = new Set<string>()
            const filteredFiles = allFiles
                .filter((file) => {
                    const needle = query.toLowerCase();
                    const haystack = (
                        file.fromWorkspacePath +
                        "/" +
                        file.fileName +
                        "@" +
                        file.symbolName
                    ).toLowerCase();
                    if (seen.has(haystack)) return false;
                    seen.add(haystack);
                    return haystack.includes(needle);
                })
                .slice(0, 50);
            console.log({allFiles, filteredFiles});
            return filteredFiles;
  },
  render: () => {
    return {
      onStart: (props) => {
        dropdownStatus.set("at") 
        filteredFiles.set(props.items.map(file => ({...file, onEnter: () => {
          console.log('onEnter called :D')
          props.command(file)}}))) // add the enter command to the files
        },
        
        onUpdate(props) {
          dropdownStatus.set("at") 
          filteredFiles.set(props.items.map(file => ({...file, onEnter: () => {
          console.log('onEnter called :D')
          props.command(file)}})))
      },

      onKeyDown(props) {
        console.log('onKeyDown suggestion plugin: props:', props)
        if (props.event.key === "Escape") {
          dropdownStatus.set('none')
          return true 
          // return true
        }
        if(props.event.key == 'ArrowUp') {
          props.event.preventDefault() //necessary to prevent moving the cursor
          focusedFileIndex.update(i => (i + $filteredFiles.length - 1) % $filteredFiles.length)
        }
        
        if(props.event.key == 'ArrowDown') {
          props.event.preventDefault() //necessary to prevent moving the cursor
          focusedFileIndex.update(i => (i + 1) % $filteredFiles.length)
        }

        
        return false
      },

      onExit() {
        dropdownStatus.set('none')
        focusedFileIndex.set(0)
      },
    }
  },
}


  let isFocused = true

  let _container: HTMLDivElement | undefined

  state.subscribe((s) => {
    isFocused = s.isOmnibarFocused
    if (isFocused) {
      focus()
    }
  })

  function sendMessage() {
    console.log("sending message")
    console.log($state.agents[$state.selectedAgentId])
    if($state.agents[$state.selectedAgentId].taskWithSubtasks?.task.status === 'done' || $state.agents[$state.selectedAgentId].taskWithSubtasks?.task.status === 'error') {
      console.log('cannot send message while agent has done or error status')
      return
    }
    if ($state.agents[$state.selectedAgentId].isStreaming) {
      console.log("cannot send messages while agent is responding")
      return
    }


    console.log("chat history")
    console.log($state.agents[$state.selectedAgentId].chatHistory)

    let appendedMessages = $state.agents[$state.selectedAgentId].chatHistory
    if(!editor) throw new Error("no editor in sendMEssage() function in OmniBar.svelte")

    appendedMessages.push({ role: "user", content: editorContent })
    console.log("appendedMessages")
    console.log(appendedMessages)

    if (!$state.agents[$state.selectedAgentId]) throw new Error()

    vscode.postMessage({
      type: "chatMessage",
      agent_id: $state.selectedAgentId,
      agent_type: $state.agents[$state.selectedAgentId].type,
      messages: appendedMessages,
      message: editorContent,
    })

    resetTextarea()
  }

  function handleValueChange({ editor, transaction }: { editor: Editor; transaction: Transaction }) {
    editorContent = editor.getText()

    let newFilteredAgents: AgentRegistryItem[] = []
    if (editorContent.trim().startsWith("/")) {
      let searchString = editorContent.substring(1).toLowerCase()
      newFilteredAgents = $state.availableAgents.filter((agent) => {
        return (
          agent.agent_type.toLowerCase().includes(searchString) ||
          agent.display_name.toLowerCase().includes(searchString)
        )
      })
      filteredAgents.set(newFilteredAgents)
    } else filteredAgents.set([])

    if (editorContent.trim().startsWith("/") && newFilteredAgents.length > 0) {
      console.log("setting slash")
      dropdownStatus.set("slash")
     } else if($dropdownStatus == 'slash') {
      console.log("setting none")
      dropdownStatus.set("none")
    }
  }

  dropdownStatus.subscribe((s) => console.log("dropdownStatus!:", s))

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter") {

      if (e.shiftKey) return
      if (!editorContent) resetTextarea()
      if($dropdownStatus == 'at') {
        $filteredFiles[$focusedFileIndex].onEnter()
        return
      }
      if (!editorContent || $dropdownStatus == "slash") return
      sendMessage()
    }
  }

  function handleRunAgent(agent_type: string) {
    if (!$state.availableAgents.map((x) => x.agent_type).includes(agent_type))
      throw new Error("attempt to run unavailable agent")
    vscode.postMessage({
      type: "createAgent",
      agent_type,
    })

    resetTextarea()
    dropdownStatus.set("none")
  }

  let onFocus = async () => {
    isFocused = true
    vscode.postMessage({
      type: "sendHasNotificationChange",
      agentId: $state.selectedAgentId,
      hasNotification: false,
    })
    vscode.postMessage({
      type: "focusOmnibar",
    })
  }

  let onBlur = () => {
    isFocused = false
    vscode.postMessage({
      type: "blurOmnibar",
    })
  }

  const focus = () => editor?.commands.focus()
  const blur = () => editor?.commands.blur()

  function resetTextarea() {
    console.log('resetting text area')
    editor?.commands.clearContent()
  }

  function disableDefaults(event: Event) {
    const e = event as KeyboardEvent

    
    if (e.code === "Enter" && $dropdownStatus != "none") event.preventDefault()
  }

let editor: Editor | undefined

  onMount(() => {
    editor = new Editor({
      element: _container,
      extensions: [
        StarterKit.configure(starterKitConfig),
        FileChip.configure({
          HTMLAttributes: {

          },
          suggestion
        }),
        Placeholder.configure({
          emptyEditorClass: "is-editor-empty",
          placeholder: "Type to chat or hit / for commands",
        }),
      ],
      editorProps: {
        attributes: {
          class: "outline-none focus:outline-none max-h-40 overflow-auto",
        },
      },
      content: "",
      onTransaction: (props) => {
        editor = editor
      },
      onFocus,
      onBlur,
      onUpdate: handleValueChange,
      onSelectionUpdate: (props) => {
      },
    })

    const editorRootElement = document.querySelector(".ProseMirror")
    if (!editorRootElement) throw new Error()
    editorRootElement.addEventListener("keydown", disableDefaults, true)
  })

  onDestroy(() => {
    const editorRootElement = document.querySelector(".ProseMirror")
    editorRootElement?.removeEventListener("keydown", disableDefaults, true)
    editor?.destroy()
  })

  let editorContent = ""
  $: {
    editorContent = editor?.getText() ?? ""
  }
</script>

<div class="p-2 border-t border-b border-[var(--vscode-input-background)] w-full relative flex flex-row">
  <div class="justify-self-end flex mr-1">
    <button on:click={() => editor?.chain().focus().insertContent('@').run()}
      class="items-center flex"
      title="Add reference to file or symbol"
      >
      <AtIconSvg />
    </button>
  </div>
  <div
    class={`w-full text-md p-2 bg-[var(--vscode-input-background)] rounded-md flex flex-row items-center border ${
      isFocused ? "border-[var(--vscode-focusBorder)]" : "border-transparent"
    }`}>
    <div
      id="omnibar"
      class="w-full bg-transparent resize-none overflow-visible hide-scrollbar max-h-40"
      bind:this={_container}
      on:keydown={handleKeyDown} />
    <div class="justify-self-end flex">
      <button on:click={sendMessage} class="items-center flex">
        <SendSvg />
      </button>
    </div>
  </div>
  {#if $dropdownStatus == "slash"}
    <SlashDropdown {handleRunAgent} />
  {/if}

  {#if $dropdownStatus == "at"}
    <AtDropdown />
  {/if}
</div>

<style>
  .hide-scrollbar::-webkit-scrollbar {
    display: none;
  }

  .hide-scrollbar {
    -ms-overflow-style: none; /* IE and Edge */
    scrollbar-width: none; /* Firefox */
  }

  :global(.ProseMirror p.is-editor-empty:first-child::before) {
    color: #adb5bd;
    content: attr(data-placeholder);
    float: left;
    height: 0;
    pointer-events: none;
  }
</style>

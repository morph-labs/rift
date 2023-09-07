<script lang="ts">
  import { onDestroy, onMount } from "svelte"
  import RiftSvg from "../icons/RiftSvg.svelte"
  import { copySvg } from "../icons/copySvg"
  export let value = ""
  // export let isNew = false;
  import { state } from "../stores"
  import CopySvg from "../icons/CopySvg.svelte"
  import { SvelteComponent } from "svelte"
  import showdown from "showdown"
  import morphdom from "morphdom"

  export let last = false
  export let scrollToBottomIfNearBottom: ((...args: any) => any) | undefined = undefined

  $: currentAgent = $state.availableAgents.filter(
    (agent) => agent.agent_type === $state.agents[$state.selectedAgentId]?.type
  )[0]

  let responseBlock: HTMLDivElement | undefined
  const converterOptions: showdown.ConverterOptions = {
    omitExtraWLInCodeBlocks: true,
    simplifiedAutoLink: true,
    excludeTrailingPunctuationFromURLs: true,
    literalMidWordUnderscores: true,
    simpleLineBreaks: true,
    openLinksInNewWindow: true,
  }
  const converter = new showdown.Converter(converterOptions)

  function textToFormattedHTML(text: string) {
    function fixCodeBlocks(response: string) {
      // Use a regular expression to find all occurrences of the substring in the string
      const REGEX_CODEBLOCK = new RegExp("```", "g")
      const matches = response.match(REGEX_CODEBLOCK)

      // Return the number of occurrences of the substring in the response, check if even
      const count = matches ? matches.length : 0
      if (count % 2 === 0) {
        return response
      } else {
        // else append ``` to the end to make the last code block complete
        return response.concat("\n```")
      }
    }

    const fixedCodeBlocksText = fixCodeBlocks(text)
    // console.log(`Original text: ${text}`);
    const text2 = fixedCodeBlocksText.replace(/\n{2,}/g, (m) => m.replace(/\n/g, "<br/>"))
    const text3 = text2.replace(/<br\/>([^<])/g, "<br/>\n\n$1")
    text = converter.makeHtml(text3)
    // console.log(`Converted text: ${text}`);
    return text
  }
  let something: string


  const preblockToCopyContent: string[] = []
  $: {
    const getHTML = (_responseBlock: HTMLDivElement) => {
      const responseBlock = _responseBlock.cloneNode(true) as HTMLDivElement
      responseBlock.innerHTML = textToFormattedHTML(value.trim())
      responseBlock.querySelectorAll("code").forEach((node) => node.classList.add("code"))
      responseBlock.querySelectorAll("pre").forEach((preblock, i) => {
        // index = i
        preblock.classList.add("p-2", "my-2", "block", "overflow-x-scroll")
        //remove prev copy buttons
        preblock.querySelectorAll("#copy").forEach((copy) => copy.parentElement?.removeChild(copy))
        const copyContent = preblock.textContent
        const copyButton = document.createElement("button")
        copyButton.id = "copy"
        copyButton.className = "flex text-sm py-1 mb-1 text-[var(--vscode-panelTitle-inactiveForeground)]"
        copyButton.appendChild(copySvg())
        const copyCodeWords = document.createElement("p")
        copyCodeWords.innerText = " copy"
        copyButton.appendChild(copyCodeWords)
        preblockToCopyContent[i] = preblock.textContent ?? ""
        copyButton.addEventListener("click", () => {

          vscode.postMessage({
            type: "copyText",
            content: preblockToCopyContent[i],
          })
        })
        preblock.insertBefore(copyButton, preblock.firstChild)
      })
      return responseBlock
    }
    if (responseBlock) {
      const dsfa = responseBlock.innerHTML
      const time = Date.now()
      const newHTML = getHTML(responseBlock)
      console.log(`gen latency is ${Date.now() - time}`)
      something = newHTML.innerHTML
      morphdom(responseBlock, newHTML)
      responseBlock.contentEditable = "false"
      scrollToBottomIfNearBottom?.()
    }
  }
</script>

<div id={last ? "last" : undefined} class="w-full pr-2 p-[10px]">
  <div
    class={`flex items-center pb-[6px] ${
      value == "" && !$state.agents[$state.selectedAgentId].isStreaming ? "hidden" : ""
    }`}>
    <div class="flex items-center justify-center h-[16px] w-[16px] mr-2 scale-125">
      {#if currentAgent.agent_icon}
        {@html currentAgent.agent_icon}
      {:else}
        <RiftSvg />
      {/if}
    </div>
    <p class="text-sm font-semibold">
      {currentAgent.display_name.toUpperCase()}
    </p>
  </div>
  <div
    class={`w-full text-md focus:outline-none flex flex-row ${
      value === "" && !$state.agents[$state.selectedAgentId].isStreaming ? "hidden" : ""
    }`}>
    <div
      contenteditable="true"
      bind:this={responseBlock}
      bind:innerHTML={something}
      id="response"
      class="w-full text-md focus:outline-none" />
  </div>
</div>

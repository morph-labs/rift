<script lang="ts">
  import { onDestroy, onMount, tick } from "svelte";
  import { state, dropdownStatus } from "../stores";
  import UserSvg from "../icons/oldUserSvg.svelte";
  import UserInput from "./UserInput.svelte";
  import Response from "./Response.svelte";
  import OmniBar from "../OmniBar.svelte";

  let observer: MutationObserver;
  let chatWindow: HTMLDivElement;
  let fixedToBottom: boolean;
  dropdownStatus.subscribe((s) => console.log("dropdownStatus!:", s));

  function scrollToBottomIfNearBottom() {
    if (fixedToBottom) chatWindow.scrollTo(0, chatWindow.scrollHeight);
  }

  $: {
    console.log("change");
    console.log(typeof chatWindow);
  }

  onMount(async () => {
    console.log("awaiting tick");
    await tick();
    chatWindow.scrollTo(0, chatWindow.scrollHeight);

    observer = new MutationObserver(scrollToBottomIfNearBottom);
    observer.observe(chatWindow, { childList: true, subtree: true });

    fixedToBottom = true;

    chatWindow.addEventListener("scroll", function () {
      if (!chatWindow.scrollTop || !chatWindow.scrollHeight) {
        console.log(chatWindow);
        console.log(chatWindow.scrollTop);
        console.log(chatWindow.scrollHeight);
        throw new Error();
      }
      fixedToBottom = Boolean(
        chatWindow.clientHeight + chatWindow.scrollTop >=
          chatWindow.scrollHeight - 10,
      );
    });
  });
  onDestroy(() => {
    observer.disconnect();
  });

  // state.subscribe(s => console.log('new webview state: ', s))
</script>

<div
  bind:this={chatWindow}
  class={`flex items-start flex-grow flex-col overflow-y-auto ${
    $dropdownStatus != "none" ? "opacity-10" : ""
  }`}
>
  {#if $state.agents[$state.selectedAgentId]?.inputRequest}
    <Response
      value={$state.agents[$state.selectedAgentId]?.inputRequest?.msg}
    />
  {:else}
    {#each $state.agents[$state.selectedAgentId]?.chatHistory ?? [] as item}
      {#if item.role == "user"}
        <UserInput value={item.content} />
      {:else}
        <Response value={item.content} />
      {/if}
    {/each}
    {#if $state.selectedAgentId in $state.agents && $state.agents[$state.selectedAgentId].isStreaming}
      <Response
        value={$state.agents[$state.selectedAgentId].streamingText}
        {scrollToBottomIfNearBottom}
        last={true}
      />
    {/if}
  {/if}
</div>

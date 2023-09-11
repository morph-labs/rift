<script lang="ts">
  import LogGreenSvg from "../icons/LogGreenSvg.svelte";
  import LogYellow from "../icons/LogYellowSvg.svelte";
  import LogRed from "../icons/LogRedSvg.svelte";
  import ArrowRightSvg from "../icons/ArrowRightSvg.svelte";
  import ArrowDownSvg from "../icons/ArrowDownSvg.svelte";
  import ChatSvg from "../icons/ChatSvg.svelte";
  import EllipsisSvg from "../icons/EllipsisSvg.svelte";
  import EllipsisDarkSvg from "../icons/EllipsisDarkSvg.svelte";
  import Log from "./Log.svelte";
  import { state } from "../stores";
  import type { WebviewState } from "../../../src/types";

  let expanded = true;
  export let id: string = "";
  export let name: string = "rift_chat";
  export let hasNotification = false;

  $: isSelected = id == $state.selectedAgentId;

  let subtasks = $state.agents[id].taskWithSubtasks?.subtasks;
  $: subtasks = $state.agents[id].taskWithSubtasks?.subtasks;

  let doneAgent = false;

  let isDropdownOpen = false; // default state (dropdown close)

  const handleDropdownClick = (event: MouseEvent) => {
    event.stopPropagation();
    isDropdownOpen = !isDropdownOpen; // togle state on click
  };

  const handleDropdownFocusLoss = ({
    relatedTarget,
    currentTarget,
  }: FocusEvent) => {
    // use "focusout" event to ensure that we can close the dropdown when clicking outside or when we leave the dropdown with the "Tab" button
    const ct = currentTarget as HTMLElement;

    if (relatedTarget instanceof HTMLElement && ct.contains(relatedTarget))
      return; // check if the new focus target doesn't present in the dropdown tree (exclude ul\li padding area because relatedTarget, in this case, will be null)
    isDropdownOpen = false;
  };

  const handleChatIconClick = (e: MouseEvent) => {
    vscode.postMessage({ type: "selectedAgentId", agentId: id });
  };

  const handleCancelAgent = (e: MouseEvent) => {
    vscode.postMessage({ type: "cancelAgent", agentId: id });
  };
  const handleDeleteAgent = (e: MouseEvent) => {
    vscode.postMessage({ type: "deleteAgent", agentId: id });
  };
</script>

<button
  on:click={handleChatIconClick}
  class:bg-[var(--vscode-editor-hoverHighlightBackground)]={isSelected}
  class="w-full py-2"
>
  <div class="flex">
    {#if expanded == false}
      <button
        class="py-2 px-4 w-[16px] h-[16px]"
        on:click={() => (expanded = !expanded)}
        on:keydown={() => (expanded = !expanded)}
      >
        <ArrowRightSvg />
      </button>
    {:else}
      <button
        class="py-2 px-4 w-[16px] h-[16px]"
        on:click={() => (expanded = !expanded)}
        on:keydown={() => (expanded = !expanded)}
      >
        <ArrowDownSvg />
      </button>
    {/if}
    <div class="flex w-full select-none items-center">
      <div class="flex">
        {#if $state.agents[id].taskWithSubtasks?.task.status == "done"}
          <div class="mx-1 mt-0.5"><LogGreenSvg /></div>
        {:else if $state.agents[id].taskWithSubtasks?.task.status == "running"}
          <div class="mx-1 mt-0.5"><LogYellow /></div>
        {:else}
          <div class="mx-1 mt-0.5"><LogRed /></div>
        {/if}
        <div>{name}</div>
      </div>
      <div
        class="relative w-fit mr-2 mt-0 ml-auto flex hover:text-[var(--vscode-list-hoverBackground)]"
      >
        {#if $state.agents[id].chatHistory.length > 0 && hasNotification}
          <div
            class="absolute bottom-auto left-auto right-0 top-0 z-10 inline-block -translate-y-1/2 translate-x-2/4 rotate-0 skew-x-0 skew-y-0 scale-x-50 scale-y-50 rounded-full bg-pink-700 p-2.5 text-xs"
          />
        {/if}
        {#if $state.agents[id].chatHistory.length > 0}
          <ChatSvg />
        {/if}
      </div>
    </div>

    <div class="dropdown left-auto flex">
      <div class="flex items-center">
        <div class="dropdown flex" on:focusout={handleDropdownFocusLoss}>
          <button class="btn pb-2.5 pt-2" on:click={handleDropdownClick}>
            {#if isDropdownOpen}
              <div class="px-2"><EllipsisDarkSvg /></div>
            {:else}
              <div class="px-2"><EllipsisSvg /></div>
            {/if}
          </button>

          <ul
            style="left: auto !important;
                        right: 0px !important;
                        opacity: 1 !important ;z-index: 99; background-color: var(--vscode-input-background);"
            class="dropdown-content absolute menu shadow rounded-box"
            style:visibility={isDropdownOpen ? "visible" : "hidden"}
          >
            <li class="list-item">
              <button class="btn px-2 text-left" on:click={handleCancelAgent}
                >Cancel</button
              >
            </li>
            <!-- cant delete the last agent bc no ui for it -->
            {#if Object.values($state.agents).filter((agent) => agent.isDeleted == false).length > 1}
              <li class="list-item text-left">
                <button class="btn px-2" on:click={handleDeleteAgent}
                  >Delete</button
                >
              </li>
            {/if}
          </ul>
        </div>
      </div>
    </div>
  </div>
  <div class="border-l ml-6 my-2 space-y-2" hidden={!expanded}>
    {#if subtasks}
      {#each subtasks as subtask}
        <Log {subtask} />
      {/each}
    {/if}
  </div>
</button>

<style>
  .list-item:hover {
    background-color: var(--vscode-button-secondaryHoverBackground);
  }
</style>

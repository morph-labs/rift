<script lang="ts">
  import { state, filteredAgents } from "../../stores";
  import type { AgentRegistryItem } from "../../../../src/types";
  import SlashDropdownCard from "./SlashDropdownCard.svelte";
  import { onMount } from "svelte";
  import type { WebviewState } from "../../../../src/types";

  export let handleRunAgent: (agent_type: string) => void;

  let availableAgents: AgentRegistryItem[] = $state.availableAgents;

  let activeId = $filteredAgents.length - 1;
  $: activeId = $filteredAgents.length - 1;

  onMount(() => {
    //response is saved to state in ChatWebview.svelte
    vscode.postMessage({ type: "listAgents" });
  });

  console.log("in dropdown: ", availableAgents);

  if (availableAgents.length < 1) throw new Error("no available agents");

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter") {
      e.preventDefault();
      // create agent
      console.log("fa:", $filteredAgents);
      console.log("activeId:", activeId);
      console.log("agent_type: " + $filteredAgents[activeId].agent_type);

      handleRunAgent($filteredAgents[activeId].agent_type);
    }
    if (e.key == "ArrowDown") {
      e.preventDefault();
      if (activeId == $filteredAgents.length - 1) activeId = 0;
      else activeId++;
      console.log("new active Id: ", activeId);
    } else if (e.key == "ArrowUp") {
      e.preventDefault();
      if (activeId == 0) activeId = $filteredAgents.length - 1;
      else activeId--;
      console.log("new active Id: ", activeId);
    } else return;
  }
</script>

<svelte:window on:keydown={handleKeyDown} />

<div
  class="absolute bottom-full left-0 pr-2 pl-6 w-full z-20 drop-shadow-[0_-4px_16px_0px_rgba(0,0,0,0.36)] max-h-64 overflow-auto"
>
  <div
    class="border border-[var(--vscode-gitDecoration-ignoredResourceForeground)]"
  >
    {#each $filteredAgents as agent, index}
      <SlashDropdownCard
        {agent}
        focused={index === activeId}
        {handleRunAgent}
      />
    {/each}
  </div>
</div>

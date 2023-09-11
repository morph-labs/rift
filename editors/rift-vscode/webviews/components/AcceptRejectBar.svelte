<script lang="ts">
  // This module handles the Accept and Reject actions in the UI.
  import Check from "./icons/AcceptRejectCheck.svelte"; // Importing the Check icon component
  import Close from "./icons/AcceptRejectClose.svelte"; // Importing the Close icon component
  import { state } from "./stores"; // Importing the state from the stores

  // This function handles the Accept action.
  function handleAccept() {
    // Post a message to vscode with the type of action (accept or reject), the selected agent ID, and the action (accept: true)
    vscode.postMessage({
      type: "acceptOrReject",
      agentId: $state.selectedAgentId,
      doesAccept: true,
    });
  }

  // This function handles the Reject action.
  function handleReject() {
    // Post a message to vscode with the type of action (accept or reject), the selected agent ID, and the action (reject: false)
    vscode.postMessage({
      type: "acceptOrReject",
      agentId: $state.selectedAgentId,
      doesAccept: false,
    });
  }
</script>

<div class="flex flex-row w-full bg-[var(--vscode-input-background)] p-1 px-2">
  <button
    on:click={handleAccept}
    class="flex bg-[#37662A] justify-center items-center flex-grow p-1 rounded-l-md"
    >Accept<Check /></button
  >
  <button
    on:click={handleReject}
    class="flex bg-[#763B22] justify-center items-center flex-grow p-1 rounded-r-md"
    >Reject<Close /></button
  >
</div>

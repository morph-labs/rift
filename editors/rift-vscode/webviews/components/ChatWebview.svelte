<script lang="ts">
  import AcceptRejectBar from "./AcceptRejectBar.svelte";
  import Chat from "./chat/Chat.svelte";
  import OmniBar from "./OmniBar.svelte";
  import { state } from "./stores";

  vscode.postMessage({ type: "refreshState" });
</script>

<div class="h-screen flex flex-col">
  <Chat />
  {#if $state.agents[$state.selectedAgentId]?.doesShowAcceptRejectBar}
    <AcceptRejectBar />
  {/if}
  <OmniBar />
</div>

<script lang="ts">
  // This script imports the Logs component and posts a message to vscode to refresh the state.

  // Import the Logs component.
  import Logs from "./logs/Logs.svelte";
  // Post a message to vscode to refresh the state.
  vscode.postMessage({ type: "refreshState" });
</script>

<!-- This is the main LogsWebview component. It displays the Logs component. -->
<div class="">
  <div>
    <!-- Display the Logs component. -->
    <Logs />
  </div>
</div>

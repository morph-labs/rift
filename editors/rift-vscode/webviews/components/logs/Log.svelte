<script lang="ts">
  import { createEventDispatcher, onMount } from "svelte";
  import LogGreenSvg from "../icons/LogGreenSvg.svelte";
  import LogYellow from "../icons/LogYellowSvg.svelte";
  import LogRed from "../icons/LogRedSvg.svelte";
  import { state } from "../stores";
  export let subtask: { description: string; status: string };

  const dispatch = createEventDispatcher();
</script>

<div class="flex select-none">
  {#if subtask.status == "done"}
    <div class="ml-4 mr-2 mt-0.5"><LogGreenSvg /></div>
  {:else if subtask.status == "running" || subtask.status == "scheduled"}
    <div class="ml-4 mr-2 mt-0.5"><LogYellow /></div>
  {:else}
    <div class="ml-4 mr-2 mt-0.5"><LogRed /></div>
  {/if}
  {(
    subtask.description.substring(0, 40).charAt(0).toUpperCase() +
    subtask.description.substring(0, 40).slice(1).toLowerCase()
  ).replaceAll("_", " ")}
</div>

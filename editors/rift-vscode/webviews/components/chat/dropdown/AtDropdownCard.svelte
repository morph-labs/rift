<script lang="ts">
  import CodeIcon from "../../icons/CodeIconSvg.svelte";
  import FileIcon from "../../icons/FileIconSvg.svelte";

  export let type: "symbol" | "file" = "file";
  export let focused: boolean = false;
  export let onClick: (...args: any) => any;
  export let displayName = "example.ts";
  export let description = "/project/server/example.ts";

  let ref: HTMLDivElement | undefined = undefined;

  $: {
    if (focused && ref) {
      ref.scrollIntoView({ block: "center" });
    }
  }
</script>

<div bind:this={ref} class="bg-[var(--vscode-editor-background)]">
  <!-- me and my homies love a11y -->
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <div
    class={`flex flex-col hover:cursor-pointer pl-2 py-2
    ${
      focused
        ? "bg-[var(--vscode-editor-hoverHighlightBackground)]"
        : "bg-[var(--vscode-editor-background)] hover:bg-[var(--vscode-list-hoverBackground)]"
    }`}
    on:click={onClick}
  >
    <div class="flex flex-row ml-[6px] items-center truncate overflow-hidden">
      <div class="mr-[3px]">
        {#if type === "file"}
          <FileIcon />
        {:else}
          <CodeIcon />
        {/if}
      </div>
      {displayName}
    </div>
    <div
      class="text-[var(--vscode-gitDecoration-ignoredResourceForeground)] truncate overflow-hidden ml-[6px]"
    >
      {description}
    </div>
  </div>
</div>

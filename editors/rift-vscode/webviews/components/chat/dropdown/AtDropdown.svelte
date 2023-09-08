<script lang="ts">
  import { filteredFiles, focusedFileIndex } from "../../stores";
  import AtDropdownCard from "./AtDropdownCard.svelte";
</script>

<div
  class="absolute bottom-full left-0 pr-2 pl-6 w-full z-20 drop-shadow-[0_-4px_16px_0px_rgba(0,0,0,0.36)] max-h-96 overflow-auto"
>
  <div
    class="border border-[var(--vscode-gitDecoration-ignoredResourceForeground)]"
  >
    {#each $filteredFiles as file, index}
      <AtDropdownCard
        focused={index === $focusedFileIndex}
        type={file.symbolName ? 'symbol' : 'file'}
        displayName={file.symbolName || file.fileName}
        description={file.fromWorkspacePath}
        onClick={file.onEnter}
      />
    {/each}
  </div>
</div>

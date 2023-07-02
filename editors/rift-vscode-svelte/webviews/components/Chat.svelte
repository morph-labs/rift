<script lang="ts">
    import { onMount } from "svelte";

    let message = ""

    onMount(async () => {
        message = tsvscode.getState();
    });

    let count = 0;

    function handleClick() {
        count += 1;
    }
</script>

<div>
    <h1>Another</h1>
    <button on:click={handleClick}>
        clicks: {count}
    </button>
    <p>{message}</p>
</div>

<script lang="ts">
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
</div>

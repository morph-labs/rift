<script lang="ts">
  import { onMount } from "svelte";
  import UserSvg from "../icons/UserSvg.svelte";
  import { Editor } from "@tiptap/core";
  import StarterKit from "@tiptap/starter-kit";
  import { FileChip } from "../FileChip";
  import { starterKitConfig } from "../stores";

  export let value: string;

  // TODO Pass in message and add parsing function???

  let textarea: HTMLDivElement; //used to be a textarea
  let editor: Editor | undefined;

  function rawTextToLiteralHTML(raw: string): string {
    const p = document.createElement("p");
    p.innerText = raw.trim();
    return p.innerHTML;
  }

  function parseProseMirrorHTMLfromMessageContent(message: string) {
    const literalHTML = rawTextToLiteralHTML(message);
    const regex = /\[(.*?)\]\((.*?)\)/g;
    const chipped = literalHTML.replace(regex, (match, uri, path) => {
      const filename: string = path.split("/").pop();
      const symbolname = filename.includes("#")
        ? filename.slice(filename.indexOf("#") + 1)
        : "";
      return `<span data-type="filechip" data-fullpath="${path}" data-filename="${filename}" data-symbolname="${symbolname}"></span>`;
    });
    return chipped;
  }

  const editorContent = parseProseMirrorHTMLfromMessageContent(value);

  onMount(() => {
    editor = new Editor({
      element: textarea,
      extensions: [StarterKit.configure(starterKitConfig), FileChip],
      editable: false,
      editorProps: {
        attributes: {
          class: "outline-none focus:outline-none max-h-40 overflow-auto",
        },
      },
      // content: `<span data-type="filechip" data-filename="example.ts" data-fullpath="path/to/example.ts"></span>`,
      content: editorContent,
      // content: `<span type="filechip" data-filename="uri" data-fullpath="/Users/brentburdick/Dev/test/nested/nothing.js"></span>`,
      onTransaction: (props) => {
        // force re-render so `editor.isActive` works as expected
        editor = editor;
      },
    });
  });
</script>

<div class="bg-[var(--vscode-input-background)] w-full p-[10px]">
  <div class="flex items-center pb-[6px]">
    <UserSvg classes="mr-2" />
    <p class="text-sm font-semibold">YOU</p>
  </div>
  <div class="text-md flex flex-row items-center">
    <div bind:this={textarea} contenteditable="false"></div>
  </div>
</div>

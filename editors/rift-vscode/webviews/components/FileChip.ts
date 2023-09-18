import { Node } from "@tiptap/core";
import { PluginKey } from "@tiptap/pm/state";
import Suggestion, { SuggestionOptions } from "@tiptap/suggestion";

export const MentionPluginKey = new PluginKey("filechip");

type FileChipOptions = {
  HTMLAttributes: Record<string, any>;
  suggestion: Omit<SuggestionOptions, "editor">;
};

/**
 * FileChip class represents a file chip in the editor.
 * It provides functionalities for creating and manipulating file chips.
 */
export const FileChip = Node.create<FileChipOptions>({
  name: "filechip",

  /**
   * addOptions method is used to add options to the FileChip.
   * It returns an object containing the options.
   */
  addOptions() {
    return {
      HTMLAttributes: {},
      suggestion: {
        char: "@",
        pluginKey: MentionPluginKey,
        command: ({ editor, range, props }) => {
          // increase range.to by one when the next node is of type "text"
          // and starts with a space character
          const nodeAfter = editor.view.state.selection.$to.nodeAfter;
          const overrideSpace = nodeAfter?.text?.startsWith(" ");

          if (overrideSpace) {
            range.to += 1;
          }

          editor
            .chain()
            .focus()
            .insertContentAt(range, [
              {
                type: this.name,
                attrs: props,
              },
              {
                type: "text",
                text: " ",
              },
            ])
            .run();

          window.getSelection()?.collapseToEnd();
        },
        allow: ({ state, range }) => {
          const $from = state.doc.resolve(range.from);
          const type = state.schema.nodes[this.name];
          const allow = !!$from.parent.type.contentMatch.matchType(type);

          return allow;
        },
      },
    };
  },

  group: "inline",

  inline: true,

  selectable: false,

  atom: true,

  /**
   * addAttributes method is used to add attributes to the FileChip.
   * It returns an object containing the attributes.
   */
  addAttributes() {
    return {
      fileName: {
        default: null,
        parseHTML: (element) => element.getAttribute("data-filename"),
        renderHTML: (attributes) => {
          if (!attributes.fileName) {
            return {};
          }

          return {
            "data-filename": attributes.fileName,
          };
        },
      },
      symbolName: {
        default: null,
        parseHTML: (element) => element.getAttribute("data-symbolname"),
        renderHTML: (attributes) => {
          if (!attributes.symbolName) {
            return undefined;
          }

          return {
            "data-symbolname": attributes.symbolName,
          };
        },
      },
      fullPath: {
        default: null,
        parseHTML: (element) => element.getAttribute("data-fullpath"),
        renderHTML: (attributes) => {
          if (!attributes.fullPath) {
            return {};
          }

          return {
            "data-fullpath": attributes.fullPath,
          };
        },
      },
    };
  },

  /**
   * parseHTML method is used to parse the HTML of the FileChip.
   * It returns an array containing the parsed HTML.
   */
  parseHTML() {
    return [
      {
        tag: `span[data-type="${this.name}"]`,
      },
    ];
  },

  /**
   * renderHTML method is used to render the HTML of the FileChip.
   * It takes an object containing a node and HTMLAttributes as parameter.
   * It returns a span element containing the rendered HTML.
   */
  renderHTML({ node, HTMLAttributes }) {
    function createFileSvg() {
      const xmlns = "http://www.w3.org/2000/svg";

      let svg = document.createElementNS(xmlns, "svg");
      svg.setAttribute("width", "10");
      svg.setAttribute("height", "10");
      svg.setAttribute("fill", "none");
      svg.setAttribute("xmlns", xmlns);

      let g = document.createElementNS(xmlns, "g");
      svg.appendChild(g);

      let path = document.createElementNS(xmlns, "path");
      let defs = document.createElementNS(xmlns, "defs");
      svg.appendChild(defs);

      let clipPath = document.createElementNS(xmlns, "clipPath");
      let rect = document.createElementNS(xmlns, "rect");

      if (!node.attrs.symbolName) {
        svg.setAttribute("viewBox", "0 0 10 10");
        path.setAttribute(
          "d",
          "M1.4248 1.22499C1.4248 1.0084 1.51085 0.800676 1.664 0.647521C1.81716 0.494367 2.02488 0.408325 2.24147 0.408325H6.31745C6.53403 0.408371 6.74172 0.494443 6.89484 0.647609L8.53552 2.28829C8.68869 2.44141 8.77476 2.6491 8.7748 2.86568V8.57499C8.7748 8.79158 8.68876 8.99931 8.53561 9.15246C8.38245 9.30562 8.17473 9.39166 7.95814 9.39166H2.24147C2.02488 9.39166 1.81716 9.30562 1.664 9.15246C1.51085 8.99931 1.4248 8.79158 1.4248 8.57499V1.22499ZM2.24147 1.02083C2.18732 1.02083 2.13539 1.04234 2.0971 1.08062C2.05882 1.11891 2.0373 1.17084 2.0373 1.22499V8.57499C2.0373 8.62914 2.05882 8.68107 2.0971 8.71936C2.13539 8.75765 2.18732 8.77916 2.24147 8.77916H7.95814C8.01229 8.77916 8.06422 8.75765 8.10251 8.71936C8.14079 8.68107 8.1623 8.62914 8.1623 8.57499V3.47083H6.52897C6.31238 3.47083 6.10466 3.38478 5.9515 3.23163C5.79835 3.07847 5.7123 2.87075 5.7123 2.65416V1.02083H2.24147ZM6.3248 1.02083V2.65416C6.3248 2.70831 6.34631 2.76024 6.3846 2.79853C6.42289 2.83681 6.47482 2.85833 6.52897 2.85833H8.1623C8.16051 2.8067 8.1392 2.75767 8.10269 2.72113L6.462 1.08044C6.42546 1.04393 6.37643 1.02262 6.3248 1.02083Z",
        );
        rect.setAttribute("width", "9.8");
        rect.setAttribute("height", "9.8");
        rect.setAttribute("fill", "white");
        rect.setAttribute("transform", "translate(0.200195)");
        clipPath.setAttribute("id", "clip0_511_33161");
        g.setAttribute("clip-path", "url(#clip0_511_33161)");
      } else {
        svg.setAttribute("viewBox", "0 0 24 24");
        path.setAttribute(
          "d",
          "M7 8L3 11.6923L7 16M17 8L21 11.6923L17 16M14 4L10 20",
        );
        path.setAttribute("stroke-width", "2");
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("stroke-linejoin", "round");
      }

      defs.appendChild(clipPath);
      path.setAttribute("fill", "#CCCCCC");
      path.setAttribute("stroke", "#CCCCCC");
      g.appendChild(path);
      clipPath.appendChild(rect);

      return svg;
    }
    const span = document.createElement("span");

    span.classList.add(
      "bg-[var(--vscode-editor-background)]",
      "text-xs",
      "inline-flex",
      "flex-row-reverse",
      "items-center",
      "h-[1.5rem]",
      "border",
      "border-[var(--vscode-minimapSlider-hoverBackground,#656565)]",
      "px-1",
      "rounded",
    );
    span.append(
      document.createTextNode(
        `${node.attrs.symbolName || node.attrs.fileName}`,
      ),
    );
    // TipTap cursor logic gets weird when the first element in the node is an svg?
    // Flex-Reverse magic to fix.
    const svg = createFileSvg();
    svg.classList.add("mr-[2px]");
    span.append(svg);

    return ["span", HTMLAttributes, span];
  },

  /**
   * renderText method is used to render the text of the FileChip.
   * It takes an object containing a node as parameter.
   * It returns a string containing the rendered text.
   */
  renderText({ node }) {
    return `[uri](${node.attrs.fullPath})`;
  },

  /**
   * addKeyboardShortcuts method is used to add keyboard shortcuts to the FileChip.
   * It returns an object containing the keyboard shortcuts.
   */
  addKeyboardShortcuts() {
    return {
      Backspace: () =>
        this.editor.commands.command(({ tr, state }) => {
          let isMention = false;
          const { selection } = state;
          const { empty, anchor } = selection;

          if (!empty) {
            return false;
          }

          state.doc.nodesBetween(anchor - 1, anchor, (node, pos) => {
            if (node.type.name === this.name) {
              isMention = true;
              tr.insertText(
                this.options.suggestion.char || "",
                pos,
                pos + node.nodeSize,
              );

              return false;
            }
          });

          return isMention;
        }),
    };
  },

  /**
   * addProseMirrorPlugins method is used to add ProseMirror plugins to the FileChip.
   * It returns an array containing the ProseMirror plugins.
   */
  addProseMirrorPlugins() {
    return [
      Suggestion({
        editor: this.editor,
        ...this.options.suggestion,
      }),
    ];
  },
});

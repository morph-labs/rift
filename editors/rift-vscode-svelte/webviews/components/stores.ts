import { writable } from 'svelte/store'
import type { SvelteStore } from '../../src/types'

export const DEFAULT_STATE: SvelteStore = {
  currentlySelectedAgentId: 'rift-chat',
  agents: {
    "rift-chat": {
      chatHistory: [{role: "assistant", content: "How can I help?"}],
      logs: [],
      description: 'ask me anything ab life bro'
    },
    "aider": {
      chatHistory: [{role: "assistant", content: "How can I aid?"}],
      logs: [],
      description: 'congrats ur now a 10x engineer'
    }
  }
};


export const state = writable<SvelteStore>(DEFAULT_STATE)
export const loading = writable(false)
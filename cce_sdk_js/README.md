# cce-client

JavaScript client and offline engine for the [Context Compression Engine](https://github.com/kareem1207/context-compression-engine): memory for AI assistants that turns long conversations into topic notes and sends a model only the context it needs.

Two parts, no dependencies, Node 18+ and browsers:

```js
// 1) Talk to the CCE REST API (run it with: pip install "context-compression-engine[rest,neural]" && cce-rest)
import { CCEClient } from "cce-client";
const cce = new CCEClient({ baseUrl: "http://localhost:9100", apiKey: process.env.CCE_API_KEY });
await cce.addMessages("chat-1", messages);
const { context } = await cce.getContext("chat-1", "what did we decide about billing?");

// 2) Or run a lightweight engine in-process, offline (also what the live demo uses)
import { createEngine } from "cce-client/lite";
const out = createEngine().process(messages, "what did we decide about billing?");
console.log(out.stats.savedPercent, out.payload.toMessages());
```

The offline engine matches on shared words (no neural model), so it works best on keyword-rich text. The Python engine with the neural embedder understands paraphrase.

- Live demo: https://kareem1207.github.io/context-compression-engine/
- Benchmarks and their limits: https://github.com/kareem1207/context-compression-engine/blob/main/benchmarks/README.md
- Licence: MIT

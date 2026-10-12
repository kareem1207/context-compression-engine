/**
 * CCE Lite: a dependency-free JavaScript port of the Context Compression Engine pipeline.
 * Runs in browsers and Node 18+, fully offline.
 *
 * Same stages as the Python engine: segment -> semantic chunking -> hierarchical
 * summaries -> hot/warm memory -> retrieval (similarity + recency + keywords) -> context builder.
 * The one difference is the embedder: Python uses a MiniLM sentence model, Lite uses
 * feature hashing of words and word pairs. It is much lighter but less semantic, so
 * topic boundaries and retrieval are good on keyword-rich text and weaker on paraphrase.
 */

export const DEFAULTS = Object.freeze({
  embedDim: 512,
  chunkSimilarityThreshold: 0.1,
  chunkMinTurns: 4,
  chunkMaxTurns: 20,
  chunkWindowSize: 3,
  microMaxTokens: 60,
  mesoMaxTokens: 80,
  macroMaxTokens: 400,
  hotTierMaxTurns: 4,
  retrievalTopK: 2,
  retrievalRecencyBoost: 0.2,
  keywordBonusWeight: 0.15,
  contextMaxTokens: 2048,
});

const STOP = new Set(
  "the a an is it to do of and or in on at for with that this was are be have has had you your we can will would could should what how why when where about just so but if not no i me my tell explain describe give show help".split(
    " "
  )
);

/** ~0.75 words per token, the same estimate the Python engine uses. */
export function countTokens(text) {
  const t = String(text || "").trim();
  if (!t) return 0;
  return Math.max(1, Math.round(t.split(/\s+/).length / 0.75));
}

export function truncateToTokens(text, maxTokens) {
  if (countTokens(text) <= maxTokens) return text;
  const words = String(text).trim().split(/\s+/);
  return words.slice(0, Math.max(1, Math.round(maxTokens * 0.75))).join(" ") + " ...";
}

// Very light stemming so "webhooks"/"webhook" and "running"/"run" land in the same hash bucket.
const stem = (w) => (w.length > 4 ? w.replace(/(ing|ed|es|s)$/, "") : w);
const words = (text) => (String(text).toLowerCase().match(/[a-z0-9]{2,}/g) || []).filter((w) => !STOP.has(w)).map(stem);
const contentWords = (text) => (String(text).toLowerCase().match(/\b[a-z]{3,}\b/g) || []).filter((w) => !STOP.has(w));

function fnv(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Feature-hashed bag of words and word pairs, L2-normalised. */
export function embed(text, dim = DEFAULTS.embedDim) {
  const v = new Float32Array(dim);
  const ws = words(text);
  const add = (token, weight) => {
    const h = fnv(token);
    v[h % dim] += (h & 0x80000000 ? -1 : 1) * weight;
  };
  ws.forEach((w, i) => {
    add(w, 1);
    if (i > 0) add(ws[i - 1] + "_" + w, 0.5);
  });
  let n = 0;
  for (let i = 0; i < dim; i++) n += v[i] * v[i];
  n = Math.sqrt(n);
  if (n > 1e-8) for (let i = 0; i < dim; i++) v[i] /= n;
  return v;
}

export function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

function meanVec(vecs) {
  const out = new Float32Array(vecs[0].length);
  vecs.forEach((v) => v.forEach((x, i) => (out[i] += x)));
  return out.map((x) => x / vecs.length);
}

const normRole = (r) => {
  r = String(r || "user").toLowerCase().trim();
  if (r === "human") return "user";
  if (["ai", "bot", "model"].includes(r)) return "assistant";
  return ["user", "assistant", "system", "tool"].includes(r) ? r : "user";
};

/** messages: [{role, content}] or a plain string with one turn per line. */
export function segment(messages) {
  if (typeof messages === "string") {
    let next = "user";
    messages = messages
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean)
      .map((line) => {
        const m = line.match(/^(user|human|assistant|ai|system)\s*:\s*(.*)$/i);
        const role = m ? normRole(m[1]) : next;
        next = role === "user" ? "assistant" : "user";
        return { role, content: m ? m[2] : line };
      });
  }
  const turns = [];
  for (const msg of messages || []) {
    const content = String(msg?.content ?? "").trim();
    if (!content) continue;
    turns.push({ index: turns.length, role: normRole(msg.role), content, tokens: countTokens(content) });
  }
  return turns;
}

function detectBoundaries(embs, cfg) {
  const n = embs.length;
  const boundaries = new Set();
  const w = cfg.chunkWindowSize;
  for (let i = 0; i < n - 1; i++) {
    const left = meanVec(embs.slice(Math.max(0, i - w + 1), i + 1));
    const right = meanVec(embs.slice(i + 1, Math.min(n, i + w + 1)));
    if (cosine(left, right) < cfg.chunkSimilarityThreshold) boundaries.add(i + 1);
  }
  let start = 0;
  for (let i = 0; i < n; i++) {
    if (i - start >= cfg.chunkMaxTurns) {
      boundaries.add(i);
      start = i;
    }
    if (boundaries.has(i)) start = i;
  }
  const clean = new Set();
  let prev = 0;
  for (const b of [...boundaries].sort((x, y) => x - y)) {
    if (b - prev >= cfg.chunkMinTurns) {
      clean.add(b);
      prev = b;
    }
  }
  return clean;
}

// Generic words that make poor topic labels.
const LABEL_SKIP = new Set(
  "from before after every each again any one add them then than into over also more most some such only very well like make made use used using need needs want back new different topic switching start keep".split(" ")
);

export function topicLabel(turns) {
  const counts = new Map();
  turns.forEach((t) =>
    contentWords(t.content).forEach((w) => LABEL_SKIP.has(w) || counts.set(w, (counts.get(w) || 0) + 1))
  );
  if (!counts.size) return `turns ${turns[0].index}-${turns[turns.length - 1].index}`;
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .slice(0, 3)
    .map(([w]) => w)
    .join(" · ");
}

export function chunk(turns, cfg = DEFAULTS) {
  if (!turns.length) return [];
  const embs = turns.map((t) => embed(t.content, cfg.embedDim));
  const cuts = turns.length > 1 ? detectBoundaries(embs, cfg) : new Set();
  const groups = [];
  let cur = [];
  turns.forEach((t, i) => {
    if (cuts.has(i) && cur.length) {
      groups.push(cur);
      cur = [];
    }
    cur.push(t);
  });
  if (cur.length) groups.push(cur);
  return groups.map((g) => ({ turns: g, topic: topicLabel(g) }));
}

/* ---------------------------- extractive summaries --------------------------- */

function extractiveSummarize(text, maxTokens) {
  const sentences = String(text).split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
  if (!sentences.length) return truncateToTokens(text, maxTokens);
  if (sentences.length === 1) return truncateToTokens(sentences[0], maxTokens);

  const freq = new Map();
  contentWords(text).forEach((w) => freq.set(w, (freq.get(w) || 0) + 1));
  const top = new Set([...freq.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20).map(([w]) => w));

  const scored = sentences
    .map((s, i) => {
      const ws = new Set(contentWords(s));
      if (!ws.size) return { s, i, score: 0 };
      let overlap = 0;
      ws.forEach((w) => top.has(w) && overlap++);
      return { s, i, score: overlap / ws.size + Math.min(s.split(/\s+/).length / 20, 1) * 0.2 };
    })
    .sort((a, b) => b.score - a.score);

  const picked = [];
  let used = 0;
  for (const c of scored) {
    const t = countTokens(c.s);
    if (used + t > maxTokens) break;
    picked.push(c);
    used += t;
  }
  if (!picked.length) return truncateToTokens(scored[0].s, maxTokens);
  return picked.sort((a, b) => a.i - b.i).map((c) => c.s).join(" ");
}

export const micro = (turn, cfg = DEFAULTS) => (turn.tokens <= cfg.microMaxTokens ? turn.content : extractiveSummarize(turn.content, cfg.microMaxTokens));

export function meso(group, cfg = DEFAULTS) {
  // Role labels are dropped so summaries read as plain notes.
  const text = group.turns.map((t) => t.content).join(" ");
  return countTokens(text) <= cfg.mesoMaxTokens ? text : extractiveSummarize(text, cfg.mesoMaxTokens);
}

export function macro(nodes, cfg = DEFAULTS) {
  const combined = nodes.map((n) => n.mesoSummary).join("\n\n");
  return countTokens(combined) <= cfg.macroMaxTokens ? combined : extractiveSummarize(combined, cfg.macroMaxTokens);
}

/** Turns -> memory nodes (topic chunks with summaries and an embedding). */
export function compress(turns, cfg = DEFAULTS) {
  return chunk(turns, cfg).map((g, i) => {
    const tokensRaw = g.turns.reduce((a, t) => a + t.tokens, 0);
    const mesoSummary = meso(g, cfg);
    return {
      id: `n${i}`,
      topicLabel: g.topic,
      turnStart: g.turns[0].index,
      turnEnd: g.turns[g.turns.length - 1].index,
      turns: g.turns,
      microSummaries: g.turns.map((t) => micro(t, cfg)),
      mesoSummary,
      embedding: meanVec(g.turns.map((t) => embed(t.content, cfg.embedDim))),
      tokensRaw,
      tokensCompressed: countTokens(mesoSummary),
    };
  });
}

/* --------------------------------- retrieval -------------------------------- */

export function retrieve(nodes, query, cfg = DEFAULTS, topK = cfg.retrievalTopK) {
  if (!nodes.length) return [];
  const q = embed(query, cfg.embedDim);
  const kw = new Set(contentWords(query));
  const maxEnd = Math.max(...nodes.map((n) => n.turnEnd), 1);
  return nodes
    .map((node) => {
      const semantic = cosine(q, node.embedding);
      const recency = (node.turnEnd / maxEnd) * cfg.retrievalRecencyBoost;
      const target = new Set(contentWords(`${node.topicLabel} ${node.mesoSummary}`));
      let overlap = 0;
      kw.forEach((w) => target.has(w) && overlap++);
      const keyword = (kw.size ? overlap / kw.size : 0) * cfg.keywordBonusWeight;
      return { node, semantic, recency, keyword, score: semantic + recency + keyword };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

/* ------------------------------- context builder ---------------------------- */

const SYSTEM_NOTE =
  "The following conversation context has been compressed by the Context Compression Engine (CCE). 'Past context' contains summarized topic blocks from earlier in the conversation. 'Recent context' contains the verbatim last few messages. Use both to answer accurately.";

function block(node) {
  return `[Topic: ${node.topicLabel}] (turns ${node.turnStart}-${node.turnEnd})\n${node.mesoSummary}\n---`;
}

export function buildContext(results, hotTurns, cfg = DEFAULTS) {
  const recentTokens = hotTurns.reduce((a, t) => a + countTokens(t.content) + 4, 0);
  const budget = cfg.contextMaxTokens - countTokens(SYSTEM_NOTE) - recentTokens - 50;
  const pastBlocks = [];
  const included = [];
  let used = 0;
  let truncated = false;
  for (const r of results) {
    const b = block(r.node);
    const t = countTokens(b);
    if (used + t > budget) {
      const remaining = budget - used;
      if (remaining > 30) {
        pastBlocks.push(truncateToTokens(b, remaining));
        included.push(r.node.id);
      }
      truncated = true;
      break;
    }
    pastBlocks.push(b);
    included.push(r.node.id);
    used += t;
  }
  const tokenCount = countTokens(SYSTEM_NOTE) + recentTokens + 50 + pastBlocks.reduce((a, b) => a + countTokens(b), 0);
  const toMessages = () => [
    { role: "system", content: [SYSTEM_NOTE, ...(pastBlocks.length ? ["", "=== PAST CONTEXT (compressed) ===", ...pastBlocks] : [])].join("\n") },
    ...hotTurns.map((t) => ({ role: t.role, content: t.content })),
  ];
  return { systemNote: SYSTEM_NOTE, pastBlocks, recentTurns: hotTurns, tokenCount, retrievedNodeIds: included, wasTruncated: truncated, toMessages };
}

/* ----------------------------------- engine --------------------------------- */

/**
 * createEngine(config?) -> { process(messages, query, opts), compress(messages), config }
 * `process` is the stateless one-shot: compress the history, retrieve for the query, build the context.
 */
export function createEngine(overrides = {}) {
  const cfg = { ...DEFAULTS, ...overrides };
  return {
    config: cfg,
    compress(messages) {
      const turns = segment(messages);
      const nodes = compress(turns, cfg);
      return { turns, nodes, macro: macro(nodes, cfg), stats: stats(turns, nodes) };
    },
    process(messages, query, { topK = cfg.retrievalTopK } = {}) {
      const turns = segment(messages);
      const nodes = compress(turns, cfg);
      const hot = turns.slice(-cfg.hotTierMaxTurns);
      const results = retrieve(nodes, query, cfg, topK);
      const payload = buildContext(results, hot, cfg);
      const st = stats(turns, nodes);
      const rawTokens = st.rawTokens;
      return {
        turns, nodes, results, payload,
        stats: { ...st, hotTurns: hot.length, contextTokens: payload.tokenCount, tokensSaved: Math.max(0, rawTokens - payload.tokenCount), savedPercent: rawTokens ? Math.round((1 - payload.tokenCount / rawTokens) * 100) : 0 },
      };
    },
  };
}

export function stats(turns, nodes) {
  const rawTokens = turns.reduce((a, t) => a + t.tokens, 0);
  const compressedTokens = nodes.reduce((a, n) => a + n.tokensCompressed, 0);
  return {
    totalTurns: turns.length,
    totalChunks: nodes.length,
    rawTokens,
    compressedTokens,
    compressionRatio: Math.round((rawTokens / Math.max(compressedTokens, 1)) * 100) / 100,
    topics: nodes.map((n) => n.topicLabel),
  };
}

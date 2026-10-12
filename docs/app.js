import { createEngine, segment } from "./lite.js";
import { SAMPLE_CONVERSATION, SAMPLE_QUERIES } from "./sample.js";

const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") e.className = v;
    else e.setAttribute(k, v);
  }
  kids.flat().forEach((k) => e.append(k instanceof Node ? k : document.createTextNode(String(k))));
  return e;
}

const toText = (messages) => messages.map((m) => `${m.role}: ${m.content}`).join("\n");

function bubbles(turns) {
  return turns.map((t) =>
    el("li", { class: `bubble ${t.role === "user" ? "me" : "ai"}` }, el("span", { class: "who" }, t.role === "user" ? "You" : "AI"), t.content)
  );
}

function engine() {
  return createEngine({ mesoMaxTokens: +$("meso").value, hotTierMaxTurns: +$("hot").value, retrievalTopK: +$("top").value });
}

let lastQuery = "";

function showChat() {
  const turns = segment($("convo").value);
  const tokens = turns.reduce((a, t) => a + t.tokens, 0);
  $("chat").replaceChildren(...bubbles(turns));
  $("chatSummary").textContent = turns.length
    ? `${turns.length} messages, about ${tokens} tokens. Sending all of this to the AI on every question is slow and costs money.`
    : "Paste or write a conversation to begin.";
}

/** A past-context block like "[Topic: x] (turns 0-5)\nsummary\nMost relevant (turn 3): ...\n---" */
function blockCard(text, index) {
  const lines = text.split("\n").filter((l) => l && l !== "---");
  const head = (lines.shift() || "").match(/^\[Topic: (.*)\] \(turns (\d+)-(\d+)\)$/);
  const title = head ? `Part ${index + 1} of the chat: messages ${+head[2] + 1} to ${+head[3] + 1}` : `Part ${index + 1} of the chat`;
  const body = [];
  for (const l of lines) {
    const m = l.match(/^(?:Most relevant|Relevant)(?: \(turn \d+\))?: (.*)$/);
    body.push(m ? el("p", { class: "match" }, el("b", {}, "Best match: "), m[1]) : el("p", {}, l));
  }
  return el("div", { class: "pick" }, el("h4", {}, title), body, head ? el("p", { class: "label" }, `topic: ${head[1]}`) : "");
}

function renderAnswer(query) {
  const q = query.trim();
  if (!q) return;
  lastQuery = q;
  const out = engine().process($("convo").value, q);
  const { stats, payload } = out;
  $("result").hidden = false;

  const pct = Math.max(0, stats.savedPercent);
  $("verdict").replaceChildren(
    stats.rawTokens
      ? el("p", { class: "big" }, `${pct}% less for the AI to read`)
      : el("p", { class: "big" }, "Add a conversation first"),
    el("p", { class: "fine" }, `Without CCE: ${stats.rawTokens} tokens. With CCE: ${stats.contextTokens} tokens.`)
  );

  const bar = (label, tokens, good) =>
    el("div", { class: "bar" + (good ? " good" : "") }, el("span", {}, label),
      el("div", { class: "track" }, el("div", { class: "fill", style: `width:${stats.rawTokens ? Math.max(3, (tokens / stats.rawTokens) * 100) : 0}%` })),
      el("span", {}, `${tokens}`));
  $("compare").replaceChildren(bar("Whole chat", stats.rawTokens, false), bar("With CCE", stats.contextTokens, true));

  $("picked").replaceChildren(...(payload.pastBlocks.length ? payload.pastBlocks.map(blockCard) : [el("p", { class: "fine" }, "Nothing older was needed.")]));
  $("recent").replaceChildren(...bubbles(payload.recentTurns));

  $("scores").replaceChildren(
    ...out.results.map((r) =>
      el("p", { class: "fine" }, `${r.node.topicLabel} (messages ${r.node.turnStart + 1}-${r.node.turnEnd + 1}): similarity ${r.semantic.toFixed(2)}, recency +${r.recency.toFixed(2)}, keywords +${r.keyword.toFixed(2)}`)
    )
  );
  $("payload").textContent = payload.toMessages().map((m) => `[${m.role}]\n${m.content}`).join("\n\n");
}

function setSample() {
  $("convo").value = toText(SAMPLE_CONVERSATION);
  showChat();
}

["meso", "hot", "top"].forEach((id) => {
  const out = { meso: "oMeso", hot: "oHot", top: "oTop" }[id];
  $(id).addEventListener("input", () => {
    $(out).textContent = $(id).value;
    if (lastQuery) renderAnswer(lastQuery);
  });
});

let timer;
$("convo").addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(() => {
    showChat();
    if (lastQuery) renderAnswer(lastQuery);
  }, 250);
});
$("useSample").addEventListener("click", () => {
  setSample();
  if (lastQuery) renderAnswer(lastQuery);
});
$("askForm").addEventListener("submit", (e) => {
  e.preventDefault();
  renderAnswer($("query").value);
});

$("chips").replaceChildren(
  ...SAMPLE_QUERIES.map((q) => {
    const b = el("button", { type: "button", class: "chip" }, q);
    b.addEventListener("click", () => {
      $("query").value = q;
      renderAnswer(q);
      $("result").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return b;
  })
);

setSample();
$("query").value = SAMPLE_QUERIES[0];
renderAnswer(SAMPLE_QUERIES[0]);

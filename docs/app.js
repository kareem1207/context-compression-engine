import { createEngine, countTokens, segment } from "./lite.js";
import { SAMPLE_CONVERSATION, SAMPLE_QUERIES } from "./sample.js";

const $ = (id) => document.getElementById(id);
const COLORS = ["var(--c0)", "var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)", "var(--c7)"];

const toText = (messages) => messages.map((m) => `${m.role}: ${m.content}`).join("\n");

function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  Object.entries(props).forEach(([k, v]) => {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else e.setAttribute(k, v);
  });
  kids.flat().forEach((k) => e.append(k instanceof Node ? k : document.createTextNode(String(k))));
  return e;
}

function engine() {
  return createEngine({
    mesoMaxTokens: +$("meso").value,
    hotTierMaxTurns: +$("hot").value,
    retrievalTopK: +$("top").value,
  });
}

let lastQuery = "";

function renderMemory() {
  const eng = engine();
  const text = $("convo").value;
  const { turns, nodes, stats } = eng.compress(text);

  $("stats").replaceChildren(
    ...[
      [stats.totalTurns, "messages"],
      [stats.rawTokens, "tokens in the full history"],
      [stats.totalChunks, "topics found"],
      [stats.compressedTokens, "tokens in topic memory"],
      [stats.compressionRatio + "x", "memory compression"],
    ].map(([b, s]) => el("div", { class: "stat" }, el("b", {}, b), el("span", {}, s)))
  );

  $("timeline").replaceChildren(
    ...nodes.map((n, i) =>
      el("i", { style: `flex:${n.turns.length};background:${COLORS[i % COLORS.length]}`, title: `${n.topicLabel} (messages ${n.turnStart + 1}-${n.turnEnd + 1})` })
    )
  );

  $("chunks").replaceChildren(
    ...nodes.map((n, i) =>
      el(
        "div",
        { class: "chunk", style: `border-left-color:${COLORS[i % COLORS.length]}` },
        el("h4", {}, n.topicLabel, el("span", { class: "meta" }, `messages ${n.turnStart + 1} to ${n.turnEnd + 1} · ${n.tokensRaw} → ${n.tokensCompressed} tokens`)),
        el("p", {}, n.mesoSummary)
      )
    )
  );

  if (lastQuery) renderAnswer(lastQuery);
  return { turns, nodes };
}

function renderAnswer(query) {
  const q = query.trim();
  if (!q) return;
  lastQuery = q;
  const out = engine().process($("convo").value, q);
  const { stats } = out;
  $("result").hidden = false;

  const bar = (label, tokens, max, good) =>
    el("div", { class: "bar" + (good ? " good" : "") },
      el("span", {}, label),
      el("div", { class: "track" }, el("div", { class: "fill", style: `width:${max ? Math.max(2, (tokens / max) * 100) : 0}%` })),
      el("span", {}, `${tokens} tokens`));

  $("compare").replaceChildren(
    bar("Send everything", stats.rawTokens, stats.rawTokens, false),
    bar("Send CCE context", stats.contextTokens, stats.rawTokens, true),
    el("div", { class: "saved" }, stats.rawTokens ? `${Math.max(0, stats.savedPercent)}% fewer tokens (${stats.tokensSaved} saved)` : "Add a conversation first")
  );

  $("ranked").replaceChildren(
    ...(out.results.length
      ? out.results.map((r, i) =>
          el("div", { class: "rank" },
            el("strong", {}, `${i + 1}. ${r.node.topicLabel}`), ` (messages ${r.node.turnStart + 1} to ${r.node.turnEnd + 1})`,
            el("div", { class: "score" },
              el("span", {}, `similarity ${r.semantic.toFixed(2)}`),
              el("span", {}, `recency +${r.recency.toFixed(2)}`),
              el("span", {}, `keywords +${r.keyword.toFixed(2)}`),
              el("span", {}, `score ${r.score.toFixed(2)}`))))
      : [el("p", { class: "fine" }, "No topics yet.")])
  );

  const msgs = out.payload.toMessages();
  $("payload").textContent = msgs.map((m) => `[${m.role}]\n${m.content}`).join("\n\n");
}

function loadSample() {
  $("convo").value = toText(SAMPLE_CONVERSATION);
  lastQuery = "";
  $("result").hidden = true;
  renderMemory();
}

function bindSliders() {
  [["meso", "oMeso"], ["hot", "oHot"], ["top", "oTop"]].forEach(([i, o]) => {
    $(i).addEventListener("input", () => {
      $(o).textContent = $(i).value;
      renderMemory();
    });
  });
}

let timer;
$("convo").addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(renderMemory, 250);
});
$("useSample").addEventListener("click", loadSample);
$("clear").addEventListener("click", () => {
  $("convo").value = "";
  lastQuery = "";
  $("result").hidden = true;
  renderMemory();
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
    });
    return b;
  })
);

bindSliders();
loadSample();
renderAnswer(SAMPLE_QUERIES[0]);
$("query").value = SAMPLE_QUERIES[0];

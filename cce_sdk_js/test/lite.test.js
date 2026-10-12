import { test } from "node:test";
import assert from "node:assert/strict";
import { createEngine, countTokens, segment, embed, cosine, DEFAULTS } from "../lite.js";
import { SAMPLE_CONVERSATION, SAMPLE_QUERIES } from "../sample.js";

const engine = createEngine();

test("countTokens follows the 0.75 words-per-token rule", () => {
  assert.equal(countTokens(""), 0);
  assert.equal(countTokens("one"), 1);
  assert.equal(countTokens("a b c d e f g h"), 11);
});

test("segment normalises roles, drops empty turns and parses plain text", () => {
  const t = segment([{ role: "Human", content: " hi " }, { role: "bot", content: "" }, { role: "AI", content: "hello" }]);
  assert.deepEqual(t.map((x) => [x.index, x.role, x.content]), [[0, "user", "hi"], [1, "assistant", "hello"]]);
  const p = segment("user: a question\nan answer line\nUser: another");
  assert.deepEqual(p.map((x) => x.role), ["user", "assistant", "user"]);
});

test("embedding is unit length and similar texts score higher than unrelated ones", () => {
  const a = embed("verify webhook signature header");
  assert.ok(Math.abs(cosine(a, a) - 1) < 1e-5);
  assert.ok(cosine(a, embed("webhook signatures are verified")) > cosine(a, embed("pasta with garlic and basil")));
  assert.equal(embed("").every((x) => x === 0), true);
});

test("compress groups the sample into several topic chunks that cover every turn once", () => {
  const { turns, nodes, stats } = engine.compress(SAMPLE_CONVERSATION);
  assert.ok(nodes.length >= 4 && nodes.length <= 8, `chunks: ${nodes.length}`);
  assert.equal(nodes.flatMap((n) => n.turns).length, turns.length);
  nodes.forEach((n, i) => i && assert.equal(n.turnStart, nodes[i - 1].turnEnd + 1));
  assert.ok(stats.compressionRatio > 1.5, `ratio ${stats.compressionRatio}`);
});

test("each sample query retrieves the matching topic first", () => {
  const expectations = [/payment|dashboard|webhook|invoice|revenue|mrr|build/, /pasta|garlic|sauce|water/, /run|mileage|shin|eat|week|ankle/, /config|key|default|yaml/];
  SAMPLE_QUERIES.forEach((q, i) => {
    const { results } = engine.process(SAMPLE_CONVERSATION, q);
    const text = (results[0].node.topicLabel + " " + results[0].node.mesoSummary).toLowerCase();
    assert.match(text, expectations[i], `query: ${q}`);
  });
});

test("the context is much smaller than the full history and keeps recent turns verbatim", () => {
  const out = engine.process(SAMPLE_CONVERSATION, SAMPLE_QUERIES[0]);
  assert.ok(out.stats.contextTokens < out.stats.rawTokens);
  assert.ok(out.stats.savedPercent > 20, `saved ${out.stats.savedPercent}%`);
  const msgs = out.payload.toMessages();
  assert.equal(msgs[0].role, "system");
  assert.equal(msgs.length, 1 + DEFAULTS.hotTierMaxTurns);
  assert.equal(msgs.at(-1).content, SAMPLE_CONVERSATION.at(-1).content);
});

test("the token budget is respected and flagged when it truncates", () => {
  const tiny = createEngine({ contextMaxTokens: 150 });
  const out = tiny.process(SAMPLE_CONVERSATION, SAMPLE_QUERIES[1]);
  assert.equal(out.payload.wasTruncated, true);
});

test("empty and tiny inputs do not crash", () => {
  assert.deepEqual(engine.compress([]).nodes, []);
  const one = engine.process([{ role: "user", content: "hello there" }], "hello");
  assert.equal(one.nodes.length, 1);
  assert.equal(engine.process([], "anything").payload.pastBlocks.length, 0);
});

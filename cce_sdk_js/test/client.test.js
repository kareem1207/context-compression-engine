import { test } from "node:test";
import assert from "node:assert/strict";
import { CCEClient, CCEError } from "../client.js";

function mockFetch(handler) {
  const calls = [];
  const fn = async (url, init) => {
    calls.push({ url, init });
    const { status = 200, body = {} } = handler(url, init);
    return { ok: status < 400, status, json: async () => body };
  };
  fn.calls = calls;
  return fn;
}

test("compress posts the right payload", async () => {
  const f = mockFetch(() => ({ body: { context: [], stats: {} } }));
  const c = new CCEClient({ baseUrl: "http://x:1/", fetch: f });
  await c.compress([{ role: "user", content: "hi" }], "q", { topK: 3 });
  assert.equal(f.calls[0].url, "http://x:1/v1/compress");
  assert.equal(f.calls[0].init.method, "POST");
  assert.deepEqual(JSON.parse(f.calls[0].init.body), { messages: [{ role: "user", content: "hi" }], query: "q", top_k: 3, fmt: "messages" });
});

test("session ids are URL-encoded", async () => {
  const f = mockFetch(() => ({ body: { sessions: [] } }));
  const c = new CCEClient({ fetch: f });
  await c.getStats("a b/c");
  assert.equal(f.calls[0].url, "http://127.0.0.1:9100/v1/sessions/a%20b%2Fc/stats");
});

test("errors carry status and the server message", async () => {
  const c = new CCEClient({ fetch: mockFetch(() => ({ status: 404, body: { detail: "Session 'x' not found" } })) });
  await assert.rejects(c.getSummary("x"), (e) => e instanceof CCEError && e.status === 404 && /not found/.test(e.message));
});

test("validation errors from the API are readable", async () => {
  const c = new CCEClient({ fetch: mockFetch(() => ({ status: 422, body: { detail: [{ msg: "too short" }] } })) });
  await assert.rejects(c.compress([], "q"), (e) => e.status === 422 && /too short/.test(e.message));
});

test("listSessions unwraps the array", async () => {
  const c = new CCEClient({ fetch: mockFetch(() => ({ body: { sessions: [{ session_id: "a" }] } })) });
  assert.deepEqual(await c.listSessions(), [{ session_id: "a" }]);
});

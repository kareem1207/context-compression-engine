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

test("getContext can ask for the explanation", async () => {
  const f = mockFetch(() => ({ body: { context: [], explain: { notes: [] } } }));
  const c = new CCEClient({ fetch: f });
  await c.getContext("s", "q", { explain: true });
  assert.deepEqual(JSON.parse(f.calls[0].init.body), { query: "q", fmt: "messages", explain: true });
  await c.getContext("s", "q");
  assert.equal(JSON.parse(f.calls[1].init.body).explain, false);
});

test("memory management calls use the right routes and bodies", async () => {
  const f = mockFetch(() => ({ body: { pins: [{ pin_id: "p1" }], pin_id: "p1" } }));
  const c = new CCEClient({ baseUrl: "http://x:1", fetch: f });
  await c.pin("s", { text: "remember" });
  assert.deepEqual(await c.listPins("s"), [{ pin_id: "p1" }]);
  await c.editPin("s", "p1", "new");
  await c.unpin("s", "p1");
  await c.forget("s", { start: 2, end: 5 });
  await c.editNote("s", "n 1", { mesoSummary: "fixed" });
  await c.editTurn("s", 3, "edited");
  await c.exportSession("s");
  await c.getContext("s", "q", { lastTurns: 4, since: "2026-01-01T00:00:00Z" });
  const seen = f.calls.map((x) => `${x.init.method} ${x.url.replace("http://x:1", "")}`);
  assert.deepEqual(seen, [
    "POST /v1/sessions/s/pins", "GET /v1/sessions/s/pins", "PUT /v1/sessions/s/pins/p1", "DELETE /v1/sessions/s/pins/p1",
    "POST /v1/sessions/s/forget", "PATCH /v1/sessions/s/notes/n%201", "PATCH /v1/sessions/s/turns/3",
    "GET /v1/sessions/s/export", "POST /v1/sessions/s/context",
  ]);
  assert.deepEqual(JSON.parse(f.calls[4].init.body), { start: 2, end: 5 });
  assert.equal(JSON.parse(f.calls[8].init.body).last_turns, 4);
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

test("apiKey is sent as a Bearer token", async () => {
  const f = mockFetch(() => ({ body: { status: "ok" } }));
  await new CCEClient({ fetch: f, apiKey: "k1" }).health();
  assert.equal(f.calls[0].init.headers.Authorization, "Bearer k1");
  const g = mockFetch(() => ({ body: { status: "ok" } }));
  await new CCEClient({ fetch: g }).health();
  assert.equal(g.calls[0].init.headers.Authorization, undefined);
});

/**
 * HTTP client for the CCE REST API (`uv run python -m cce_rest.app`).
 * Uses the global fetch of Node 18+ and browsers. No dependencies.
 */

export class CCEError extends Error {
  constructor(message, status, body) {
    super(message);
    this.name = "CCEError";
    this.status = status;
    this.body = body;
  }
}

export class CCEClient {
  /**
   * @param {{ baseUrl?: string, fetch?: typeof fetch, timeoutMs?: number, apiKey?: string }} [options]
   */
  constructor({ baseUrl = "http://127.0.0.1:9100", fetch: fetchImpl, timeoutMs = 60000, apiKey } = {}) {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this._fetch = fetchImpl || globalThis.fetch;
    this.timeoutMs = timeoutMs;
    if (!this._fetch) throw new Error("No fetch available: use Node 18+ or pass options.fetch");
  }

  async _request(method, path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await this._fetch(this.baseUrl + path, {
        method,
        headers: {
          ...(body === undefined ? {} : { "Content-Type": "application/json" }),
          ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      let data = null;
      try {
        data = await res.json();
      } catch {
        /* empty or non-JSON body */
      }
      if (!res.ok) {
        const detail = data && data.detail;
        const message = typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `HTTP ${res.status}`;
        throw new CCEError(message, res.status, data);
      }
      return data;
    } catch (err) {
      if (err.name === "AbortError") throw new CCEError(`Request timed out after ${this.timeoutMs} ms`, 0, null);
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  health() {
    return this._request("GET", "/health");
  }

  /** One-shot: compress `messages` and get the context for `query`. Nothing is stored. */
  compress(messages, query, { topK, fmt = "messages" } = {}) {
    return this._request("POST", "/v1/compress", { messages, query, top_k: topK, fmt });
  }

  listSessions() {
    return this._request("GET", "/v1/sessions").then((r) => r.sessions);
  }

  addMessages(sessionId, messages) {
    return this._request("POST", `/v1/sessions/${encodeURIComponent(sessionId)}/messages`, { messages });
  }

  /** Context to send to your LLM right before each call. */
  getContext(sessionId, query, { topK, fmt = "messages", explain = false, since, until, lastTurns } = {}) {
    return this._request("POST", `/v1/sessions/${encodeURIComponent(sessionId)}/context`, {
      query, top_k: topK, fmt, explain, since, until, last_turns: lastTurns,
    });
  }

  getSummary(sessionId) {
    return this._request("GET", `/v1/sessions/${encodeURIComponent(sessionId)}/summary`);
  }

  getStats(sessionId) {
    return this._request("GET", `/v1/sessions/${encodeURIComponent(sessionId)}/stats`);
  }

  /** Everything stored for a session as plain data. */
  exportSession(sessionId) {
    return this._request("GET", `/v1/sessions/${encodeURIComponent(sessionId)}/export`);
  }

  /** Pin a text (or a stored turn by index): it is sent with every context of the session. */
  pin(sessionId, { text, turn } = {}) {
    return this._request("POST", `/v1/sessions/${encodeURIComponent(sessionId)}/pins`, { text, turn });
  }

  listPins(sessionId) {
    return this._request("GET", `/v1/sessions/${encodeURIComponent(sessionId)}/pins`).then((r) => r.pins);
  }

  editPin(sessionId, pinId, text) {
    return this._request("PUT", `/v1/sessions/${encodeURIComponent(sessionId)}/pins/${encodeURIComponent(pinId)}`, { text });
  }

  unpin(sessionId, pinId) {
    return this._request("DELETE", `/v1/sessions/${encodeURIComponent(sessionId)}/pins/${encodeURIComponent(pinId)}`);
  }

  /** Forget a note ({ nodeId }) or turns ({ start, end }). Every overlapping note is deleted whole. */
  forget(sessionId, { nodeId, start, end } = {}) {
    return this._request("POST", `/v1/sessions/${encodeURIComponent(sessionId)}/forget`, { node_id: nodeId, start, end });
  }

  editNote(sessionId, nodeId, { topicLabel, mesoSummary } = {}) {
    return this._request("PATCH", `/v1/sessions/${encodeURIComponent(sessionId)}/notes/${encodeURIComponent(nodeId)}`,
      { topic_label: topicLabel, meso_summary: mesoSummary });
  }

  editTurn(sessionId, index, content) {
    return this._request("PATCH", `/v1/sessions/${encodeURIComponent(sessionId)}/turns/${index}`, { content });
  }

  closeSession(sessionId, { checkpoint = true } = {}) {
    return this._request("POST", `/v1/sessions/${encodeURIComponent(sessionId)}/close?checkpoint=${checkpoint}`);
  }

  deleteSession(sessionId) {
    return this._request("DELETE", `/v1/sessions/${encodeURIComponent(sessionId)}`);
  }
}

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
  getContext(sessionId, query, { topK, fmt = "messages", explain = false } = {}) {
    return this._request("POST", `/v1/sessions/${encodeURIComponent(sessionId)}/context`, { query, top_k: topK, fmt, explain });
  }

  getSummary(sessionId) {
    return this._request("GET", `/v1/sessions/${encodeURIComponent(sessionId)}/summary`);
  }

  getStats(sessionId) {
    return this._request("GET", `/v1/sessions/${encodeURIComponent(sessionId)}/stats`);
  }

  closeSession(sessionId, { checkpoint = true } = {}) {
    return this._request("POST", `/v1/sessions/${encodeURIComponent(sessionId)}/close?checkpoint=${checkpoint}`);
  }

  deleteSession(sessionId) {
    return this._request("DELETE", `/v1/sessions/${encodeURIComponent(sessionId)}`);
  }
}

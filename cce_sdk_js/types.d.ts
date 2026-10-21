export interface Message {
  role: "user" | "assistant" | "system" | "tool" | string;
  content: string;
}

export type Format = "messages" | "string" | "dict";

/* ------------------------------ REST client ------------------------------ */

export class CCEError extends Error {
  status: number;
  body: unknown;
}

export interface ClientOptions {
  baseUrl?: string;
  fetch?: typeof fetch;
  timeoutMs?: number;
  apiKey?: string;
}

export interface ContextOptions {
  topK?: number;
  fmt?: Format;
  /** Also return why the context looks as it does (REST `explain`). */
  explain?: boolean;
  /** Only notes and recent turns from this moment on (ISO 8601 string). */
  since?: string;
  /** Only notes and recent turns up to this moment (ISO 8601 string). */
  until?: string;
  /** Only the last N turns of the session and the notes that reach into them. */
  lastTurns?: number;
}

export interface Pin {
  pin_id: string;
  text: string;
  source: string | null;
  created_at: string;
}

export interface SessionExport {
  format: "cce-session-export/1";
  exported_at: string;
  session_id: string;
  stats: SessionStats;
  turns: Array<{ index: number; role: string; content: string; tokens: number; timestamp: string; state: "hot" | "pending" | "done" }>;
  notes: Array<{
    node_id: string;
    topic: string;
    turns: [number, number];
    summary: string;
    message_summaries: string[];
    tokens_original: number;
    created_at: string;
    time: [string, string] | null;
  }>;
  pins: Pin[];
  session_summary: Record<string, unknown> | null;
}

/** Why a context looks as it does: chosen notes and quotes, scores, the token budget split, what was dropped. */
export interface Explain {
  query: {
    text: string;
    mode: "search" | "overview";
    top_k: number;
    bm25_terms: string[];
    embedder: { name: string; dim: number; model?: string };
    notes_in_session: number;
    notes_retrieved: number;
  };
  budget: {
    max_tokens: number;
    usable: number;
    system_note: number;
    overhead: number;
    recent_turns: number;
    pending_turns: number;
    past_notes: number;
    total: number;
  };
  recent: { turns: number; shortened_or_dropped: boolean };
  pending: { turns_included: number; turns_available: number; cut: boolean };
  notes: Array<{
    node_id: string;
    topic: string;
    turns: [number, number];
    included: boolean;
    tokens: number;
    scores: { dense: number; recency: number; bm25: number; fused: number };
    quotes: string[];
    overview_lines: string[];
  }>;
  dropped_for_budget: string[];
}

export interface ContextResponse {
  context: Message[] | string | Record<string, unknown>;
  token_count: number;
  was_truncated: boolean;
  retrieved_node_ids: string[];
  explain?: Explain;
}

/** GET /v1/sessions/{id}/stats */
export interface SessionStats {
  session_id: string;
  turns_ingested: number;
  raw_tokens: number;
  hot_turns: number;
  pending_turns: number;
  warm_nodes: number;
  warm_tokens_original: number;
  warm_tokens_compressed: number;
  raw_turns_stored: number;
  /** What sending the whole session would cost now: notes as sent + hot and pending turns. */
  tokens_sent: number;
  tokens_saved: number;
  /** raw_tokens / tokens_sent; null while the session is not smaller than its raw text. */
  compression_ratio: number | null;
  cold_summary_exists: boolean;
  opened_at: string;
}

export interface CompressResponse {
  context: Message[] | string | Record<string, unknown>;
  stats: {
    total_turns: number;
    total_chunks: number;
    raw_tokens: number;
    compressed_tokens: number;
    compression_ratio: number;
    topics: string[];
    hot_turns: number;
    retrieved_nodes: number;
  };
}

export class CCEClient {
  constructor(options?: ClientOptions);
  health(): Promise<{ status: string; active_sessions: number }>;
  compress(messages: Message[], query: string, options?: ContextOptions): Promise<CompressResponse>;
  listSessions(): Promise<Array<Record<string, unknown>>>;
  addMessages(sessionId: string, messages: Message[]): Promise<{ session_id: string; turns_added: number }>;
  getContext(sessionId: string, query: string, options?: ContextOptions): Promise<ContextResponse>;
  getSummary(sessionId: string): Promise<{ session_id: string; macro_summary: string; node_count: number; topics: string[] }>;
  getStats(sessionId: string): Promise<SessionStats>;
  exportSession(sessionId: string): Promise<SessionExport>;
  pin(sessionId: string, options: { text?: string; turn?: number }): Promise<{ pin_id: string }>;
  listPins(sessionId: string): Promise<Pin[]>;
  editPin(sessionId: string, pinId: string, text: string): Promise<{ pin_id: string; updated: boolean }>;
  unpin(sessionId: string, pinId: string): Promise<{ pin_id: string; removed: boolean }>;
  forget(
    sessionId: string,
    options: { nodeId?: string; start?: number; end?: number }
  ): Promise<{ notes_removed: Array<{ node_id: string; topic: string; turns: [number, number] }>; turns_removed: number }>;
  editNote(sessionId: string, nodeId: string, options: { topicLabel?: string; mesoSummary?: string }): Promise<{ node_id: string; updated: boolean }>;
  editTurn(sessionId: string, index: number, content: string): Promise<{ index: number; updated: boolean }>;
  closeSession(sessionId: string, options?: { checkpoint?: boolean }): Promise<{ session_id: string; closed: boolean }>;
  deleteSession(sessionId: string): Promise<{ session_id: string; deleted: boolean }>;
}

/* --------------------------- Offline engine (Lite) --------------------------- */

export interface LiteConfig {
  embedDim: number;
  chunkSimilarityThreshold: number;
  chunkMinTurns: number;
  chunkMaxTurns: number;
  chunkWindowSize: number;
  microMaxTokens: number;
  mesoMaxTokens: number;
  macroMaxTokens: number;
  hotTierMaxTurns: number;
  retrievalTopK: number;
  retrievalRecencyBoost: number;
  keywordBonusWeight: number;
  contextMaxTokens: number;
}

export interface Turn {
  index: number;
  role: string;
  content: string;
  tokens: number;
}

export interface MemoryNode {
  id: string;
  topicLabel: string;
  turnStart: number;
  turnEnd: number;
  turns: Turn[];
  microSummaries: string[];
  mesoSummary: string;
  embedding: Float32Array;
  turnEmbeddings: Float32Array[];
  tokensRaw: number;
  tokensCompressed: number;
}

export interface RetrievalResult {
  node: MemoryNode;
  bestTurn: Turn;
  semantic: number;
  recency: number;
  keyword: number;
  score: number;
}

export interface ContextPayload {
  systemNote: string;
  pastBlocks: string[];
  recentTurns: Turn[];
  tokenCount: number;
  retrievedNodeIds: string[];
  wasTruncated: boolean;
  toMessages(): Message[];
}

export interface CompressionStats {
  totalTurns: number;
  totalChunks: number;
  rawTokens: number;
  compressedTokens: number;
  compressionRatio: number;
  topics: string[];
}

export interface LiteEngine {
  config: LiteConfig;
  compress(messages: Message[] | string): { turns: Turn[]; nodes: MemoryNode[]; macro: string; stats: CompressionStats };
  process(
    messages: Message[] | string,
    query: string,
    options?: { topK?: number }
  ): {
    turns: Turn[];
    nodes: MemoryNode[];
    results: RetrievalResult[];
    payload: ContextPayload;
    stats: CompressionStats & { hotTurns: number; contextTokens: number; tokensSaved: number; savedPercent: number };
  };
}

export const DEFAULTS: Readonly<LiteConfig>;
export function createEngine(overrides?: Partial<LiteConfig>): LiteEngine;
export function countTokens(text: string): number;
export function segment(messages: Message[] | string): Turn[];
export function compress(turns: Turn[], config?: LiteConfig): MemoryNode[];
export function retrieve(nodes: MemoryNode[], query: string, config?: LiteConfig, topK?: number, ignoreFrom?: number): RetrievalResult[];
export function buildContext(results: RetrievalResult[], hotTurns: Turn[], config?: LiteConfig): ContextPayload;
export function embed(text: string, dim?: number): Float32Array;
export function cosine(a: Float32Array, b: Float32Array): number;

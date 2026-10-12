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
}

export interface ContextResponse {
  context: Message[] | string | Record<string, unknown>;
  token_count: number;
  was_truncated: boolean;
  retrieved_node_ids: string[];
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
  getStats(sessionId: string): Promise<Record<string, unknown>>;
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

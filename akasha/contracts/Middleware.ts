// Akasha Contract — Middleware / PipelineContext / Message I/O
//
// Python 一次ソース:
//   src/pneuma_core/protocols/middleware.py
//   src/pneuma_core/models/message.py
//
// RuntimeEngine のパイプラインに任意処理を差し込むためのフック。

import type { Character } from "./Character";
import type { EmotionalState } from "./EmotionalState";
import type { GoalTree } from "./Goals";
import type { EpisodicMemory, SemanticMemory } from "./Memory";

export type SenderType = "human" | "character" | "system";

export interface MessageInput {
  /** 空文字は不可 (Python 側で `__post_init__` がエラー)。 */
  content: string;
  sender_id: string;
  sender_name: string;
  sender_type: SenderType;
  channel?: string | null;
  metadata?: Record<string, unknown>;
}

export interface ToolCall {
  name: string;
  arguments?: Record<string, unknown>;
}

export interface SystemMessage {
  /** "warning" | "info" | "error" など (自由文字列)。 */
  type: string;
  message: string;
  /** "memory_search" | "emotion" | "todo" | "llm" など。 */
  component: string;
}

export interface DiagnosticInfo {
  /** 任意の診断情報。Python 側は dataclass `DiagnosticInfo`。 */
  [k: string]: unknown;
}

export interface ChangeRecord {
  id: string;
  character_id: string;
  type: string;
  before?: Record<string, unknown> | null;
  after: Record<string, unknown>;
  reason: string;
  /** ISO 8601 datetime 文字列。 */
  timestamp: string;
}

export interface MessageOutput {
  content: string;
  emotion: EmotionalState;
  thought?: string | null;
  action?: string | null;
  tool_calls?: ToolCall[];
  internal_changes?: ChangeRecord[];
  diagnostic?: DiagnosticInfo | null;
  system_messages?: SystemMessage[];
}

/** ミドルウェアチェーン全体で共有される状態。 */
export interface PipelineContext {
  character: Character;
  emotion: EmotionalState;
  goals: GoalTree;
  memories: Array<EpisodicMemory | SemanticMemory>;
  system_prompt: string;
  /** OpenAI/Anthropic 互換の chat history (`{role, content}` 等)。 */
  history: Array<Record<string, unknown>>;
  turn_count: number;
  metadata: Record<string, unknown>;
}

/**
 * 1 ターンの処理に介入する境界。
 *   pre_process : LLM 呼び出し前 — 入力を変更可
 *   post_process: LLM 呼び出し後 — 出力を変更・副作用を起こす
 */
export interface Middleware {
  pre_process(msg: MessageInput, context: PipelineContext): Promise<MessageInput>;
  post_process(
    msg: MessageInput,
    output: MessageOutput,
    context: PipelineContext,
  ): Promise<MessageOutput>;
}

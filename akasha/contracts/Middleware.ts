// Akasha Contract — Middleware / PipelineContext
//
// Python 一次ソース: src/pneuma_core/protocols/middleware.py
// RuntimeEngine のパイプラインに任意処理を差し込むためのフック。

import type { Character } from "./Character";
import type { EmotionalState } from "./EmotionalState";
import type { GoalTree } from "./Goals";
import type { EpisodicMemory, SemanticMemory } from "./Memory";

export interface MessageInput {
  role: "user" | "system" | "assistant" | "tool";
  content: string;
  /** 送信者識別 (キャラクター・人間・システム)。 */
  sender_id?: string | null;
  metadata?: Record<string, unknown>;
}

export interface MessageOutput {
  content: string;
  metadata?: Record<string, unknown>;
}

/** ミドルウェアチェーン全体で共有される状態。 */
export interface PipelineContext {
  character: Character;
  emotion: EmotionalState;
  goals: GoalTree;
  memories: Array<EpisodicMemory | SemanticMemory>;
  system_prompt: string;
  history: Array<{ role: string; content: string }>;
  turn_count: number;
  metadata: Record<string, unknown>;
}

/**
 * 1 ターンの処理に介入する境界。
 * pre_process: LLM 呼び出し前 (入力を変更可)
 * post_process: LLM 呼び出し後 (出力を変更・副作用を起こす)
 */
export interface Middleware {
  pre_process(msg: MessageInput, context: PipelineContext): Promise<MessageInput>;
  post_process(
    msg: MessageInput,
    output: MessageOutput,
    context: PipelineContext,
  ): Promise<MessageOutput>;
}

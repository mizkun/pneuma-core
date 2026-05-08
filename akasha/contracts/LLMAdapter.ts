// Akasha Contract — LLMAdapter / LLMRequest / LLMResponse / ModelConfig
//
// Python 一次ソース: src/pneuma_core/protocols/llm.py
//
// 値域チェック (temperature ∈ [0, 2.0], max_tokens > 0) は Python 側で行う。

export interface LLMRequest {
  /**
   * system プロンプト本体。
   * Python 側の規約: messages に system ロールを混入させてはならない (#1, fix #2)。
   */
  system_prompt: string;

  /**
   * Chat messages。Python 側の型は `list[dict]` で role の制約は持たない
   * (キャッシュ系統で system_prompt と分離させる方針)。
   */
  messages: Array<Record<string, unknown>>;

  model?: string | null;

  /** [0.0, 2.0]。 */
  temperature: number;

  /** > 0。 */
  max_tokens: number;

  /** プロンプトキャッシュ対象とする静的な system 部 (任意)。 */
  system_prompt_cached?: string | null;

  /** 動的な system 部 (任意)。キャッシュからは除外される。 */
  system_prompt_dynamic?: string | null;
}

export interface LLMResponse {
  content: string;
  model: string;
  usage: Record<string, unknown>;
}

export interface ModelConfig {
  model: string;
  /** [0.0, 2.0]。 */
  temperature: number;
  /** > 0。 */
  max_tokens: number;
}

/**
 * LLM 通信アダプタの境界。
 * 実装は `pneuma_core.llm.*` 配下に限定し、コア層 (runtime / memory) は
 * これを型としてのみ参照する。
 */
export interface LLMAdapter {
  generate(request: LLMRequest): Promise<LLMResponse>;
}

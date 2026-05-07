// Akasha Contract — LLMAdapter
//
// Python 一次ソース: src/pneuma_core/protocols/llm.py
// この型は Python の `LLMAdapter` Protocol と意味的に同期している必要がある。
// 値の正規化 (temperature, max_tokens の検証) は Python 側で行う。

export interface LLMRequest {
  /** system プロンプト本体。messages に system ロールを混入させてはならない (#1)。 */
  system_prompt: string;

  /** OpenAI/Anthropic 互換の chat messages。各要素は { role, content } 形式。 */
  messages: Array<{ role: "user" | "assistant" | "tool"; content: string }>;

  /** 省略時はアダプタの既定モデルを使用。 */
  model?: string | null;

  /** [0.0, 2.0]。範囲外は実行時に弾かれる。 */
  temperature: number;

  /** > 0。範囲外は実行時に弾かれる。 */
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
 * 実装は `pneuma_core.llm.*` 配下に限定し、`pneuma_core.runtime` などのコアは
 * これを型としてのみ参照する。
 */
export interface LLMAdapter {
  generate(request: LLMRequest): Promise<LLMResponse>;
}

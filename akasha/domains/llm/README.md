# Domain — llm

Python source: [`src/pneuma_core/llm/`](../../../src/pneuma_core/llm)

## 役割

LLM プロバイダ・Embedding サービスとの境界を成す層。
Protocol (`protocols.llm`, `protocols.embedding`) で抽象化された型に対する
具体実装を提供する。

## モジュール

| モジュール | 役割 |
|-----------|------|
| `adapter` | `LLMAdapter` Protocol と `LLMRequest` / `LLMResponse` / `ModelConfig` の dataclass |
| `claude` | Anthropic Claude API の `LLMAdapter` 実装 (`pneuma-core[anthropic]` extras) |
| `embedding` | `EmbeddingService` Protocol と OpenAI Embedding 実装 (`pneuma-core[openai]` extras) |

## 同梱実装

| 実装 | プロバイダ | extras |
|------|----------|--------|
| `ClaudeAdapter` | anthropic | `pneuma-core[anthropic]` |
| `OpenAIEmbeddingService` | openai | `pneuma-core[openai]` |

## 不変条件

- コア (`pneuma_core.runtime`, `pneuma_core.memory` 等) は anthropic / openai に
  直接依存しない。直接依存を許されるのは `pneuma_core.llm.*` の実装のみ。
- `LLMRequest` は system プロンプトを `system_prompt` フィールドに集約し、
  `messages` には system ロールを混入させない。
  > 過去 Incident: `_build_messages_for_llm` が system ロールを `messages` に挿入し、
  > Claude API でエラー (#1, fix #2)。messages は `user` / `assistant` のみ。

## 関連ドメイン

- [`runtime`](../runtime/) — `LLMAdapter` を呼び出す消費者
- [`memory`](../memory/) — `EmbeddingService` を消費
- [`protocols`](../protocols/) — Protocol 定義の一次ソース

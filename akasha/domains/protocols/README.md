# Domain — protocols

Python source: [`src/pneuma_core/protocols/`](../../../src/pneuma_core/protocols)

## 役割

pneuma-core の "外側" との接続境界を一括で定義する Protocol 群。
すべて `typing.Protocol` ベースの pure interface で、実装詳細は持たない。

## Protocol 一覧

| Protocol | モジュール | 責務 |
|---------|-----------|------|
| `LLMAdapter` | `llm` | LLM プロバイダ抽象 (`generate(LLMRequest) -> LLMResponse`) |
| `EmbeddingService` | `embedding` | テキスト → ベクトル変換 |
| `StorageBackend` | `storage` | 全データの統合永続化境界 |
| `MemoryStore` | `memory_store` | 記憶ストア (ベクトル類似検索を含む) |
| `Middleware` | `middleware` | パイプライン拡張フック (pre / post) |
| `VoiceProtocol` (`TTSAdapter` / `STTService`) | `voice` | 音声入出力抽象 |
| `TaskBackend` | `task` | タスク管理抽象 |

## 関連 dataclass (補助型)

- `LLMRequest` / `LLMResponse` / `ModelConfig` (`protocols.llm`)
- `PipelineContext` (`protocols.middleware`)

## 不変条件

- Protocol は pure interface であり、実装詳細を持たない。
- Python ランタイムにおける一次ソースは `src/pneuma_core/protocols/` 配下。
- Akasha 側の `akasha/contracts/*.ts` は外側からの "仕様" として参照され、
  Python の Protocol 定義と意味的に同期している必要がある。

## 関連ドメイン

- [`llm`](../llm/) / [`storage`](../storage/) / [`memory`](../memory/) /
  [`runtime`](../runtime/) — Protocol の実装または消費

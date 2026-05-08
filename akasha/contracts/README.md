# Contract Pack — pneuma-core

Akasha **main** の `defineContract()` に準拠した Producer→Consumer 宣言の正本は
[`index.mjs`](./index.mjs)。Akasha がロード時に `contractEntry` 経由で動的 import し、
singleton registry に登録する。

並置されている `*.ts` ファイルは **補助的な型仕様** で、Python の dataclass /
Protocol との同期を視覚化するためのもの。Akasha の registry には登録されない。

## 登録 contracts (13 件)

| id | layer | producers | consumers (主) |
|----|-------|-----------|-------------|
| `character` | 4F-domain | models.character, character_sheet | runtime.engine, prompt_builder, storage.backend |
| `emotional-state` | 4F-domain | models.emotion, emotion.baseline, emotion.decay, runtime.emotion_engine | runtime.engine, prompt_builder, storage.backend |
| `episodic-memory` | 4F-domain | models.memory, memory.store, memory.consolidator | memory.search, semantic_consolidator, runtime.engine, storage.backend |
| `semantic-memory` | 4F-domain | models.memory, memory.semantic_consolidator | memory.search, runtime.engine, storage.backend |
| `embedding-vector` | 3F-external-service | llm.embedding | memory.search, memory.store, memory.consolidator |
| `llm-request` | 3F-external-service | runtime.engine, prompt_builder | llm.adapter, llm.claude |
| `llm-response` | 3F-external-service | llm.adapter, llm.claude | runtime.engine, response_parser |
| `goal-tree` | 4F-domain | models.goals | runtime.engine, prompt_builder, storage.backend |
| `relation` | 4F-domain | models.relation | runtime.engine, storage.backend |
| `message-input` | 4F-domain | models.message | runtime.engine, runtime.middleware |
| `message-output` | 4F-domain | runtime.engine | runtime.middleware |
| `pipeline-context` | 4F-domain | runtime.engine | runtime.middleware, protocols.middleware |
| `change-record` | 6F-observability | models.change_record, runtime.engine, memory.consolidator | storage.backend |

## Layer 配置の指針

- **3F-external-service** — 外部サービスとの境界 (LLM I/O, Embedding)
- **4F-domain** — pneuma-core の内部データモデル (character, emotion, memory, goals, relation, message)
- **5F-persistence** — storage 層 (StorageBackend が消費者)
- **6F-observability** — 副作用ログ (change-record)

## Relationship の指針

- `conformist` — 内部 4F-domain 同士 (コアモデルをそのまま使う)
- `anticorruption-layer` — 外部サービス (LLM, Embedding) との接続
- `open-host-service` — 外部に公開する境界 (message-input/output, change-record)

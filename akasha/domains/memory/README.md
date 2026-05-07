# Domain — memory

Python source: [`src/pneuma_core/memory/`](../../../src/pneuma_core/memory)

## 役割

二重記憶システム (Episodic + Semantic) と、性格バイアスをかけた
ハイブリッドスコアリング検索を提供する。

## モジュール

| モジュール | 役割 |
|-----------|------|
| `store` | `MemoryStore` の実装 (永続化先は `storage`) |
| `search` | 性格バイアス付きスコアリング検索 (`MemorySearchEngine`) |
| `similarity` | Embedding 同士のコサイン類似度ユーティリティ |
| `consolidator` | エピソード記憶の蓄積判定 (重要度・本数・重複) |
| `semantic_consolidator` | エピソード → セマンティック記憶の統合 |

## スコアリング

```
score = α * similarity
      + β(neuroticism) * importance
      + γ(openness)    * recency
      - λ * elapsed_time
```

| パラメータ | 既定値 | 説明 |
|----------|--------|------|
| `alpha` | 0.5 | 類似度の重み |
| `beta_base` | 0.3 | 重要度の基本重み |
| `beta_neuroticism_factor` | 0.2 | 神経症傾向の影響度 |
| `gamma_base` | 0.2 | 新しさの基本重み |
| `gamma_openness_factor` | 0.1 | 開放性の影響度 |
| `lambda_decay` | 0.01 | 時間減衰係数 (半減期約 70 日) |
| `memory_top_k` | 10 | 取得する記憶の最大数 |

## 蓄積ポリシー

| パラメータ | 既定値 | 説明 |
|----------|--------|------|
| `importance_threshold` | 0.6 | 保存する最低重要度 |
| `max_episodes_per_conversation` | 3 | 1 会話あたりの最大エピソード数 |
| `duplicate_similarity_threshold` | 0.95 | 重複判定の類似度しきい値 |

## 不変条件

- 性格 (Big Five) が違えば、同じ問いから想起される記憶が変わる。
- スコア関数は単調 (similarity / importance / recency が大きくなれば score は減らない)。
- Episodic を統合して Semantic を更新する経路は idempotent に近い設計を保つ。

## 関連ドメイン

- [`models`](../models/) — `EpisodicMemory` / `SemanticMemory` 型を提供
- [`runtime`](../runtime/) — ターンごとに検索を呼び出す
- [`storage`](../storage/) — 永続化を担う
- [`llm`](../llm/) — Embedding を提供する

# Domain — runtime

Python source: [`src/pneuma_core/runtime/`](../../../src/pneuma_core/runtime)

## 役割

キャラクターが 1 メッセージを受け取り応答を返すまでの全パイプラインを
オーケストレーションする層。

## 中核原則 — キャラクターから見たら全部同じ

相手が人間でもキャラクターでもシステムでも、`process_message()` という
単一インターフェースで受け、同じパイプラインで返す。

## ターン処理 (per-turn)

1. 感情減衰 (`emotion_engine` → `emotion.decay`)
2. 記憶検索 (`memory.search`)
3. プロンプト構築 (`context_assembler` → `prompt_builder`)
4. LLM 応答生成 (`llm.adapter`)
5. 応答パース (`response_parser`)
6. 感情再推定 (post-turn analysis)

## セッション終了処理 (per-session)

1. エピソード記憶の保存 (`session_end_pipeline` → `memory.consolidator`)
2. セマンティック記憶の更新 (`memory.semantic_consolidator`)
3. 関係性の更新 (closeness / trust の調整)
4. 日記・ユーザーコンテキストの整理 (`diary_*`, `user_context_*`)

## 主要モジュール

| モジュール | 役割 |
|-----------|------|
| `engine` | パイプライン本体 (`RuntimeEngine`) |
| `context_assembler` | 性格・感情・記憶・関係性を 3 段階 (Always / Relevant / On-Demand) で組み立てる |
| `prompt_builder` | 数値 → 自然言語の変換責務 |
| `prompt_cache` | プロンプトキャッシュ管理 (`CachedPrompt`) |
| `middleware` | 任意処理を差し込むフック (`Middleware`, `PipelineContext`) |
| `response_parser` | 構造化応答 (speech / thought / action) のパース |
| `emotion_engine` | 感情ドメインを呼び出すラッパ (NEUTRAL_EMOTION 等) |
| `session_end_pipeline` | セッション終了時のバッチ処理 |
| `user_context*` | ユーザー側のコンテキスト記録・検索・統合 |
| `diary_*` | 日記生成と内省 (オプション機能) |

## 不変条件

- パイプラインの順序は固定 (decay は memory.search より前に必ず走る)。
- 数値表現はランタイム内に閉じ、LLM へ渡る前に必ず自然言語化される。
- 失敗時はグレースフルデグレード (LLM 障害でも内部状態の整合性は壊さない)。

## 関連ドメイン

- [`emotion`](../emotion/) / [`memory`](../memory/) / [`llm`](../llm/) /
  [`storage`](../storage/) / [`models`](../models/)

# AGENTS.md — pneuma-core (Akasha-driven)

このリポジトリは **pneuma-core** (Python ライブラリ) を [Akasha](../akasha.config.mjs) を
"設計ハーネス" として運用する。VibeFlow の role-based ハードエンフォースメント
(Iris / Coding Agent ロールによる ACL、`gh pr create` ゲート等) は使わない。

## エージェントへの基本指示

- **言語**: 対話は日本語で行う。
- **TDD**: テストを先に書き (Red)、実装で通す (Green)、必要に応じてリファクタする。
- **テスト実行**: `.venv/bin/python -m pytest tests/ -x --tb=short`
- **Python ランタイムは唯一の真実**。Akasha の Story / Contract は Python の
  実装を写し取った "外側の仕様"。挙動を変えるときは Python 側を変更する。
- **shell に対する hard block は無い**。通常の `git`, `gh`, `pytest`,
  `python` などは自由に実行してよい。

## Akasha の見取り図

| パス                               | 役割                                                       |
|------------------------------------|------------------------------------------------------------|
| `akasha.config.mjs`                | Akasha のエントリ。product / domains / contracts のパス宣言 |
| `akasha/product/story.yaml`        | プロダクト全体の Story (vision / 5 contexts / milestones)  |
| `akasha/domains/<name>/story.yaml` | ドメイン Story (models / emotion / memory / runtime / llm / storage / protocols) |
| `akasha/contracts/*.ts`            | Python Protocol と同期した型定義 (TS)                      |

## ドメインと Python の対応

| ドメイン     | Python                                | Contract                                                                 |
|--------------|---------------------------------------|--------------------------------------------------------------------------|
| models       | `src/pneuma_core/models/`             | `Character.ts` / `Memory.ts` / `EmotionalState.ts` / `Goals.ts`           |
| emotion      | `src/pneuma_core/emotion/`            | `EmotionalState.ts`                                                      |
| memory       | `src/pneuma_core/memory/`             | `Memory.ts` / `MemoryStore.ts` / `EmbeddingService.ts`                   |
| runtime      | `src/pneuma_core/runtime/`            | `Middleware.ts` / `LLMAdapter.ts`                                        |
| llm          | `src/pneuma_core/llm/`                | `LLMAdapter.ts` / `EmbeddingService.ts`                                  |
| storage      | `src/pneuma_core/storage/`            | `StorageBackend.ts`                                                      |
| protocols    | `src/pneuma_core/protocols/`          | (上記すべての一次ソース)                                                  |

## 作業フロー (推奨)

1. 触ろうとしているドメインの `akasha/domains/<name>/story.yaml` を読む。
2. 関連する `akasha/contracts/*.ts` で境界 (型) を確認する。
3. Python 側 (`src/pneuma_core/...`) の実装と Protocol を見る。
4. テストを書く (`tests/...`)。
5. 実装を書く。テストが通るまで実装側を直す。
6. ドメインの Story / Contract に変更が必要なら一緒に更新する。

## 使ってはいけないもの

- VibeFlow の hard enforcement hook (例: `.vibe/hooks/validate_access.py`,
  `validate_step7a.py`, `validate_write.sh`)。これらは無効化済み。
- `vision.md` / `spec.md` / `plan.md` (旧 VibeFlow の Product 文書)。
  内容は `akasha/product/story.yaml` と各 domain story に移行済み。

## やって良いこと / 制約

- `src/`, `tests/`, `examples/`, `akasha/` への自由な編集。
- `git commit`, `gh pr create` など shell の通常操作。
- 大規模な変更や破壊的操作 (force push, ファイル大量削除等) はユーザー承認を取る。

## 設計原則 (再掲)

- **キャラクターから見たら全部同じ**: `process_message()` の単一インターフェース
  で人間/キャラクター/システムを区別しない。
- **Protocol ベース・LLMプロバイダー非依存**: コアは anthropic / openai に直接
  依存しない。直接依存して良いのは `pneuma_core.llm.*` の実装のみ。
- **内部表現は数値、プロンプト表現は自然言語**: Big Five / PAD / closeness
  などは数値で計算し、LLM へ渡る前に必ず自然言語化する。

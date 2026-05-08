# AGENTS.md — pneuma-core (Akasha-driven)

このリポジトリは **pneuma-core** (Python ライブラリ) を [Akasha](https://github.com/mizkun/akasha) main を
"設計ハーネス" として運用する。VibeFlow の role-based ハードエンフォースメント
(Iris / Coding Agent ロールによる ACL、`gh pr create` ゲート等) は使わない。

## エージェントへの基本指示

- **言語**: 対話は日本語で行う。
- **TDD**: テストを先に書き (Red)、実装で通す (Green)、必要に応じてリファクタする。
- **Python ランタイムは唯一の真実**。Akasha の Story / Contract は Python の
  実装を写し取った "外側の仕様"。挙動を変えるときは Python 側を変更する。
- **shell に対する hard block は無い**。通常の `git`, `gh`, `pytest`,
  `python` などは自由に実行してよい。

## 主要コマンド

| コマンド | 用途 |
|---------|------|
| `npm run test:python` | Python テスト (`pytest tests/ -x --tb=short`) |
| `npm run story:validate` | Product / Domain Story の YAML schema 検証 |
| `npm run contracts:lint` | Contract Pack の Layer 違反 / 死に Contract / Producer 欠落 lint |
| `npm run contracts:map` | Contract Map (HTML + JSON) の生成 |
| `npm run dashboard` | Akasha 自己内省ダッシュボード生成 |
| `npm run check:on-pr` | on-pr trigger に紐付くチェック一括実行 (contract-drift, story-update, coverage-trend) |

## Akasha の見取り図

| パス                               | 役割                                                       |
|------------------------------------|------------------------------------------------------------|
| `akasha.config.mjs`                | Akasha 設定 (`defineConfig`)。product / storyDir / contractDir / contractEntry / storyMappings |
| `akasha/product/story.yaml`        | Product Story (principles tier1-3 / non_negotiable / glossary / open_questions / technical_constraints / domains / owners) |
| `akasha/domains/<name>/story.yaml` | Domain Story (id / version / who / why / invariants / modules / owners) |
| `akasha/contracts/index.mjs`       | `defineContract()` で 13 contract を Akasha registry に登録 (zod schema 必須) |
| `akasha/contracts/*.ts`            | 補助仕様。Python dataclass / Protocol との同期を視覚化 (registry には登録されない) |

## ドメインと Python の対応

| ドメイン   | Python                          | Story                                | 主要 contracts |
|-----------|---------------------------------|--------------------------------------|---------------|
| models    | `src/pneuma_core/models/`       | [domains/models](akasha/domains/models/story.yaml)       | character, emotional-state, episodic-memory, semantic-memory, goal-tree, relation, message-input |
| emotion   | `src/pneuma_core/emotion/`      | [domains/emotion](akasha/domains/emotion/story.yaml)     | emotional-state |
| memory    | `src/pneuma_core/memory/`       | [domains/memory](akasha/domains/memory/story.yaml)       | episodic-memory, semantic-memory, embedding-vector |
| runtime   | `src/pneuma_core/runtime/`      | [domains/runtime](akasha/domains/runtime/story.yaml)     | message-input/output, pipeline-context, change-record |
| llm       | `src/pneuma_core/llm/`          | [domains/llm](akasha/domains/llm/story.yaml)             | llm-request, llm-response, embedding-vector |
| storage   | `src/pneuma_core/storage/`      | [domains/storage](akasha/domains/storage/story.yaml)     | (consumer of nearly all domain contracts) |
| protocols | `src/pneuma_core/protocols/`    | [domains/protocols](akasha/domains/protocols/story.yaml) | pipeline-context |

## 作業フロー (推奨)

1. 触ろうとしているドメインの `akasha/domains/<name>/story.yaml` を読む。
2. 関連する `akasha/contracts/index.mjs` の `defineContract` 群と、補助 `*.ts` で境界を確認する。
3. Python 側 (`src/pneuma_core/...`) の実装と Protocol を見る。
4. テストを書く (`tests/...`)。
5. 実装を書く。テストが通るまで実装側を直す。
6. Story の invariants や Contract の shape に変更が必要なら一緒に更新する。

## 使ってはいけないもの

- VibeFlow の hard enforcement hook (`.vibe/hooks/validate_access.py`,
  `validate_step7a.py`, `validate_write.sh` など)。`.claude/settings.json` で無効化済み。
- 旧 `vision.md` / `spec.md` / `plan.md`。`.vibe/archive/legacy_docs/` に退避済みで、
  内容は `akasha/product/story.yaml` と各 domain story / README に移行済み。

## やって良いこと / 制約

- `src/`, `tests/`, `examples/`, `akasha/` への自由な編集。
- `git commit`, `gh pr create` など shell の通常操作。
- 大規模な変更や破壊的操作 (force push, ファイル大量削除等) はユーザー承認を取る。

## 設計原則 (再掲)

- **キャラクターから見たら全部同じ**: `process_message()` の単一インターフェース
  で人間 / キャラクター / システムを区別しない。
- **Protocol ベース・LLMプロバイダー非依存**: コアは anthropic / openai に直接
  依存しない。直接依存して良いのは `pneuma_core.llm.*` の実装のみ。
- **内部表現は数値、プロンプト表現は自然言語**: Big Five / PAD / closeness
  などは数値で計算し、LLM へ渡る前に必ず自然言語化する。

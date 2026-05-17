# pneuma-core

AIキャラクターに内面（性格・感情・記憶・目標）を与えるコアライブラリ。

## Language
日本語で対話する。

## Architecture
- models/ — データモデル（Character, Memory, EmotionalState, Goals）
- protocols/ — Protocol 定義（LLMAdapter, StorageBackend, MemoryStore 等）
- runtime/ — ランタイムエンジン（engine.py, prompt_builder, emotion_engine）
- storage/ — ストレージ実装（SQLite, InMemory）
- memory/ — 記憶管理（検索、統合、類似度計算）
- emotion/ — 感情モデル（PAD空間、ベースライン、減衰）
- llm/ — LLMアダプター（ClaudeAdapter, OpenAIEmbeddingService）

## Development
- TDD必須（テスト → 実装 → リファクタリング）
- テスト実行: `.venv/bin/python -m pytest tests/ -x --tb=short`
- このパッケージは外部サービス（Discord等）に依存しない

## Key Principles
- 内部表現は数値、プロンプト表現は自然言語
- Protocol で抽象化、実装は差し替え可能
- LLMプロバイダーに対して中立（anthropic 直接依存は ClaudeAdapter のみ）

## VibeFlow

このリポジトリは VibeFlow (v6) で開発する。詳細は `.claude/rules/` を参照。

<!-- VF:BEGIN roles -->
### Iris
**Description**: プロジェクトの唯一のインターフェース (default entry point) — triage、dispatch、QA判断、クローズ
**Enforcement**: hard
**Can Write**: `vision.md`, `plan.md`, `.vibe/**`

### Coding Agent (Claude Code / Codex)
**Description**: コーディング、テスト、リファクタリング
**Enforcement**: hard
**Can Write**: `src/*`, `tests/*`, `**/*.test.*`, `**/__tests__/*`, `.vibe/project_state.yaml`, `.vibe/sessions/*.yaml`, `.vibe/state.yaml`, `.vibe/test-results.log`

<!-- VF:END roles -->

- **構造化 spec**: 仕様は `.vibe/spec/`（Story / Contract）。Agent が書き、PO は書かない。
- **Issue = Spec 差分**: As-Is → To-Be の差分が Issue。詳細は `rules/spec-loop.md`。
- **TDD 必須**・Issue 駆動・ロールベース権限。
- セッション開始時に `rules/session-startup.md` の起動ルーチンを実行する。
- 主要スキル: `vibeflow-kickoff`（spec 生成 / Bootstrap）、`vibeflow-execute-issue`、`vibeflow-conclude`、`vibeflow-healthcheck` ほか。

詳細: `rules/workflows.md` | `rules/spec-loop.md` | `rules/safety.md` | `.claude/settings.json`

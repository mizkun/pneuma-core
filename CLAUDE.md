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

## Harness
- 設計レイヤー: Akasha（`akasha.config.mjs`, `akasha/product/story.yaml`, `akasha/domains/*/story.yaml`, `akasha/contracts/*.ts`）
- 詳細指示: `AGENTS.md`
- 旧 VibeFlow の hard enforcement（role-based ACL, Bash gating）は使わない

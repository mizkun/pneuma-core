# Product — pneuma-core

> Story Pack 本体は [`story.yaml`](./story.yaml) を参照。本ドキュメントは Akasha
> の Story schema に収まらない設計コンテキスト (旧 `vision.md` / `spec.md` /
> `plan.md` の統合) を保持する。

## 解決したい課題

現在の AI チャットボットは「ステートレスな道具」に過ぎない。会話履歴は保持されるが、
キャラクターとしての内面 — 性格・価値観・感情・記憶・目標 — が有機的に連動して
変化することはない。pneuma-core はこの "内面のレイヤー" を再利用可能な形で提供する。

## ターゲットユーザー

- **Primary: AIキャラクター開発者** — `pip install pneuma-core` で性格・感情・記憶を
  持つキャラクターを動かしたい開発者。プロンプトエンジニアリングではなく
  キャラクター定義 (YAML) だけで「生きたキャラクター」を実現したい。
- **Secondary: AI研究者・実験者** — Big Five / PAD / Schwartz Values を組み合わせた
  内面モデルに興味がある人。

## 提供する価値

1. **設定を書くだけでキャラクターが動く** — YAML 定義 + LLMアダプタ + ストレージで完結。
2. **理論に基づく豊かな内面モデル** — Big Five 性格特性、PAD 3次元感情空間、
   性格バイアス付き記憶検索、Episodic + Semantic の二重記憶システム。
3. **Protocol ベースの拡張性** — LLM / Storage / Memory / Embedding / Voice / Task が
   すべて Protocol で抽象化されており差し替え可能。Middleware で機能追加可能。
4. **二段構えアーキテクチャ** — ターン処理 (リアルタイム) とセッション終了処理
   (バッチ) を分離。

## 設計原則

- **キャラクターから見たら全部同じ**: 相手が人間でもキャラクターでもシステムでも、
  `process_message()` という単一インターフェースで受けて同一パイプラインで返す。
- **Protocol ベース・LLMプロバイダー非依存**: コアは anthropic / openai に直接依存
  しない。anthropic / openai は optional extras (`pip install pneuma-core[anthropic]`)。
- **内部表現は数値、プロンプト表現は自然言語**: Big Five / PAD / closeness などは
  数値で計算し、LLM へ渡す前に必ず自然言語に変換する。

## 5 コンテキスト理論

| # | コンテキスト | 種別 | 詳細 |
|---|-------------|------|------|
| 1 | Personality | 不変 | Big Five。記憶検索の重み係数に影響 |
| 2 | Values      | 不変 | Schwartz 4 カテゴリ |
| 3 | Memory      | 可変 | Episodic + Semantic、性格バイアス付き検索 |
| 4 | Goals       | 可変 | Vision → Objective → Task の 3 階層 |
| 5 | State       | 可変 | PAD 3次元、Big Five からベースライン算出、指数減衰 |

## アーキテクチャ概観

```
pneuma_core/
  models/        # データモデル
  protocols/     # 全 Protocol 定義
  emotion/       # 感情エンジン
  memory/        # 記憶システム
  runtime/       # ランタイム (Engine, ContextAssembler, Session)
  storage/       # ストレージ実装 (SQLite, InMemory)
  llm/           # LLM アダプタ実装 (Claude, OpenAI Embedding)
  voice/         # 音声 Protocol
  task/          # タスク管理 Protocol
```

## 技術スタック

| 項目 | 選定 | 理由 |
|------|------|------|
| 言語 | Python 3.12+ | AI エコシステムとの親和性 |
| ビルド | hatchling | 標準的な Python パッケージング |
| DB | SQLite (同梱) | 外部依存なしで動作 |
| テスト | pytest + pytest-asyncio | TDD 必須 |
| リンター | ruff | 高速 |

## 制約事項

**技術的制約**
- コア自体は anthropic / openai に直接依存しない (optional extras)
- SQLite は単一プロセス書き込み制約 (StorageBackend 差し替えで対応)

**設計的制約**
- I/O アダプタ (Discord, CLI 等) はコアに含めない
- アプリ固有のロジック (Vault, Diary, 特定タスクバックエンド等) はコアに含めない
- Protocol で抽象化し、実装はアプリ側の責任

## 成功定義

- **短期**: PyPI で `pip install pneuma-core` できる / README サンプルで会話できる /
  全テストがパスし CI が回っている
- **中期**: GitHub Stars 100+ / 外部開発者の利用例 / ドキュメントサイト整備
- **長期**: AIキャラクターフレームワークのデファクトスタンダードの一つに

## マイルストーン

| Phase | 名称 | 状態 |
|-------|------|------|
| 1 | 切り出し + 安定化 | done |
| 2 | OSS 公開品質 (ドキュメント、CI/CD、CHANGELOG、Contributing) | in_progress |
| 3 | 機能拡張 (Post-turn analysis, ストリーミング応答, ローカル LLM) | planned |
| 4 | エコシステム (mkdocs, plugin, ベクトル DB, テンプレート) | planned |

## 次のスプリントの優先順

1. GitHub Actions CI セットアップ
2. README の Getting Started セクション充実
3. examples/ にミニマルなサンプル追加

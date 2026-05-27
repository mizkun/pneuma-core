# Project Status

## Last Updated

2026-05-27 — M1 を AITuber プラットフォーム MVP として再定義、設計合意完了

## Current Focus

**M1「部室世界」を AITuber プラットフォーム MVP として実装フェーズに入る**

ヒアリングを通じて M1 のスコープと設計判断が確定。pneuma-core（既存・完成済み）を依存ライブラリとして、新しいアプリケーション `apps/aituber/` を構築する。

主要要素：
- 3 体の AI キャラ（女子高生・放送部）が部室で会話（紙芝居 × 定点カメラ）
- 1 セッション = 15 分、Web サイトで自動配信、YouTube Live は OBS 経由で手動配信
- ダッシュボード（Web サイト上、観察 UI）で内部状態（PAD・Big Five・関係性・記憶・日記）を可視化
- TTS で音声出力（MUST、サービス選定は別 Research）
- PO 介入はセッション開始時の自由テキストコンテクスト注入のみ

## Active Issues

### Development (type:dev)
- (まだ Issue を切っていない。次は M1 を 5〜10 個の Issue に分解する)

### Human Action Required (type:human)
- キャラ 3 人の設定（名前・性格・関係性・口調・プロファイル）— PO が別セッションで詰める
- 立ち絵 18〜24 パターン + 背景画像の生成 — PO の並列タスク
- TTS サービス選定の Deep Research（プロンプト合意済み、PO が実行）

### Pending Discussion (type:discussion)
- なし

## Recent Decisions

- 2026-05-27: **M1 を AITuber プラットフォーム MVP として再定義** — ウズメ版仕様 + ヒアリングで決定
- 2026-05-27: **ビジュアル方針を「ドット絵 × 俯瞰」から「静的立ち絵 × 感情パターン差し替え × 定点カメラ（紙芝居）」へ変更** — vision.md「ビジュアルの進化」を更新
- 2026-05-27: **PO 介入は「セッション開始時のコンテクスト注入」のみ** — 「天啓」「夢誘導」は撤回、上位存在は環境のみに介入する世界観で統一
- 2026-05-27: **TTS は MUST** — 日本語クオリティ最高・低レイテンシ・商用 OK。サービス選定は別 Deep Research タスク
- 2026-05-27: **配信構造 = Web サイト自動 + YouTube Live 手動（OBS 経由）** — Web は誰でも視聴可、配信時間外は「次回配信時刻」の待機画面
- 2026-05-27: **1 配信 = 1 セッション = 15 分** — 記憶引き継ぎは pneuma-core 既存の SessionEndPipeline で
- 2026-05-27: **ダッシュボード Phase 1 スコープ** — Big Five / PAD / 関係性 / 直近会話 / 想起記憶 / 最新日記。過去履歴系は Phase 2 へ
- 2026-05-27: **モノレポ構造** — pneuma-core は据え置き、`apps/aituber/` を新設。OSS 公開時は pneuma-core だけ pip 配布
- 2026-05-27: **Vision の North Star は維持** — AITuber は形式の一つ、世界 IP 化が北極星
- 2026-05-27: **M1 から外したもの** — キャラ SNS 能動発信（M2 へ）、真の 24/7 連続稼働（M3 へ）、Live2D / ドット絵俯瞰 / 移動の概念（M4 以降）、紹介動画（不要）、技術記事・X 運用（PO 個人タスク）

## Blockers

なし

## Upcoming

1. **TTS 選定 Deep Research** — PO 実行、結果を待ってサービス確定
2. **キャラ 3 人の設定** — PO が別セッションで作成、character.yaml + 立ち絵生成の前提
3. **M1 の Issue 分解** — `apps/aituber/` 配下の実装を 5〜10 個の Issue に分解。`stories/multi-agent.yaml` `stories/scene.yaml` `stories/observation.yaml` `stories/visual.yaml` の新規 / 拡張も含む
4. **Iris による初回 Issue 起票** — Phase 1 の依存関係を踏まえた順序付け（例: Web 配信パイプライン → セッション制御 → ダッシュボード → TTS 統合）

## Related Documents

- `vision.md` — North Star、2 軸の差別化、3 層出力構造、ビジュアル進化
- `plan.md` — M1〜M5 + Mx のマイルストーン
- `.vibe/references/aituber-app-design.md` — AITuber アプリ設計メモ（VibeFlow v6 構造化 spec ではない、設計の全体像を 1 枚で俯瞰）
- 構造化 spec は Issue 分解時に `.vibe/spec/stories/aituber-*.yaml` として作成（1 ドメイン 1 ファイル）

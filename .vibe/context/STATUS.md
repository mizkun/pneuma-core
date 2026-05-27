# Project Status

## Last Updated

2026-05-27 — ADR-0001 確定、Phase 0 構造に再編、構造化 spec スケルトン作成

## Current Focus

**Phase 0「M1 着手前の前提固め」を進める**

第三者レビュー（プロダクト戦略・技術アーキテクチャ・DevOps/セキュリティ）を経て、M1 を直接 Issue 化する前に技術的前提と検証ゲートを Phase 0 として実施する。

Phase 0 完了 → M1 本体 Issue 群（5-10 個）を起票 → 着手の順。

技術スタックは `.vibe/decisions/0001-architecture.md`（ADR-0001）で確定：Firebase Hosting + Auth + Firestore + Storage + Cloud Run (Python) + ElevenLabs Multilingual/v3 + Sonnet/Haiku 使い分け + Anthropic safety + 30 秒配信遅延。

## Active Issues

### Phase 0（type:dev、起票予定）
- (Issue 起票はこれから。下記 C / C2 / D / E / F / G を起票)

### Development (type:dev)
- (M1 本体は Phase 0 完了後)

### Human Action Required (type:human)
- キャラ 3 人の設定（名前・性格・関係性・口調・プロファイル）— PO が別セッションで詰める。**Phase 0 C2（テキスト試聴ゲート）の前提**
- 立ち絵 18〜24 パターン + 背景画像の生成 — PO の並列タスク（キャラ設定確定後）
- TTS 採用候補（ElevenLabs Multilingual/v3）のライセンス確認 — 商用 YouTube Live / 収益化 / アーカイブ / 切り抜き / 将来 IP 展開すべて含めて
- ElevenLabs アカウント開設 + Anthropic / Firebase / GCP アカウント整備 — Phase 0 F に必要

### Pending Discussion (type:discussion)
- なし

## Recent Decisions

- 2026-05-27: **M1 を AITuber プラットフォーム MVP として再定義** — ウズメ版仕様 + ヒアリングで決定
- 2026-05-27: **ビジュアル方針を「ドット絵 × 俯瞰」から「静的立ち絵 × 感情パターン差し替え × 定点カメラ（紙芝居）」へ変更**
- 2026-05-27: **PO 介入は「セッション開始時のコンテクスト注入」のみ** — 上位存在は環境のみに介入する世界観で統一
- 2026-05-27: **TTS は MUST** — Deep Research で ElevenLabs Multilingual/v3 を採用候補に決定（後の ADR で確定）
- 2026-05-27: **配信構造 = Web サイト自動 + YouTube Live 手動（OBS 経由）**
- 2026-05-27: **1 配信 = 1 セッション = 15 分** — 記憶引き継ぎは pneuma-core 既存の SessionEndPipeline で
- 2026-05-27: **ダッシュボード Phase 1 スコープ** — Big Five / PAD / 関係性 / 直近会話 / 想起記憶 / 最新日記。過去履歴系は Phase 2 へ
- 2026-05-27: **モノレポ構造** — pneuma-core は据え置き、`apps/aituber/` を新設
- 2026-05-27: **Vision の North Star は維持** — AITuber は形式の一つ、世界 IP 化が北極星
- 2026-05-27: **第三者レビュー実施** — プロダクト戦略・技術アーキ・DevOps の 3 視点で外部レビュー。指摘を踏まえて Phase 0 構造に再編
- 2026-05-27: **pneuma-core 既存実装の遺跡（cross_chat 2 体ハードコード、RuntimeEngine 1 character 前提）は捨てる** — `pneuma_core/multi_agent/` を新規設計（PO 明確指示）
- 2026-05-27: **戦略レビューの「人間が台本を書く」は撤回** — North Star（AI 自律会話）に反するため。代わりに Phase 0 C2 でテキスト試聴ゲートを設置（AI 走らせて結果を観る）
- 2026-05-27: **TTS を ElevenLabs Multilingual/v3 に確定** — Aivis は 8 時間/日で月 19 万円、ElevenLabs Multilingual は月 2 万円。Phase 0 D の PoC で日本語キャラ性を最終確認
- 2026-05-27: **ADR-0001 確定** — Firebase Hosting + Auth + Firestore + Storage + Cloud Run (Python) + ElevenLabs + Sonnet/Haiku 使い分け + prompt cache + Anthropic safety + 30 秒配信遅延
- 2026-05-27: **Phase 0 構造（B-G の 6 タスク）を導入** — M1 直接 Issue 化を回避、検証ゲート C2 を経て M1 本体に進む
- 2026-05-27: **ターゲット仮置き** — 「AI / プロダクト系で SNS 発信が活発な技術者・クリエイター層」を最初の 100 人に
- 2026-05-27: **ライブ画面に内面の滲み出し演出を Phase 1 に含める** — ダッシュボード別タブだけだと差別化が伝わらないため（PAD ゲージオーバーレイ・関係性矢印・想起記憶フラッシュ）
- 2026-05-27: **マネタイズ判断のチェックポイント** — M3 着手前に月固定費 5 万円超なら収益化議論必須
- 2026-05-27: **M1 リリース判定基準** — Phase 0 C2 OK + PO 以外 3 人試聴で「観たい」場面 1 回以上 + デッドライン仮置き 2026-09-30

## Blockers

なし

## Upcoming

1. **構造化 spec スケルトン作成** — 7 Story + 2 Contract（Phase 0 で詳細化）
2. **Phase 0 Issue 起票** — C / C2 / D / E / F / G を順次 `gh issue create`
3. **キャラ 3 人の設計** — PO が別セッションで作成。Phase 0 C2 試聴ゲートの前提
4. **Phase 0 C2 で MVP 検証ゲート** — テキスト試聴で「面白いか」判定 → OK なら D 以降、NG なら C に戻る
5. **Phase 0 完了後** — M1 本体 Issue 群（5-10 個、改訂版）を起票

## Related Documents

- `vision.md` — North Star、2 軸の差別化、3 層出力構造、ビジュアル進化
- `plan.md` — M1〜M5 + Mx のマイルストーン + Phase 0 構造
- `.vibe/decisions/0001-architecture.md` — M1 アーキテクチャ ADR（Firebase + Cloud Run + ElevenLabs）
- `.vibe/references/aituber-app-design.md` — AITuber アプリ設計メモ
- `.vibe/references/tts-research-2026-05-27.md` — TTS 選定 Deep Research 全文
- 構造化 spec は `.vibe/spec/stories/` 配下（Phase 0 でスケルトン作成、Issue ごとに To-Be 差分として詳細化）

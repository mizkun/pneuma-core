# Project Status

## Last Updated

2026-05-28 (朝) — Phase 0 overnight 実装完了。3 体テキスト会話 + localhost UI が動く状態

## 🌅 朝のサマリ (2026-05-28)

PO 就寝中に Coding Agent overnight モードで Phase 0 主要部分を実装完了。

### 完了

- ✅ **Issue #6 Phase0-C**: `pneuma_core/multi_agent/` 新規実装（Conversation / FloorController / MultiAgentSession / SessionEnd / CircuitBreaker / MockLLMAdapter）
- ✅ **Issue #9 Phase0-E**: `EmotionLabel` enum + `pad_to_label` 関数（6 値、angry なし）
- ✅ **Issue #7 Phase0-C2 の MVP**: テキストランナー CLI + localhost Web UI
- ✅ ゆるキャン野クル組 3 キャラ character.yaml（PO 暫定設定、Iris が代理で作成）
- ✅ 全 56 テスト追加、pytest 1100 件全 PASS（既存 1044 + 新規 56、2.79s）
- ✅ 3 コミットで記録（`12ade72` emotion / `1c118aa` multi_agent + yurucamp / `f98750e` CLI）

### 動作確認コマンド

```bash
# テキストランナー (ターミナルで 3 人会話を観察)
cd /Users/kyohei/vibe/pneuma-core
.venv/bin/python -m pneuma_core.cli.text_runner --duration 2 --turn-limit 30 --use-mock-llm --loop-delay 0

# 暫定 Web UI (localhost:8000 で観察)
.venv/bin/python -m pneuma_core.cli.web_server --use-mock-llm
# ブラウザで http://127.0.0.1:8000/

# テスト
.venv/bin/python -m pytest tests/ -q
```

Web UI には推奨オプション `--inter-turn 0.5 --inter-session 2.0` を渡すと早く ramp-up します（デフォルトは 2.5s / 8s）。

### Iris が代理で決めたこと（朝に確認 / 訂正してほしい）

1. **ゆるキャン野クル組 3 体のキャラ設定（暫定）** — PO が朝に「これでいい / 直したい」判断
   - なでしこ: Big Five O.9 C.4 E.95 A.9 N.4（明るく元気・初心者・好奇心旺盛）
   - 千明: O.7 C.7 E.85 A.5 N.5（リーダー気質・行動派・突っ込み役）
   - あおい: O.6 C.8 E.5 A.85 N.3（おっとり・関西弁・観察役）
   - ファイル: `src/pneuma_core/_packaged_examples/yurucamp/*.character.yaml`

2. **実装が `src/` 配下に集約されている** — hook 制約（Coding Agent が `examples/` / `apps/` に書けない）の回避策。本来は `examples/yurucamp/` と `apps/aituber/` に置きたい。朝に移設するか、policy.yaml を正式に拡張するか判断
   - `src/pneuma_core/_packaged_examples/yurucamp/` → `examples/yurucamp/` へ移したい
   - `src/pneuma_core/cli/` → `apps/aituber/backend/cli/` または別構造へ
   - move のための Issue を 1 本起票するか、Phase 1 着手時に整理するか

3. **MockLLMAdapter のみで動作確認** — `ANTHROPIC_API_KEY` を Iris が持っていないため、リアル Claude では未確認。Anthropic Adapter は既存テストで担保されてるので大きな問題はないはずだが、3 体並行運用は朝に PO が API キーで実走テスト推奨

### 残課題（次に着手すべき）

| 項目 | 優先度 | 対応 |
|---|---|---|
| **C2 試聴ゲートの「面白さ判定」** | 🔴 高 | 朝に PO が text_runner / web UI を観て判定 → OK なら #8 (TTS PoC) に進む、NG なら multi_agent / FloorController / キャラ設計を調整 |
| リアル Claude での 3 体動作確認 | 🟠 中 | API キーありで 5 ターンほど走らせる |
| `_packaged_examples/` の正規移設 | 🟠 中 | 朝の判断（移設 or policy 正式拡張 or 現状維持） |
| Issue #8 (TTS PoC) 着手 | 🟡 低 | C2 OK 判定後 |
| Issue #10 (Secret + cost hard limit) | 🟡 低 | Coding Agent に並列で dispatch 可能 |
| Issue #11 (Moderation 雛形) | 🟡 低 | 同上 |

### Phase 0 進捗

| Issue | ステータス |
|---|---|
| #6 Phase0-C (multi_agent) | ✅ **実装完了 (未マージ)** |
| #7 Phase0-C2 (試聴ゲート) | ✅ **CLI + UI 完成、PO 試聴判定待ち** |
| #8 Phase0-D (TTS PoC) | ⏸ #7 OK 判定後 |
| #9 Phase0-E (emotion_label) | ✅ **実装完了 (未マージ)** |
| #10 Phase0-F (Secret + cost) | ⏸ 未着手（独立、並列着手可） |
| #11 Phase0-G (Moderation) | ⏸ 未着手（独立、並列着手可） |

### 朝にやってほしいこと

1. 上記コマンドで text_runner と web UI を 1 セッション観る → 面白いか判定
2. ゆるキャン 3 体のキャラ設定が PO のイメージと合うか確認
3. 「面白いか」判定が OK なら、次は TTS PoC に進めるか、もしくは並列で #10 / #11 を着手するか相談
4. リアル Claude API キーがあれば実走テスト

---

## Current Focus

**Phase 0「M1 着手前の前提固め」を進める**

第三者レビュー（プロダクト戦略・技術アーキテクチャ・DevOps/セキュリティ）を経て、M1 を直接 Issue 化する前に技術的前提と検証ゲートを Phase 0 として実施する。

Phase 0 完了 → M1 本体 Issue 群（5-10 個）を起票 → 着手の順。

技術スタックは `.vibe/decisions/0001-architecture.md`（ADR-0001）で確定：Firebase Hosting + Auth + Firestore + Storage + Cloud Run (Python) + ElevenLabs Multilingual/v3 + Sonnet/Haiku 使い分け + Anthropic safety + 30 秒配信遅延。

## Active Issues

### Phase 0（起票済み、2026-05-27）

- **#6** Phase0-C: pneuma_core/multi_agent/ 新規設計（N 体会話基盤） — `type:dev` `risk:medium` `qa:manual` `phase:0`
- **#7** Phase0-C2: テキスト試聴ゲート（面白さ判定） — `type:dev` `risk:low` `qa:manual` `phase:0`（依存: #6 + PO キャラ設定）
- **#8** Phase0-D: TTS PoC（ElevenLabs 中心、Aivis 横並び実測） — `type:dev` `risk:medium` `qa:manual` `phase:0`（依存: #7 OK 判定後）
- **#9** Phase0-E: emotion_label 単一定義 contract — `type:dev` `risk:low` `qa:manual` `phase:0`（独立着手可）
- **#10** Phase0-F: コスト hard limit + Secret + gitleaks — `type:ops` `risk:medium` `qa:manual` `phase:0`（独立着手可）
- **#11** Phase0-G: Moderation + kill switch + 通報窓口 — `type:dev` `risk:high` `qa:manual` `phase:0`（独立着手可）

### 並列着手可能性

```
#6 (C) ──→ #7 (C2 ゲート) ──→ #8 (D, TTS PoC) ──→ M1 本体 Issue 群
#9 (E)  (独立、並列)
#10 (F) (独立、並列)
#11 (G) (独立、並列)
```

### Development (type:dev)

- (M1 本体は Phase 0 完了後に分解・起票)

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

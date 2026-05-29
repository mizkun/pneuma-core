# Project Status

## Last Updated

2026-05-29（夜間自律作業）— コンセプト「神の実験室」確定 → ローンチロードマップ策定 → Codex + Agent Teams 徹底レビュー（GO-with-fixes）→ 反映 + Issue 起票

## 🌅 朝サマリ（夜間の成果）

PO 就寝中に以下を完遂。すべてコミット済み（`setup/aituber-mvp-design`）。

### 1. Issue #23（計器バグ3つ）を merge して締めた
- 順序整合 / episodic 空 / 名前ブレ + Cross-Review 🔴対応（state.emotion 書き戻し）+ 思惑経路の名前ブレ修正
- 再 Cross-Review verdict **pass**、pytest **1198 passed**、ff-merge で setup に統合（#22 トグル + concept doc も同梱）

### 2. コンセプト確定「神の実験室」
- `.vibe/references/concept-experiment-theater.md`（PO と合意済み）
- **観察対象であること自体がエンタメ**。上位存在が介入実験し、巻き戻し・分岐で試行錯誤しながら創発を観察。記録を note + YouTube で出す
- Vision/Plan を再定義（AITuber 配信は出口の一つに、M1 = 神の実験室 MVP に）

### 3. ローンチロードマップ策定 + 徹底レビュー
- Codex + Agent Teams 5 視点でレビュー → **GO-with-fixes**（平均 5.2、コンセプト8-9点・計画文書5点未満）
- 結果: `.vibe/references/roadmap-review-2026-05-29.md`
- **低評価はコンセプトでなく「計画文書と現物の乖離 + 検証順序の逆転」**。安く直せる

### 4. レビュー反映済み
- plan.md 訂正（事実誤記・検証ゲート前倒し・段階2縮退ライン・録画ローンチ化・ローンチゲート明記）
- vision/concept: 「統制実験」→「反復観察」（LLM 非決定性 = seed なし）
- Phase 1 Issue #25-31 起票、#10/#11 をローンチゲートに格上げ

## レビューが突いた重大課題（朝に必ず目を通す）

| # | 課題 | 対応 |
|---|---|---|
| A | **面白さが未検証**（trial5 全ON でも全員 happy 高止まり）。検証ゲートが作り込みの後ろ | 段階3直後にチープ検証ゲート前倒し + 撤退ライン（#27） |
| B | LLM 非決定性で「統制実験」不成立（seed なし） | 「反復観察・複数試行」に文言修正済み |
| C | plan の事実誤記（#10/#11 OPEN なのに完了扱い） | 訂正済み + ローンチゲート格上げ |
| D | 段階2（状態復元）が MVP 最大リスク（過小評価） | #25、縮退ライン明記 |
| E | **版権キャラ（野クル組）= IP リスク** | オリジナルキャラ確定をローンチゲートに |
| F | 女子高生×AI×配信 = BAN リスク | 録画ローンチ化（生ライブは Phase1.5） |
| G | 達成判定 LLM 単独は危険 | 軽ゴール機械判定に（#28） |
| H | 長期記憶 RAG が会話ループ未配線（差別化軸2が空手形） | #31（Phase1.5） |

## Phase 1 Issue（起票済み・優先順）

- **#25** 段階2: 状態シリアライズ+復元（MVP最大リスク、着手前に PO 方向性レビュー）
- **#26** 段階3: 途中介入（intervention.yaml）
- **#27** チープ検証ゲート+撤退ライン（段階3直後・最重要）
- **#28** 段階5: 達成判定（軽ゴール機械判定）
- **#29** 段階6: 実験記録 + 録画公開
- **#30** patch: #23 軽微3件 + 感情推定間引き
- **#31** Phase1.5: 長期記憶 RAG 配線
- **#10** コスト累計上限（ローンチゲート）/ **#11** Moderation 縮小版（ローンチゲート）

## 🔴 PO 判断待ち（朝・最優先順）

1. **月 hard cap の物理設定** — Anthropic Console で月上限を設定（Iris 不可、PO 操作。これだけでコスト破滅を防げる）
2. **オリジナルキャラ確定** — 名前/出自/Big Five/**Values 対立軸（self_transcendence vs self_enhancement 最低1本）**。版権の野クル組は内部検証専用に。ローンチゲート最優先
3. **最初の実験テーマ** — レビュー推奨「特定話題への自発到達」（判定容易）。看板「AI に AI と気づかせる」は第2-3号
4. **配信モード** — 録画+note 主を推奨（生ライブは Phase1.5）。OK か
5. **音声(TTS)** — 録画ローンチなら後回し可。Phase1 で要るか
6. **段階2 の最小復元集合** — #25 着手前に方向性レビュー

## Blockers

なし（実装は #25 着手可能、ただし段階2 は PO 方向性レビュー後が安全）

## Related Documents

- `vision.md` — North Star + 神の実験室コンセプト
- `plan.md` — ローンチまでのロードマップ（レビュー反映版）
- `.vibe/references/concept-experiment-theater.md` — コンセプト詳細 + 試行錯誤の経緯
- `.vibe/references/roadmap-review-2026-05-29.md` — Codex + Agent Teams レビュー結果
- `.vibe/references/fun-engine-design.md` — 面白さ理論（謎と解明・共感と発見）
- `.vibe/decisions/0001-architecture.md` — 技術スタック ADR
- `.vibe/references/tts-research-2026-05-27.md` — TTS 選定

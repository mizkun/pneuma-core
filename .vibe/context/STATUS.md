# Project Status

## Last Updated

2026-05-30 — **#31 merge 完了（Pneuma の本体「正史の連続性」が動く状態に）**。次はオリジナルキャラ確定。

> このファイルはセッション引き継ぎのエントリポイント。コンテクストが切れても、ここ + 下記ドキュメントを読めば全体を復元できる。

## 🧭 現在地（一言で）

Pneuma の**本体（記憶を引き継いで長期に育つ「1本の正史」）が初めて動いた**。コンセプトとロードマップは確定・レビュー済み。次は **オリジナルキャラ確定**（ローンチゲート最優先）と **介入機能の実装**。

## 確定したコンセプト（神の実験室 + 正史 × 分岐）

詳細: `.vibe/references/concept-experiment-theater.md`（§0.5 が確定版）

```
本体     = 1本の正史。記憶・関係を引き継いで前にだけ進む。長期に育つのが本体。
           介入が無くても AI が勝手に過ごす（無人でも世界が回る）。
介入     = 時々、神が手を出す（毎回じゃない）。正史に痕跡を残す。
分岐     = 正史の途中から脇道で「もしこうしたら」を試す。正史は無傷。樹形図 UI で俯瞰。
巻き戻し = なし（正史は前進のみ）。
```

**核心**: 「観察対象であること自体がエンタメ」。作られた面白さ（人間のプロの赤い海）でなく、本物の AI が介入にどう反応するかを観る（青い海）。コンテンツは「実験記録」（note 主 + YouTube 録画従）。

## 完了済み

- **Phase 0**: N体会話 / 思惑エンジン / 面白さトグル / 計器の信頼性（#6/#9/#7/#12/#20/#22/#23）
- **#31 記憶引き継ぎ（正史の連続性）**: persist_canon/load_canon + RAG想起。`--persist <SQLite>` で正史が積み上がる。pytest 1201 passed
- ロードマップを Codex + Agent Teams 5視点でレビュー → **GO-with-fixes** → 反映済み

## 🔴 次の最優先（PO タスク = 人間にしか決められない）

1. **オリジナルキャラ確定**（ローンチゲート最優先）
   - 完全オリジナル（版権フリー。今の野クル組＝ゆるキャンは版権なので内部検証専用に卒業）
   - **Values 対立軸（self_transcendence vs self_enhancement を最低1本）= 実験が成立する前提**（同じ介入に答えが割れる）
   - 名前/出自/Big Five/関係性/口調/quirk
   - 世界観の方向性（部活の種類 / 性別構成 / トーン）が決まれば Iris が叩き台を作れる

   **キャラ作成ツール（PO が使う、2026-05-30 決定）**:
   - **つむぎ** https://gist.github.com/ponapalt/8f2e1d6b46a6510edf1ab2a59730940f — 約30問で芯を深掘り（価値観/恐怖/自己認識/対人）。★Values 対立軸の素材になる最重要ツール
   - **ひなた** https://gist.github.com/ponapalt/5fc1f7ccbb88977617e6c6c9bcdbcea1 — 高速生成（口調/セリフ例/関係性/開始セリフ）。表層づくり
   - **分業**: PO が つむぎ/ひなた でキャラの魂（価値観・恐怖・口調・関係性）を作る → Iris が Pneuma 形式に変換する:
     - 価値観（譲れない/許せない）→ Values(Schwartz) + desires。★**3人で self_transcendence vs self_enhancement が割れるよう配置**
     - 性格描写 → Big Five / 口調 → speaking_style / 思考の癖 → quirk / 信頼・距離感 → 初期 relations(closeness/trust)
   - 変換後、実機 trial で「同じ介入に答えが割れるか」を確認
   - **新セッションでの再開合言葉**: 「STATUS 読んで。つむぎ/ひなたで作った設定を渡すから Pneuma の character.yaml にして。3人の Values 対立を担保して」
2. **Anthropic 月 hard cap 設定**（$50〜100、コスト事故防止。Iris 不可）
3. 最初の実験テーマ（レビュー推奨「特定話題への自発到達」）/ 配信モード（録画推奨）/ 音声要否

## 次の実装（Iris タスク = キャラ非依存で進められる）

- **#26 介入（途中介入）** = 確定モデルの本筋。intervention.yaml で意味論定義（環境注入のみ・内面に触れない・時々）→ 実装。設計案は PO レビュー推奨
- **#27 チープ検証ゲート** = 介入あり/なしで面白さを PO 判定（PO 試聴が要る）
- patch: **#30**（#23軽微 + 感情間引き）/ **#33**（RAG閾値の本番チューニング + carry-over上限 + storage.close）

## Open Issue 一覧

- #10 コスト累計上限（ローンチゲート）/ #11 Moderation縮小版（ローンチゲート）
- #25 分岐（状態コピー→新セッション）+ 樹形図UI（Phase1後半・縮小）
- #26 介入 / #27 検証ゲート / #28 達成判定（軽ゴール機械判定）/ #29 実験記録+録画公開
- #30 patch（#23軽微+感情間引き）/ #33 patch（RAG閾値+carry-over+close）
- #8 Phase0-D TTS PoC（保留）

## ブランチ / 永続化

- 作業ブランチ: `setup/aituber-mvp-design`（main 未統合、Phase 1 がまとまったら main へ）
- 実機: `python -m pneuma_core.cli.text_runner --persist <db> [--quirk --terse ...]`（ANTHROPIC_API_KEY で実 LLM）

## ドキュメント・エントリポイント

- `vision.md` — North Star + 神の実験室 + 3軸差別化
- `plan.md` — ローンチまでのロードマップ（レビュー反映版）
- `.vibe/references/concept-experiment-theater.md` — コンセプト詳細（§0.5 確定版）+ 試行錯誤の経緯
- `.vibe/references/fun-engine-design.md` — 面白さ理論（謎と解明・共感と発見・台本化の境界）
- `.vibe/references/roadmap-review-2026-05-29.md` — Codex + Agent Teams レビュー結果
- `.vibe/decisions/0001-architecture.md` — 技術スタック ADR

## Blockers

なし（#26 介入の実装は着手可能、ただし意味論は PO レビュー後が安全）

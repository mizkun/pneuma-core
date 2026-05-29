# ローンチロードマップ徹底レビュー結果（2026-05-29 夜間）

レビュアー: Codex(OpenAI 系) + Agent Teams 5 視点(技術/コンセプト/コンテンツ/リスク/ローンチ戦略) + 統合
対象: vision.md / plan.md / concept-experiment-theater.md / fun-engine-design.md

## 結論: GO-with-fixes（条件付き続行）

- 平均 5.2 / 10。内訳は **「コンセプト 8-9 点・計画文書としての実行可能性 5 点未満」の合成**。
- 低いのはコンセプトではなく **計画と現物の乖離 + 検証順序の逆転**。
- 5 人全員 + Codex が「戦う土俵の選び方（介入実験）は正しい」と認めた。捨てる計画ではない。
- ただし現 plan のまま段階2-6に直進するのは危険。**優先アクション1-4を着手前ゲートに入れれば GO**。

## 両レビューが一致した重大課題（critical / high）

### A. 面白さが未検証のまま、検証ゲートが段階2-6の「後ろ」
最新実機ログ（trial5 / 全トグル ON）ですら全員 happy 高止まり・対立ゼロ・予定調和で、PO 旧診断が未解消。「介入すれば面白くなる」は未論証の希望。
→ **段階3直後に「チープ検証ゲート」を前倒し**（同一起点で介入あり/なしを並べ、PO が 3 点チェック: (a)最後まで観られた (b)計器に観測可能な差分 (c)台本でない創発が1つ以上）。NO なら段階4-6投資をやり直す**撤退ライン(kill criteria)** を明記。

### B. LLM 非決定性で「統制実験」が技術的に成立しない
Anthropic API に seed なし（コード確認済み: protocols/llm.py に seed フィールド0件）。同じ起点から分岐しても出力は毎回変わり、差が介入由来かノイズか分離できない。
→ Vision/Plan/concept の「統制実験／同じ起点から変数だけ変えて比較」を **「反復観察・複数試行」** に正直修正。「同一介入は N 回試行で傾向評価」を invariant 化。RAG は初回「記憶凍結・分岐間で再埋め込みしない」割り切りで決定性確保。

### C. plan.md の事実誤記（計画の起点が虚偽）
- #10（コスト累計上限）・#11（Moderation/kill switch）が **OPEN のまま** なのに、plan は「Phase0 完了」扱い + 「サーキットブレーカー(実装済み)」と誤記。実在する CircuitBreaker は **session 単位のみ**＝巻き戻し/分岐（多 session 高速生成）の死角。
- コード grep: moderation / kill-switch / 累計コスト上限 / restore は **全て 0 件**。
→ plan を実態に訂正。#10/#11 を **ローンチゲート** に格上げ。**Anthropic Console の月 hard cap を物理設定**（コードより先、これだけで破滅を防ぐ）。

### D. 段階2（状態の完全シリアライズ+復元）を過小評価＝MVP 最大の技術リスク
restore/branch/checkout/to_state/from_state は **0 件**。現 snapshot() は round 丸め・episodic[-5:]・history[-20:] 切り詰めの **lossy 観測ダンプ** で復元基盤にならない。
→ snapshot() は観測用に温存し、**to_state()/from_state()（無損失）を別物として新規分離**。restore は新 session 再構築の classmethod（in-place は frozen Character/LLM 参照で事故る）。`from_state(to_state(s)) はターン進行が等価` を Red テストで縛る。詰まったら **DAG を後回しにし「snapshot+単純巻き戻しのみ」で縮退ローンチ** する判断ルートを plan に明記。

### E. 版権キャラ（ゆるキャン野クル組）放置 = IP/法務リスク
他者 IP は「世界を IP として育てる」North Star と真っ向矛盾。収益化・公開で BAN/権利問題に直結。リスク表に存在しない。加えて「3 キャラの Values が似ていて割れない」構造欠陥（同一介入に全員仲良く受け流す）が独立タスク化されていない。
→ **オリジナルキャラ確定をローンチゲートに格上げ**。最小要件: 名前/出自/Big Five/**Values 対立軸（self_transcendence vs self_enhancement を最低1本）**。受け入れ条件に「同一介入に答えが割れることを実機 trial で確認」。野クル組は内部検証専用に。

### F. ローンチを「生ライブ」から「録画+note 主」に倒す
女子高生3体 × AI 自律発言 × YouTube は #11 自記の最高 BAN リスク構成。BAN 一発でチャンネル終了 = IP 育成の前提消滅。
→ 初回は **非ライブ（録画→人手レビュー→note 主 + YouTube アーカイブ従）**。Moderation 初版を「録画の人手レビュー + 未成年キャラ題材の構造ガード + kill switch」に縮小。生配信は数本回した後の Phase1.5。「編集でオチを担保」という PO 理論とも整合。

### G. 達成判定 LLM 単独は危険 / 看板実験（自己認識）は最難関
判定がブレると実験記録の信頼性＝コンテンツの根幹が崩れる。
→ Phase1 は **機械判定できる軽ゴール**（closeness 閾値 / PAD 遷移 / episodic に特定事実記録）に限定。LLM judge は「判定理由+根拠ターンの引用」必須（bool 単独を信用しない）。看板（AI に AI と気づかせる）は **クリティカルパス外**、第2-3号に。

### H. 長期記憶 RAG が会話ループに未配線（差別化軸2が空手形）
session.py は episodic[-3:] スライスのみ。memory/search.py（RAG）/embedding が multi_agent から呼ばれていない。SessionEnd の永続化もオプショナル。aituber-broadcast.yaml の `session-end-state-persisted` invariant がコード未実装の **spec-code drift**。
→ SessionEnd→storage 永続化→次セッション RAG recall→prompt 注入のループを閉じる。ただしリピーター/IP 育成向けなので **ローンチブロッカーではなく Phase1.5 候補**。

### その他（medium/low）
- 途中介入の意味論未設計（intervention.yaml を先に invariant 化）
- 分岐 DAG のストレージ未決（content-addressable JSON 木を推奨）
- コスト試算が楽観（試行錯誤・分岐で指数増、感情推定を毎ターン全キャラ実行→発話者+関係者のみに間引き）
- Web フロント段階6過小（計器=既存 web_server 拡張で十分 / 世界線ツリー UI=新規別工数→Phase2 降格検討）
- FloorController の substring 一致（embedding 化は後）

## 優先アクション（owner 付き）

| # | アクション | owner |
|---|---|---|
| 1 | plan 事実誤記訂正 + #10/#11 をローンチゲート格上げ + **月 hard cap 物理設定** | PO判断（Iris が plan/Issue 整備） |
| 2 | 段階3直後に**チープ検証ゲート + 撤退ライン**前倒し | PO判断（試聴）+ Iris（ゲート設計） |
| 3 | **録画ローンチ化** + Moderation 縮小 | PO判断（公開形態）+ Iris（実装） |
| 4 | **キャラのオリジナル確定**（Values 対立軸）をゲート化 | PO個人タスク |
| 5 | 「統制実験」→「反復観察」文言修正 + invariant | PO判断（文言）+ Iris（spec） |
| 6 | 段階2: to_state/from_state 分離 + round-trip Red テスト + 縮退ライン | Iris実装（最小集合は PO レビュー） |
| 7 | 達成判定の分割（軽ゴール機械判定）+ 根拠ターン必須 | PO判断（看板優先度）+ Iris |
| 8 | 長期記憶 RAG 配線（Phase1.5 候補） | Iris実装 + PO判断 |

## 夜間の反映方針（PO 確認不要 = 反映済み / PO 判断待ち = 朝）

**夜間に反映（明白に正しい訂正・整備）:**
- plan.md の事実誤記訂正、#10/#11 ゲート格上げ、検証ゲート+撤退ライン、段階2縮退ライン、Web 分離、コスト再試算注記
- vision/plan/concept の「統制実験」→「反復観察・複数試行」文言修正
- 録画ローンチを「推奨」として plan に反映（最終決定は PO）
- 新 Issue 起票（実装タスク + patch のバックログ整備）

**PO 判断待ち（朝）:**
- 月 hard cap の物理設定（PO のアカウント操作、Iris 不可）
- キャラのオリジナル確定（PO 個人タスク、ローンチゲート最優先）
- 配信モード / 最初の実験テーマ / 音声(TTS)の要否
- 段階2 の最小復元集合の方向性レビュー → 実装着手 GO

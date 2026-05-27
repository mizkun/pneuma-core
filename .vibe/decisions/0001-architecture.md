# ADR-0001: M1 アーキテクチャ採用

## ステータス

**Accepted** — 2026-05-27

## コンテクスト

Pneuma の M1（AITuber プラットフォーム MVP）を着手するにあたり、技術スタックを決める必要があった。要件：

- **Web ベース**（誰でも視聴可、OBS でブラウザキャプチャ → YouTube Live 配信）
- **3 体のキャラが部室で自律会話**（pneuma-core を依存ライブラリとして使う）
- **観察ダッシュボード**（PAD / 関係性 / 想起記憶 / 日記をリアルタイム表示）
- **音声出力**（TTS、3 体それぞれ別の声）
- **PO 管理者 UI**（コンテクスト注入）
- **個人プロジェクト**（運用人員 1 人、月コストを抑えたい）
- **pneuma-core は Python**（既存資産を活かす）

第三者レビュー（プロダクト戦略・技術アーキテクチャ・DevOps）を経て、以下の論点が浮上：

- pneuma-core 既存実装（`cross_chat.py`、RuntimeEngine、SessionManager）は 2 体・1 character 前提でハードコード気味。3 体並行運用には新規設計が必要
- ダッシュボードのリアルタイム同期方式（Firestore Realtime / WebSocket / SSE）の選択
- バックエンドのランタイム（Cloud Functions / Cloud Run / VPS）の選択
- TTS は Aivis Cloud API（日本語キャラ声）か ElevenLabs（多言語・低レイテンシ）かの選択
- コスト hard limit・Moderation・Secret 管理など運用ガードレールが未設計

## 決定

### スタック

| レイヤー | 採用 |
|---|---|
| **Web フロント** | Next.js + **Firebase Hosting** |
| **認証（PO 管理者 UI）** | **Firebase Authentication** |
| **永続化** | **Firestore**（リアルタイム同期）+ Firebase Storage（画像） |
| **バックエンド** | **Cloud Run (Python)** — FastAPI + uvicorn |
| **キャラ会話エンジン** | **pneuma-core の `multi_agent/` 新規モジュール**（既存 cross_chat / RuntimeEngine の 2 体・1 character 前提は捨てる） |
| **リアルタイム配信（ダッシュボード）** | **Firestore onSnapshot**（自前 WebSocket は使わない） |
| **音声ストリーミング（フロント）** | Cloud Run → フロントへ HTTPS stream（Web Audio API でデコード） |
| **TTS** | **ElevenLabs Multilingual/v3**（Aivis Cloud API は不採用、Phase 0 D で PoC して逆転がなければ確定） |
| **LLM** | **Anthropic Claude（Sonnet + Haiku 使い分け + prompt cache）** |
| **Moderation** | **Anthropic safety + 自前 NG ワードリスト + 配信遅延 30 秒バッファ** |
| **Secret 管理** | **Google Secret Manager**（本番）+ dotenv（ローカル）+ gitleaks pre-commit |
| **監視・エラー通知** | **Cloud Logging + Sentry**（無料枠） |
| **コスト hard limit** | 各プロバイダのコンソール billing alert + アプリ側カウンター（閾値超過で配信プロセス kill） |
| **配信** | OBS で Web ブラウザキャプチャ → YouTube Live（PO 手動） |

### 月コスト試算（Phase 1、15 分/日 × 30 日）

- Firebase Hosting / Auth / Firestore / Storage: 無料枠で足りる
- Cloud Run: 約 $5（最小インスタンス常駐）
- ElevenLabs Multilingual/v3: 約 $13.50
- Anthropic Claude（Sonnet + Haiku + cache）: 約 $20-50
- Sentry / Cloud Logging: 無料枠
- **合計: 約 1 万円/月**

### 月コスト試算（参考、8 時間/日 × 30 日、メリハリ運用）

- Cloud Run: 約 $5
- ElevenLabs Multilingual/v3: 約 2 万円
- Anthropic Claude（Sonnet + Haiku + cache + 沈黙・睡眠時間あり）: 約 5-8 万円
- **合計: 約 7-10 万円/月**

## 理由

### 1. Firebase 採用の理由

- **PO が Firebase に慣れている**（個人プロジェクトでは技術習熟度が立ち上げ速度に直結）
- **Firestore のリアルタイム同期が強力** — ダッシュボード（PAD / 関係性 / 想起記憶）の更新を自前 WebSocket 実装なしで実現
- **Firebase Authentication が無料で強力** — PO 管理 UI のログインが即実装
- **無料枠が広い** — Phase 1 のフロント / Auth / Firestore はほぼタダ
- **デプロイが単純** — `firebase deploy` で完結

### 2. Cloud Run（Python）採用の理由

- **pneuma-core が Python** — Cloud Functions Gen2 でも動くが、WebSocket・長時間実行・ストリーミング音声に Cloud Run の方が素直
- **コールドスタート問題は minimum instances で回避**（Phase 1 は最小 1 インスタンスで $5/月程度）
- **15 分セッションを連続で回せる** — Cloud Functions の 60 分制限は問題ないが、ストリーミング処理は Cloud Run の方が安定

### 3. ElevenLabs Multilingual/v3 採用の理由

- **コスト** — Aivis 従量プランは 8 時間/日で月 19 万円に達するが、ElevenLabs Multilingual は月 2 万円程度
- **多言語対応・感情演技の幅** — v3 は感情・演技タグが強い
- **ストリーミング対応** — Flash 75ms / Turbo 250-300ms
- **声ライブラリが豊富** — 3 キャラの識別性を作りやすい
- **Phase 0 D の PoC で Aivis と横並び実測**して、日本語キャラ性で大差なければ確定。日本語が極端に弱ければ Aivis に戻す

### 4. pneuma-core の multi_agent モジュール新規設計

既存 `cross_chat.py` は 298 行のデモスクリプトで、2 体ハードコード・`peer_session.messages.append(...)` で会話履歴を手動ミラーリング。3 体への「拡張」ではなく、`pneuma_core/multi_agent/` を新規設計する：

- `Conversation` クラス（N 体の参加者を保持）
- `FloorController`（発話権制御、A→B→C→A 固定ではなく性格・文脈関連度・発話頻度のスコアリング）
- `MultiAgentSession`（N 体 ↔ 1 セッション）
- `SessionEndPipeline` を N 体対応に拡張（または `MultiAgentSessionEndPipeline` を別途）

### 5. LLM の Sonnet / Haiku 使い分け

| 呼び出し用途 | モデル | 理由 |
|---|---|---|
| 思考 + 発言生成（コア） | **Sonnet** | キャラの深みに直結 |
| 感情評価（PAD 更新） | Haiku | 軽量判定 |
| 会話履歴要約 | Haiku | 文字数削減のみ |
| 想起記憶のスコアリング | Haiku | 数値判定 |
| 日記の自動要約 | Haiku | 内省的要約 |
| SessionEnd 統合更新 | Sonnet | 重要、間違うと記憶が壊れる |

これと prompt cache（pneuma-core 既存の PromptCache、input の 70% 程度をキャッシュ）で、LLM コストを約 1/8 に削減。

### 6. Moderation の必要性

3 体の AI が女子高生キャラとして自律発言する設計は YouTube BAN リスクが構造的に高い。CSAM 疑い・差別表現・実在人物中傷を防ぐため、配信遅延 30 秒 + Anthropic safety + 自前 NG ワードリストで多層防御。

## 不採用案と理由

### 案 A. 純 Firebase（Cloud Functions だけでバックエンド）

- ❌ Python Gen2 は WebSocket をネイティブサポートしていない
- ❌ 長時間実行（ストリーミング音声・15 分セッション継続）に向かない
- ❌ コールドスタートが配信開始時に効く

### 案 B. Cloudflare Pages + Fly.io（Iris の初期案）

- ✅ Lock-in 最小
- ✅ Python ネイティブ
- ❌ ダッシュボードのリアルタイム同期を自前 WebSocket で実装する手間
- ❌ 認証も自前 or 別サービスで構築する手間
- ❌ PO の習熟度が Firebase より低い

### 案 C. Aivis Cloud API（TTS）

- ✅ 日本語キャラ声らしさが強い
- ❌ プレミアム 1,980 円/月だが 10 req/min レート制限 → 3 体ストリーミング設計が厳しい
- ❌ 従量プランは 8 時間/日で月 19 万円、24/7 で月 57 万円と高い
- 🟡 Phase 0 D の PoC で逆転があれば再採用検討

### 案 D. SQLite を本番でも使い続ける

- ✅ pneuma-core 既存資産そのまま
- ❌ Cloud Run は揮発的ストレージ（永続ボリュームには別途設定が必要）
- ❌ Firestore のリアルタイム同期を使えない
- 🟡 開発時は SQLite、本番は Firestore というハイブリッドにする

## 影響

### pneuma-core への影響

1. **`pneuma_core/multi_agent/` 新規モジュール** — Conversation / FloorController / MultiAgentSession
2. **`SessionEndPipeline` のマルチパーティ拡張**（または別クラス）
3. **`pneuma_core/storage/firestore.py` 新規追加** — StorageBackend Protocol の Firestore 実装（SQLite と並列で本番デプロイ可能に）
4. **`pneuma_core/llm/anthropic.py` の Sonnet/Haiku 使い分け対応** — 用途ごとにモデルを切り替えられる adapter
5. **TTS adapter（`pneuma_core/voice/elevenlabs.py`）追加** — 既存の `TTSAdapter` Protocol を実装

### apps/aituber/ 新規追加

```
apps/aituber/
├── SPEC.md（実装時に .vibe/references/aituber-app-design.md から転写）
├── frontend/         # Next.js + Firebase SDK
│   ├── app/          # ライブ視聴 + ダッシュボード
│   ├── lib/          # Firebase / API クライアント
│   └── ...
├── backend/          # Python + FastAPI + Cloud Run
│   ├── main.py       # エンドポイント定義
│   ├── conversation/ # 3 体会話制御（multi_agent 呼び出し）
│   ├── tts/          # ElevenLabs 統合 + ストリーミング
│   ├── moderation/   # 出力フィルタ + 30 秒遅延バッファ
│   └── ...
├── infra/            # Firebase config / Cloud Run deploy script / Secret 管理
│   ├── firebase.json
│   ├── firestore.rules
│   ├── cloudrun-deploy.sh
│   └── ...
└── scripts/          # 運用スクリプト（kill switch / コスト監視）
```

### 構造化 spec への影響

新規 Story / Contract（Issue ごとに To-Be 差分として詳細化）：

| 種別 | ID | ドメイン |
|---|---|---|
| Story | `multi-agent` | N 体ターン制・FloorController・観察ターン |
| Story | `aituber-scene` | 場（部室）・コンテクスト注入 |
| Story | `aituber-observation` | ダッシュボード・内部状態可視化 |
| Story | `aituber-visual` | 立ち絵切替・感情マッピング |
| Story | `aituber-broadcast` | Web 配信制御・セッション境界 |
| Story | `aituber-tts` | 音声出力・3 体音声割当・レート制御 |
| Story | `aituber-moderation` | 出力フィルタ・配信遅延・kill switch |
| Contract | `aituber-emotion-label` | PAD → label → TTS/立ち絵 の単一参照経路 |
| Contract | `aituber-session-context` | PO コンテクスト注入の形 |
| 拡張 | `runtime`, `character`, `storage` | multi_agent 対応・Firestore backend・モデル使い分け |

## フォールバック / 移行戦略

- **Firebase が想定外に高い場合**: Firestore → Postgres（Supabase）に移行。フロントは Next.js のままで切替可能
- **Cloud Run のコストが想定外に高い場合**: Fly.io へ移行（コンテナベースなので移植容易）
- **ElevenLabs の日本語が想定外に弱い場合**: Aivis Cloud API に切替（adapter pattern で差し替え可能、Phase 0 D の PoC で判断）
- **pneuma-core の multi_agent モジュールが想定外に複雑化**: スコープを縮めて 3 体固定の特化実装に降格（汎用 N 体は M3 に先送り）

## 関連ドキュメント

- `vision.md` — North Star
- `plan.md` — マイルストーン
- `.vibe/references/aituber-app-design.md` — アプリ設計メモ
- `.vibe/references/tts-research-2026-05-27.md` — TTS Deep Research

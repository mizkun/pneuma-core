# TTS 選定 Deep Research

実施日: 2026-05-27
依頼: Pneuma Phase 1 用 TTS サービス選定
出力: Aivis Cloud API を本命、ElevenLabs をバックアップ、OpenAI TTS を LLM 統合検討用とする

## 結論

Pneuma の Phase 1 なら、**第一候補は Aivis Cloud API、第二候補は ElevenLabs、第三候補は OpenAI TTS / Realtime** が現実的。日本語の "キャラ声" らしさ、2 秒以内の体感応答、API 実装、商用 YouTube Live 利用のバランスを見ると、Aivis がかなり刺さる。

**にじボイスは 2026 年 2 月 4 日にサービス終了済み** なので、比較対象としては重要だが、採用候補からは外す。

コスト試算は、**日本語 1 分あたり約 300 文字**、Phase 1 は **15 分/日 × 30 日 = 450 分 = 約 13.5 万文字/月**、24/7 は **43,200 分 = 約 1,296 万文字/月** として概算。

---

## 1. サマリ表

凡例: **◎ = 第一候補級、○ = PoC 候補、△ = 条件付き、× = 採用不可または非推奨**

| サービス | 低レイテンシ / ストリーミング | 日本語クオリティ・感情・声数 | 商用利用 / API | 月額コスト目安 | Pneuma 適性 |
|---|---|---|---|---|---|
| **Aivis Cloud API / Aivis Speech** | 公称で最短 0.3 秒再生開始、ストリーミング対応。30 秒音声も最短 0.7 秒以下という説明あり | 日本語キャラ声向き。AivisHub のモデルを使え、辞書・SSML サブセット・プライベートモデル運用も可能。各モデルのライセンス確認は必須 | Cloud API あり。ローカルの AivisSpeech Engine はサーバー高負荷用途には最適化されていないため、リアルタイム配信では Cloud API / Citoras 側が本命 | 従量: 約 5,940 円/月、24/7 約 57 万円/月。Aivis プレミアムなら 1,980 円/月、10 req/min 内で文字数制限なし | **◎ 最有力** |
| **ElevenLabs** | Flash v2.5 は公称 75ms、Turbo は 250〜300ms、ストリーミング出力対応 | 日本語対応。自然さ、抑揚、表情づけが強い。v3 は感情・演技タグ、多話者、70 以上の言語を訴求。声ライブラリやクローンも豊富 | REST / Python SDK。商用権は有料プラン前提。声クローンは権利・同意管理が重要 | Flash/Turbo 換算: 約 $6.75/月、24/7 約 $648。Multilingual/v3 換算: 約 $13.50/月、24/7 約 $1,296。実請求はプラン・クレジット次第 | **◎ 有力** |
| **OpenAI TTS / tts-1-hd / Realtime API** | Speech API は chunk transfer によるリアルタイム音声ストリーミング対応。低遅延用途は WAV / PCM 推奨。Realtime API は常時接続の会話向け | `gpt-4o-mini-tts` は声の感情幅、イントネーション、速度、トーン、囁きなどを instructions で制御可能。公式は音色が英語寄りに最適化されていると明記。標準声は 13 種類 | REST / Python SDK。TTS 音声が AI 生成であることの開示が必要。Custom voice は対象顧客向けで、同意録音とサンプル録音が必要 | tts-1: 約 $2.03/月、24/7 約 $194。tts-1-hd: 約 $4.05/月、24/7 約 $389。Realtime は gpt-realtime-2 が音声出力 $64/100 万 tokens などのトークン課金 | **○〜◎ LLM 統合重視なら強い** |
| **Google Cloud Text-to-Speech / Chirp 3 HD / Gemini TTS** | Chirp 3 HD はストリーミング合成対応。Gemini TTS も streaming outputs、多話者、スタイル制御を訴求 | Chirp 3 HD は 30 声、ja-JP 対応。Gemini TTS は感情・アクセント・ペース・トーンのプロンプト制御、多話者に対応。Chirp の streaming では SSML 非対応という制約あり | Google Cloud の REST / SDK。Instant custom voice は許可制・同意ありの領域 | Chirp 3 HD: 約 $4.05/月、24/7 約 $389。Gemini 2.5 Flash TTS は出力音声 25 tokens/sec、$10/100 万音声 tokens なので、音声出力だけで約 $6.75/月、24/7 約 $648 | **○ 企業運用・安定重視** |
| **Azure Speech Service** | SDK / REST でリアルタイム合成、バッチ合成対応 | 日本語音声は Nanami、Keita、Aoi、Daichi、Mayu、Shiori など複数。SSML で style、role、pitch、rate など制御可能 | 商用クラウド利用向き。Custom Neural Voice は制限付きアクセスで、学習・権利手続きが重い | Neural $15/100 万文字なら約 $2.03/月、24/7 約 $194。Neural HD $22/100 万文字なら約 $2.97/月、24/7 約 $285 | **○ 堅牢だがキャラ性は要検証** |
| **Fish Audio** | WebSocket / streaming、低レイテンシ設定あり。balanced は約 300ms を示す | S2 は自然言語の感情指示、S1 は happy/sad/angry などのタグ。S2-Pro は multi-speaker dialogue に対応。クローンも強い | API / SDK あり。商用利用は有料プラン・所有/検証済み声など条件確認が必要 | API は $15/100 万 UTF-8 bytes。日本語を 3 bytes/字で見ると約 $6.08/月 + プラン費、24/7 約 $583 + プラン費 | **○ 注目候補** |
| **Cartesia Sonic 3.5** | TTFA as low as 40ms、低レイテンシ・ストリーミングを強く訴求。日本語対応 | 日本語ローカライズ音声、感情・ペーシング制御を訴求。キャラ声としての相性は要試聴 | API 提供。商用・価格は契約条件を確認 | 価格は要確認 | **○ 速度重視の PoC 候補** |
| **Amazon Polly Neural / Generative** | API 安定性は高い。Generative engine はストリーミング可能な decoder を説明 | 企業ナレーションには強いが、3 人の女子高生キャラ会話としては声の個性・演技幅を要検証 | AWS 標準 API | Neural $16/100 万文字: 約 $2.16/月、24/7 約 $207。Generative $30/100 万文字: 約 $4.05/月、24/7 約 $389 | **△ 企業用途寄り** |
| **にじボイス** | 旧 API あり、感情表現パラメータも存在 | 100 体以上のキャラクター声があり、AITuber 適性は高かった | **2026 年 2 月 4 日に終了済み** | 旧価格は 825 円/1 万文字。今は利用不可 | **× 採用不可** |
| **Style-Bert-VITS2 自前ホスティング** | API サーバーあり。GPU 構成次第で低遅延化可能だが、クラウド SaaS の公称 SLA はない | 日本語向け JP-Extra、スタイル・感情制御が強い。独自キャラ声を作るなら自由度が高い | OSS。ただしコードライセンス、モデル、学習音声、話者同意の管理が本体 | GPU 代、運用費、モデル制作費。小規模なら安いが運用者コストが重い | **○ 技術内製できるなら強い** |
| **VOICEVOX** | ローカル HTTP API、Docker/実行ファイルあり。基本は文単位合成で、チャンクストリーミング型ではない | 無料で使いやすい。話者が多く、イントネーション調整も可能。最新商用 TTS ほど自然な会話感・感情演技は出にくい | 商用・非商用利用可だが、生成音声は各キャラクター規約に従う。クレジット表記も必要 | ソフト無料、サーバー費のみ | **△ プロトタイプ・予備声向き** |
| **COEIROINK** | ローカル API あり | MYCOEIROINK など多様な声。声ごとの規約確認が必須 | ソフトは商用・非商用可だが、生成音声の権利は声の提供者側にあり、声ライブラリごとの利用条件が優先 | ソフト無料、サーバー費のみ | **△ 規約管理できるなら PoC 向き** |
| **Coqui TTS / XTTS 系** | XTTSv2 は streaming <200ms をうたう実装あり | 多言語・音声変換・学習ツールとしては強いが、日本語キャラ会話の即戦力はモデル選定次第 | コードは MPL-2.0 系だが、各モデルのライセンスは別確認。旧 Coqui の事業終了・保守系譜にも注意 | OSS + GPU/運用費 | **△ 研究開発向き** |
| **Bark** | 研究・デモ寄り。リアルタイムライブの安定運用には不向き | 笑い、ため息、泣き声など非言語表現は面白いが、台本忠実性が弱く、意図しない脱線があり得る | MIT、pretrained checkpoints は commercial use ready と説明。ただし出力制御・品質保証が課題 | OSS + GPU/運用費 | **△ 実験用** |

---

## 2. 推奨 Top 3

### 1 位: Aivis Cloud API

**Pneuma の本命。** 日本語キャラ声、低レイテンシ、ストリーミング、API、料金の現実性が最も噛み合う。特に Phase 1 の 7.5 時間/月なら、従量でも約 5,940 円、Aivis プレミアムなら月 1,980 円で試せる可能性。3 キャラそれぞれに、明るい子、落ち着いた子、ツッコミ役のような声を割り振る設計にも向く。

**採用条件**: 使う 3 声のライセンスを厳密に固定すること。商用 YouTube Live、収益化、アーカイブ、切り抜き、SNS 転載、将来のグッズやアプリ展開まで想定し、非商用限定や声の再配布禁止に引っかからない声を選ぶべき。

### 2 位: ElevenLabs

**品質・声数・実装速度のバランスが強い。** 75ms 級の Flash、250〜300ms 級の Turbo、ストリーミング、巨大な声ライブラリ、voice cloning があり、3 キャラの識別性を作りやすい。

**弱点**: 24/7 化時のコストと、日本語キャラ芝居の相性。Phase 1 では十分予算内だが、常時配信では $648〜$1,296/月級まで膨らむ。Aivis を主、ElevenLabs を高品質バックアップまたは特別回用にする構成が綺麗。

### 3 位: OpenAI TTS / Realtime API

**LLM 会話制御まで含めた統合候補。** OpenAI で会話生成、モデレーション、キャラ人格、TTS、場合によっては Realtime までまとめると、システム全体が単純になる。Speech API はストリーミング対応で、`gpt-4o-mini-tts` は感情・速度・トーンの指示ができる。

**弱点**: 声のバリエーションは ElevenLabs や Aivis より少なく、標準声は英語寄り最適化と明記されている。3 人の女子高生キャラとしては、声質だけでなく、話し方プロンプト、口癖、間、音量、EQ、BGM との混ざり方でキャラ差を補う必要あり。

---

## 3. 落とし穴・注意点

### 声の権利は、最初に潰すべきラスボス

にじボイスの終了は Pneuma にとって警鐘。法的に明確な侵害でなくても、「声が特定の俳優・声優に酷似している」と受け取られれば、配信・収益化・継続運営のリスクになる。

避けるべき:
- 既存声優・VTuber・配信者に似せたキャラ設計
- 本人の同意がない voice cloning
- 利用規約が曖昧な野良モデルの商用利用

### 「商用 OK」は、ソフトではなく "声ごと" に見る

VOICEVOX、COEIROINK、AivisHub、Style-Bert-VITS2 系モデルでは、エンジン自体が商用利用可能でも、**声・キャラクター・学習データ・生成音声の規約**が別。声の使用許諾をドキュメント化し、後から声を差し替えると視聴者には "魂が入れ替わった" 違和感が出る。

### 2 秒以内は TTS だけでなく、全パイプラインで測る

入力から音声出力まで 2 秒以内にしたい場合、TTS の API レスポンスだけを見ても足りない。実際には、コメント取得、LLM 応答生成、発話文の分割、TTS first audio byte、音声デコード、OBS へのルーティング、キャラの口パク同期まで足される。

実装では、**1 発話を長文で投げず、1〜2 文単位で TTS に流す** のが基本。PCM / WAV / Opus の低遅延設定、先頭無音カット、相づちキャッシュ、「えっと」「なるほどね」などの定型音声キャッシュを組み合わせると体感が一気に良くなる。

### 感情制御は、強すぎると事故る

怒り、喜び、悲しみのタグや instructions は便利だが、ライブ配信では過剰演技がノイズになる。

Pneuma では、感情を **`neutral / happy / teasing / surprised / embarrassed / sad-lite`** くらいに抑え、`angry` は基本使わないほうが安全。女子高生 3 人の部室会話なら、強い怒りより「むくれる」「茶化す」「小声で拗ねる」のほうがキャラが立つ。

### 3 声の識別は、声質だけに頼らない

3 人とも女子高生にすると、声質が近くなりがち。声だけでなく、話速、語尾、相づち、口癖、音量、EQ、立ち位置、字幕色、Live2D 表情をセットで差別化したほうがよい。

おすすめ：
- **キャラ A**: 明るく早口、少し高め
- **キャラ B**: 落ち着き、やや低め、間が長い
- **キャラ C**: ツッコミ役、中速、語尾が鋭い

### 24/7 化では、コストより先にレート制限とキューが詰まる

Phase 1 の月 7.5 時間はほとんどの候補で安い。問題は 24/7。文字数は約 96 倍になり、ElevenLabs、Google Gemini TTS、Fish Audio などは数百ドルから千ドル超の領域に入る。

Aivis プレミアムのような定額プランは魅力的だが、10 req/min のようなレート制限あり。常時配信では、1 リクエストあたりの文量、3 キャラの発話順、割り込み、キャンセル、再生成、失敗時リトライを設計しないと、配信の裏で音声キューが渋滞する。

### OpenAI 利用時は AI 音声の開示が必要

OpenAI の TTS ドキュメントは、エンドユーザーに対して、聞いている音声が AI 生成であり人間の声ではないと明確に開示する必要があると説明。Pneuma では配信概要欄、固定コメント、番組説明、必要なら画面内テロップに入れておくのが安全。

### 本番前の PoC は、同一台本で横並び実測する

最初の PoC では、Aivis、ElevenLabs、OpenAI、Google Gemini/Chirp、Fish Audio を同じ台本で比べる。台本は、通常会話、ツッコミ、照れ、長めの説明、固有名詞の 5 種類を用意。

測るべき値：
- first audio byte
- 再生開始まで
- 全文生成完了まで
- イントネーション崩れ
- 固有名詞読み
- 声の識別性
- OBS に流したときの聞き取りやすさ

机上比較だけだと、部室の空気までは聴こえてこない。Pneuma は耳で決める部分が大きい。

---

## 参考リンク

[1] Aivis Realtime Streaming Demo: https://api.aivis-project.com/v1/demo/realtime-streaming
[2] note - Aivis Project: https://note.com/aivis_project/n/nd7840ae2b903
[3] AivisSpeech Engine README: https://github.com/Aivis-Project/AivisSpeech-Engine/blob/master/README.md
[4] Aivis Premium 価格: https://note.com/aivis_project/n/ne08001f5985a
[5] ElevenLabs TTS API: https://elevenlabs.io/text-to-speech-api
[6] ElevenLabs Pricing: https://elevenlabs.io/pricing/api
[7] OpenAI TTS Guide: https://developers.openai.com/api/docs/guides/text-to-speech
[8] OpenAI tts-1-hd: https://developers.openai.com/api/docs/models/tts-1-hd
[9] Google Chirp 3 HD: https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd
[10] Google Instant Custom Voice: https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice
[11] Google TTS Pricing: https://cloud.google.com/text-to-speech/pricing
[12] Azure Speech Service: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/text-to-speech
[13] Azure Language Support: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
[14] Azure Professional Voice: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/professional-voice-train-voice
[15] Azure Speech Pricing: https://azure.microsoft.com/zh-tw/pricing/details/speech/
[16] Fish Audio TTS Docs: https://docs.fish.audio/developer-guide/core-features/text-to-speech
[17] Fish Audio Developers: https://fish.audio/developers/
[18] Cartesia Japan: https://cartesia.ai/regions/japan
[19] Cartesia Sonic 3.5: https://docs.cartesia.ai/build-with-cartesia/tts-models/sonic-3-5
[20] AWS Polly Generative: https://docs.aws.amazon.com/polly/latest/dg/generative-voices.html
[21] AWS Polly Pricing: https://aws.amazon.com/polly/pricing/
[22] にじボイス GameMakers: https://gamemakers.jp/article/2024_12_11_87974/
[23] にじボイス PC Watch: https://pc.watch.impress.co.jp/docs/news/2065958.html
[24] にじボイス終了 Algomatic: https://algomatic.jp/news/notice_nijivoice_20251121
[25] Style-Bert-VITS2: https://github.com/litagin02/Style-Bert-VITS2
[26] VOICEVOX Engine: https://github.com/VOICEVOX/voicevox_engine
[27] VOICEVOX: https://voicevox.hiroshiba.jp/
[28] VOICEVOX 利用規約: https://voicevox.hiroshiba.jp/term/
[29] COEIROINK ヘルプ: https://coeiroink.com/help/002
[30] COEIROINK 利用規約: https://coeiroink.com/terms
[31] Coqui TTS: https://github.com/idiap/coqui-ai-TTS
[32] Bark: https://github.com/suno-ai/bark

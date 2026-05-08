# Domain — emotion

Python source: [`src/pneuma_core/emotion/`](../../../src/pneuma_core/emotion)

## 役割

PAD 3 次元感情空間 (Pleasure / Arousal / Dominance) を内部表現とし、
Big Five 性格特性をベースラインとして連動させる。3 層モデル
(Emotion / Mood / Personality) のうち、Emotion (秒〜分) と Mood (時間〜日) の
遷移をランタイムで扱う。

## モジュール

| モジュール | 役割 |
|-----------|------|
| `pad_mapping` | Big Five → PAD ベースラインの数式変換 |
| `baseline` | Personality に基づくベースライン算出と保持 |
| `decay` | 経過時間と半減期から現在値をベースラインへ回帰させる |

## 不変条件

- Big Five が同一なら同一の PAD ベースラインが得られる。
- 入力イベントがない時、PAD は Baseline へ単調収束する。
- Pleasure / Arousal / Dominance はそれぞれ `[-1, 1]` (`models.emotion` と同じ)。

## チューニングパラメータ

| 名前 | 既定値 | 単位 | 説明 |
|------|--------|------|------|
| `emotion_half_life` | 3600 | 秒 | 感情がベースラインへ向かって減衰する半減期 |

## 関連ドメイン

- [`models`](../models/) — `EmotionalState` / `Mood` 型を提供
- [`runtime`](../runtime/) — 各ターンで decay → 推論 → 更新を駆動

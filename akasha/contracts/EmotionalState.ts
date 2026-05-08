// Akasha Contract — EmotionalState / Mood
//
// Python 一次ソース: src/pneuma_core/models/emotion.py
//
// PAD 3 次元感情。値域は [-1, 1]。
// EmotionalState は離散ラベル + 状況説明を併せ持ち、Mood は PAD のみの移動平均。

export interface EmotionalState {
  /** 不快 → 快。 */
  pleasure: number;
  /** 沈静 → 興奮。 */
  arousal: number;
  /** 服従 → 支配。 */
  dominance: number;
  /** 表示用の離散ラベル (例: "中立", "喜び")。 */
  emotion_label: string;
  /** 現在の状況を表す 1 文。 */
  situation: string;
}

/** 中期感情 (時間〜日)。EmotionalState を入力として指数移動平均で更新する。 */
export interface Mood {
  pleasure: number;
  arousal: number;
  dominance: number;
}

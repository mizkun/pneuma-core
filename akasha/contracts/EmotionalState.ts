// Akasha Contract — EmotionalState
//
// Python 一次ソース: src/pneuma_core/models/emotion.py
// PAD 3次元感情空間。値域は [-1, 1]。Big Five からベースラインを計算し、
// 指数減衰でベースラインへ回帰する。

export interface PAD {
  /** 快 / 不快。[-1, 1]。 */
  pleasure: number;
  /** 覚醒。[-1, 1]。 */
  arousal: number;
  /** 支配 / 服従。[-1, 1]。 */
  dominance: number;
}

export interface EmotionalState {
  character_id: string;
  current: PAD;
  baseline: PAD;
  /** 直近の更新時刻 (ISO 8601)。decay 計算の基準。 */
  updated_at: string;
}

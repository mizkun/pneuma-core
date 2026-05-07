// Akasha Contract — Character / Personality / Values
//
// Python 一次ソース:
//   src/pneuma_core/models/character.py
//   src/pneuma_core/models/personality.py
//   src/pneuma_core/models/values.py
//
// 不変属性 (性格・価値観) と自由記述 (プロフィール・外見・口調等) を持つ identity。

/** Big Five 性格モデル。各値は [0, 1]。 */
export interface Personality {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

/** Schwartz 4 カテゴリ価値観モデル。各値は [0, 1]。重視判定しきい値は 0.6 以上。 */
export interface Values {
  self_transcendence: number;
  self_enhancement: number;
  openness_to_change: number;
  conservation: number;
}

/**
 * キャラクターの Identity。
 * personality / values は不変、プロフィール系は自由記述 (任意)。
 */
export interface Character {
  id: string;
  name: string;
  personality: Personality;
  values: Values;
  profile?: string | null;
  appearance?: string | null;
  speaking_style?: string | null;
  background?: string | null;
  /** YAML 直書きまたは LLM 生成 (Phase 1.5) の自然言語表現。 */
  personality_description?: string | null;
  values_description?: string | null;
}

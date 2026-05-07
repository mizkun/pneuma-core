// Akasha Contract — Character / CharacterSheet
//
// Python 一次ソース:
//   src/pneuma_core/models/character.py
//   src/pneuma_core/character_sheet.py
//   src/pneuma_core/models/personality.py
//   src/pneuma_core/models/values.py
//
// キャラクターを構成する不変的・可変的な情報。
// 内部表現は数値、プロンプト表現は自然言語。

/** Big Five の各特性。値域は [0, 1]。 */
export interface BigFive {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

/**
 * Schwartz の 4 上位カテゴリの相対重み。値域は [0, 1] 程度。
 * (Self-Transcendence / Self-Enhancement / Openness-to-Change / Conservation)
 */
export interface SchwartzValues {
  self_transcendence: number;
  self_enhancement: number;
  openness_to_change: number;
  conservation: number;
}

export interface CharacterProfile {
  display_name: string;
  pronouns?: string | null;
  age?: number | null;
  bio?: string | null;
  /** 口調・話し方の指針 (自然言語)。 */
  speech_style?: string | null;
}

export interface CharacterSheet {
  id: string;
  profile: CharacterProfile;
  personality: BigFive;
  values: SchwartzValues;
  /** 任意のメタデータ。実装依存。 */
  metadata?: Record<string, unknown>;
}

export interface Character {
  sheet: CharacterSheet;
  /** ストレージから復元される動的フィールド (任意)。 */
  created_at?: string;
  updated_at?: string;
}

// Akasha Contract — Memory models
//
// Python 一次ソース: src/pneuma_core/models/memory.py
// 二重記憶システム (Episodic + Semantic) のデータ表現。

export interface EpisodicMemory {
  id: string;
  character_id: string;
  content: string;
  /** 0.0..1.0 程度に正規化された重要度。 */
  importance: number;
  /** 観測時の埋め込みベクトル。 */
  embedding: number[];
  /** ISO 8601 時刻。 */
  created_at: string;
  /** 観測時の感情スナップショット (PAD)。任意。 */
  emotional_snapshot?: { pleasure: number; arousal: number; dominance: number } | null;
  metadata?: Record<string, unknown>;
}

export interface SemanticMemory {
  id: string;
  character_id: string;
  /** 汎化された知識・パターンの自然言語表現。 */
  content: string;
  /** 統合された出来事の数 (信頼度の代理指標)。 */
  consolidation_count: number;
  embedding: number[];
  created_at: string;
  updated_at: string;
  metadata?: Record<string, unknown>;
}

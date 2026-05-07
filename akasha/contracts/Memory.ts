// Akasha Contract — Memory models
//
// Python 一次ソース: src/pneuma_core/models/memory.py
//
// 二重記憶システム (Episodic + Semantic) のデータ表現。

export interface EpisodicMemory {
  id: string;
  character_id: string;
  /** 出来事の自然言語表現。 */
  content: string;
  /** ISO 8601 形式の datetime 文字列 (Python 側は datetime)。 */
  timestamp: string;
  /** [-1, 1]。不快 → 快。 */
  emotional_valence: number;
  /** [0, 1]。重要度。 */
  importance: number;
  conversation_id?: string | null;
  /** 1536 次元程度の埋め込みベクトル (未生成時は null)。 */
  embedding?: number[] | null;
}

export interface SemanticMemory {
  id: string;
  character_id: string;
  /** 汎化された知識・パターンの自然言語表現。 */
  content: string;
  /** [0, 1]。裏付けエピソード数で増加する信頼度。 */
  confidence: number;
  /** 統合元のエピソード ID 集合。 */
  source_episode_ids: string[];
  embedding?: number[] | null;
}

// Akasha Contract — MemoryStore
//
// Python 一次ソース: src/pneuma_core/protocols/memory_store.py
// StorageBackend から分離された記憶専用の境界。
// 記憶はベクトル類似検索を必要とし、他のデータとはアクセスパターンが異なるため独立。

import type { EpisodicMemory, SemanticMemory } from "./Memory";

export interface MemoryStore {
  add_episodic(memory: EpisodicMemory): Promise<void>;

  add_semantic(memory: SemanticMemory): Promise<void>;
  update_semantic(memory: SemanticMemory): Promise<void>;
  delete_semantic(memory_id: string): Promise<void>;

  get_episodic_by_character(character_id: string): Promise<EpisodicMemory[]>;
  get_semantic_by_character(character_id: string): Promise<SemanticMemory[]>;

  /** 類似度 >= threshold のエピソード記憶を返す。 */
  find_similar_episodic(
    character_id: string,
    embedding: number[],
    threshold: number,
  ): Promise<EpisodicMemory[]>;
}

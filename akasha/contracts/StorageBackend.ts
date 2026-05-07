// Akasha Contract — StorageBackend
//
// Python 一次ソース: src/pneuma_core/protocols/storage.py
//
// 全データの統合永続化境界。Character / Memory / Goals / State / Relation /
// ChangeRecord / Todo の CRUD を提供する。

import type { Character } from "./Character";
import type { EmotionalState } from "./EmotionalState";
import type { GoalTree } from "./Goals";
import type { EpisodicMemory, SemanticMemory } from "./Memory";
import type { ChangeRecord } from "./Middleware";

export interface TodoItem {
  id: string;
  content: string;
  /** "habit" | "deadline" | "this_week" | "someday" など。 */
  label: string;
  /** "must" | "want"。 */
  kind: string;
  /** "pending" | "done" | "skipped"。 */
  status: string;
  /** 1 (高) 〜 3 (低)。 */
  priority: number;
  /** ISO 8601 date 文字列。 */
  due_date?: string | null;
  /** "daily" | "weekly" | "weekdays" | null。 */
  recurrence?: string | null;
  /** ISO 8601 datetime 文字列。 */
  created_at: string;
  /** ISO 8601 datetime 文字列。 */
  completed_at?: string | null;
  /** 既定値: "user"。 */
  owner_id: string;
}

export interface Relation {
  id: string;
  owner_id: string;
  target_id: string;
  target_name: string;
  /** "partner" | "friend" | "family" | "mentor" など。 */
  relationship_type: string;
  description: string;
  /** [0, 1]。 */
  closeness: number;
  /** [0, 1]。 */
  trust: number;
  /** ISO 8601 datetime 文字列。 */
  updated_at: string;
  notes?: string | null;
}

export interface StorageBackend {
  // ─── Character ───
  save_character(character: Character): Promise<void>;
  get_character(character_id: string): Promise<Character | null>;
  list_characters(): Promise<Character[]>;

  // ─── Memory ───
  save_episodic_memory(memory: EpisodicMemory): Promise<void>;
  get_episodic_memories(character_id: string): Promise<EpisodicMemory[]>;
  find_similar_memories(
    character_id: string,
    embedding: number[],
    threshold: number,
  ): Promise<EpisodicMemory[]>;
  save_semantic_memory(memory: SemanticMemory): Promise<void>;
  update_semantic_memory(memory: SemanticMemory): Promise<void>;
  delete_semantic_memory(memory_id: string): Promise<void>;
  get_semantic_memories(character_id: string): Promise<SemanticMemory[]>;

  // ─── Goals ───
  save_goals(character_id: string, goals: GoalTree): Promise<void>;
  get_goals(character_id: string): Promise<GoalTree | null>;

  // ─── State ───
  save_emotional_state(
    character_id: string,
    state: EmotionalState,
  ): Promise<void>;
  get_emotional_state(character_id: string): Promise<EmotionalState | null>;

  // ─── ChangeLog ───
  save_change(change: ChangeRecord): Promise<void>;
  get_changes(character_id: string, limit?: number): Promise<ChangeRecord[]>;

  // ─── Todo ───
  save_todo(todo: TodoItem): Promise<void>;
  get_todo(todo_id: string): Promise<TodoItem | null>;
  list_todos(opts?: {
    status?: string | null;
    label?: string | null;
    owner_id?: string | null;
  }): Promise<TodoItem[]>;
  update_todo(todo: TodoItem): Promise<void>;
  delete_todo(todo_id: string): Promise<void>;

  // ─── Relation ───
  save_relation(relation: Relation): Promise<void>;
  get_relation(relation_id: string): Promise<Relation | null>;
  list_relations(opts?: { owner_id?: string | null }): Promise<Relation[]>;
}

// Akasha Contract — StorageBackend
//
// Python 一次ソース: src/pneuma_core/protocols/storage.py
// 全データの統合永続化境界。Character / Memory / Goals / State / Relation /
// ChangeLog / Todo の CRUD を提供する。
//
// 実装は SQLite / InMemory が同梱、PostgreSQL 等は外部実装に委ねる。

import type { Character } from "./Character";
import type { EmotionalState } from "./EmotionalState";
import type { GoalTree } from "./Goals";
import type { EpisodicMemory, SemanticMemory } from "./Memory";

export interface ChangeRecord {
  id: string;
  character_id: string;
  kind: string;
  detail: Record<string, unknown>;
  created_at: string;
}

export interface TodoItem {
  id: string;
  owner_id?: string | null;
  title: string;
  status: string;
  label?: string | null;
  metadata?: Record<string, unknown>;
}

export interface Relation {
  id: string;
  owner_id: string;
  target_id: string;
  closeness: number;
  trust: number;
  metadata?: Record<string, unknown>;
}

export interface StorageBackend {
  // Character
  save_character(character: Character): Promise<void>;
  get_character(character_id: string): Promise<Character | null>;
  list_characters(): Promise<Character[]>;

  // Memory
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

  // Goals
  save_goals(character_id: string, goals: GoalTree): Promise<void>;
  get_goals(character_id: string): Promise<GoalTree | null>;

  // State
  save_emotional_state(character_id: string, state: EmotionalState): Promise<void>;
  get_emotional_state(character_id: string): Promise<EmotionalState | null>;

  // ChangeLog
  save_change(change: ChangeRecord): Promise<void>;
  get_changes(character_id: string, limit?: number): Promise<ChangeRecord[]>;

  // Todo
  save_todo(todo: TodoItem): Promise<void>;
  get_todo(todo_id: string): Promise<TodoItem | null>;
  list_todos(opts?: {
    status?: string | null;
    label?: string | null;
    owner_id?: string | null;
  }): Promise<TodoItem[]>;
  update_todo(todo: TodoItem): Promise<void>;
  delete_todo(todo_id: string): Promise<void>;

  // Relation
  save_relation(relation: Relation): Promise<void>;
  get_relation(relation_id: string): Promise<Relation | null>;
  list_relations(opts?: { owner_id?: string | null }): Promise<Relation[]>;
}

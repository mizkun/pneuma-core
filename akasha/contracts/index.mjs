// Akasha Contract Registry — pneuma-core
//
// Akasha の defineContract() は Producer→Consumer の情報フローを宣言する。
// id / version / description / producers / consumers の 5 フィールドが必須。
//
// ここで登録する Contract は、pneuma-core の "ドメイン境界をまたいで流れる情報" の一覧。
// TS 型としての shape は同ディレクトリの `*.ts` に並置されている (Python dataclass と同期)。

import { defineContract } from "@mizkun/akasha";

// ─── 1. Identity / Personality ─────────────────────────────────────────────
export const characterContract = defineContract({
  id: "character",
  version: "1.0.0",
  description:
    "キャラクターの不変属性 (Big Five Personality, Schwartz Values) と自由記述プロフィール",
  producers: ["models.character", "character_sheet"],
  consumers: [
    "runtime.engine",
    "runtime.prompt_builder",
    "runtime.context_assembler",
    "storage.backend",
  ],
});

export const personalityContract = defineContract({
  id: "personality",
  version: "1.0.0",
  description: "Big Five 性格特性 (5 trait, 各 [0,1])",
  producers: ["models.personality"],
  consumers: [
    "runtime.engine",
    "runtime.prompt_builder",
    "memory.search",
    "emotion.pad_mapping",
    "emotion.baseline",
  ],
});

export const valuesContract = defineContract({
  id: "values",
  version: "1.0.0",
  description: "Schwartz 4 カテゴリ価値観 (各 [0,1])",
  producers: ["models.values"],
  consumers: ["runtime.engine", "runtime.prompt_builder"],
});

// ─── 2. Emotion ────────────────────────────────────────────────────────────
export const emotionalStateContract = defineContract({
  id: "emotional-state",
  version: "1.0.0",
  description:
    "PAD 3 次元感情状態 (pleasure / arousal / dominance ∈ [-1, 1] + emotion_label + situation)",
  producers: [
    "models.emotion",
    "runtime.emotion_engine",
    "emotion.baseline",
    "emotion.decay",
  ],
  consumers: [
    "runtime.engine",
    "runtime.prompt_builder",
    "runtime.context_assembler",
    "storage.backend",
  ],
});

export const padBaselineContract = defineContract({
  id: "pad-baseline",
  version: "1.0.0",
  description: "Big Five → PAD のベースライン値",
  producers: ["emotion.pad_mapping"],
  consumers: ["emotion.baseline", "emotion.decay", "runtime.emotion_engine"],
});

// ─── 3. Memory ────────────────────────────────────────────────────────────
export const episodicMemoryContract = defineContract({
  id: "episodic-memory",
  version: "1.0.0",
  description:
    "具体的な出来事の記録 (content / timestamp / emotional_valence / importance / embedding)",
  producers: ["models.memory", "memory.store", "memory.consolidator"],
  consumers: [
    "memory.search",
    "memory.semantic_consolidator",
    "runtime.engine",
    "storage.backend",
  ],
});

export const semanticMemoryContract = defineContract({
  id: "semantic-memory",
  version: "1.0.0",
  description:
    "汎化された知識 (content / confidence / source_episode_ids / embedding)",
  producers: [
    "models.memory",
    "memory.semantic_consolidator",
    "memory.consolidator",
  ],
  consumers: ["memory.search", "runtime.engine", "storage.backend"],
});

export const memorySearchResultContract = defineContract({
  id: "memory-search-result",
  version: "1.0.0",
  description:
    "性格バイアス付きスコアリング検索の結果 (Episodic / Semantic 混在)",
  producers: ["memory.search"],
  consumers: ["runtime.engine", "runtime.context_assembler"],
});

// ─── 4. LLM I/O ────────────────────────────────────────────────────────────
export const embeddingContract = defineContract({
  id: "embedding-vector",
  version: "1.0.0",
  description: "テキスト埋め込みベクトル (1536 次元程度)",
  producers: ["llm.embedding"],
  consumers: ["memory.search", "memory.store", "memory.consolidator"],
});

export const llmRequestContract = defineContract({
  id: "llm-request",
  version: "1.0.0",
  description:
    "LLM 呼び出し要求 (system_prompt / messages / model / temperature / max_tokens / cache 制御)",
  producers: ["runtime.engine", "runtime.prompt_builder"],
  consumers: ["llm.adapter", "llm.claude"],
});

export const llmResponseContract = defineContract({
  id: "llm-response",
  version: "1.0.0",
  description: "LLM からの応答 (content / model / usage)",
  producers: ["llm.adapter", "llm.claude"],
  consumers: ["runtime.engine", "runtime.response_parser"],
});

// ─── 5. Goals / Relation / I/O ────────────────────────────────────────────
export const goalTreeContract = defineContract({
  id: "goal-tree",
  version: "1.0.0",
  description:
    "Vision → Objective → Task 3 階層 (フラットリスト + ID 参照)",
  producers: ["models.goals"],
  consumers: ["runtime.engine", "runtime.prompt_builder", "storage.backend"],
});

export const relationContract = defineContract({
  id: "relation",
  version: "1.0.0",
  description: "エンティティ間の関係性 (closeness / trust ∈ [0, 1])",
  producers: ["models.relation"],
  consumers: ["runtime.engine", "storage.backend"],
});

export const messageInputContract = defineContract({
  id: "message-input",
  version: "1.0.0",
  description:
    "統一入力 (content / sender_id / sender_name / sender_type: human|character|system)",
  producers: ["models.message"],
  consumers: ["runtime.engine", "runtime.middleware"],
});

export const messageOutputContract = defineContract({
  id: "message-output",
  version: "1.0.0",
  description:
    "統一出力 (content / emotion / thought / action / tool_calls / internal_changes / system_messages)",
  producers: ["runtime.engine"],
  consumers: ["runtime.middleware"],
});

// ─── 6. Pipeline / Side-effects ───────────────────────────────────────────
export const pipelineContextContract = defineContract({
  id: "pipeline-context",
  version: "1.0.0",
  description:
    "Middleware チェーン全体で共有されるターン状態 (character / emotion / goals / memories / system_prompt / history / turn_count / metadata)",
  producers: ["runtime.engine"],
  consumers: ["runtime.middleware", "protocols.middleware"],
});

export const changeRecordContract = defineContract({
  id: "change-record",
  version: "1.0.0",
  description: "内部状態変化の記録 (type / before / after / reason / timestamp)",
  producers: [
    "models.change_record",
    "runtime.engine",
    "memory.consolidator",
  ],
  consumers: ["storage.backend"],
});

// 全 Contract を配列としても露出する (lint / map 用)。
export const contracts = [
  characterContract,
  personalityContract,
  valuesContract,
  emotionalStateContract,
  padBaselineContract,
  episodicMemoryContract,
  semanticMemoryContract,
  memorySearchResultContract,
  embeddingContract,
  llmRequestContract,
  llmResponseContract,
  goalTreeContract,
  relationContract,
  messageInputContract,
  messageOutputContract,
  pipelineContextContract,
  changeRecordContract,
];

export default contracts;

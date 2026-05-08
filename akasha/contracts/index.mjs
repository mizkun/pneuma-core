// Akasha Contract Registry — pneuma-core
//
// Akasha main の defineContract() に準拠した Producer→Consumer 宣言。
// 必須フィールド:
//   id (lowercase-kebab) / version (int) / semver / provider_contract /
//   consumer_contracts / layer / allowed_consumer_layers / relationship /
//   latency_budget_ms / sync_call / owners / story / producers / consumers
//
// provider_contract / consumer_contracts は zod schema でなければならない
// (defineContract は instanceof 風に Zod schema を判定する)。
//
// Python の一次ソース (src/pneuma_core/...) と意味を同期する。
// ここで参照される Python dataclass:
//   models.character / personality / values / emotion / memory / goals /
//   message / relation / change_record
// および protocols.llm の LLMRequest / LLMResponse。

import { defineContract } from "@mizkun/akasha/contracts";
import { z } from "zod";

// ─── 共通の zod ヘルパ ─────────────────────────────────────────────────────
const unitInterval = z.number().min(0).max(1);
const padRange = z.number().min(-1).max(1);
const isoDateTime = z.string(); // ISO 8601 (Python 側は datetime; 直列化された文字列で扱う)

// ─── 1. Character / Personality / Values ───────────────────────────────────
const personalitySchema = z.object({
  openness: unitInterval,
  conscientiousness: unitInterval,
  extraversion: unitInterval,
  agreeableness: unitInterval,
  neuroticism: unitInterval,
});

const valuesSchema = z.object({
  self_transcendence: unitInterval,
  self_enhancement: unitInterval,
  openness_to_change: unitInterval,
  conservation: unitInterval,
});

const characterSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  personality: personalitySchema,
  values: valuesSchema,
  profile: z.string().nullable().optional(),
  appearance: z.string().nullable().optional(),
  speaking_style: z.string().nullable().optional(),
  background: z.string().nullable().optional(),
  personality_description: z.string().nullable().optional(),
  values_description: z.string().nullable().optional(),
});

export const characterContract = defineContract({
  id: "character",
  version: 1,
  semver: "1.0.0",
  provider_contract: characterSchema,
  consumer_contracts: {
    "runtime.engine": characterSchema,
    "runtime.prompt_builder": characterSchema.pick({
      name: true,
      personality: true,
      values: true,
      profile: true,
      speaking_style: true,
      background: true,
      personality_description: true,
      values_description: true,
    }),
    "storage.backend": characterSchema,
  },
  layer: "4F-domain",
  allowed_consumer_layers: ["2F-state-machine", "4F-domain", "5F-persistence"],
  relationship: "conformist",
  latency_budget_ms: 5,
  sync_call: true,
  owners: ["mizkun"],
  story: "models",
  producers: ["models.character", "character_sheet"],
  consumers: ["runtime.engine", "runtime.prompt_builder", "storage.backend"],
  rationale:
    "キャラクターの不変属性 (Personality / Values) と自由記述プロフィール。" +
    "prompt_builder は描画に必要な記述系のみ pick して使う。",
});

// ─── 2. EmotionalState (PAD) ───────────────────────────────────────────────
const emotionalStateSchema = z.object({
  pleasure: padRange,
  arousal: padRange,
  dominance: padRange,
  emotion_label: z.string(),
  situation: z.string(),
});

export const emotionalStateContract = defineContract({
  id: "emotional-state",
  version: 1,
  semver: "1.0.0",
  provider_contract: emotionalStateSchema,
  consumer_contracts: {
    "runtime.engine": emotionalStateSchema,
    "runtime.prompt_builder": emotionalStateSchema.pick({
      pleasure: true,
      arousal: true,
      dominance: true,
      emotion_label: true,
      situation: true,
    }),
    "storage.backend": emotionalStateSchema,
  },
  layer: "4F-domain",
  allowed_consumer_layers: ["2F-state-machine", "4F-domain", "5F-persistence"],
  relationship: "conformist",
  latency_budget_ms: 2,
  sync_call: true,
  owners: ["mizkun"],
  story: "emotion",
  producers: [
    "models.emotion",
    "emotion.baseline",
    "emotion.decay",
    "runtime.emotion_engine",
  ],
  consumers: ["runtime.engine", "runtime.prompt_builder", "storage.backend"],
  rationale: "PAD 3 次元感情。各値は [-1, 1]。離散ラベルと状況説明を併せ持つ。",
});

// ─── 3. Memory (Episodic / Semantic) ───────────────────────────────────────
const episodicMemorySchema = z.object({
  id: z.string().min(1),
  character_id: z.string().min(1),
  content: z.string(),
  timestamp: isoDateTime,
  emotional_valence: padRange,
  importance: unitInterval,
  conversation_id: z.string().nullable().optional(),
  embedding: z.array(z.number()).nullable().optional(),
});

export const episodicMemoryContract = defineContract({
  id: "episodic-memory",
  version: 1,
  semver: "1.0.0",
  provider_contract: episodicMemorySchema,
  consumer_contracts: {
    "memory.search": episodicMemorySchema,
    "memory.semantic_consolidator": episodicMemorySchema.pick({
      id: true,
      character_id: true,
      content: true,
      importance: true,
      embedding: true,
    }),
    "runtime.engine": episodicMemorySchema.pick({
      id: true,
      content: true,
      timestamp: true,
      emotional_valence: true,
    }),
    "storage.backend": episodicMemorySchema,
  },
  layer: "4F-domain",
  allowed_consumer_layers: ["2F-state-machine", "4F-domain", "5F-persistence"],
  relationship: "conformist",
  latency_budget_ms: 30,
  sync_call: true,
  owners: ["mizkun"],
  story: "memory",
  producers: ["models.memory", "memory.store", "memory.consolidator"],
  consumers: [
    "memory.search",
    "memory.semantic_consolidator",
    "runtime.engine",
    "storage.backend",
  ],
  rationale:
    "具体的出来事のレコード。embedding は遅延生成可 (None 初期値) のため optional。",
});

const semanticMemorySchema = z.object({
  id: z.string().min(1),
  character_id: z.string().min(1),
  content: z.string(),
  confidence: unitInterval,
  source_episode_ids: z.array(z.string()),
  embedding: z.array(z.number()).nullable().optional(),
});

export const semanticMemoryContract = defineContract({
  id: "semantic-memory",
  version: 1,
  semver: "1.0.0",
  provider_contract: semanticMemorySchema,
  consumer_contracts: {
    "memory.search": semanticMemorySchema,
    "runtime.engine": semanticMemorySchema.pick({
      id: true,
      content: true,
      confidence: true,
    }),
    "storage.backend": semanticMemorySchema,
  },
  layer: "4F-domain",
  allowed_consumer_layers: ["2F-state-machine", "4F-domain", "5F-persistence"],
  relationship: "conformist",
  latency_budget_ms: 30,
  sync_call: true,
  owners: ["mizkun"],
  story: "memory",
  producers: [
    "models.memory",
    "memory.semantic_consolidator",
    "memory.consolidator",
  ],
  consumers: ["memory.search", "runtime.engine", "storage.backend"],
  rationale:
    "汎化された知識。confidence は裏付けエピソード数で増加し、source_episode_ids で系譜を保つ。",
});

// ─── 4. Embedding ──────────────────────────────────────────────────────────
// Python 一次ソース:
//   src/pneuma_core/protocols/embedding.py — `EmbeddingService.embed(text) -> list[float]`
//                                            `EmbeddingService.embed_batch(texts) -> list[list[float]]`
//   src/pneuma_core/llm/embedding.py       — OpenAIEmbeddingService 実装
// 補助 TS 仕様: akasha/contracts/EmbeddingService.ts
//
// 単一の埋め込みベクトルそのものを契約とする (フラットな number[])。
const embeddingVectorSchema = z.array(z.number());

export const embeddingVectorContract = defineContract({
  id: "embedding-vector",
  version: 1,
  semver: "1.0.0",
  provider_contract: embeddingVectorSchema,
  consumer_contracts: {
    "memory.search": embeddingVectorSchema,
    "memory.store": embeddingVectorSchema,
    "memory.consolidator": embeddingVectorSchema,
  },
  layer: "3F-external-service",
  allowed_consumer_layers: ["2F-state-machine", "4F-domain"],
  relationship: "anticorruption-layer",
  latency_budget_ms: 200,
  sync_call: false,
  owners: ["mizkun"],
  story: "memory",
  producers: ["llm.embedding"],
  consumers: ["memory.search", "memory.store", "memory.consolidator"],
  rationale:
    "Python の EmbeddingService.embed() は list[float] を返す。" +
    "契約はそのフラットなベクトル (number[]) 自体。複数返す API は number[][]。",
});

// ─── 5. LLM I/O ────────────────────────────────────────────────────────────
const llmRequestSchema = z.object({
  system_prompt: z.string(),
  // Python の messages は list[dict]。役割の制約は持たないが、
  // system は system_prompt に集約する規約。
  messages: z.array(z.record(z.string(), z.unknown())),
  model: z.string().nullable().optional(),
  temperature: z.number().min(0).max(2),
  max_tokens: z.number().int().positive(),
  system_prompt_cached: z.string().nullable().optional(),
  system_prompt_dynamic: z.string().nullable().optional(),
});

export const llmRequestContract = defineContract({
  id: "llm-request",
  version: 1,
  semver: "1.0.0",
  provider_contract: llmRequestSchema,
  consumer_contracts: {
    "llm.adapter": llmRequestSchema,
    "llm.claude": llmRequestSchema,
  },
  layer: "3F-external-service",
  allowed_consumer_layers: ["3F-external-service"],
  relationship: "anticorruption-layer",
  latency_budget_ms: 5000,
  sync_call: false,
  owners: ["mizkun"],
  story: "llm",
  producers: ["runtime.engine", "runtime.prompt_builder"],
  consumers: ["llm.adapter", "llm.claude"],
  rationale:
    "LLM 呼び出し要求。system_prompt と messages を厳密に分離する (incident #1, fix #2)。",
});

const llmResponseSchema = z.object({
  content: z.string(),
  model: z.string(),
  usage: z.record(z.string(), z.unknown()),
});

export const llmResponseContract = defineContract({
  id: "llm-response",
  version: 1,
  semver: "1.0.0",
  provider_contract: llmResponseSchema,
  consumer_contracts: {
    "runtime.engine": llmResponseSchema,
    "runtime.response_parser": llmResponseSchema.pick({ content: true }),
  },
  layer: "3F-external-service",
  allowed_consumer_layers: ["2F-state-machine", "4F-domain"],
  relationship: "anticorruption-layer",
  latency_budget_ms: 5000,
  sync_call: false,
  owners: ["mizkun"],
  story: "llm",
  producers: ["llm.adapter", "llm.claude"],
  consumers: ["runtime.engine", "runtime.response_parser"],
  rationale: "LLM からの応答。usage は cost / cache 観測に使う。",
});

// ─── 6. Goals ──────────────────────────────────────────────────────────────
const visionSchema = z.object({
  id: z.string().min(1),
  character_id: z.string().min(1),
  content: z.string(),
});

const objectiveSchema = z.object({
  id: z.string().min(1),
  character_id: z.string().min(1),
  vision_id: z.string().min(1),
  content: z.string(),
  status: z.enum(["active", "achieved", "abandoned"]),
  progress: unitInterval,
});

const taskSchema = z.object({
  id: z.string().min(1),
  character_id: z.string().min(1),
  objective_id: z.string().min(1),
  content: z.string(),
  status: z.enum(["pending", "in_progress", "completed", "abandoned"]),
});

const goalTreeSchema = z.object({
  visions: z.array(visionSchema),
  objectives: z.array(objectiveSchema),
  tasks: z.array(taskSchema),
});

export const goalTreeContract = defineContract({
  id: "goal-tree",
  version: 1,
  semver: "1.0.0",
  provider_contract: goalTreeSchema,
  consumer_contracts: {
    "runtime.engine": goalTreeSchema,
    "runtime.prompt_builder": goalTreeSchema,
    "storage.backend": goalTreeSchema,
  },
  layer: "4F-domain",
  allowed_consumer_layers: ["2F-state-machine", "4F-domain", "5F-persistence"],
  relationship: "conformist",
  latency_budget_ms: 5,
  sync_call: true,
  owners: ["mizkun"],
  story: "models",
  producers: ["models.goals"],
  consumers: ["runtime.engine", "runtime.prompt_builder", "storage.backend"],
  rationale:
    "Vision → Objective → Task の 3 階層をフラットなリスト + ID 参照で保持する。",
});

// ─── 7. Relation ───────────────────────────────────────────────────────────
const relationSchema = z.object({
  id: z.string().min(1),
  owner_id: z.string().min(1),
  target_id: z.string().min(1),
  target_name: z.string(),
  relationship_type: z.string(),
  description: z.string(),
  closeness: unitInterval,
  trust: unitInterval,
  updated_at: isoDateTime,
  notes: z.string().nullable().optional(),
});

export const relationContract = defineContract({
  id: "relation",
  version: 1,
  semver: "1.0.0",
  provider_contract: relationSchema,
  consumer_contracts: {
    "runtime.engine": relationSchema,
    "storage.backend": relationSchema,
  },
  layer: "4F-domain",
  allowed_consumer_layers: ["2F-state-machine", "4F-domain", "5F-persistence"],
  relationship: "conformist",
  latency_budget_ms: 5,
  sync_call: true,
  owners: ["mizkun"],
  story: "models",
  producers: ["models.relation"],
  consumers: ["runtime.engine", "storage.backend"],
  rationale:
    "エンティティ間の関係性。closeness / trust ∈ [0, 1] で双方向ではなく owner→target で持つ。",
});

// ─── 8. Message I/O ────────────────────────────────────────────────────────
const messageInputSchema = z.object({
  content: z.string().min(1),
  sender_id: z.string().min(1),
  sender_name: z.string().min(1),
  sender_type: z.enum(["human", "character", "system"]),
  channel: z.string().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export const messageInputContract = defineContract({
  id: "message-input",
  version: 1,
  semver: "1.0.0",
  provider_contract: messageInputSchema,
  consumer_contracts: {
    "runtime.engine": messageInputSchema,
    "runtime.middleware": messageInputSchema,
  },
  layer: "4F-domain",
  allowed_consumer_layers: ["2F-state-machine", "4F-domain"],
  relationship: "open-host-service",
  latency_budget_ms: 1,
  sync_call: true,
  owners: ["mizkun"],
  story: "runtime",
  producers: ["models.message"],
  consumers: ["runtime.engine", "runtime.middleware"],
  rationale:
    "process_message() 単一エントリの入力。sender_type で送信者種別を区別するが、" +
    "ランタイムの本体はそれで分岐しない (char-eye-view)。",
});

const toolCallSchema = z.object({
  name: z.string(),
  arguments: z.record(z.string(), z.unknown()).optional(),
});

const systemMessageSchema = z.object({
  type: z.string(),
  message: z.string(),
  component: z.string(),
});

const messageOutputSchema = z.object({
  content: z.string(),
  emotion: emotionalStateSchema,
  thought: z.string().nullable().optional(),
  action: z.string().nullable().optional(),
  tool_calls: z.array(toolCallSchema).optional(),
  internal_changes: z.array(z.unknown()).optional(),
  diagnostic: z.unknown().nullable().optional(),
  system_messages: z.array(systemMessageSchema).optional(),
});

export const messageOutputContract = defineContract({
  id: "message-output",
  version: 1,
  semver: "1.0.0",
  provider_contract: messageOutputSchema,
  consumer_contracts: {
    "runtime.middleware": messageOutputSchema,
  },
  layer: "4F-domain",
  allowed_consumer_layers: ["2F-state-machine", "4F-domain"],
  relationship: "open-host-service",
  latency_budget_ms: 1,
  sync_call: true,
  owners: ["mizkun"],
  story: "runtime",
  producers: ["runtime.engine"],
  consumers: ["runtime.middleware"],
  rationale:
    "process_message() の応答。content 以外に emotion / thought / action / 副作用ログ等を含む。",
});

// ─── 9. Pipeline / Side-effects ────────────────────────────────────────────
const pipelineContextSchema = z.object({
  character: characterSchema,
  emotion: emotionalStateSchema,
  goals: goalTreeSchema,
  memories: z.array(z.union([episodicMemorySchema, semanticMemorySchema])),
  system_prompt: z.string(),
  history: z.array(z.record(z.string(), z.unknown())),
  turn_count: z.number().int().nonnegative(),
  metadata: z.record(z.string(), z.unknown()),
});

export const pipelineContextContract = defineContract({
  id: "pipeline-context",
  version: 1,
  semver: "1.0.0",
  provider_contract: pipelineContextSchema,
  consumer_contracts: {
    "runtime.middleware": pipelineContextSchema,
    "protocols.middleware": pipelineContextSchema,
  },
  layer: "4F-domain",
  allowed_consumer_layers: ["2F-state-machine", "4F-domain"],
  relationship: "conformist",
  latency_budget_ms: 1,
  sync_call: true,
  owners: ["mizkun"],
  story: "runtime",
  producers: ["runtime.engine"],
  consumers: ["runtime.middleware", "protocols.middleware"],
  rationale:
    "Middleware チェーン全体で共有される 1 ターンの状態。書き換えで副作用を伝播させる。",
});

const changeRecordSchema = z.object({
  id: z.string().min(1),
  character_id: z.string().min(1),
  type: z.string(),
  before: z.record(z.string(), z.unknown()).nullable(),
  after: z.record(z.string(), z.unknown()),
  reason: z.string(),
  timestamp: isoDateTime,
});

export const changeRecordContract = defineContract({
  id: "change-record",
  version: 1,
  semver: "1.0.0",
  provider_contract: changeRecordSchema,
  consumer_contracts: {
    "storage.backend": changeRecordSchema,
  },
  layer: "6F-observability",
  allowed_consumer_layers: ["5F-persistence", "6F-observability"],
  relationship: "open-host-service",
  latency_budget_ms: 50,
  sync_call: false,
  owners: ["mizkun"],
  story: "runtime",
  producers: ["models.change_record", "runtime.engine", "memory.consolidator"],
  consumers: ["storage.backend"],
  rationale:
    "内部状態変化の監査ログ。後追いデバッグと before/after 比較に使う。",
});

// ─── Aggregate export ──────────────────────────────────────────────────────
export const contracts = [
  characterContract,
  emotionalStateContract,
  episodicMemoryContract,
  semanticMemoryContract,
  embeddingVectorContract,
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

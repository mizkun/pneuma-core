# Domain — models

Python source: [`src/pneuma_core/models/`](../../../src/pneuma_core/models)

## 役割

キャラクターの内面 (性格・価値観・感情・記憶・目標・関係性) を表す純粋データ層。
I/O や副作用は持たない。すべて `@dataclass(frozen=True)` で不変オブジェクトとして
扱う (一部、可変な `GoalTree` / `Relation` / `TodoItem` を除く)。

## 主要モデル

| モデル | 種別 | 値域・キー |
|--------|------|-----------|
| `Character` | identity | `id, name, personality, values, profile, appearance, speaking_style, background, personality_description, values_description` |
| `Personality` | Big Five | `openness, conscientiousness, extraversion, agreeableness, neuroticism` ∈ [0, 1] |
| `Values` | Schwartz 4 軸 | `self_transcendence, self_enhancement, openness_to_change, conservation` ∈ [0, 1] |
| `EmotionalState` | PAD | `pleasure, arousal, dominance` ∈ [-1, 1] + `emotion_label, situation` |
| `Mood` | PAD 移動平均 | `Mood.update()` で `(1 - α) * mood + α * emotion` |
| `EpisodicMemory` | 出来事 | `id, character_id, content, timestamp, emotional_valence ∈ [-1, 1], importance ∈ [0, 1], conversation_id?, embedding?` |
| `SemanticMemory` | 汎化 | `id, character_id, content, confidence ∈ [0, 1], source_episode_ids[], embedding?` |
| `GoalTree` | 3 階層 | `visions[], objectives[], tasks[]` (フラットリスト + ID 参照) |
| `Vision` | 5–10 年 | `id, character_id, content` |
| `Objective` | 四半期〜年 | + `vision_id, status: active|achieved|abandoned, progress ∈ [0, 1]` |
| `Task` | 日〜週 | + `objective_id, status: pending|in_progress|completed|abandoned` |
| `Relation` | 関係性 | `owner_id, target_id, target_name, relationship_type, description, closeness ∈ [0, 1], trust ∈ [0, 1], updated_at, notes?` |
| `MessageInput` | 入力 | `content, sender_id, sender_name, sender_type: human|character|system, channel?, metadata` |
| `MessageOutput` | 出力 | `content, emotion, thought?, action?, tool_calls[], internal_changes[], diagnostic?, system_messages[]` |
| `ChangeRecord` | 内部変化 | `id, character_id, type, before?, after, reason, timestamp` |
| `TodoItem` | TODO + 習慣 | `id, content, label, kind, status, priority, due_date?, recurrence?, ...` |

## 不変条件

- Big Five と Schwartz Values は作成時に決まり、ランタイム中は不変として扱う。
- PAD は `[-1, 1]`、Big Five / Values / importance / confidence / closeness / trust は `[0, 1]`。
- これらのモデルは I/O や副作用を持たない (純粋なデータ層)。

## Non-goals

- 永続化の責務は持たない (それは [`storage`](../storage/))。
- LLM への変換ロジックは持たない (それは [`runtime`](../runtime/) の prompt builder)。

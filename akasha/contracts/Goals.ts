// Akasha Contract — Goals (Vision / Objective / Task / GoalTree)
//
// Python 一次ソース: src/pneuma_core/models/goals.py
//
// 3 階層構造を「フラットなリスト + ID 参照」で持つ。
// 階層的に nest はしない (`get_objectives_for_vision()` 等で関連付ける)。

export type ObjectiveStatus = "active" | "achieved" | "abandoned";
export type TaskStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "abandoned";

export interface Vision {
  id: string;
  character_id: string;
  /** 5–10 年スパンの長期ビジョン。 */
  content: string;
}

export interface Objective {
  id: string;
  character_id: string;
  vision_id: string;
  content: string;
  status: ObjectiveStatus;
  /** [0, 1]。 */
  progress: number;
}

export interface Task {
  id: string;
  character_id: string;
  objective_id: string;
  content: string;
  status: TaskStatus;
}

/** 3 階層の goal を保持する mutable コンテナ。 */
export interface GoalTree {
  visions: Vision[];
  objectives: Objective[];
  tasks: Task[];
}

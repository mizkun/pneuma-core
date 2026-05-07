// Akasha Contract — Goals (3 階層)
//
// Python 一次ソース: src/pneuma_core/models/goals.py
// Vision → Objective → Task の階層。

export interface Task {
  id: string;
  description: string;
  status: "pending" | "in_progress" | "done" | "abandoned";
  due_at?: string | null;
}

export interface Objective {
  id: string;
  description: string;
  tasks: Task[];
}

export interface Vision {
  id: string;
  description: string;
  objectives: Objective[];
}

export interface GoalTree {
  character_id: string;
  visions: Vision[];
  updated_at: string;
}

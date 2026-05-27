"""Multi-agent conversation framework (Issue #6 / Phase 0 C).

N 体のキャラクターが 1 つのセッションを共有して自律会話する基盤。

- Conversation: 参加者リストと共通 history を保持
- FloorController: 性格・関連度・発話頻度ペナルティで次の発話者を決める
- MultiAgentSession: 各キャラの内部状態（PAD・記憶・関係性）を更新するセッション
- SessionEndPipeline (N 体対応): 既存 SessionEndPipeline をループ実行 + 関係性クロス更新
- CircuitBreaker: turn / token / 経過時間で session を強制終了するセーフティ
- MockLLMAdapter: ANTHROPIC_API_KEY 未設定環境で動かすための固定応答プール

詳細は `.vibe/spec/stories/multi-agent.yaml` 参照。
"""

from pneuma_core.multi_agent.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.floor_controller import (
    FloorController,
    Utterance,
)
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter
from pneuma_core.multi_agent.session import (
    MultiAgentSession,
    TurnResult,
)
from pneuma_core.multi_agent.session_end import MultiAgentSessionEndPipeline

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "Conversation",
    "FloorController",
    "MockLLMAdapter",
    "MultiAgentSession",
    "MultiAgentSessionEndPipeline",
    "TurnResult",
    "Utterance",
]

"""FunEngineConfig — 面白さエンジンのトグル設定 (Issue #22).

設計の土台: .vibe/references/fun-engine-design.md

面白さ = 共感 × 発見。発見を「台本化せず創発で作る」ために、複数の独立した
創発要素をトグルで ON/OFF できるようにする。各要素は「出力（セリフ/展開/結末）を
決める（台本）」のではなく「入力（性格/クセ/欲求/お題/締切）を設計する（創発）」で
実装する。具体的な出力は LLM の自律に委ねる。

invariants（multi-agent Story 参照）:
- fun-engine-toggleable: 各創発要素は独立に ON/OFF でき、OFF 時は既存挙動を
  変えない。デフォルトは enable_intent のみ True（#20 の挙動を保つ）。

トグル一覧:
- enable_intent           : 思惑（表/裏のゴール、#20 実装）。デフォルト True。
- enable_quirk            : キャラの思考のクセ（character.quirk を発話に注入）。
- enable_terse            : テンポ（発話を短く、max_tokens 抑制 + 「短く」指示）。
- enable_lateral_thinking : 水平思考的飛躍（連想・アナロジーを促す）。
- enable_emotion_dynamics : 感情の温度差（全員同方向への高止まりを解消）。
- enable_structured_ending: 構造的オチ（終盤に締切/決定の外圧で収束を促す）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FunEngineConfig:
    """面白さエンジンの各創発要素を独立に ON/OFF するフラグ群.

    デフォルトは enable_intent のみ True、他は全て False。これにより #20
    （思惑エンジン）の挙動を保ったまま、各要素を明示的に ON にして A/B 検証
    できる（fun-engine-toggleable）。
    """

    # 思惑（#20 由来）。デフォルト True で挙動不変。
    enable_intent: bool = True
    # 思考のクセ（千明=勝負脳 / なでしこ=食い意地 / あおい=達観 等）
    enable_quirk: bool = False
    # テンポ（短い発話、大喜利的）
    enable_terse: bool = False
    # 水平思考的飛躍（連想・アナロジー）
    enable_lateral_thinking: bool = False
    # 感情の温度差・収束（高止まり解消）
    enable_emotion_dynamics: bool = False
    # 構造的オチ（締切/決定の外圧で終盤に収束）
    enable_structured_ending: bool = False

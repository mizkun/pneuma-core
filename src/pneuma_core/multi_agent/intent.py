"""思惑エンジン — 長期欲求 → 短期ゴール（思惑）の創発 (Issue #20).

invariants:
- intent-driven-utterance: 各キャラは長期欲求（values/goals）と状況から短期
  ゴール（思惑）を持ち、発話はその思惑に駆動される。
- intent-emergent-not-scripted: 短期ゴールは長期欲求 + 状況から創発的に生成され、
  展開（いつ何が起きるか）は台本化しない。
- speech-thought-separation: 各ターンは表の発言（speech）と裏の本音（thought）を
  分離生成し、両者にギャップがありうる（視聴者は thought を観測できる）。

設計（fun-engine-design.md 採用案 A）:
    キャラの長期欲求（Values + profile + 任意 desires + GoalTree.Vision）
      + セッションの状況（shared_context）
      → 短期ゴール（思惑）が創発 ← 台本じゃない

LLM 呼び出しは structured_completion（Tool Use 強制, Issue #12）を使い、
surface_goal / hidden_goal / intensity を schema で構造化する。
EmotionEngine と同じく Tool Use 経路 / generate fallback 経路の両対応で、
失敗時はキャラの Values から導出した degraded な Intent を返す（クラッシュしない）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.llm.retry import RetryableLLMCall
from pneuma_core.llm.validation import ResponseValidator, StructuredResponseError
from pneuma_core.models.character import Character
from pneuma_core.runtime.response_parser import strip_code_fences

logger = logging.getLogger(__name__)

_MAX_GOAL_LEN = 200


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class Intent:
    """キャラの短期ゴール（思惑）.

    invariant: speech-thought-separation を支える構造 — surface_goal は公言できる
    表のゴール、hidden_goal は裏の本音（言わない動機）。両者にギャップがありうる。

    Attributes:
        character_id: 思惑の持ち主.
        surface_goal: 表のゴール（公言できる、発話を駆動する）. 非空が前提.
        hidden_goal: 裏の本音（言わない動機）. None 可.
        intensity: 思惑の強さ 0.0〜1.0（強いほど発話を強く駆動する）.
    """

    character_id: str
    surface_goal: str
    intensity: float
    hidden_goal: str | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError(
                f"intensity must be between 0.0 and 1.0, got {self.intensity}"
            )


class IntentEvaluation(BaseModel):
    """構造化された思惑生成レスポンス (Tool Use / strict validation 用).

    invariant: structured-output-via-tool-use — schema は Tool Use 経由で強制。
    invariant: llm-response-validated-strict — strict Pydantic 検証。

    Note: intensity の範囲は Field で強制せず、_to_intent() で clamp する
        （LLM が範囲外を返すケースを degraded 受容する。EmotionEngine と同方針）。
    """

    surface_goal: str
    hidden_goal: str
    intensity: float


# JSON Schema for Tool Use (Anthropic API)
_INTENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "surface_goal": {
            "type": "string",
            "description": (
                "このキャラが今この会話で達成したい『表のゴール（思惑）』。"
                "公言できる短期ゴール。1 文で具体的に。"
            ),
        },
        "hidden_goal": {
            "type": "string",
            "description": (
                "表のゴールの裏にある『本音（言わない動機）』。"
                "表と食い違ってよい。無ければ空文字。"
            ),
        },
        "intensity": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "思惑の強さ (0.0=どうでもいい 〜 1.0=絶対に通したい)",
        },
    },
    "required": ["surface_goal", "hidden_goal", "intensity"],
}


_SYSTEM_PROMPT_BASE = """\
あなたはキャラクターの動機分析エキスパートです。
以下のキャラクターが、与えられた状況で「今この会話で達成したい短期ゴール（思惑）」を
創発させてください。

## キャラクター
名前: {name}
プロフィール: {profile}

## 性格 (Big Five, 0〜1)
- 開放性: {openness}
- 誠実性: {conscientiousness}
- 外向性: {extraversion}
- 協調性: {agreeableness}
- 神経症傾向: {neuroticism}

## 価値観 (Schwartz, 0〜1)
- 自己超越（他者・調和を重んじる）: {self_transcendence}
- 自己高揚（達成・権力を重んじる）: {self_enhancement}
- 開放性（変化・刺激を求める）: {openness_to_change}
- 保守（安全・伝統を重んじる）: {conservation}

## 長期欲求（このキャラが根底で求めているもの）
{desires}

## 今の状況
{situation}

## 指示
- 長期欲求と価値観から、この状況で自然に生まれる『短期ゴール（思惑）』を 1 つ作る。
- 思惑は内発的なものにする。視聴者を意識した「面白くしよう」は禁止。
  あくまでキャラ自身がこの会話で達成したいこと。
- 表のゴール（公言できる）と、その裏の本音（言わない動機）を分けてよい。
  本音は表と食い違ってよい（例: 表「企画を通したい」/ 裏「負けたくないだけ」）。
- 展開そのものは決めない。今の時点の思惑だけを出す。
"""


def _character_desires(character: Character) -> str:
    """キャラの長期欲求テキストを組み立てる.

    優先順位: desires（あれば）> values_description > Values から導出した一文。
    profile は別途プロンプトに入るので、ここでは「根底の欲求」に絞る。
    """
    parts: list[str] = []
    desires = getattr(character, "desires", None)
    if desires:
        parts.append(str(desires).strip())
    if character.values_description:
        parts.append(character.values_description.strip())
    if not parts:
        parts.append(_values_to_desire_hint(character))
    return "\n".join(p for p in parts if p)


def _values_to_desire_hint(character: Character) -> str:
    """Values から「根底で何を求めるか」のヒント文を導出する（fallback 用）."""
    v = character.values
    pairs = [
        (v.self_enhancement, "認められたい・抜きん出たい"),
        (v.self_transcendence, "人を支えたい・調和を保ちたい"),
        (v.openness_to_change, "新しいこと・刺激を求めたい"),
        (v.conservation, "安全・いつもの心地よさを守りたい"),
    ]
    pairs.sort(key=lambda x: x[0], reverse=True)
    top = pairs[0][1]
    return f"根底では「{top}」という欲求が強い。"


def _fallback_intent(character: Character, shared_context: str) -> Intent:
    """LLM が使えない / 失敗したときの degraded Intent.

    invariant: intent-driven-utterance — どんな状況でも surface_goal は非空にする
    （思惑ゼロ = 駆動力ゼロを避ける）。キャラの最も強い価値観から導出する。
    """
    hint = _values_to_desire_hint(character)
    situation = shared_context or "この場"
    surface = f"{situation}で、{hint.rstrip('。')}"
    # intensity は最も強い価値観の値（弱め下限 0.3 で必ず駆動する）
    v = character.values
    strongest = max(
        v.self_enhancement,
        v.self_transcendence,
        v.openness_to_change,
        v.conservation,
    )
    return Intent(
        character_id=character.id,
        surface_goal=surface,
        hidden_goal=None,
        intensity=_clamp01(max(0.3, strongest)),
    )


class IntentGenerator:
    """長期欲求 + 状況 → 短期ゴール（思惑）を生成する.

    EmotionEngine と同じ二経路戦略:
        1. adapter が structured_completion を持つ（real Claude / Mock）→
           Tool Use API で schema を強制 + retry。
        2. それ以外（テストの素朴な stub）→ generate() + strip + strict validate。
    どちらも失敗したら _fallback_intent を返す（クラッシュしない）。
    """

    def __init__(
        self,
        llm: LLMAdapter,
        model: str | None = None,
        max_retries: int = 2,
        temperature: float = 0.9,
    ) -> None:
        self._llm = llm
        self._model = model
        self._max_retries = max_retries
        # 思惑は創発性が要る（intent-emergent-not-scripted）ので温度高め。
        self._temperature = temperature

    async def generate(
        self,
        character: Character,
        shared_context: str,
    ) -> Intent:
        """1 キャラぶんの Intent を生成する.

        invariants: intent-driven-utterance, intent-emergent-not-scripted,
            structured-output-via-tool-use, llm-response-validated-strict
        """
        system_prompt = self._build_system_prompt(character, shared_context)
        validator = ResponseValidator(IntentEvaluation)
        retry = RetryableLLMCall(max_retries=self._max_retries, initial_delay=0.0)

        if self._supports_tool_use():
            call = self._make_tool_use_call(system_prompt, validator)
        else:
            call = self._make_generate_call(system_prompt, validator)

        try:
            evaluation = await retry.run(call)
        except StructuredResponseError as e:
            logger.warning(
                "Intent generation failed after retries for %s (%s), using fallback",
                character.name, e,
            )
            return _fallback_intent(character, shared_context)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Intent LLM call failed for %s: %s: %s, using fallback",
                character.name, type(e).__name__, e,
            )
            return _fallback_intent(character, shared_context)

        return self._to_intent(character, evaluation)

    def _to_intent(self, character: Character, ev: IntentEvaluation) -> Intent:
        surface = ev.surface_goal.strip()[:_MAX_GOAL_LEN]
        hidden_raw = ev.hidden_goal.strip()[:_MAX_GOAL_LEN]
        hidden = hidden_raw or None
        if not surface:
            # LLM が空 surface を返した → fallback に切り替え（駆動力を保証）
            return _fallback_intent(character, character.profile or "")
        return Intent(
            character_id=character.id,
            surface_goal=surface,
            hidden_goal=hidden,
            intensity=_clamp01(ev.intensity),
        )

    def _build_system_prompt(
        self, character: Character, shared_context: str
    ) -> str:
        p = character.personality
        v = character.values
        return _SYSTEM_PROMPT_BASE.format(
            name=character.name,
            profile=character.profile or "（記載なし）",
            openness=p.openness,
            conscientiousness=p.conscientiousness,
            extraversion=p.extraversion,
            agreeableness=p.agreeableness,
            neuroticism=p.neuroticism,
            self_transcendence=v.self_transcendence,
            self_enhancement=v.self_enhancement,
            openness_to_change=v.openness_to_change,
            conservation=v.conservation,
            desires=_character_desires(character),
            situation=shared_context or "（特になし）",
        )

    def _make_tool_use_call(
        self,
        system_prompt: str,
        validator: ResponseValidator[IntentEvaluation],
    ):
        async def call() -> IntentEvaluation:
            payload = await self._llm.structured_completion(  # type: ignore[attr-defined]
                system_prompt=system_prompt,
                messages=[{
                    "role": "user",
                    "content": "今のあなたの思惑（短期ゴール）を 1 つ作ってください。",
                }],
                tool_name="report_intent",
                tool_description=(
                    "Report the character's emergent short-term goal (intent) "
                    "for this conversation, with a surface goal, a hidden true "
                    "motive, and an intensity."
                ),
                schema=_INTENT_SCHEMA,
                model=self._model,
                temperature=self._temperature,
            )
            return validator.validate(payload)
        return call

    def _make_generate_call(
        self,
        system_prompt: str,
        validator: ResponseValidator[IntentEvaluation],
    ):
        request = LLMRequest(
            system_prompt=system_prompt,
            messages=[{
                "role": "user",
                "content": (
                    "今のあなたの思惑を JSON で出力してください: "
                    '{"surface_goal": "...", "hidden_goal": "...", "intensity": 0.0}'
                ),
            }],
            model=self._model,
            temperature=self._temperature,
        )

        async def call() -> IntentEvaluation:
            response = await self._llm.generate(request)
            raw = response.content or ""
            stripped = strip_code_fences(raw)
            try:
                data = json.loads(stripped)
            except (json.JSONDecodeError, TypeError) as e:
                raise StructuredResponseError(
                    f"failed to JSON-parse intent response: {e}; raw={raw[:200]!r}"
                ) from e
            return validator.validate(data)
        return call

    def _supports_tool_use(self) -> bool:
        marker = getattr(type(self._llm), "supports_structured_completion", None)
        return bool(marker)

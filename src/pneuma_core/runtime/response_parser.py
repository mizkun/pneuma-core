"""Response parser: parse structured LLM output (speech/thought/action).

Follows the same JSON parsing pattern as diary_processor._parse_json_response.

invariant: llm-response-pre-strip — Tool Use を使えないレガシーパス（テスト・
モック・後方互換）でも、マークダウンコードブロック (```json ... ```) や前後の
説明文は前処理で除去してから JSON parse する。
"""

from __future__ import annotations

import json
import re

from pneuma_core.models.message import StructuredResponse

# Keys that identify a structured response
_STRUCTURED_KEYS = frozenset({"speech", "thought", "action"})

# Markdown code fence with optional language hint (json/yaml/etc.)
_FENCE_RE = re.compile(
    r"```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)\n?```",
    re.DOTALL,
)


def strip_code_fences(text: str) -> str:
    """JSON 文字列を、マークダウンコードブロックや前後の説明文から取り出す.

    Strategy (in order):
        1. ``` ... ``` (with or without language tag) があれば中身を抜く
        2. 残ったテキストから最初の `{` と最後の `}` を見つけて切り出す
        3. どちらも見つからなければ原文をそのまま返す（呼び出し側で fallback）

    Args:
        text: LLM の生レスポンス文字列.

    Returns:
        JSON parse 可能な可能性が高い文字列、あるいは原文.

    invariant: llm-response-pre-strip
    """
    if not text:
        return text

    stripped = text.strip()
    if not stripped:
        return text

    # 1. コードフェンス内の中身を取り出す
    fence_match = _FENCE_RE.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()

    # 2. 前後説明文を剥がす — 最初の { から最後の } まで
    first_open = stripped.find("{")
    last_close = stripped.rfind("}")
    if first_open != -1 and last_close != -1 and last_close > first_open:
        return stripped[first_open : last_close + 1]

    # 3. JSON 構造が見つからない → 原文を返す（呼び出し側で fallback）
    return text


def parse_structured_response(raw: str) -> StructuredResponse:
    """Parse a raw LLM response into a StructuredResponse.

    Parsing strategy:
        1. Strip markdown code blocks (```json ... ```)
        2. Try json.loads()
        3. If valid dict with at least one structured key -> StructuredResponse
        4. Otherwise, fallback: entire raw text becomes speech

    Args:
        raw: Raw text from LLM response.

    Returns:
        StructuredResponse with speech/thought/action fields.
    """
    text = raw.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Not valid JSON -> fallback to plain text
        return StructuredResponse(speech=raw)

    if not isinstance(parsed, dict):
        # Non-dict JSON (e.g. array) -> fallback
        return StructuredResponse(speech=raw)

    # Check if it has at least one structured key
    if not _STRUCTURED_KEYS & set(parsed.keys()):
        # No structured keys -> fallback
        return StructuredResponse(speech=raw)

    return StructuredResponse(
        speech=parsed.get("speech"),
        thought=parsed.get("thought"),
        action=parsed.get("action"),
    )

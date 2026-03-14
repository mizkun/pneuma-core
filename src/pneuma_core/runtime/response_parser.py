"""Response parser: parse structured LLM output (speech/thought/action).

Follows the same JSON parsing pattern as diary_processor._parse_json_response.
"""

from __future__ import annotations

import json

from pneuma_core.models.message import StructuredResponse

# Keys that identify a structured response
_STRUCTURED_KEYS = frozenset({"speech", "thought", "action"})


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

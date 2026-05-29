"""Regression tests for instrumentation reliability — Issue #23 バグ3.

キャラ名のブレ（「リン」混入）除去と呼称の固定
(invariant: cast-naming-fixed).

AC-3: 3 キャラの呼称が固定され「リン」等のブレが起きない。
"""

from __future__ import annotations

import json

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter, _RESPONSE_POOL
from pneuma_core.multi_agent.session import MultiAgentSession

# 正しいキャスト本名
_CAST = ("なでしこ", "千明", "あおい")
# 旧デモ cross_chat 由来の混入してはいけない名前
_FORBIDDEN_NAMES = ("リン", "アイネ")


def _make_char(char_id: str, name: str, extraversion: float = 0.5) -> Character:
    return Character(
        id=char_id,
        name=name,
        personality=Personality(
            openness=0.6, conscientiousness=0.5, extraversion=extraversion,
            agreeableness=0.7, neuroticism=0.4,
        ),
        values=Values(
            self_transcendence=0.5, self_enhancement=0.5,
            openness_to_change=0.5, conservation=0.5,
        ),
        profile="テストキャラ",
        speaking_style="普通",
    )


def _has_forbidden(text: str) -> str | None:
    """禁止名（「リン」等）が含まれていれば最初の語を返す.

    「プリン」「リング」等の部分一致を誤検出しないように、前後の文字も見る簡易判定。
    本テストの応答プールは短文なので、リン単独 or リンちゃん等を拾えれば十分。
    """
    for bad in _FORBIDDEN_NAMES:
        idx = 0
        while True:
            pos = text.find(bad, idx)
            if pos == -1:
                break
            after = text[pos + len(bad): pos + len(bad) + 1]
            before = text[pos - 1: pos]
            # 「リン」は人名なら直後が「ちゃん/さん/くん/、/。/！/空白/文末」になりやすい。
            # 「プリン」など直前が別文字でも混入名なので、ここでは単純に検出する。
            # 例外: カタカナ語の一部（リング/プリン）は before/after でゆるく除外。
            if bad == "リン" and (before == "プ" or after in ("グ", "ゴ")):
                idx = pos + len(bad)
                continue
            return bad
    return None


# ─────────────── MockLLM 応答プールの静的検査 ───────────────


def test_response_pool_has_no_forbidden_names() -> None:
    """AC-3: MockLLM の固定セリフプールに「リン」等の旧名が含まれない."""
    for name, pool in _RESPONSE_POOL.items():
        for speech, action in pool:
            assert _has_forbidden(speech) is None, (
                f"{name} の speech に禁止名: {speech!r}"
            )
            assert _has_forbidden(action) is None, (
                f"{name} の action に禁止名: {action!r}"
            )


def test_response_pool_only_references_valid_cast() -> None:
    """AC-3: プール内でキャラ名に言及するときは正しいキャストのみ.

    なでしこのプールに「リンちゃん」があった回帰。正しい呼称(ちかちゃん等)へ。
    """
    forbidden_found = []
    for name, pool in _RESPONSE_POOL.items():
        for speech, _action in pool:
            bad = _has_forbidden(speech)
            if bad:
                forbidden_found.append((name, speech, bad))
    assert not forbidden_found, f"禁止名混入: {forbidden_found}"


# ─────────────── キャスト呼称表（cast roster）───────────────


def test_cast_roster_present_in_speech_prompt() -> None:
    """AC-3: 発話プロンプト先頭にキャスト呼称表が固定で示される.

    本名と呼称（他キャラからの呼び方）を明示して、生成時の逸脱を防ぐ。
    """
    from pneuma_core.multi_agent.session import build_cast_roster_section

    chars = [
        _make_char("a", "なでしこ", 0.95),
        _make_char("b", "千明", 0.7),
        _make_char("c", "あおい", 0.5),
    ]
    roster = build_cast_roster_section(chars)
    # 全キャストの本名が含まれる
    for name in _CAST:
        assert name in roster
    # 禁止名は含まれない
    assert _has_forbidden(roster) is None


@pytest.mark.asyncio
async def test_speech_prompt_includes_cast_roster() -> None:
    """AC-3: run_turn の発話 system_prompt にキャスト呼称表が注入される."""
    captured: list[str] = []

    class _Capturing(MockLLMAdapter):
        async def generate(self, request):  # type: ignore[override]
            captured.append(request.system_prompt)
            return await super().generate(request)

    chars = [
        _make_char("a", "なでしこ", 0.95),
        _make_char("b", "千明", 0.7),
        _make_char("c", "あおい", 0.5),
    ]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(conversation=conv, llm=_Capturing(seed=1))
    await session.run_turn()
    speech_prompt = captured[0]
    # キャスト本名がプロンプトに載る（呼称固定）
    for name in _CAST:
        assert name in speech_prompt
    assert _has_forbidden(speech_prompt) is None


@pytest.mark.asyncio
async def test_generated_speech_has_no_forbidden_names() -> None:
    """AC-3: 何ターン回しても生成 speech に禁止名が出ない."""
    chars = [
        _make_char("a", "なでしこ", 0.95),
        _make_char("b", "千明", 0.7),
        _make_char("c", "あおい", 0.5),
    ]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(conversation=conv, llm=MockLLMAdapter(seed=7))
    for _ in range(15):
        await session.run_turn()
    for u in conv.history:
        assert _has_forbidden(u.speech or "") is None, (
            f"生成 speech に禁止名: {u.speech!r}"
        )
        assert _has_forbidden(u.thought or "") is None, (
            f"生成 thought に禁止名: {u.thought!r}"
        )

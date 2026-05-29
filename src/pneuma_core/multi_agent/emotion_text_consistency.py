"""内心(PAD/emotion)とテキスト(thought)の整合 (Issue #23 バグ1).

invariant: pad-text-consistency — 確定した emotion_label と、生成された thought
（内心）の感情方向が正反対にならない。矛盾したときは label を thought 側に
合わせる（観測の信頼性を守る）。

設計:
- emotion_label / thought をそれぞれ「感情方向」(+1 正 / -1 負 / 0 中立・不明)に
  写像する。
- 方向が正反対（積が負）なら矛盾とみなし、label を thought の方向に合わせた
  代表ラベルへ補正する。
- どちらかが中立(0)なら過補正を避けて何もしない。

ここではキーワードベースの軽量判定にとどめる（real Claude でも mock でも
決定論的に効かせるため）。将来 sentiment 推定に差し替え可能。
"""

from __future__ import annotations

from pneuma_core.models.emotion import EmotionLabel

# 感情ラベル → 方向 (+1 正 / -1 負 / 0 中立)。
# AITuber 6 種ラベル (Issue #9) を快/不快の軸で分類する。
# surprised は快/不快が定まらないので中立扱い（過補正しない）。
_LABEL_DIRECTION: dict[str, int] = {
    EmotionLabel.HAPPY.value: 1,
    EmotionLabel.TEASING.value: 1,
    EmotionLabel.SAD_LITE.value: -1,
    EmotionLabel.EMBARRASSED.value: -1,
    EmotionLabel.SURPRISED.value: 0,
    EmotionLabel.NEUTRAL.value: 0,
}

# thought（内心）が正方向であることを示す語。
_POSITIVE_CUES: tuple[str, ...] = (
    "嬉し", "楽し", "うれし", "たのし", "やった", "わくわく", "ワクワク",
    "好き", "幸せ", "しあわせ", "最高", "いける", "勝てる", "自信",
    "ありがと", "安心", "満たさ", "誇らし", "ほっと",
)

# thought（内心）が負方向であることを示す語。
_NEGATIVE_CUES: tuple[str, ...] = (
    "悲し", "かなし", "つらい", "辛い", "最悪", "嫌", "いや", "怖い",
    "こわい", "不安", "悔し", "くやし", "落ち込", "がっかり", "ショック",
    "情けな", "恥ずかし", "はずかし", "むかつ", "腹立", "泣き", "苦し",
    "寂し", "さみし", "ダメだ", "だめだ", "失敗",
)

# 方向ごとの補正先ラベル（thought に合わせる代表ラベル）。
_DIRECTION_FALLBACK_LABEL: dict[int, str] = {
    1: EmotionLabel.HAPPY.value,
    -1: EmotionLabel.SAD_LITE.value,
}


def label_emotion_direction(emotion_label: str) -> int:
    """emotion_label を感情方向(+1/-1/0)に写像する.

    未知ラベルは中立(0)扱い。
    """
    return _LABEL_DIRECTION.get((emotion_label or "").strip().lower(), 0)


def thought_emotion_direction(thought: str) -> int:
    """thought（内心）の感情方向を推定する.

    正/負どちらの感情語が多いかで方向を決める。同数 or 感情語なしは中立(0)。
    """
    text = thought or ""
    if not text:
        return 0
    pos = sum(1 for cue in _POSITIVE_CUES if cue in text)
    neg = sum(1 for cue in _NEGATIVE_CUES if cue in text)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def is_contradictory(emotion_label: str, thought: str) -> bool:
    """emotion_label と thought の感情方向が正反対か.

    どちらかが中立(0)なら矛盾とはみなさない（過補正しない）。
    """
    l_dir = label_emotion_direction(emotion_label)
    t_dir = thought_emotion_direction(thought)
    return l_dir * t_dir < 0


def resolve_emotion_text_consistency(
    emotion_label: str, thought: str
) -> tuple[str, bool]:
    """emotion_label と thought の整合をとる.

    invariant: pad-text-consistency。

    Returns:
        (corrected_label, changed):
          矛盾していれば label を thought の方向に合わせた代表ラベルへ補正し、
          (補正ラベル, True) を返す。矛盾していなければ (元ラベル, False)。
    """
    if not is_contradictory(emotion_label, thought):
        return emotion_label, False
    t_dir = thought_emotion_direction(thought)
    corrected = _DIRECTION_FALLBACK_LABEL.get(t_dir, emotion_label)
    return corrected, corrected != emotion_label

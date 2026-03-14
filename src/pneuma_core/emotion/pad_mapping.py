"""PAD → discrete emotion label mapping (8 octants + neutral)."""

_DEFAULT_THRESHOLD = 0.3

# 8-octant emotion label table (Discussion #11)
_OCTANT_TABLE: list[tuple[bool, bool, bool, str]] = [
    # (P+, A+, D+) → label
    (True, True, True, "喜び（高揚）"),
    (True, True, False, "感動"),
    (True, False, True, "余裕"),
    (True, False, False, "安らぎ"),
    (False, True, True, "怒り"),
    (False, True, False, "恐怖"),
    (False, False, True, "倦怠"),
    (False, False, False, "絶望"),
]


def pad_to_emotion_label(
    pleasure: float,
    arousal: float,
    dominance: float,
    threshold: float = _DEFAULT_THRESHOLD,
) -> str:
    """PAD 値から離散感情ラベルを返す.

    各軸の絶対値が threshold 未満の場合は「中間」として扱い、
    全軸が中間なら「中立」を返す。
    一部の軸だけ中間の場合は、その軸を正として扱い最も近いオクタントを選択する。
    """
    p_sign = pleasure >= 0 if abs(pleasure) >= threshold else None
    a_sign = arousal >= 0 if abs(arousal) >= threshold else None
    d_sign = dominance >= 0 if abs(dominance) >= threshold else None

    # 全軸が中間 → 中立
    if p_sign is None and a_sign is None and d_sign is None:
        return "中立"

    # 中間の軸はデフォルトで正（>= 0）として処理
    p_pos = p_sign if p_sign is not None else (pleasure >= 0)
    a_pos = a_sign if a_sign is not None else (arousal >= 0)
    d_pos = d_sign if d_sign is not None else (dominance >= 0)

    for pos_p, pos_a, pos_d, label in _OCTANT_TABLE:
        if p_pos == pos_p and a_pos == pos_a and d_pos == pos_d:
            return label

    return "中立"

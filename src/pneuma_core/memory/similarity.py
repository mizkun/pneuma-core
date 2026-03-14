"""Cosine similarity utility for memory embeddings."""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """2つのベクトル間のコサイン類似度を計算.

    Returns:
        -1.0〜1.0 の類似度。ゼロベクトルの場合は 0.0。

    Raises:
        ValueError: ベクトルの次元が異なる場合。
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

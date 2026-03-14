"""Exponential decay for emotion → baseline regression."""

import math


def exponential_decay(
    current: float,
    baseline: float,
    elapsed_seconds: float,
    half_life: float,
) -> float:
    """感情値を指数減衰でベースラインに近づける.

    decay = exp(-ln(2) / half_life * elapsed_seconds)
    result = baseline + (current - baseline) * decay

    Args:
        current: 現在の感情値
        baseline: ベースライン値
        elapsed_seconds: 経過秒数
        half_life: 半減期（秒）

    Returns:
        減衰後の感情値
    """
    if half_life <= 0.0:
        raise ValueError(f"half_life must be positive, got {half_life}")

    if elapsed_seconds <= 0.0:
        return current

    decay_rate = math.log(2) / half_life
    decay = math.exp(-decay_rate * elapsed_seconds)

    return baseline + (current - baseline) * decay

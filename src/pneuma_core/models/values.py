"""Schwartz values model."""

from dataclasses import dataclass

_IMPORTANCE_THRESHOLD = 0.6

_DIMENSIONS = ("self_transcendence", "self_enhancement", "openness_to_change", "conservation")


def _validate_range(value: float, name: str) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
    return value


@dataclass(frozen=True)
class Values:
    """Schwartz 4 カテゴリ価値観モデル.

    各パラメータは 0.0〜1.0 の範囲。
    対立軸: self_transcendence ↔ self_enhancement, openness_to_change ↔ conservation
    しきい値: 0.6 以上で「重視する」
    """

    self_transcendence: float
    self_enhancement: float
    openness_to_change: float
    conservation: float

    def __post_init__(self) -> None:
        for dim in _DIMENSIONS:
            _validate_range(getattr(self, dim), dim)

    def is_important(self, dimension: str) -> bool:
        """指定した価値観が重視されている（0.6 以上）かを判定."""
        return getattr(self, dimension) >= _IMPORTANCE_THRESHOLD

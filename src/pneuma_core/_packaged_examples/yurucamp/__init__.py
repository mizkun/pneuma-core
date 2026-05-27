"""ゆるキャン野クル組 3 キャラ (Issue #7 Phase 0 C2 dataset).

なでしこ / 千明 / あおい の character.yaml + helper to load them as a set.
"""

from __future__ import annotations

from pathlib import Path

from pneuma_core.character_sheet import CharacterSheet

_DIR = Path(__file__).parent


def load_yurucamp_sheets() -> list[CharacterSheet]:
    """野クル組 3 体を CharacterSheet として読み込む.

    Returns:
        [なでしこ, 千明, あおい] の順で固定（外向性高い順 ≈ 自然な発話順）。
    """
    nadeshiko = CharacterSheet.load(_DIR / "nadeshiko.character.yaml")
    chiaki = CharacterSheet.load(_DIR / "chiaki.character.yaml")
    aoi = CharacterSheet.load(_DIR / "aoi.character.yaml")
    return [nadeshiko, chiaki, aoi]


__all__ = ["load_yurucamp_sheets"]

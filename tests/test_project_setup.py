"""Tests for Issue #1: プロジェクト構造のセットアップ.

受け入れ基準:
- `from pneuma_core import ...` が動く
- パッケージ構造が spec.md のリポジトリ構成に一致する
"""


def test_pneuma_core_package_importable():
    """pneuma_core パッケージがインポートできる."""
    import pneuma_core

    assert pneuma_core is not None


def test_core_models_importable():
    """pneuma_core.models がインポートできる."""
    from pneuma_core import models

    assert models is not None


def test_core_runtime_importable():
    """pneuma_core.runtime がインポートできる."""
    from pneuma_core import runtime

    assert runtime is not None


def test_core_emotion_importable():
    """pneuma_core.emotion がインポートできる."""
    from pneuma_core import emotion

    assert emotion is not None


def test_core_memory_importable():
    """pneuma_core.memory がインポートできる."""
    from pneuma_core import memory

    assert memory is not None


def test_core_storage_importable():
    """pneuma_core.storage がインポートできる."""
    from pneuma_core import storage

    assert storage is not None


def test_core_llm_importable():
    """pneuma_core.llm がインポートできる."""
    from pneuma_core import llm

    assert llm is not None


def test_core_character_sheet_importable():
    """pneuma_core.character_sheet がインポートできる."""
    from pneuma_core import character_sheet

    assert character_sheet is not None

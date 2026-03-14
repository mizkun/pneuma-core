"""Vault path resolution utility.

PNEUMA_VAULT_PATH 環境変数からサブパスを解決するユーティリティ。
vault/ ディレクトリにキャラクター定義・ユーザーコンテキスト・
ランタイムデータを一元管理する。
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_VAULT_PATH = "./vault"
_ENV_VAR = "PNEUMA_VAULT_PATH"


def get_vault_path() -> Path:
    """Vault のルートパスを返す.

    PNEUMA_VAULT_PATH 環境変数が設定されていればその値を使い、
    未設定の場合はデフォルト ``./vault`` にフォールバックする。
    """
    return Path(os.environ.get(_ENV_VAR, _DEFAULT_VAULT_PATH))


def get_characters_dir() -> Path:
    """キャラクター定義ディレクトリ ``vault/characters/`` のパスを返す."""
    return get_vault_path() / "characters"


def get_user_context_dir() -> Path:
    """ユーザーコンテキストディレクトリ ``vault/user/`` のパスを返す."""
    return get_vault_path() / "user"


def get_logs_dir() -> Path:
    """ログディレクトリ ``vault/logs/`` のパスを返す."""
    return get_vault_path() / "logs"


def get_db_path() -> Path:
    """データベースファイル ``vault/pneuma.db`` のパスを返す."""
    return get_vault_path() / "pneuma.db"


def get_entity_dir(entity_name: str) -> Path:
    """エンティティデータディレクトリ ``vault/{entity_name}/`` のパスを返す.

    vault/{entity}/ パターンでエンティティ固有のデータを管理するために使用する。
    例: vault/mira/, vault/user/, vault/aine/ など。
    """
    return get_vault_path() / entity_name


def get_entity_diary_dir(entity_name: str) -> Path:
    """エンティティの日記ディレクトリ ``vault/{entity_name}/diary/`` のパスを返す."""
    return get_entity_dir(entity_name) / "diary"


def get_entity_relations_path(entity_name: str) -> Path:
    """エンティティの関係性ファイル ``vault/{entity_name}/relations.yaml`` のパスを返す."""
    return get_entity_dir(entity_name) / "relations.yaml"

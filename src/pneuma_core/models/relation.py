"""Relation: エンティティ間の関係性モデル."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Relation:
    """エンティティ間の関係性.

    ユーザー-キャラクター、キャラクター-キャラクター間の関係を表す。
    owner_id が関係の「主体」、target_id が「相手」。
    """

    id: str
    owner_id: str              # 関係の主体 (user, mira, etc.)
    target_id: str             # 関係の相手
    target_name: str           # 相手の表示名
    relationship_type: str     # partner, friend, family, mentor, etc.
    description: str           # 関係の説明
    closeness: float           # 親密度 0.0-1.0
    trust: float               # 信頼度 0.0-1.0
    updated_at: datetime
    notes: str | None = None   # 追加メモ

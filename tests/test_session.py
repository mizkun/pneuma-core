"""Tests for session management (Issue #71)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from pneuma_core.runtime.session import (
    ConversationSession,
    SessionConfig,
    SessionManager,
)

NOW = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)


# --- SessionConfig ---


class TestSessionConfig:
    """SessionConfig の検証."""

    def test_default_timeout(self) -> None:
        config = SessionConfig()
        assert config.timeout_minutes == 30

    def test_custom_timeout(self) -> None:
        config = SessionConfig(timeout_minutes=60)
        assert config.timeout_minutes == 60


# --- ConversationSession ---


class TestConversationSession:
    """ConversationSession モデルの検証."""

    def test_creation(self) -> None:
        session = ConversationSession(
            session_id="sess-001",
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            started_at=NOW,
            last_active_at=NOW,
        )
        assert session.session_id == "sess-001"
        assert session.character_id == "aine-001"
        assert session.user_id == "user-001"
        assert session.channel_id == "ch-001"
        assert session.started_at == NOW
        assert session.last_active_at == NOW
        assert session.messages == []

    def test_messages_default_empty(self) -> None:
        session = ConversationSession(
            session_id="sess-001",
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            started_at=NOW,
            last_active_at=NOW,
        )
        assert session.messages == []
        assert isinstance(session.messages, list)

    def test_messages_mutable(self) -> None:
        session = ConversationSession(
            session_id="sess-001",
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            started_at=NOW,
            last_active_at=NOW,
        )
        msg = {"role": "user", "content": "hello"}
        session.messages.append(msg)
        assert len(session.messages) == 1
        assert session.messages[0] == msg


# --- SessionManager ---


class TestSessionManager:
    """SessionManager の検証."""

    def test_create_new_session(self) -> None:
        manager = SessionManager()
        session, is_new = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        assert is_new is True
        assert session.character_id == "aine-001"
        assert session.user_id == "user-001"
        assert session.channel_id == "ch-001"
        assert session.started_at == NOW
        assert session.last_active_at == NOW
        assert session.messages == []
        assert session.session_id  # session_id が生成されている

    def test_get_existing_session_before_timeout(self) -> None:
        """タイムアウト前は既存セッションを返す."""
        manager = SessionManager()
        session1, is_new1 = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        assert is_new1 is True

        # 10分後にアクセス（タイムアウト前）
        later = NOW + timedelta(minutes=10)
        session2, is_new2 = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=later,
        )
        assert is_new2 is False
        assert session2.session_id == session1.session_id
        assert session2.last_active_at == later

    def test_create_new_session_after_timeout(self) -> None:
        """タイムアウト後は新しいセッションを作成."""
        manager = SessionManager()
        session1, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )

        # 31分後にアクセス（タイムアウト後）
        later = NOW + timedelta(minutes=31)
        session2, is_new = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=later,
        )
        assert is_new is True
        assert session2.session_id != session1.session_id
        assert session2.started_at == later

    def test_custom_timeout(self) -> None:
        """カスタムタイムアウト値を設定できる."""
        config = SessionConfig(timeout_minutes=60)
        manager = SessionManager(config=config)

        session1, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )

        # 45分後（デフォルト30分超だがカスタム60分以内）
        later = NOW + timedelta(minutes=45)
        session2, is_new = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=later,
        )
        assert is_new is False
        assert session2.session_id == session1.session_id

    def test_session_key_includes_channel_type(self) -> None:
        """セッション識別は channel_type:channel_id:user_id の組み合わせ."""
        manager = SessionManager()

        # discord チャンネル
        session_discord, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )

        # cli チャンネル（同じ channel_id, user_id でも別セッション）
        session_cli, is_new = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="cli",
            now=NOW,
        )
        assert is_new is True
        assert session_discord.session_id != session_cli.session_id

    def test_different_users_get_different_sessions(self) -> None:
        """異なるユーザーは別セッション."""
        manager = SessionManager()

        session_user1, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )

        session_user2, is_new = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-002",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        assert is_new is True
        assert session_user1.session_id != session_user2.session_id

    def test_add_message(self) -> None:
        """メッセージを追加すると messages に蓄積される."""
        manager = SessionManager()
        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )

        msg = {"role": "user", "content": "こんにちは"}
        manager.add_message(session.session_id, msg)

        assert len(session.messages) == 1
        assert session.messages[0] == msg

    def test_add_message_updates_last_active(self) -> None:
        """メッセージ追加で last_active_at が更新される."""
        manager = SessionManager()
        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )

        later = NOW + timedelta(minutes=5)
        msg = {"role": "user", "content": "こんにちは"}
        manager.add_message(session.session_id, msg, now=later)

        assert session.last_active_at == later

    def test_add_message_nonexistent_session_raises(self) -> None:
        """存在しないセッションへのメッセージ追加はエラー."""
        manager = SessionManager()
        with pytest.raises(KeyError):
            manager.add_message("nonexistent", {"role": "user", "content": "hello"})

    def test_end_session(self) -> None:
        """セッション終了で ConversationSession が返される."""
        manager = SessionManager()
        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        sid = session.session_id
        msg = {"role": "user", "content": "こんにちは"}
        manager.add_message(sid, msg)

        ended = manager.end_session(sid)
        assert ended.session_id == sid
        assert len(ended.messages) == 1

    def test_end_session_removes_from_active(self) -> None:
        """終了後のセッションは active から削除される."""
        manager = SessionManager()
        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        sid = session.session_id
        manager.add_message(sid, {"role": "user", "content": "hello"})
        manager.end_session(sid)

        # 次のアクセスは新しいセッション
        session2, is_new = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        assert is_new is True
        assert session2.session_id != sid

    def test_end_session_nonexistent_raises(self) -> None:
        """存在しないセッションの終了はエラー."""
        manager = SessionManager()
        with pytest.raises(KeyError):
            manager.end_session("nonexistent")

    def test_end_session_empty_returns_none(self) -> None:
        """空のセッション（メッセージなし）の終了は None を返す."""
        manager = SessionManager()
        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        result = manager.end_session(session.session_id)
        assert result is None

    def test_check_timeout_false_before_expiry(self) -> None:
        """タイムアウト前は False."""
        manager = SessionManager()
        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        later = NOW + timedelta(minutes=10)
        assert manager.check_timeout(session.session_id, now=later) is False

    def test_check_timeout_true_after_expiry(self) -> None:
        """タイムアウト後は True."""
        manager = SessionManager()
        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        later = NOW + timedelta(minutes=31)
        assert manager.check_timeout(session.session_id, now=later) is True

    def test_check_timeout_exact_boundary(self) -> None:
        """ちょうど30分はタイムアウトしない（超過のみ）."""
        manager = SessionManager()
        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        exactly_30 = NOW + timedelta(minutes=30)
        assert manager.check_timeout(session.session_id, now=exactly_30) is False

    def test_check_timeout_nonexistent_raises(self) -> None:
        """存在しないセッションの timeout チェックはエラー."""
        manager = SessionManager()
        with pytest.raises(KeyError):
            manager.check_timeout("nonexistent", now=NOW)

    def test_conversation_id_for_memory_linkage(self) -> None:
        """session_id が conversation_id として EpisodicMemory と紐づく."""
        manager = SessionManager()
        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        # session_id は文字列で、EpisodicMemory の conversation_id として使える
        assert isinstance(session.session_id, str)
        assert len(session.session_id) > 0


# --- SessionManager callback ---


class TestSessionManagerCallback:
    """セッション終了時のコールバック検証."""

    def test_on_session_end_callback_called(self) -> None:
        """セッション終了時に on_session_end コールバックが呼ばれる."""
        callback = MagicMock()
        manager = SessionManager(on_session_end=callback)

        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        manager.add_message(session.session_id, {"role": "user", "content": "hello"})
        manager.end_session(session.session_id)

        callback.assert_called_once()
        called_session = callback.call_args[0][0]
        assert called_session.session_id == session.session_id

    def test_on_session_end_not_called_for_empty_session(self) -> None:
        """空セッション終了時はコールバック不発火."""
        callback = MagicMock()
        manager = SessionManager(on_session_end=callback)

        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        manager.end_session(session.session_id)

        callback.assert_not_called()

    def test_on_session_end_callback_on_timeout_replacement(self) -> None:
        """タイムアウトによるセッション置換時もコールバックが呼ばれる."""
        callback = MagicMock()
        manager = SessionManager(on_session_end=callback)

        session1, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        manager.add_message(
            session1.session_id,
            {"role": "user", "content": "hello"},
            now=NOW,
        )

        # タイムアウト後に新セッション作成 → 旧セッションのコールバック発火
        later = NOW + timedelta(minutes=31)
        session2, is_new = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=later,
        )
        assert is_new is True

        callback.assert_called_once()
        called_session = callback.call_args[0][0]
        assert called_session.session_id == session1.session_id

    def test_no_callback_set(self) -> None:
        """コールバック未設定でもエラーにならない."""
        manager = SessionManager()  # no callback
        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        manager.add_message(session.session_id, {"role": "user", "content": "hello"})
        ended = manager.end_session(session.session_id)
        assert ended is not None  # コールバックなしでも正常終了


# --- SessionManager: get_or_create_session with timeout auto-end ---


class TestSessionManagerTimeoutAutoEnd:
    """タイムアウト時の自動セッション終了."""

    def test_timeout_auto_ends_old_session(self) -> None:
        """タイムアウト後の get_or_create で旧セッションが自動終了される."""
        manager = SessionManager()
        session1, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        manager.add_message(
            session1.session_id,
            {"role": "user", "content": "first message"},
            now=NOW,
        )

        # タイムアウト後
        later = NOW + timedelta(minutes=31)
        session2, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=later,
        )

        # 旧セッションは active から削除されている
        with pytest.raises(KeyError):
            manager.check_timeout(session1.session_id, now=later)

    def test_timeout_empty_session_no_callback(self) -> None:
        """空セッションがタイムアウトした場合はコールバック不発火."""
        callback = MagicMock()
        manager = SessionManager(on_session_end=callback)

        session1, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )

        # メッセージなしでタイムアウト
        later = NOW + timedelta(minutes=31)
        manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=later,
        )

        callback.assert_not_called()


# --- SessionManager async callback (#130) ---


class TestSessionManagerAsyncCallback:
    """async コールバック対応の検証 (#130)."""

    @pytest.mark.asyncio
    async def test_async_callback_awaited_on_end_session(self) -> None:
        """async コールバックが end_session で await される."""
        callback = AsyncMock()
        manager = SessionManager(on_session_end=callback)

        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        manager.add_message(session.session_id, {"role": "user", "content": "hello"})
        await manager.end_session_async(session.session_id)

        callback.assert_awaited_once()
        called_session = callback.call_args[0][0]
        assert called_session.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_async_callback_awaited_on_timeout(self) -> None:
        """タイムアウト時も async コールバックが await される."""
        callback = AsyncMock()
        manager = SessionManager(on_session_end=callback)

        session1, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        manager.add_message(
            session1.session_id,
            {"role": "user", "content": "hello"},
            now=NOW,
        )

        later = NOW + timedelta(minutes=31)
        session2, is_new = await manager.get_or_create_session_async(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=later,
        )
        assert is_new is True

        callback.assert_awaited_once()
        called_session = callback.call_args[0][0]
        assert called_session.session_id == session1.session_id

    @pytest.mark.asyncio
    async def test_async_end_session_empty_returns_none(self) -> None:
        """空セッションの async end は None を返す."""
        callback = AsyncMock()
        manager = SessionManager(on_session_end=callback)

        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        result = await manager.end_session_async(session.session_id)
        assert result is None
        callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sync_callback_still_works_with_async_end(self) -> None:
        """sync コールバックでも async end メソッドが動作する."""
        callback = MagicMock()
        manager = SessionManager(on_session_end=callback)

        session, _ = manager.get_or_create_session(
            character_id="aine-001",
            user_id="user-001",
            channel_id="ch-001",
            channel_type="discord",
            now=NOW,
        )
        manager.add_message(session.session_id, {"role": "user", "content": "hello"})
        await manager.end_session_async(session.session_id)

        callback.assert_called_once()

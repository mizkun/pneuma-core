"""Session management: conversation session tracking with timeout."""

from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable


@dataclass
class SessionConfig:
    """Tunable parameters for session management."""

    timeout_minutes: int = 30


@dataclass
class ConversationSession:
    """A conversation session between a user and a character.

    Attributes:
        session_id: Unique session identifier (also used as conversation_id for EpisodicMemory).
        character_id: The character participating in this session.
        user_id: The user participating in this session.
        channel_id: The channel where the conversation takes place.
        started_at: When the session started.
        last_active_at: When the last message was sent/received.
        messages: List of message dicts (role/content) in this session.
    """

    session_id: str
    character_id: str
    user_id: str
    channel_id: str
    started_at: datetime
    last_active_at: datetime
    messages: list[dict] = field(default_factory=list)


class SessionManager:
    """Manages conversation sessions with timeout-based expiry.

    Sessions are identified by the combination of channel_type:channel_id:user_id.
    When a session times out (no activity for timeout_minutes), it is automatically
    ended and a new session is created.

    Args:
        config: Session configuration (timeout etc.).
        on_session_end: Optional callback invoked when a non-empty session ends.
            Called with the ended ConversationSession as the sole argument.
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        on_session_end: Callable[[ConversationSession], None] | None = None,
    ) -> None:
        self._config = config or SessionConfig()
        self._on_session_end = on_session_end

        # session_id -> ConversationSession
        self._sessions: dict[str, ConversationSession] = {}

        # session_key (channel_type:channel_id:user_id) -> session_id
        self._key_to_session_id: dict[str, str] = {}

    @staticmethod
    def _make_session_key(channel_type: str, channel_id: str, user_id: str) -> str:
        """Build the session lookup key."""
        return f"{channel_type}:{channel_id}:{user_id}"

    def get_or_create_session(
        self,
        character_id: str,
        user_id: str,
        channel_id: str,
        channel_type: str,
        now: datetime | None = None,
    ) -> tuple[ConversationSession, bool]:
        """Get an existing session or create a new one.

        If an existing session has timed out, it is ended (triggering on_session_end
        if the session had messages) and a new session is created.

        Args:
            character_id: The character for this conversation.
            user_id: The user for this conversation.
            channel_id: The channel identifier.
            channel_type: The channel type (e.g. "discord", "cli").
            now: Current time (defaults to UTC now).

        Returns:
            Tuple of (session, is_new) where is_new indicates a fresh session.
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)

        key = self._make_session_key(channel_type, channel_id, user_id)
        timeout_delta = timedelta(minutes=self._config.timeout_minutes)

        # Check for existing session
        existing_sid = self._key_to_session_id.get(key)
        if existing_sid is not None and existing_sid in self._sessions:
            existing = self._sessions[existing_sid]
            elapsed = now - existing.last_active_at
            if elapsed <= timeout_delta:
                # Session still active — update last_active_at and return
                existing.last_active_at = now
                return existing, False
            else:
                # Session timed out — end it
                self._end_session_internal(existing_sid)

        # Create new session
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        session = ConversationSession(
            session_id=session_id,
            character_id=character_id,
            user_id=user_id,
            channel_id=channel_id,
            started_at=now,
            last_active_at=now,
        )
        self._sessions[session_id] = session
        self._key_to_session_id[key] = session_id
        return session, True

    def add_message(
        self,
        session_id: str,
        message: dict,
        now: datetime | None = None,
    ) -> None:
        """Add a message to a session and update last_active_at.

        Args:
            session_id: The session to add the message to.
            message: The message dict (role/content).
            now: Current time (defaults to UTC now).

        Raises:
            KeyError: If the session does not exist.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")

        if now is None:
            now = datetime.now(tz=timezone.utc)

        session = self._sessions[session_id]
        session.messages.append(message)
        session.last_active_at = now

    def end_session(self, session_id: str) -> ConversationSession | None:
        """End a session explicitly.

        If the session has messages, returns the session and triggers on_session_end.
        If the session is empty (no messages), returns None and does not trigger callback.

        Args:
            session_id: The session to end.

        Returns:
            The ended ConversationSession if it had messages, None otherwise.

        Raises:
            KeyError: If the session does not exist.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")

        return self._end_session_internal(session_id)

    def check_timeout(
        self,
        session_id: str,
        now: datetime | None = None,
    ) -> bool:
        """Check if a session has timed out.

        Args:
            session_id: The session to check.
            now: Current time (defaults to UTC now).

        Returns:
            True if the session has timed out, False otherwise.

        Raises:
            KeyError: If the session does not exist.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")

        if now is None:
            now = datetime.now(tz=timezone.utc)

        session = self._sessions[session_id]
        timeout_delta = timedelta(minutes=self._config.timeout_minutes)
        elapsed = now - session.last_active_at
        return elapsed > timeout_delta

    def _end_session_internal(self, session_id: str) -> ConversationSession | None:
        """Internal session end logic (sync version).

        Removes the session from active tracking and triggers callback if non-empty.
        For async callbacks, use end_session_async() or get_or_create_session_async().

        Returns:
            The ended session if it had messages, None otherwise.
        """
        session = self._sessions.pop(session_id)

        # Remove from key mapping
        keys_to_remove = [
            k for k, v in self._key_to_session_id.items() if v == session_id
        ]
        for k in keys_to_remove:
            del self._key_to_session_id[k]

        if not session.messages:
            return None

        # Trigger callback for non-empty sessions
        if self._on_session_end is not None:
            self._on_session_end(session)

        return session

    async def _end_session_internal_async(
        self, session_id: str
    ) -> ConversationSession | None:
        """Internal session end logic (async version).

        Supports both sync and async callbacks via inspect.isawaitable.

        Returns:
            The ended session if it had messages, None otherwise.
        """
        session = self._sessions.pop(session_id)

        # Remove from key mapping
        keys_to_remove = [
            k for k, v in self._key_to_session_id.items() if v == session_id
        ]
        for k in keys_to_remove:
            del self._key_to_session_id[k]

        if not session.messages:
            return None

        # Trigger callback for non-empty sessions (sync or async)
        if self._on_session_end is not None:
            result = self._on_session_end(session)
            if inspect.isawaitable(result):
                await result

        return session

    async def end_session_async(
        self, session_id: str
    ) -> ConversationSession | None:
        """End a session explicitly (async version).

        Supports both sync and async on_session_end callbacks.

        Args:
            session_id: The session to end.

        Returns:
            The ended ConversationSession if it had messages, None otherwise.

        Raises:
            KeyError: If the session does not exist.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")

        return await self._end_session_internal_async(session_id)

    async def get_or_create_session_async(
        self,
        character_id: str,
        user_id: str,
        channel_id: str,
        channel_type: str,
        now: datetime | None = None,
    ) -> tuple[ConversationSession, bool]:
        """Get an existing session or create a new one (async version).

        Same as get_or_create_session but awaits async callbacks on timeout.
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)

        key = self._make_session_key(channel_type, channel_id, user_id)
        timeout_delta = timedelta(minutes=self._config.timeout_minutes)

        # Check for existing session
        existing_sid = self._key_to_session_id.get(key)
        if existing_sid is not None and existing_sid in self._sessions:
            existing = self._sessions[existing_sid]
            elapsed = now - existing.last_active_at
            if elapsed <= timeout_delta:
                # Session still active — update last_active_at and return
                existing.last_active_at = now
                return existing, False
            else:
                # Session timed out — end it (async)
                await self._end_session_internal_async(existing_sid)

        # Create new session
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        session = ConversationSession(
            session_id=session_id,
            character_id=character_id,
            user_id=user_id,
            channel_id=channel_id,
            started_at=now,
            last_active_at=now,
        )
        self._sessions[session_id] = session
        self._key_to_session_id[key] = session_id
        return session, True

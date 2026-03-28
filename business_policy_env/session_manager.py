from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import time

from .environment import BusinessPolicyComplianceEnv


class SessionCapacityError(RuntimeError):
    """Raised when the session store is at capacity."""


class RateLimitError(RuntimeError):
    """Raised when a client exceeds the configured request budget."""


@dataclass
class SessionEntry:
    env: BusinessPolicyComplianceEnv
    created_at: float
    last_access: float


class SessionManager:
    def __init__(
        self,
        *,
        max_sessions: int = 100,
        session_ttl_seconds: int = 3600,
        rate_limit_per_minute: int = 120,
        base_seed: int = 20260328,
    ) -> None:
        self._max_sessions = max_sessions
        self._session_ttl_seconds = session_ttl_seconds
        self._rate_limit_per_minute = rate_limit_per_minute
        self._base_seed = base_seed
        self._sessions: dict[str, SessionEntry] = {}
        self._rate_windows: dict[str, deque[float]] = defaultdict(deque)

    def _stable_seed(self, key: str) -> int:
        return self._base_seed + sum((index + 1) * ord(char) for index, char in enumerate(key))

    def _now(self) -> float:
        return time.time()

    def _evict_expired_sessions(self, now: float | None = None) -> None:
        current_time = self._now() if now is None else now
        expired = [
            session_id
            for session_id, entry in self._sessions.items()
            if current_time - entry.last_access > self._session_ttl_seconds
        ]
        for session_id in expired:
            self.close(session_id)

    def _prune_rate_window(self, key: str, now: float) -> deque[float]:
        window = self._rate_windows[key]
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if not window:
            self._rate_windows.pop(key, None)
            window = deque()
            self._rate_windows[key] = window
        return window

    def enforce_rate_limit(self, client_host: str, session_id: str, now: float | None = None) -> None:
        current_time = self._now() if now is None else now
        self._evict_expired_sessions(current_time)
        key = f"{client_host}:{session_id}"
        window = self._prune_rate_window(key, current_time)
        if len(window) >= self._rate_limit_per_minute:
            raise RateLimitError("Rate limit exceeded. Please retry later.")
        window.append(current_time)

    def get_or_create(self, session_id: str, now: float | None = None) -> BusinessPolicyComplianceEnv:
        current_time = self._now() if now is None else now
        self._evict_expired_sessions(current_time)
        if session_id in self._sessions:
            entry = self._sessions[session_id]
            entry.last_access = current_time
            return entry.env

        if len(self._sessions) >= self._max_sessions:
            raise SessionCapacityError("Session capacity reached. Please retry later.")

        env = BusinessPolicyComplianceEnv(seed=self._stable_seed(session_id))
        self._sessions[session_id] = SessionEntry(env=env, created_at=current_time, last_access=current_time)
        return env

    def get(self, session_id: str, now: float | None = None) -> BusinessPolicyComplianceEnv | None:
        current_time = self._now() if now is None else now
        self._evict_expired_sessions(current_time)
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        entry.last_access = current_time
        return entry.env

    def close(self, session_id: str) -> None:
        entry = self._sessions.pop(session_id, None)
        if entry is not None:
            entry.env.close()

    def close_all(self) -> None:
        for session_id in list(self._sessions):
            self.close(session_id)
        self._rate_windows.clear()

    def active_session_count(self) -> int:
        self._evict_expired_sessions()
        return len(self._sessions)


default_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    return default_session_manager

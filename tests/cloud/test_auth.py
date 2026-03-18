"""Tests for HMAC-SHA256 session token auth."""

from maddening.cloud._auth import generate_session_token, validate_session_token


class TestSessionTokens:
    def test_generate_returns_hex_string(self):
        token = generate_session_token("session-123", "my-secret")
        assert isinstance(token, str)
        assert len(token) == 64  # SHA-256 hex digest

    def test_validate_correct_token(self):
        secret = "test-secret-key"
        sid = "session-abc"
        token = generate_session_token(sid, secret)
        assert validate_session_token(sid, token, secret)

    def test_validate_wrong_token(self):
        secret = "test-secret-key"
        assert not validate_session_token("session-1", "bogus", secret)

    def test_validate_wrong_session_id(self):
        secret = "test-secret-key"
        token = generate_session_token("session-1", secret)
        assert not validate_session_token("session-2", token, secret)

    def test_validate_wrong_secret(self):
        token = generate_session_token("session-1", "secret-a")
        assert not validate_session_token("session-1", token, "secret-b")

    def test_deterministic(self):
        t1 = generate_session_token("s1", "key")
        t2 = generate_session_token("s1", "key")
        assert t1 == t2

    def test_different_sessions_different_tokens(self):
        t1 = generate_session_token("s1", "key")
        t2 = generate_session_token("s2", "key")
        assert t1 != t2

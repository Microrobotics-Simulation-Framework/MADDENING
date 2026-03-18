"""HMAC-SHA256 session token generation and validation.

Tokens authenticate WebRTC signaling connections so that only the
session owner (or holders of the shared secret) can connect.
"""

import hashlib
import hmac


def generate_session_token(session_id: str, secret: str) -> str:
    """Generate an HMAC-SHA256 token for *session_id*."""
    return hmac.new(
        secret.encode("utf-8"),
        session_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def validate_session_token(
    session_id: str,
    token: str,
    secret: str,
) -> bool:
    """Validate *token* against *session_id* using constant-time compare."""
    expected = generate_session_token(session_id, secret)
    return hmac.compare_digest(token, expected)

"""Password hashing for local accounts (self-serve signup/login).

Uses scrypt from the standard library (memory-hard, no extra dependency). Stored as
``scrypt$<salt_b64>$<hash_b64>``; verification is constant-time. OIDC/SSO (Keycloak) is
the enterprise path and replaces this without touching call sites — auth still yields a
:class:`Principal` either way.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

_N, _R, _P, _DKLEN = 2**14, 8, 1, 32


def hash_password(password: str) -> str:
    """Hash a password for storage (random per-password salt)."""
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN)
    return f"scrypt${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of a password against a stored hash. False on any malformed input."""
    try:
        scheme, salt_b64, dk_b64 = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    try:
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
    except (ValueError, TypeError):
        return False
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN)
    return hmac.compare_digest(dk, expected)


# A throwaway hash to verify against when an account does NOT exist, so a login attempt does
# the same scrypt work whether or not the email is registered — otherwise the response time
# leaks account existence (timing-based enumeration, QA ISS-008).
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def verify_dummy(password: str) -> bool:
    """Burn the same work as a real verification for a non-existent account. Always returns False."""
    verify_password(password, _DUMMY_HASH)
    return False

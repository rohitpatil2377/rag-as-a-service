"""Authentication & authorization.

Phase-0 ships a **dev-auth stub**: HS256 JWTs signed with a dev-only secret, so
tenancy/RBAC/ABAC are fully testable end-to-end without standing up an IdP. The
:class:`Principal` and the FastAPI dependencies are the stable contract; Phase-1
swaps ``dev_jwt`` for real OIDC (Keycloak JWKS) with no call-site changes
(TDD §8: federated SSO + MFA).

RBAC via ``roles``; ABAC via ``attrs`` (e.g. CAG officer hierarchy
Dealing-Hand -> AAO -> SAO -> DAG). Authorization helpers are pure and unit-testable.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import jwt
from fastapi import Depends, Header

from rag_common.config import get_settings
from rag_common.errors import ForbiddenError, UnauthorizedError


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller. ``tenant_id`` is authoritative for isolation."""

    subject: str
    tenant_id: str
    roles: frozenset[str] = field(default_factory=frozenset)
    attrs: dict[str, Any] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        return role in self.roles


def _decode_dev_jwt(token: str) -> Principal:
    settings = get_settings()
    try:
        claims = jwt.decode(token, settings.dev_jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise UnauthorizedError("invalid token", details={"reason": str(exc)}) from exc
    tenant_id = claims.get("tenant_id")
    subject = claims.get("sub")
    if not tenant_id or not subject:
        raise UnauthorizedError("token missing sub/tenant_id")
    return Principal(
        subject=subject,
        tenant_id=tenant_id,
        roles=frozenset(claims.get("roles", [])),
        attrs=claims.get("attrs", {}),
    )


_jwk_clients: dict[str, Any] = {}  # jwks_url -> cached PyJWKClient (keys refreshed by the client)


def _oidc_signing_key(token: str, jwks_url: str) -> Any:
    """The RSA/EC public key for a token's `kid`, fetched from the IdP's JWKS (cached).
    The JWKS host must be egress-allowlisted — an IdP is an external dependency, gated
    exactly like any other outbound call."""
    from rag_common.sovereignty import assert_sovereign  # noqa: PLC0415 - avoid import cycle

    assert_sovereign(jwks_url)
    client = _jwk_clients.get(jwks_url)
    if client is None:
        client = jwt.PyJWKClient(jwks_url)
        _jwk_clients[jwks_url] = client
    return client.get_signing_key_from_jwt(token).key


def _has_mfa(claims: dict[str, Any], mfa_values: set[str]) -> bool:
    """True when the token evidences multi-factor auth via `amr` (methods) or `acr` (context)."""
    amr = claims.get("amr") or []
    if isinstance(amr, list) and (set(amr) & mfa_values or "mfa" in amr):
        return True
    return str(claims.get("acr", "")) in mfa_values


def _decode_oidc_jwt(token: str) -> Principal:
    """Verify an IdP-issued JWT against the provider JWKS and map claims → Principal (FR: SSO).
    MFA (if required) is asserted from the token, never trusted from the client."""
    settings = get_settings()
    if not settings.oidc_jwks_url:
        raise UnauthorizedError("OIDC is enabled but RAG_OIDC_JWKS_URL is not configured")
    try:
        key = _oidc_signing_key(token, settings.oidc_jwks_url)
        claims: dict[str, Any] = jwt.decode(
            token,
            key,
            algorithms=["RS256", "RS512", "ES256", "ES384"],
            audience=settings.oidc_audience or None,
            issuer=settings.oidc_issuer or None,
            options={"verify_aud": bool(settings.oidc_audience), "verify_iss": bool(settings.oidc_issuer)},
        )
    except jwt.PyJWTError as exc:
        raise UnauthorizedError("invalid OIDC token", details={"reason": str(exc)}) from exc

    if settings.oidc_require_mfa:
        mfa_values = {v.strip() for v in settings.oidc_acr_mfa_values.split(",") if v.strip()}
        if not _has_mfa(claims, mfa_values):
            raise UnauthorizedError("multi-factor authentication required")

    subject = claims.get(settings.oidc_subject_claim) or claims.get("sub")
    tenant_id = claims.get(settings.oidc_tenant_claim) or claims.get("org") or claims.get("tenant_id")
    if not subject or not tenant_id:
        raise UnauthorizedError("OIDC token missing subject or tenant/org claim")
    roles = claims.get(settings.oidc_roles_claim) or []
    groups = claims.get(settings.oidc_groups_claim) or []
    return Principal(
        subject=str(subject),
        tenant_id=str(tenant_id),
        roles=frozenset(str(r) for r in roles) if isinstance(roles, list) else frozenset(),
        attrs={"groups": list(groups) if isinstance(groups, list) else [], "idp": True},
    )


def authenticate(authorization: str | None) -> Principal:
    """Verify a bearer token and return the :class:`Principal` (fail-closed)."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise UnauthorizedError("missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    settings = get_settings()
    if settings.auth_mode == "dev_jwt":
        return _decode_dev_jwt(token)
    if settings.auth_mode == "oidc":
        return _decode_oidc_jwt(token)
    raise UnauthorizedError(f"unsupported auth mode: {settings.auth_mode}")


# --- API-key resolution seam (FR-I4) ----------------------------------------
# The key store lives in the control plane (Postgres/in-memory); auth in libs/common
# stays store-agnostic. The control plane registers an async resolver at startup.
_ApiKeyResolver = Callable[[str], Awaitable[Principal | None]]
_api_key_resolver: _ApiKeyResolver | None = None
_API_KEY_PREFIX = "ek_"


def register_api_key_resolver(resolver: _ApiKeyResolver) -> None:
    """Wire the API-key → Principal resolver (called once at app startup)."""
    global _api_key_resolver  # noqa: PLW0603 — one-time startup registration hook
    _api_key_resolver = resolver


async def get_principal(
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """FastAPI dependency: authenticated principal for the request (bearer JWT or API key)."""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        if token.startswith(_API_KEY_PREFIX):
            if _api_key_resolver is None:
                raise UnauthorizedError("API keys are not enabled")
            principal = await _api_key_resolver(token)
            if principal is None:
                raise UnauthorizedError("invalid or revoked API key")
            return principal
    return authenticate(authorization)


def require_roles(*required: str) -> Any:
    """FastAPI dependency factory enforcing that the principal has ALL roles (RBAC)."""

    async def _dep(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        missing = {r for r in required if not principal.has_role(r)}
        if missing:
            raise ForbiddenError("missing required roles", details={"missing": sorted(missing)})
        return principal

    return _dep


def issue_dev_jwt(
    subject: str,
    tenant_id: str,
    roles: list[str] | None = None,
    attrs: dict[str, Any] | None = None,
    *,
    ttl_seconds: int | None = None,
) -> str:
    """Mint a dev JWT (tests, demo CLI, local console). NOT for production use.

    ``ttl_seconds`` adds a standard ``exp`` claim (PyJWT verifies it on decode) — used for
    short-lived scoped tokens such as the embeddable-widget session token (FR-I6)."""
    settings = get_settings()
    payload: dict[str, Any] = {"sub": subject, "tenant_id": tenant_id, "roles": roles or [], "attrs": attrs or {}}
    if ttl_seconds is not None:
        payload["exp"] = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
    return jwt.encode(payload, settings.dev_jwt_secret, algorithm="HS256")

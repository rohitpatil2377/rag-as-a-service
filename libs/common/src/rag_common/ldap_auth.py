"""On-prem Active Directory / LDAP bind authentication (banking pack).

Many Indian banks run internal Active Directory. This lets a user sign in with their AD
credentials: the platform performs an LDAP *bind* as that user against the bank's directory and,
on success, issues the normal platform JWT — no separate password store. Complements OIDC (for
IdPs that speak OIDC) and dev-JWT.

The heavy dependency (``ldap3``) is the optional ``rag-common[ldap]`` extra, imported on use so
the base image still boots. The AD server host is put through ``assert_sovereign`` (operator-
allowlisted) so directory traffic stays within the sovereign boundary. Off by default
(``ldap_enabled``). A ``binder`` can be injected for tests (no live directory needed).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rag_common.config import get_settings
from rag_common.logging import get_logger
from rag_common.sovereignty import assert_sovereign

_log = get_logger(__name__)


@dataclass(slots=True)
class LdapIdentity:
    email: str
    name: str
    groups: list[str] = field(default_factory=list)


# A binder attempts an LDAP bind and returns the user's attributes, or None on auth failure.
# Signature: (server, user_dn, password, base_dn) -> LdapIdentity | None
Binder = Callable[[str, str, str, str], LdapIdentity | None]


def role_for_groups(groups: list[str], role_map: str, default: str = "reader") -> str:
    """Map AD group names to a platform role via ``ldap_role_map`` ('grp:role,grp2:role2').
    Highest-privilege wins so a user in both an admin and a reader group is an admin."""
    mapping: dict[str, str] = {}
    for pair in role_map.split(","):
        if ":" in pair:
            grp, role = pair.split(":", 1)
            mapping[grp.strip().lower()] = role.strip()
    order = {"owner": 4, "admin": 3, "reviewer": 2, "ingestor": 1, "reader": 0}
    best = default
    for g in groups:
        mapped = mapping.get(g.strip().lower())
        if mapped and order.get(mapped, -1) > order.get(best, -1):
            best = mapped
    return best


def _default_binder(server: str, user_dn: str, password: str, base_dn: str) -> LdapIdentity | None:  # pragma: no cover
    """Real ldap3 bind — only exercised when the [ldap] extra + a live directory are present."""
    import ldap3  # noqa: PLC0415

    conn = ldap3.Connection(ldap3.Server(server, get_info=ldap3.NONE), user=user_dn, password=password)
    if not conn.bind():
        return None
    attrs = ["mail", "displayName", "cn", "memberOf"]
    conn.search(base_dn or user_dn, f"(distinguishedName={user_dn})", attributes=attrs) if base_dn else None
    entry = conn.entries[0] if conn.entries else None
    email = str(entry.mail) if entry and "mail" in entry else user_dn
    name = str(entry.displayName) if entry and "displayName" in entry else user_dn
    groups = [str(g).split(",")[0].split("=")[-1] for g in (entry.memberOf if entry and "memberOf" in entry else [])]
    conn.unbind()
    return LdapIdentity(email=email, name=name, groups=groups)


def authenticate_ldap(username: str, password: str, *, binder: Binder | None = None) -> LdapIdentity | None:
    """Bind ``username``/``password`` against the configured AD/LDAP server. Returns the identity
    on success, or None on invalid credentials / misconfiguration (never raises to the caller)."""
    s = get_settings()
    if not s.ldap_enabled or not s.ldap_server:
        return None
    if not password:  # never allow an anonymous/unauthenticated bind to pass as a login
        return None
    try:
        assert_sovereign(s.ldap_server)  # AD host must be operator-allowlisted (sovereign)
        user_dn = (s.ldap_user_dn_template or "{username}").format(username=username)
        bind = binder or _default_binder
        return bind(s.ldap_server, user_dn, password, s.ldap_base_dn)
    except Exception as exc:
        _log.warning("ldap_auth_failed", error=str(exc)[:180])
        return None

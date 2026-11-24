"""Air-gap preflight — pure checks for a fully offline deployment.

Sovereign mode (`sovereign_enforce` + `allowed_egress_hosts` in :mod:`rag_common.config`)
allow-lists specific outbound hosts; it does not promise there are none. Air-gap mode is a
stricter contract for sites with NO outbound connectivity at all: every setting that would
still dial out — a hosted model/OCR backend, SMTP, an external IdP's JWKS, a remote SIEM, or
a missing update-verification key — must be absent before the site starts. This module never
makes a network call itself; it only inspects a settings object and reports what it finds.
"""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel


class AirGapViolation(BaseModel):
    """One setting that would breach air-gap mode if the site went fully offline."""

    setting: str
    value: str
    reason: str


def _host_of(value: str) -> str:
    """Best-effort hostname from a URL or a bare ``host[:port]`` string (e.g. a syslog target)."""
    if not value:
        return ""
    netloc_form = value if "//" in value else f"//{value}"
    return (urlparse(netloc_form).hostname or "").lower()


def _is_offsite(host: str) -> bool:
    """True if `host` is not something air-gap treats as the operator's own infra.

    Treated as local (NOT a violation): loopback/RFC1918-private IPs, bare docker/intranet
    service names with no dot (how this codebase names its own containers throughout
    config.py, e.g. "qdrant", "vllm", "reranker"), and the ".local"/".internal"
    private-network TLD conventions used elsewhere in config.py (e.g. smtp_from's
    "ragaas.local", console_base_url's "rag.acme.internal" example). Anything
    else — a real public FQDN such as "www.datalab.to" or "api.openai.com" — is off-box.
    """
    if not host:
        return False
    if host in {"localhost", "0.0.0.0", "::1"}:  # noqa: S104 - membership check, not a bind
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        pass  # not a literal IP address
    else:
        return not (addr.is_private or addr.is_loopback)
    if "." not in host:
        return False  # bare service name resolved on the operator's own network
    return not (host.endswith(".local") or host.endswith(".internal"))


def _backend_violation(
    settings: Any, *, backend_attr: str, hosted_values: set[str], url_attr: str, label: str
) -> AirGapViolation | None:
    """Flag a model/OCR backend selection whose configured endpoint resolves off-box.

    Gated on the backend actually being set to one of `hosted_values` — an
    `openai_compat` backend pointed at the operator's own intranet vLLM/TEI is not a
    violation; only a URL that resolves off-box under the selected backend is.
    """
    backend_value = getattr(settings, backend_attr, "")
    if backend_value not in hosted_values:
        return None
    url = getattr(settings, url_attr, "") or ""
    if not _is_offsite(_host_of(url)):
        return None
    return AirGapViolation(
        setting=url_attr,
        value=url,
        reason=(
            f"{label} backend ({backend_attr}={backend_value!r}) resolves off-box; "
            "air-gap requires a self-hosted/intranet endpoint"
        ),
    )


# Each entry checks one backend selector against the URL it actually dials when set to a
# "hosted" value. `chandra_api` always means Datalab's hosted API (datalab_url); `chandra`
# and `openai_compat` are self-hostable, so they're violations only if the operator pointed
# them off-box.
_BACKEND_CHECKS: list[dict[str, Any]] = [
    {"backend_attr": "llm_backend", "hosted_values": {"openai_compat"}, "url_attr": "vllm_url", "label": "LLM"},
    {
        "backend_attr": "embed_backend",
        "hosted_values": {"openai_compat"},
        "url_attr": "tei_url",
        "label": "embedding",
    },
    {
        "backend_attr": "rerank_backend",
        "hosted_values": {"openai_compat"},
        "url_attr": "rerank_url",
        "label": "reranker",
    },
    {"backend_attr": "ocr_backend", "hosted_values": {"chandra_api"}, "url_attr": "datalab_url", "label": "OCR"},
    {"backend_attr": "ocr_backend", "hosted_values": {"chandra"}, "url_attr": "chandra_url", "label": "OCR"},
    {
        "backend_attr": "vision_llm_backend",
        "hosted_values": {"openai_compat"},
        "url_attr": "vision_llm_url",
        "label": "vision-LLM",
    },
    {
        "backend_attr": "vision_llm_backend",
        "hosted_values": {"chandra_api"},
        "url_attr": "datalab_url",
        "label": "vision-OCR",
    },
    {
        "backend_attr": "vision_llm_backend",
        "hosted_values": {"chandra"},
        "url_attr": "chandra_url",
        "label": "vision-OCR",
    },
    # Sarvam translate-then-extract dials translate_sarvam_url (api.sarvam.ai by default) at
    # runtime; without this entry air-gap preflight FALSE-PASSED a live hosted egress.
    {
        "backend_attr": "translate_backend",
        "hosted_values": {"sarvam"},
        "url_attr": "translate_sarvam_url",
        "label": "translate",
    },
]


def airgap_violations(settings: Any) -> list[AirGapViolation]:
    """Every setting on `settings` that would breach air-gap mode, in check order.

    Pure — reads attributes off `settings` via `getattr` (sane defaults for anything
    missing) and returns data; makes no network/DB call of its own. Checks:

    1. Hosted/external model or OCR backends: LLM/embedding/reranker/vision-LLM set to
       `openai_compat`, or OCR/vision-OCR set to `chandra`/`chandra_api` — each flagged
       only if the URL that backend actually dials resolves off-box (see `_is_offsite`).
       `chandra_api` is Datalab's hosted API and is inherently remote by default.
    2. `smtp_host` set — any SMTP relay is outbound email; air-gapped sites must leave
       it empty and use the admin-copies-a-link activation/reset fallback instead.
    3. `oidc_jwks_url` set — JWKS is fetched live on every token verification; that has
       no offline substitute, so it's flagged whenever configured.
    4. `siem_enabled` with `siem_endpoint` resolving off-box — an on-prem SIEM collector
       is fine, a remote one is egress.
    5. `update_release_public_key` EMPTY — an air-gapped site can never phone home to
       check an update bundle's provenance, so it MUST be able to verify locally; an
       empty key silently disables that check, which is itself a real risk.
    """
    violations: list[AirGapViolation] = []

    for check in _BACKEND_CHECKS:
        result = _backend_violation(settings, **check)
        if result is not None:
            violations.append(result)

    smtp_host = getattr(settings, "smtp_host", "") or ""
    if smtp_host:
        violations.append(
            AirGapViolation(
                setting="smtp_host",
                value=smtp_host,
                reason="SMTP relay configured; email is outbound egress with no offline substitute",
            )
        )

    jwks_url = getattr(settings, "oidc_jwks_url", "") or ""
    if jwks_url:
        violations.append(
            AirGapViolation(
                setting="oidc_jwks_url",
                value=jwks_url,
                reason=(
                    "OIDC JWKS is fetched live on every token verification; "
                    "an air-gapped site has no offline substitute"
                ),
            )
        )

    if getattr(settings, "siem_enabled", False):
        siem_endpoint = getattr(settings, "siem_endpoint", "") or ""
        if siem_endpoint and _is_offsite(_host_of(siem_endpoint)):
            violations.append(
                AirGapViolation(
                    setting="siem_endpoint",
                    value=siem_endpoint,
                    reason="SIEM forwarding enabled with an off-box endpoint; air-gap requires an on-prem collector",
                )
            )

    release_key = getattr(settings, "update_release_public_key", "") or ""
    if not release_key:
        violations.append(
            AirGapViolation(
                setting="update_release_public_key",
                value="",
                reason=(
                    "empty — an air-gapped site cannot phone home to check update provenance "
                    "and MUST verify bundles locally"
                ),
            )
        )

    return violations


def assert_air_gap(settings: Any) -> None:
    """Raise ValueError naming every violation if the site is air-gapped and misconfigured.

    No-op when `air_gap_enabled` is falsy/missing — air-gap is opt-in, so a site that
    hasn't turned it on is never blocked by this check.
    """
    if not getattr(settings, "air_gap_enabled", False):
        return
    violations = airgap_violations(settings)
    if not violations:
        return
    detail = "; ".join(f"{v.setting}={v.value!r} ({v.reason})" for v in violations)
    raise ValueError(f"air-gap mode is enabled but {len(violations)} setting(s) breach it: {detail}")


def airgap_preflight(settings: Any) -> dict[str, Any]:
    """Report shape for an ops preflight endpoint. Never raises.

    Violations are computed regardless of `air_gap_enabled` so an operator can preview
    "would this deployment pass air-gap" before flipping the flag on. `clean` is simply
    whether the violation list is empty; when `air_gap_enabled` is false nothing is
    currently being enforced, but the report still shows what would need fixing.
    """
    violations = airgap_violations(settings)
    return {
        "air_gap_enabled": bool(getattr(settings, "air_gap_enabled", False)),
        "clean": not violations,
        "violations": [v.model_dump() for v in violations],
    }

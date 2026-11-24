"""A revoke must not be able to disable a host the DEPLOYMENT configured.

Happened live: an admin revoked `models-uat.example.internal` in the console's egress panel. That
host is the generation model's endpoint, set in the deployment's own environment -- so every
answer stopped working, with a sovereignty error that pointed at the guard rather than at the
revoke. A restart silently fixed it, because the base list comes back from the environment.

The approval workflow exists to WIDEN the allowlist at runtime. Narrowing it below what the
operator configured is not something that surface should be able to do, and the asymmetry has to
be enforced rather than documented.
"""

from __future__ import annotations

import pytest
from svc_control_plane import egress_requests as eg

from rag_common import get_settings


@pytest.fixture(autouse=True)
def _restore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(eg._EGRESS_ENV, "localhost,qdrant,models-uat.example.internal")
    get_settings.cache_clear()
    eg.reset_base_hosts()


def test_a_runtime_approved_host_can_be_revoked() -> None:
    eg._apply_host_to_allowlist("extra.example.com")
    assert "extra.example.com" in get_settings().allowed_egress_hosts
    eg._remove_host_from_allowlist("extra.example.com")
    assert "extra.example.com" not in get_settings().allowed_egress_hosts


def test_a_deployment_configured_host_survives_a_revoke() -> None:
    """The model endpoint the platform was configured with stays reachable."""
    eg._remove_host_from_allowlist("models-uat.example.internal")
    assert "models-uat.example.internal" in get_settings().allowed_egress_hosts


def test_revoking_a_base_host_does_not_disturb_the_others() -> None:
    eg._apply_host_to_allowlist("extra.example.com")
    eg._remove_host_from_allowlist("qdrant")
    hosts = get_settings().allowed_egress_hosts
    for h in ("localhost", "qdrant", "models-uat.example.internal", "extra.example.com"):
        assert h in hosts

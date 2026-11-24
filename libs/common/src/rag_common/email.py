"""Sovereign transactional email (member activation / password reset).

Sends via an operator-configured SMTP relay whose host MUST be on the sovereignty egress
allowlist (``assert_sovereign``) — no external email provider is ever used. When SMTP is not
configured, :func:`send_email` returns ``False`` and the caller falls back to an admin-copied
link, so a fully air-gapped deployment works with no mail server at all.
"""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage

from rag_common.config import get_settings
from rag_common.logging import get_logger
from rag_common.sovereignty import assert_sovereign

_log = get_logger(__name__)


def email_configured() -> bool:
    """True if an SMTP relay is configured (so the platform can send mail itself)."""
    return bool(get_settings().smtp_host)


def send_email(to: str, subject: str, body: str) -> bool:
    """Send a plain-text email via the configured (allowlisted) SMTP relay. Returns True on
    send, False when SMTP isn't configured OR the send fails — the caller then surfaces a
    copyable link instead, so the flow never hard-depends on mail."""
    s = get_settings()
    if not s.smtp_host:
        return False
    assert_sovereign(f"smtp://{s.smtp_host}:{s.smtp_port}")  # fail closed if host not allowlisted
    msg = EmailMessage()
    msg["From"] = s.smtp_from
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=15) as smtp:
            if s.smtp_starttls:
                smtp.starttls(context=ssl.create_default_context())
            if s.smtp_user:
                smtp.login(s.smtp_user, s.smtp_password)
            smtp.send_message(msg)
    except Exception as exc:  # never let a mail failure break the request; caller falls back
        _log.warning("email_send_failed", to=to, error=str(exc)[:160])
        return False
    _log.info("email_sent", to=to, subject=subject)
    return True

"""Resolve an external source's credentials BY REFERENCE, per provider.

Secrets never travel in a request or the catalog: the source carries a NAME, and the platform
``SecretsProvider`` resolves it. S3 wants an access key + secret key. Azure Blob wants the account
name (taken from the endpoint host, never typed) and either an account key or a SAS token. Both
are returned in the same (access, secret) shape so every fetcher shares one call signature.
"""

from __future__ import annotations

from typing import Any


def _get(secrets: Any, name: str) -> str | None:
    try:
        value = secrets.get(name)
    except (KeyError, LookupError):
        return None
    return value or None


def _azure_account(endpoint: str) -> str:
    """The storage account an endpoint addresses.

    Azure has two addressing styles and they put the account in different places. A real account
    is a SUBDOMAIN (``acct.blob.core.windows.net``); the emulator and path-style endpoints put it
    in the PATH (``127.0.0.1:10000/devstoreaccount1``). Reading the first host label in the second
    case yields "127", and every call then fails as an opaque credential error.
    """
    host_and_path = endpoint.split("://", 1)[-1].strip("/")
    if "/" in host_and_path:
        return host_and_path.rsplit("/", 1)[-1]
    return host_and_path.split(":", 1)[0].split(".", 1)[0]


def external_credentials(source: Any, secrets: Any) -> tuple[str, str]:
    """``(access_key, secret_key)`` for an ``ExternalObjectSource``, resolved via ``secrets``.

    Azure: ``access_key`` is the storage ACCOUNT NAME (see :func:`_azure_account`) and
    ``secret_key`` is ``{ref}_account_key`` if present, else ``{ref}_sas_token``. A missing Azure
    credential is a hard error naming the reference, so a misconfigured connector fails at the
    sync call rather than as an opaque 403 from Azure.
    """
    ref = source.credentials_ref
    if getattr(source, "provider", "s3") == "azure":
        account = _azure_account(str(source.endpoint))
        secret = _get(secrets, f"{ref}_account_key") or _get(secrets, f"{ref}_sas_token")
        if not secret:
            raise KeyError(f"no Azure credential for reference {ref!r}: set {ref}_account_key or {ref}_sas_token")
        return account, secret
    return secrets.get(f"{ref}_access_key"), secrets.get(f"{ref}_secret_key")

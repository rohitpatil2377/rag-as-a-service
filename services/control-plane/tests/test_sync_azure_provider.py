"""Folder sync must honour the source's PROVIDER, exactly as single-object fetch does.

Reported live from the Connectors page: selecting "Azure Blob Storage" and syncing a folder
failed with `missing secret: RAG_AZURITE_ACCESS_KEY` -- an S3 credential name. The single-object
path had been taught about providers; this one had not, so it resolved S3 credentials and called
the S3 fetcher regardless of what the user picked. Azure has no access/secret pair, so the sync
could never have worked whatever the user put in the form.
"""

from __future__ import annotations

import pytest
from svc_control_plane import sync_jobs
from svc_control_plane.sync_jobs import SyncSource

from rag_schemas.envelope import ExternalObjectSource


class _Secrets:
    """Only Azure-shaped names exist here, so an S3 lookup raises exactly as it did live."""

    def __init__(self, store: dict[str, str]) -> None:
        self._s = store

    def get(self, name: str) -> str:
        if name not in self._s:
            raise KeyError(f"missing secret: {name}")
        return self._s[name]


class _Fetcher:
    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

    async def list_keys(self, **kw: object) -> list[str]:
        self.seen = kw
        return []

    async def fetch(self, **kw: object) -> bytes:
        return b""


def _source(provider: str) -> SyncSource:
    return SyncSource(
        external_source=ExternalObjectSource(
            endpoint="acct.blob.core.windows.net",
            bucket="sops",
            secure=True,
            provider=provider,
            credentials_ref="azurite",
        ),
        prefix="pharma/",
    )


@pytest.mark.asyncio
async def test_an_azure_sync_resolves_azure_credentials_and_the_azure_fetcher(
    inmemory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetcher = _Fetcher()
    chosen: list[str] = []

    def _for(provider: str) -> _Fetcher:
        chosen.append(provider)
        return fetcher

    monkeypatch.setattr(sync_jobs, "external_fetcher_for", _for)
    monkeypatch.setattr(sync_jobs, "get_secrets_provider", lambda: _Secrets({"azurite_account_key": "K"}))
    monkeypatch.setattr(sync_jobs, "assert_sovereign", lambda url: None)

    await sync_jobs.run_sync("kb", _source("azure"), actor="tester")

    assert chosen == ["azure"], "the S3 fetcher was used for an Azure source"
    # Azure's account NAME comes from the endpoint, and the key from {ref}_account_key.
    assert fetcher.seen["access_key"] == "acct"
    assert fetcher.seen["secret_key"] == "K"


@pytest.mark.asyncio
async def test_an_s3_sync_is_unchanged(inmemory: None, monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = _Fetcher()
    monkeypatch.setattr(sync_jobs, "external_fetcher_for", lambda provider: fetcher)
    monkeypatch.setattr(
        sync_jobs,
        "get_secrets_provider",
        lambda: _Secrets({"acme_access_key": "AK", "acme_secret_key": "SK"}),
    )
    monkeypatch.setattr(sync_jobs, "assert_sovereign", lambda url: None)

    src = SyncSource(
        external_source=ExternalObjectSource(
            endpoint="obj.local", bucket="b", secure=True, provider="s3", credentials_ref="acme"
        ),
        prefix="p/",
    )
    await sync_jobs.run_sync("kb", src, actor="tester")
    assert fetcher.seen["access_key"] == "AK"
    assert fetcher.seen["secret_key"] == "SK"

"""AzureBlobExternalFetcher against a fake SDK client — no Azure account needed."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import pytest

from rag_clients.azure_fetcher import AzureBlobExternalFetcher
from rag_clients.external_creds import external_credentials
from rag_common import get_settings


@pytest.fixture(autouse=True)
def _allow_azure_host(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Allowlist the fake account host for the sovereignty guard the way the other client tests
    do: per test, via monkeypatch, settings cache cleared on both sides. An import-time
    os.environ default was order-dependent -- it passed alone and failed after any earlier test
    had cached settings without the host."""
    monkeypatch.setenv("RAG_ALLOWED_EGRESS_HOSTS", "localhost,acct.blob.core.windows.net")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _NotFoundError(Exception):
    pass


_NotFoundError.__name__ = "ResourceNotFoundError"


class _Blob:
    def __init__(self, data: bytes | None) -> None:
        self._data = data

    def download_blob(self) -> _Blob:
        if self._data is None:
            raise _NotFoundError("no such blob")
        return self

    def readall(self) -> bytes:
        return self._data or b""


class _Container:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def list_blobs(self, name_starts_with: str | None = None) -> Iterator[dict[str, str]]:
        for n in self._names:
            if not name_starts_with or n.startswith(name_starts_with):
                yield {"name": n}


class _Client:
    calls: ClassVar[list[tuple[str, object]]] = []

    def __init__(self, account_url: str, credential: object) -> None:
        _Client.calls.append((account_url, credential))
        self.blobs = {"sops/a.pdf": b"AAA", "sops/b.docx": b"BB"}

    def get_blob_client(self, container: str, blob: str) -> _Blob:
        return _Blob(self.blobs.get(blob))

    def get_container_client(self, name: str) -> _Container:
        return _Container([*self.blobs.keys(), "sops/folder/"])


@pytest.mark.asyncio
async def test_fetch_and_list_through_the_seam() -> None:
    f = AzureBlobExternalFetcher(client_factory=_Client)
    data = await f.fetch(
        endpoint="acct.blob.core.windows.net",
        bucket="c",
        key="sops/a.pdf",
        access_key="acct",
        secret_key="k3y",
        secure=True,
    )
    assert data == b"AAA"
    keys = await f.list_keys(
        endpoint="acct.blob.core.windows.net",
        bucket="c",
        prefix="sops/",
        access_key="acct",
        secret_key="k3y",
        secure=True,
        limit=10,
    )
    assert keys == ["sops/a.pdf", "sops/b.docx"]  # the pseudo-folder entry is dropped
    assert _Client.calls[0][0] == "https://acct.blob.core.windows.net"
    assert _Client.calls[0][1] == {"account_name": "acct", "account_key": "k3y"}


@pytest.mark.asyncio
async def test_missing_blob_is_keyerror_like_every_other_store() -> None:
    f = AzureBlobExternalFetcher(client_factory=_Client)
    with pytest.raises(KeyError):
        await f.fetch(
            endpoint="acct.blob.core.windows.net",
            bucket="c",
            key="nope.pdf",
            access_key="acct",
            secret_key="k3y",
            secure=True,
        )


def test_sas_token_is_passed_verbatim() -> None:
    f = AzureBlobExternalFetcher(client_factory=_Client)
    assert f._credential("acct", "?sv=2024&sig=abc") == "?sv=2024&sig=abc"


class _Src:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.endpoint = "acct.blob.core.windows.net"
        self.credentials_ref = "clientazure"


class _Secrets:
    def __init__(self, store: dict[str, str]) -> None:
        self._s = store

    def get(self, name: str) -> str:
        return self._s[name]


def test_azure_credentials_resolve_account_from_endpoint_and_key_by_reference() -> None:
    acc, sec = external_credentials(_Src("azure"), _Secrets({"clientazure_account_key": "K"}))
    assert (acc, sec) == ("acct", "K")
    acc, sec = external_credentials(_Src("azure"), _Secrets({"clientazure_sas_token": "?sig=x"}))
    assert (acc, sec) == ("acct", "?sig=x")
    with pytest.raises(KeyError):
        external_credentials(_Src("azure"), _Secrets({}))


class _Ep:
    def __init__(self, endpoint: str) -> None:
        self.provider = "azure"
        self.endpoint = endpoint
        self.credentials_ref = "azurite"


def test_the_account_name_comes_from_a_real_azure_host() -> None:
    acc, _ = external_credentials(_Ep("acct.blob.core.windows.net"), _Secrets({"azurite_account_key": "K"}))
    assert acc == "acct"


def test_the_account_name_comes_from_the_path_on_an_emulator_endpoint() -> None:
    """Azurite addresses an account by PATH (host:port/devstoreaccount1), not by subdomain.
    Taking the first host label there yields "127" and every call 403s -- reported live from the
    Connectors page against 127.0.0.1:10000/devstoreaccount1."""
    secrets = _Secrets({"azurite_account_key": "K"})
    for endpoint in (
        "127.0.0.1:10000/devstoreaccount1",
        "http://127.0.0.1:10000/devstoreaccount1",
        "azurite:10000/devstoreaccount1/",
    ):
        acc, _ = external_credentials(_Ep(endpoint), secrets)
        assert acc == "devstoreaccount1", endpoint


def test_a_bare_host_with_no_path_still_uses_its_first_label() -> None:
    acc, _ = external_credentials(_Ep("mystore.blob.core.windows.net:443"), _Secrets({"azurite_account_key": "K"}))
    assert acc == "mystore"

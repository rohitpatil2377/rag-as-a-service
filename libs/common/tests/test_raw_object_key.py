"""raw_object_key: URL doc ids become object-store-safe keys; simple ids pass through."""

from __future__ import annotations

from rag_common.tenancy import raw_object_key


def test_simple_ids_pass_through() -> None:
    assert raw_object_key("brief") == "raw/brief"
    assert raw_object_key("Acme_Night_Shift.pdf") == "raw/Acme_Night_Shift.pdf"
    assert raw_object_key("reports/q1.pdf") == "raw/reports/q1.pdf"  # single-slash path is fine


def test_url_ids_are_encoded_safely() -> None:
    key = raw_object_key("http://localhost:8899/latest/")
    assert ":" not in key and "//" not in key  # MinIO/S3-illegal chars gone
    assert key.startswith("raw/")


def test_encoding_is_deterministic_and_unique() -> None:
    a = raw_object_key("https://docs.example.com/a")
    b = raw_object_key("https://docs.example.com/b")
    assert a == raw_object_key("https://docs.example.com/a")  # deterministic
    assert a != b  # different URLs → different keys (hash disambiguates)

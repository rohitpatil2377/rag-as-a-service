"""Unit tests for FR-A9 web-crawl pure logic."""

from __future__ import annotations

from rag_core.crawl import canonical, extract_links, in_scope

_HTML = """<html><body>
<a href="/docs/a">A</a>
<a href="/docs/b#frag">B</a>
<a href="https://other.example/x">offsite</a>
<a href="mailto:x@y.z">mail</a>
<a href="/logo.png">img</a>
<a href="https://site.local/docs/a">dup</a>
</body></html>"""


def test_canonical_strips_fragment_and_trailing_slash() -> None:
    assert canonical("https://s.local/a#top") == "https://s.local/a"
    assert canonical("https://s.local/a/") == "https://s.local/a"
    assert canonical("https://s.local/") == "https://s.local/"  # root slash kept


def test_extract_links_absolute_dedup_and_filters() -> None:
    links = extract_links(_HTML, "https://site.local/docs/index")
    assert "https://site.local/docs/a" in links
    assert "https://site.local/docs/b" in links  # fragment stripped
    assert "https://other.example/x" in links
    assert not any(link.endswith(".png") for link in links)  # assets skipped
    assert not any("mailto" in link for link in links)
    assert links.count("https://site.local/docs/a") == 1  # deduped


def test_in_scope_host_and_prefix() -> None:
    seed = "https://site.local/docs/index"
    assert in_scope("https://site.local/docs/a", seed, same_host_only=True)
    assert not in_scope("https://other.example/x", seed, same_host_only=True)
    assert in_scope("https://other.example/x", seed, same_host_only=False)
    assert not in_scope("https://site.local/blog/z", seed, same_host_only=True, path_prefix="/docs")
    assert in_scope("https://site.local/docs/a", seed, same_host_only=True, path_prefix="/docs")

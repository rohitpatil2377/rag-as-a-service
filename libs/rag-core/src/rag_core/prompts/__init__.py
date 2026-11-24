"""Prompt templates as FILES, loaded by name (RAG v2; discovery §8.5 scopes this as its own work).

Every prompt in this codebase has been an inline Python constant — the answer prompt in
``rag_core.grounding``, the system line in ``rag_clients.llm_gateway``. The upgrade plan
writes its new prompts as ``prompts/*.md`` and names ``prompts/doc_profile.md`` in its Phase 3
text, and the reason is not cosmetic: a document profile prompt is the one artifact a domain
reviewer can read and correct without reading Python, and the facet taxonomy it elicits is what
Phases 4 and 8 group on. So it lives in a file, reviewable on its own.

THE DIRECTORY IS INSIDE THE PACKAGE, not at the repo root that the plan's path implies. The API
ships as a wheel built from ``src/rag_core`` (``libs/rag-core/pyproject.toml``), so a
repo-root ``prompts/`` would not be in the image at all and the first profile call in a deployment
would fail on a missing file — a packaging bug that no test run from a source checkout could see.
Inside the package the template is built, shipped and versioned with the code that reads it.

SUBSTITUTION IS ``string.Template`` (``$name``), NOT ``str.format``. These templates contain a
literal JSON schema, so they are full of ``{`` and ``}`` and ``format`` would raise on every one
of them. Only the TEMPLATE is scanned for placeholders, never the substituted values, so a ``$``
inside a document's own text cannot be read as a placeholder.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from string import Template

__all__ = ["load_prompt", "render_prompt"]


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    """The raw template text for ``name`` (no extension), e.g. ``"doc_profile"``.

    Cached: a prompt is read once per process rather than once per document. Raises
    ``FileNotFoundError`` for an unknown name — a missing prompt is a build/packaging fault, and a
    caller that silently fell back to an empty string would send the model an empty instruction and
    then blame the model for the reply.
    """
    return (resources.files(__package__) / f"{name}.md").read_text(encoding="utf-8")


def render_prompt(name: str, /, **values: str) -> str:
    """``load_prompt(name)`` with ``$placeholders`` substituted.

    ``Template.substitute`` (not ``safe_substitute``) so a placeholder the caller forgot raises
    here, where the template and the call site are both in view, rather than reaching the model as
    the literal text ``$breadcrumbs``.
    """
    return Template(load_prompt(name)).substitute(**values)

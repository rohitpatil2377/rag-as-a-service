# Indic cross-lingual query retrieval

## The gap

A query is retrieved as-is. If the corpus is indexed in English but a user asks in Hindi or
Marathi, the query embedding/lexical terms only match Hindi/Marathi text — the English documents
that actually answer the question are missed. The reverse fails too: an English query misses
Indic-language documents. For a platform serving India's **22 scheduled languages**, retrieving
only in the language the question happened to be typed in silently under-recalls the corpus.

## The flow: retrieve on original + translation, union the evidence

`cross_lingual.py` expands the query rather than re-indexing the corpus:

1. **Detect script** — `detect_script(text)` classifies the query by the majority of its
   alphabetic characters using Unicode ranges: Devanagari (Hindi/Marathi, U+0900–097F),
   Bengali (U+0980–09FF), Tamil (U+0B80–0BFF), Telugu (U+0C00–0C7F), Latin (ASCII), else
   `other`. Pure arithmetic — no model, no language-ID dependency.
2. **Decide** — `needs_translation(text, target_script=...)` is true when the query's dominant
   script differs from the corpus script. Same script → nothing to do.
3. **Expand** — `expand_query_cross_lingual(...)` returns the list of query variants to retrieve
   on. When cross-lingual, it is `[original, translation]`; otherwise `[original]`. The parent
   retrieves on every variant and **unions the hit-sets**, so the answer surfaces whether the
   phrasing matched the Indic side or the English side.

We keep the original as well as the translation on purpose: proper nouns, codes, and numerals
often retrieve better untranslated, while the translation recovers the cross-lingual matches.
The union is strictly higher-recall than either query alone.

## Injected-gateway design

Translation is an **injected async callback**, not an import:

```python
TranslateFn = Callable[[str, str], Awaitable[str]]  # (text, target_lang) -> translated
```

The parent supplies `translate_fn` — the platform's LLM gateway — when it wires this module into
the retrieval path behind a flag. The core never imports a model, DB, or network client, so:

- **Purity / testability** — script detection and expansion are unit-tested with a fake
  `translate_fn`; no model is needed in CI.
- **Sovereignty** — translation runs through the tenant's own governed gateway (the same seam as
  `translation_backend` in config), not a hard-coded provider.
- **Resilience** — retrieval must never fail because translation did. Any `translate_fn`
  exception is logged and swallowed, falling back to the original query alone. A blank or
  identical translation is deduped away, so the variant list is always non-empty and clean.

## Wiring (done by the parent, not here)

This module creates only the pure core. The parent enables it behind a flag, passes the corpus
`target_lang` / `target_script` and a `translate_fn` bound to the gateway, and retrieves on the
returned variants before merging results.

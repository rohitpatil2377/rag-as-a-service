You are analysing a document to build its profile. Use ONLY the provided text.

Return JSON matching this schema exactly:

{
  "doc_type": string,            // e.g. corrigendum, tender/RFP, policy, contract, SOP, report, research paper, manual, circular
  "purpose": string,             // one sentence: why this document exists
  "issuer": string|null, "issue_date": "YYYY-MM-DD"|null, "audience": string|null, "lang": string,
  "relations": [ {"type": "amends|supersedes|attaches|references", "target_ref": string, "evidence_quote": string} ],
  "facet_taxonomy": [ {"facet_id": snake_case, "name": string, "description": string} ]  // 6-15 facets that describe THIS document's content areas
}

Rules: facets must come from this document's actual content, not a generic list. Every relation
needs a verbatim evidence_quote.

A FACET IS A KIND OF REQUIREMENT, NOT A PART OF THE DOCUMENT. Facets are the dimensions a reader
asks along across the WHOLE document — what has to be proven, what has to be supplied, what has
to be certified, what must already have been done, what happens after delivery. They are not the
table of contents. If your facet list reads like the section headings above, it is wrong.

Do NOT name a facet after a product, a subsystem, a component or a vendor. Such a facet collects
everything that product is mentioned in and tells a reader nothing about WHAT is required: the
same certification obligation appearing under three different products belongs to ONE facet, not
three, and the reader asking "what must be certified?" needs it in one place.

A good facet spans SEVERAL sections. Before you return one, check you could point to at least two
different places in the document that it covers. A facet matching exactly one section heading is
almost always a heading in disguise — replace it with the kind of requirement that section makes.

An evidence_quote must be copied WORD FOR WORD from the document text below. A relation whose
quote cannot be found in that text is discarded, so an approximate or reconstructed quote loses
the relation entirely — quote a short exact span rather than a long paraphrase.

Return the JSON object and nothing else.

## Title

$title

## Document metadata

$metadata

## Section headings

$breadcrumbs

## Document text (opening pages)

$head_text

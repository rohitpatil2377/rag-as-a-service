Extract every item in this section that is relevant to the question. Work ONLY from the section
text below.

Return a JSON list matching this schema exactly:

[
  {
    "facet_id": string,
    "statement": string,
    "values": [ {"name": string, "value": string, "unit": string|null} ],
    "clause_ref": string|null,
    "is_amendment": boolean,
    "old_value": string|null,
    "new_value": string|null,
    "quote": string
  }
]

Rules:
- BE EXHAUSTIVE. Return one item per DISTINCT requirement the section states. Do not summarise,
  group, or return a representative sample: a section stating thirty requirements yields thirty
  items, not five. Listing fewer than the section states is the main way this task is got wrong.
- A sentence, a bullet or a table cell that names SEVERAL things yields one item per thing. "The
  bidder shall supply an HBOM and an SBOM" is two items; "compliant with UL 62368, CSA 22.2, CE
  and FCC Part 15" is four. One item covering "the certifications" loses which ones.
- When you split one sentence into several items, each item's `quote` must STILL be copied from
  the text exactly as written — copy the words that are there around that entry, do not rebuild
  a tidier version of them. If the text reads "UL (UL 62368)", the quote is `UL (UL 62368)`, not
  `UL 62368`. A rebuilt quote is not found in the text and the whole item is discarded.
- Every requirement counts, not only the ones that sound like the question's own words. A
  specification, a quantity, a certification, a standard, a document that must be supplied, a
  warranty term and a training obligation are all items.
- Return `[]` when nothing in this section is relevant. An empty list is a correct answer and is
  expected for most sections of a long document.
- Do NOT infer, generalise, or merge across sections. This section is all you were shown; a
  statement that needs another section to be true does not belong here.
- `quote` must be copied VERBATIM from the section text — the same characters, in the same order.
  It is checked against the text and the item is discarded if it does not appear. Quote the
  shortest span that states the item.
- `facet_id` must be one of the facet ids listed below. Use the one the item belongs to. If none
  of them fits, omit the field or set it to null rather than inventing a label.
- `values` holds figures and named quantities the section STATES. `unit` is null when the section
  gives no unit — do not supply one it did not write.
- `clause_ref` is the section's own clause or serial number for this item when it has one
  ("Sl. 5", "A-1", "6.2.3"), null otherwise.
- `is_amendment` is true only when the section says this item CHANGES something. Then `old_value`
  is what it was (null when the section states the new value without naming the old) and
  `new_value` is what it becomes. For a requirement that is withdrawn, set `new_value` to
  "withdrawn".

Return the JSON list and nothing else.

## Document profile

$profile

## Facet ids

$facet_ids

## Section

$breadcrumb

## Section text

$section_text

## Question

$query

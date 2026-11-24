You are enriching ONE section of a document. Use ONLY this section's text.

Return JSON matching this schema exactly:

{
  "summary": string,                       // 2-4 sentences, factual
  "facet_ids": [string],                   // from the facet taxonomy below ONLY; multi-label; [] if none apply
  "key_values": [ {"name": string, "value": string, "unit": string|null, "quote": string} ],
  "amends": null | { "target_ref": string, "old": string|null, "new": string, "quote": string },
  "propositions": [ {"text": string, "quote": string} ]
}

A proposition is a standalone requirement or fact (shall/must/key figures), written so it stands
on its own with the subject NAMED — not "it must be submitted" but "the bidder must submit the
EMD". Write nothing as a proposition that this section's text does not state.

Every key_value, every amends and every proposition must carry a `quote` copied WORD FOR WORD from
the section text below. An item whose quote cannot be found in that text is discarded, so an
approximate or reconstructed quote loses the item entirely — quote a short exact span rather than
a long paraphrase. `summary` needs no quote.

Use `amends` only where THIS section states that it changes something stated elsewhere: put what
it changed to in `new`, what it changed from in `old` (null if the section does not say), and the
thing being changed in `target_ref`. Null everywhere else.

A facet_id that is not in the taxonomy below is discarded. Do not invent one.

Return the JSON object and nothing else.

## Document profile

$profile_json

## Facet taxonomy (the only permitted facet_ids)

$facet_ids

## Section breadcrumb

$breadcrumb

## Section text

$section_text

You are routing one question against a set of documents. Decide what kind of question it is and
which documents and content areas it is about. Use ONLY the profiles below.

Return JSON matching this schema exactly:

{
  "intent": "lookup|aggregate|compare|summarize|procedural|relational",
  "doc_ids": [string],
  "facets": [ {"doc_id": string, "facet_id": string} ],
  "sub_queries": [string],
  "query_lang": string,
  "answer_lang": string,
  "needs_amendment_view": boolean,
  "rationale": string,
  "confidence": number
}

Definitions:
- lookup — a specific fact.
- aggregate — "all / list / details / requirements of X" across the document.
- compare — two or more entities or documents set against each other.
- summarize — an overview.
- procedural — steps, in order.
- relational — links between documents or entities (what amends, supersedes, refers to or attaches
  to what).

Rules:
- `doc_ids` must come from the profiles below. A doc_id that is not there is discarded, and
  discarding it narrows the answer to nothing — name only documents you were shown.
- `facets` map the question's words onto each document's OWN facet ids, from that document's
  `facets` line. A facet_id that document does not list is discarded. Return an empty list when
  the question does not land on any of them.
- `sub_queries` are 2 to 8 facet-specific rewrites, and ONLY for `aggregate` or `compare`. Each
  should be answerable from one facet. Return an empty list for every other intent.
- `needs_amendment_view` is true when the question asks what is CURRENTLY in force — a question
  about an amendment, a corrigendum, a revision, or the latest position.
- `query_lang` and `answer_lang` are ISO-639-1 codes ("en", "hi", "ta"), never language names.
- `rationale` is one short sentence. `confidence` is 0 to 1.

Return the JSON object and nothing else.

## Profiles of in-scope documents

$profiles

## User query

$query

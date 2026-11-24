// The product lockup, in ONE place.
//
// This file exists because the brand drifted: the console shipped the product name in six
// separate hard-coded copies while the design deck and the intro animation disagreed with
// all of them, and nothing could catch it because no two copies were the same string. Import
// from here rather than re-typing the words — a rename is then one edit, not a grep.
//
// LEAD renders in the heading colour and MARK in the accent colour, so the two halves are
// styled separately and must stay separate strings; BRAND_NAME is the joined form for the
// places that want one run of text (document titles, the password screen eyebrow).
export const BRAND_LEAD = 'RAG'
export const BRAND_MARK = 'as a Service'
export const BRAND_NAME = `${BRAND_LEAD} ${BRAND_MARK}`
export const TAGLINE = 'Grounded answers, with citations'

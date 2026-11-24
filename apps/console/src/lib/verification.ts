// What the console should DISPLAY for one answer's verification — as data, not as JSX.
//
// WHY THIS IS A PURE MODULE. Every decision worth getting wrong lives here: whether a score may
// be shown, which methods are named and in what order, whether a degradation warning belongs, and
// whether findings exist independently of all of that. Rendering a `<span>` is not where the risk
// is. Keeping the decisions pure means the five cases can be tested without a DOM, and means the
// components cannot quietly introduce a sixth rule of their own.
//
// THE RULE: display every distinction the API makes, and invent none it does not.

import type { GroundedAnswer, Verification, VerificationMethod } from './api'

// `lexical` and `lexical_fallback` ARE DIFFERENT and the difference must survive to the label.
// A deployment whose configured checker IS the lexical one is running what it was asked to;
// warning about it would make a deliberate choice look like a permanent incident.
// `lexical_fallback` ran because NLI failed, and a reader has to know that happened.
const METHOD: Record<VerificationMethod, { label: string; concerning: boolean }> = {
  nli: { label: 'NLI', concerning: false },
  lexical: { label: 'Lexical', concerning: false },
  lexical_fallback: { label: 'Lexical fallback', concerning: true },
  unavailable: { label: 'Not checked', concerning: true },
}

export const REASON_TEXT: Record<string, string> = {
  nli_failed_fallback_used: 'the NLI checker was unreachable, so a weaker lexical check ran instead',
  nli_failed_fallback_failed: 'the NLI checker and its fallback both failed, so nothing verified this',
  nli_failed_fallback_unconfigured: 'the NLI checker was unreachable and no fallback is configured',
}

export interface MethodDisplay {
  label: string
  count: number
  concerning: boolean
}

export interface VerificationDisplay {
  /** The faithfulness score, or null when no single score is meaningful. */
  score: number | null
  /** What to say when `score` is null. Empty when a score is shown. */
  noScoreLabel: string
  /** Every method with its count, in API order. Never collapsed to a majority. */
  methods: MethodDisplay[]
  /** Set when the intended checker was not the one that ran. Independent of `score`. */
  degradedReason: string | null
  /** Claims a check flagged, whatever the answer's status. Independent of everything above. */
  findings: string[]
  /** False when the API reported no verification at all — different from `unavailable`. */
  reported: boolean
}

/**
 * The display decision for one answer.
 *
 * `score` IS DRIVEN BY `outcome` AND NOTHING ELSE. Not by `degraded`, not by the presence of a
 * method, not by comparing counts. An all-fallback run is `scored` -- one method throughout, so
 * its scores are comparable -- AND degraded, and both of those facts come out of here separately.
 */
export function describeVerification(answer: GroundedAnswer): VerificationDisplay {
  const findings = answer.unsupported_claims ?? []
  const v: Verification | null | undefined = answer.verification
  if (!v) {
    // No provenance reported. Show the bare score if one exists, so this can never display less
    // than the console did before provenance existed; claim nothing otherwise.
    return {
      score: answer.faithfulness_score ?? null,
      noScoreLabel: '',
      methods: [],
      degradedReason: null,
      findings,
      reported: false,
    }
  }
  const methods = v.methods.map((m) => ({
    label: METHOD[m.method].label,
    count: m.count,
    concerning: METHOD[m.method].concerning,
  }))
  const codes = [...new Set((v.findings ?? []).map((f) => f.reason_code).filter((c): c is string => !!c))]
  const degradedReason = v.degraded ? codes.map((c) => REASON_TEXT[c] ?? c).join('; ') || 'verification did not use its intended method' : null
  // EXHAUSTIVE: a new outcome in the API type breaks this at compile time rather than falling
  // through to a reassuring default.
  switch (v.outcome) {
    case 'scored':
      return { score: v.score, noScoreLabel: '', methods, degradedReason, findings, reported: true }
    case 'mixed':
      return { score: null, noScoreLabel: 'mixed verification', methods, degradedReason, findings, reported: true }
    case 'unavailable':
      return { score: null, noScoreLabel: 'not scored', methods, degradedReason, findings, reported: true }
  }
}

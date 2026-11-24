// The five verification cases, plus the combinations that matter.
//
// These test the DECISIONS, not the markup: whether a score may be shown, which methods are
// named and in what order, whether a degradation warning belongs, and whether findings exist
// independently of all three. Those are the only places a bug can hide — a `<span>` rendering a
// string it was handed cannot invent a distinction.

import { describe, expect, it } from 'vitest'

import type { GroundedAnswer, MethodCount, Verification, VerificationMethod } from './api'
import { describeVerification } from './verification'

function answer(verification: Verification | null, extra: Partial<GroundedAnswer> = {}): GroundedAnswer {
  return {
    tenant_id: 't',
    query: 'q',
    status: 'grounded',
    answer: 'a',
    citations: [],
    confidence: 0.9,
    abstain_reason: null,
    faithfulness_score: null,
    unsupported_claims: [],
    review_id: null,
    trace_id: null,
    model: null,
    verification,
    ...extra,
  }
}

const counts = (...pairs: [VerificationMethod, number][]): MethodCount[] =>
  pairs.map(([method, count]) => ({ method, count }))

const common = { degraded: false, methods: counts(['nli', 1]), batches: [], findings: undefined }

describe('ordinary NLI verification', () => {
  it('shows the score and names no concern', () => {
    const d = describeVerification(
      answer({ ...common, outcome: 'scored', score: 0.95, methods: counts(['nli', 12]) }),
    )
    expect(d.score).toBe(0.95)
    expect(d.noScoreLabel).toBe('')
    expect(d.methods).toEqual([{ label: 'NLI', count: 12, concerning: false }])
    expect(d.degradedReason).toBeNull()
  })
})

describe('intentional lexical verification', () => {
  it('shows its score and is NOT warned about', () => {
    // Running the checker you configured is not a downgrade. Warning here would turn a
    // deployment decision into a permanent alarm.
    const d = describeVerification(
      answer({ ...common, outcome: 'scored', score: 0.9, methods: counts(['lexical', 3]) }),
    )
    expect(d.score).toBe(0.9)
    expect(d.degradedReason).toBeNull()
    expect(d.methods).toEqual([{ label: 'Lexical', count: 3, concerning: false }])
  })
})

describe('mixed verification', () => {
  const mixed = answer({
    outcome: 'mixed',
    degraded: true,
    methods: counts(['nli', 12], ['lexical_fallback', 1]),
    batches: [],
    findings: [{ method: 'lexical_fallback', reason_code: 'nli_failed_fallback_used', detail: 'ReadError: ' }],
  })

  it('shows NO score, because an NLI score and a lexical score are not the same measurement', () => {
    expect(describeVerification(mixed).score).toBeNull()
    expect(describeVerification(mixed).noScoreLabel).toBe('mixed verification')
  })

  it('names EVERY method in API order, never the majority', () => {
    // The real run that prompted this had 12 NLI and 1 fallback. Rendering "NLI" because it was
    // the majority tells the reader the whole answer was NLI-verified, and the one batch that was
    // not is precisely the one a reviewer needs.
    expect(describeVerification(mixed).methods).toEqual([
      { label: 'NLI', count: 12, concerning: false },
      { label: 'Lexical fallback', count: 1, concerning: true },
    ])
  })

  it('explains the degradation in words, not a code', () => {
    expect(describeVerification(mixed).degradedReason).toContain('NLI checker was unreachable')
  })
})

describe('unavailable verification', () => {
  it('shows no score and says so, rather than leaving the reader to infer it', () => {
    const d = describeVerification(
      answer({
        outcome: 'unavailable',
        degraded: true,
        methods: counts(['nli', 1], ['unavailable', 1]),
        batches: [],
        findings: [{ method: 'unavailable', reason_code: 'nli_failed_fallback_failed', detail: null }],
      }),
    )
    expect(d.score).toBeNull()
    expect(d.noScoreLabel).toBe('not scored')
    expect(d.degradedReason).toContain('both failed')
    expect(d.methods.map((m) => m.label)).toEqual(['NLI', 'Not checked'])
  })
})

describe('a grounded answer carrying findings', () => {
  it('reports the findings WITHOUT turning them into a verification failure', () => {
    // The fifth case, and the one most easily got wrong in the other direction: the numeric guard
    // can flag a claim on an answer that verified cleanly. Showing a degradation warning for it
    // would invent a verification problem that did not happen.
    const d = describeVerification(
      answer({ ...common, outcome: 'scored', score: 0.97 }, { unsupported_claims: ['The total is 42 units.', 'X is Y.'] }),
    )
    expect(d.score).toBe(0.97)
    expect(d.degradedReason).toBeNull()
    expect(d.findings).toHaveLength(2)
  })
})

describe('the dimensions are independent', () => {
  it('an all-fallback run is SCORED and ALSO degraded', () => {
    // Two different facts. Every batch used the same method, so its scores are comparable; and
    // the intended mechanism never ran, which a reader has to know. Collapsing them loses one.
    const d = describeVerification(
      answer({
        outcome: 'scored',
        score: 0.8,
        degraded: true,
        methods: counts(['lexical_fallback', 3]),
        batches: [],
        findings: [{ method: 'lexical_fallback', reason_code: 'nli_failed_fallback_used', detail: null }],
      }),
    )
    expect(d.score).toBe(0.8)
    expect(d.degradedReason).not.toBeNull()
  })

  it('no verification block falls back to the bare score and claims nothing else', () => {
    // Different from `unavailable`: there, verification RAN and reached no judgement. Here it
    // never reported at all, which is what an older API or a disabled gate looks like.
    const d = describeVerification(answer(null, { faithfulness_score: 0.88 }))
    expect(d.reported).toBe(false)
    expect(d.score).toBe(0.88)
    expect(d.methods).toEqual([])
    expect(d.degradedReason).toBeNull()
  })

  it('no verification and no score shows nothing rather than implying a check', () => {
    const d = describeVerification(answer(null))
    expect(d.reported).toBe(false)
    expect(d.score).toBeNull()
  })
})

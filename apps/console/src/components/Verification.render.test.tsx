// The five cases through the ACTUAL components, not just the decision module.
//
// `lib/verification.test.ts` establishes what the console should DECIDE. This establishes what it
// RENDERS: the labels a reader sees, which methods appear and in what order, whether a warning
// pill is present, and whether the findings are reachable. A correct decision wired to the wrong
// component is still a console that misleads.
//
// RENDERED WITH `react-dom/server`, which is already a dependency. A full DOM stack (jsdom +
// testing-library) would be a heavier commitment than this needs: these components take a plain
// descriptor and emit static markup, so server rendering exercises every branch that matters.
// Interaction -- expanding the findings list -- is the one thing it cannot reach, and that is
// noted on the test that covers the closed state.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { GroundedAnswer, MethodCount, Verification, VerificationMethod } from '../lib/api'
import { VerificationStrip } from './Verification'

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

const render = (a: GroundedAnswer) => renderToStaticMarkup(<VerificationStrip answer={a} />)

describe('case 1 — ordinary NLI verification', () => {
  it('renders the score meter and the method, with no warning', () => {
    const html = render(answer({ outcome: 'scored', score: 0.95, degraded: false, methods: counts(['nli', 12]), batches: [] }))
    expect(html).toContain('faithfulness')
    expect(html).toContain('0.95')
    expect(html).toContain('NLI')
    expect(html).toContain('12')
    expect(html).not.toContain('Verification degraded')
    expect(html).not.toContain('provenance not reported')
  })
})

describe('case 2 — intentional lexical verification', () => {
  it('renders its score and does NOT warn about the method it was configured to use', () => {
    const html = render(answer({ outcome: 'scored', score: 0.9, degraded: false, methods: counts(['lexical', 3]), batches: [] }))
    expect(html).toContain('0.90')
    expect(html).toContain('Lexical')
    expect(html).not.toContain('Lexical fallback')
    expect(html).not.toContain('Verification degraded')
  })
})

describe('case 3 — mixed verification', () => {
  const html = render(
    answer({
      outcome: 'mixed',
      degraded: true,
      methods: counts(['nli', 12], ['lexical_fallback', 1]),
      batches: [],
      findings: [{ method: 'lexical_fallback', reason_code: 'nli_failed_fallback_used', detail: 'ReadError: ' }],
    }),
  )

  it('renders no score, and says why instead of leaving a gap', () => {
    expect(html).toContain('mixed verification')
    expect(html).not.toContain('faithfulness')
  })

  it('renders BOTH methods, NLI first, exactly as the API ordered them', () => {
    expect(html).toContain('NLI')
    expect(html).toContain('Lexical fallback')
    expect(html.indexOf('NLI')).toBeLessThan(html.indexOf('Lexical fallback'))
  })

  it('renders the degradation warning', () => {
    expect(html).toContain('Verification degraded')
  })
})

describe('case 4 — unavailable verification', () => {
  it('renders "not scored" and the degradation, never a number', () => {
    const html = render(
      answer({
        outcome: 'unavailable',
        degraded: true,
        methods: counts(['nli', 1], ['unavailable', 1]),
        batches: [],
        findings: [{ method: 'unavailable', reason_code: 'nli_failed_fallback_failed', detail: null }],
      }),
    )
    expect(html).toContain('not scored')
    expect(html).toContain('Not checked')
    expect(html).toContain('Verification degraded')
    expect(html).not.toMatch(/faithfulness/)
  })
})

describe('case 5 — a grounded answer carrying findings', () => {
  it('renders the finding count WITHOUT a degradation warning', () => {
    // The symmetric error to case 3: inventing a verification problem out of a flagged claim.
    // The answer verified cleanly; a check flagged a sentence. Both facts, neither inflated.
    const html = render(
      answer(
        { outcome: 'scored', score: 0.97, degraded: false, methods: counts(['nli', 1]), batches: [] },
        { unsupported_claims: ['The total is 42 units.', 'X is Y.'] },
      ),
    )
    expect(html).toContain('0.97')
    expect(html).toContain('2 claims need attention')
    expect(html).not.toContain('Verification degraded')
    // Static rendering shows the CLOSED state, which is the one that must not leak the claims
    // before a reader asks. The expansion is a `useState` toggle and is out of reach here.
    expect(html).toContain('aria-expanded="false"')
    expect(html).not.toContain('The total is 42 units.')
  })

  it('says "claim needs" for one and "claims need" for several', () => {
    const one = render(
      answer({ outcome: 'scored', score: 0.9, degraded: false, methods: counts(['nli', 1]), batches: [] }, { unsupported_claims: ['Only this.'] }),
    )
    expect(one).toContain('1 claim needs attention')
  })
})

describe('the legacy path is visibly conservative', () => {
  it('shows the bare score, marks the provenance as absent, and infers no method', () => {
    const html = render(answer(null, { faithfulness_score: 0.88 }))
    expect(html).toContain('0.88')
    expect(html).toContain('provenance not reported')
    expect(html).not.toContain('NLI')
    expect(html).not.toContain('Lexical')
    expect(html).not.toContain('Verification degraded')
  })

  it('renders nothing at all when there is neither provenance nor a score', () => {
    // Not an empty badge, not a zero: no claim about verification was made, so none is shown.
    const html = render(answer(null))
    expect(html).not.toContain('faithfulness')
    expect(html).not.toContain('not scored')
    expect(html).not.toContain('provenance not reported')
  })
})

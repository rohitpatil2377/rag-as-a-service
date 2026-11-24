// Verification provenance, rendered as the independent dimensions it is.
//
// FOUR DIMENSIONS, NOT FIVE WIDGETS. A reader needs four separate facts about an answer, and
// every one of them can vary independently of the others:
//
//   grounding      are the claims tied to cited evidence?        -> AnswerPill, untouched
//   outcome        is a single faithfulness score meaningful?    -> VerificationOutcome
//   method         which checker produced it, and how many each? -> MethodBreakdown
//   degradation    was the INTENDED checker available?           -> DegradedNote
//   findings       what did a check flag, whatever the status?   -> FindingsDisclosure
//
// They compose. A grounded answer can carry findings; an all-fallback run can show a real score
// AND a degradation warning; an intentionally-lexical deployment shows neither warning nor
// apology. Building one component per combination would have produced five that disagree.
//
// THE RULE THIS FILE EXISTS TO HOLD: display every distinction the API makes, and invent none it
// does not. `outcome` is the only thing that decides whether a score is shown — never `degraded`,
// never the presence of a method, never a count comparison.

import type { ReactNode } from 'react'
import { useState } from 'react'

import type { GroundedAnswer } from '../lib/api'
import type { VerificationDisplay } from '../lib/verification'
import { describeVerification } from '../lib/verification'
import { Icon } from './icons'
import { FaithfulnessMeter, toneStyle } from './ui'


/** The score, or the reason there isn't one. Driven by `outcome` and nothing else. */
export function VerificationScore({ display }: { display: VerificationDisplay }) {
  if (display.score != null) {
    // The existing meter, unchanged: one method produced every score, so the single number it
    // draws is a real one. This path must never show less than the console did before.
    //
    // THE LEGACY PATH IS MARKED, not silently equated with a verified one. When the API reported
    // no `verification` block at all -- an older server, or a gate that never ran -- the bare
    // score is still worth showing, but a reader must not read it as "one method verified all of
    // this": nothing said which method, or whether the intended one was reachable. It carries no
    // inferred method and says plainly that the provenance is absent. When the API migration
    // makes the block unconditional, this branch and `reported` can be deleted together.
    return (
      <span className="flex items-center gap-2">
        <FaithfulnessMeter score={display.score} />
        {!display.reported && (
          <span
            className="text-xs"
            style={{ color: 'var(--text-muted)' }}
            title="This answer carries a faithfulness score but no verification provenance: the API did not report which checker produced it, or whether the intended one was available. The score is shown as recorded and nothing about its method is inferred."
          >
            provenance not reported
          </span>
        )}
      </span>
    )
  }
  if (!display.noScoreLabel) return null
  return (
    <span
      className="text-xs"
      style={{ color: 'var(--text-muted)' }}
      title={
        display.noScoreLabel === 'mixed verification'
          ? 'Different batches were verified by different methods. An NLI score and a lexical score are not the same measurement, so no single score is reported — read the per-method breakdown instead.'
          : 'At least one part of this answer was never verified. No score is shown, because a number here would describe claims nothing examined.'
      }
    >
      {display.noScoreLabel}
    </span>
  )
}

/** Every method and its count, in the order the API gave them. */
export function MethodBreakdown({ display }: { display: VerificationDisplay }) {
  if (display.methods.length === 0) return null
  return (
    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
      {display.methods.map((m, i) => (
        <span key={m.label}>
          {i > 0 && <span aria-hidden> · </span>}
          <span
            style={m.concerning ? { color: 'var(--warning)' } : undefined}
            title={m.concerning ? 'this is not the intended verification method' : undefined}
          >
            {m.label}: <span className="mono num">{m.count}</span>
          </span>
        </span>
      ))}
    </span>
  )
}

/** Shown when the intended checker was not the one that ran. Independent of the score. */
export function DegradedNote({ display }: { display: VerificationDisplay }) {
  if (!display.degradedReason) return null
  return (
    <span className="pill" style={toneStyle('--warning')} title={display.degradedReason}>
      <Icon.alert size={13} /> Verification degraded
    </span>
  )
}

/**
 * What a check flagged, whatever the answer's status.
 *
 * A GROUNDED answer can carry these: the numeric guard can report a finding on an answer that
 * routes GROUNDED on its score, and those findings used to reach the audit payload and nowhere a
 * reader could see. A bare count would be its own version of the same problem, so the claims are
 * one click away.
 */
export function FindingsDisclosure({ display }: { display: VerificationDisplay }) {
  const [open, setOpen] = useState(false)
  if (display.findings.length === 0) return null
  const n = display.findings.length
  return (
    <span className="text-xs">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="pill"
        style={{ ...toneStyle('--warning'), cursor: 'pointer' }}
        aria-expanded={open}
        title="A verification check did not find these statements supported by the cited sources. The answer's grounding is unchanged — these are flagged for a reader's attention."
      >
        <Icon.alert size={13} /> {n} {n === 1 ? 'claim needs' : 'claims need'} attention
      </button>
      {open && (
        <ul className="mt-2 ml-1 flex flex-col gap-1" style={{ color: 'var(--text-muted)' }}>
          {display.findings.map((c, i) => (
            <li key={i} className="text-xs">
              — {c}
            </li>
          ))}
        </ul>
      )}
    </span>
  )
}

/** The four dimensions composed, for one answer. */
export function VerificationStrip({ answer }: { answer: GroundedAnswer }): ReactNode {
  const display = describeVerification(answer)
  return (
    <span className="flex items-center gap-2 flex-wrap">
      <VerificationScore display={display} />
      <MethodBreakdown display={display} />
      <DegradedNote display={display} />
      <FindingsDisclosure display={display} />
    </span>
  )
}

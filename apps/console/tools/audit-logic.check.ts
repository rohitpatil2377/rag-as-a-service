// Checks the Audit page's pure helpers (summary line, search haystack, hash-chain verdict)
// against a REAL audit trail exported from a running deployment — the payload shapes the
// platform actually writes, not fixtures that agree with the code by construction.
//
//   TOKEN=$(curl -s -X POST localhost:8000/v1/dev/token -H 'content-type: application/json' \
//     -d '{"tenant_id":"<kb-id>","subject":"you@example.com","roles":["admin","owner"]}' \
//     | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
//   curl -s "localhost:8000/v1/kb/<kb-id>/audit?limit=250" -H "authorization: Bearer $TOKEN" > /tmp/audit.json
//   npx esbuild tools/audit-logic.check.ts --bundle --platform=node --format=esm --outfile=/tmp/chk.mjs
//   node /tmp/chk.mjs /tmp/audit.json
//
// The console has no test runner wired up, so this is a standalone script rather than a spec.
import { ACTION_META, summarise, verifyChain, eventHaystack } from '../src/pages/Audit'
import type { AuditEvent } from '../src/lib/api'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const rows: AuditEvent[] = JSON.parse(readFileSync(process.argv[2], 'utf8'))
let fails = 0
const check = (name: string, ok: boolean, detail = '') => {
  if (!ok) { fails++; console.log(`  FAIL  ${name} ${detail}`) } else console.log(`  pass  ${name} ${detail}`)
}

console.log(`\n== ${rows.length} real events ==`)

// 1. Every action present produces a non-empty, human-readable summary line.
const byAction = new Map<string, AuditEvent[]>()
for (const r of rows) { const l = byAction.get(r.action) ?? []; l.push(r); byAction.set(r.action, l) }
console.log('\n-- summaries --')
for (const [action, list] of byAction) {
  const sample = summarise(list[0])
  check(`summarise(${action})`, sample.trim().length > 0, `→ "${sample.slice(0, 68)}"`)
}

// 2. The filter the user actually tried: typing "index"/an action name must match that action.
console.log('\n-- search ("the filter is not at all working") --')
for (const action of byAction.keys()) {
  const hits = rows.filter((r) => eventHaystack(r).includes(action.toLowerCase()))
  const ofThat = hits.filter((r) => r.action === action).length
  check(`search "${action}" finds its own events`, ofThat === byAction.get(action)!.length,
    `→ ${ofThat}/${byAction.get(action)!.length} matched (${hits.length} rows total)`)
}

// 3. Searching an actor, and a word from a real question, must match.
const actor = rows[0].actor
check('search by actor', rows.filter((r) => eventHaystack(r).includes(actor.toLowerCase())).length > 0, `"${actor}"`)
const withQuery = rows.find((r) => typeof r.payload.query === 'string' && (r.payload.query as string).length > 12)
if (withQuery) {
  const word = (withQuery.payload.query as string).split(/\s+/).filter((w) => w.length > 4)[0] ?? ''
  check('search a word from a real question', rows.filter((r) => eventHaystack(r).includes(word.toLowerCase())).length > 0, `"${word}"`)
}
// A hash prefix, the way an auditor pastes one.
const h = rows[0].event_hash ?? ''
check('search by event-hash prefix', rows.filter((r) => eventHaystack(r).includes(h.slice(0, 12).toLowerCase())).length === 1, h.slice(0, 12))
// Nonsense must match nothing (and so hit the "no match" empty state, not the "nothing recorded" one).
check('nonsense matches nothing', rows.filter((r) => eventHaystack(r).includes('zzzqqq')).length === 0)

// 4. Chain verification must NOT cry tampering on a healthy trail with concurrent writes.
console.log('\n-- hash chain --')
// Sweep EVERY window size, not a few samples: a fork straddling the window boundary only
// misreports at one specific n, so sampling would miss exactly the case worth catching.
// A truncated window: no size may raise a finding on a healthy trail.
const badWindows: string[] = []
for (let n = 2; n < rows.length; n++) {
  const c = verifyChain(rows.slice(0, n), { complete: false })
  if (!c || c.verdict !== 'intact') badWindows.push(`n=${n}:${c ? c.verdict : 'null'}`)
}
check('no truncated window raises a finding', badWindows.length === 0,
  badWindows.length ? `bad at ${badWindows.slice(0, 8).join(', ')}` : `swept 2..${rows.length - 1}`)
// And the same sweep treated as a complete trail must not false-alarm either, since that is the
// mode in which a missing parent IS reported.
const badComplete: string[] = []
for (let n = 2; n <= rows.length; n++) {
  const c = verifyChain(rows.slice(0, n), { complete: true })
  if (!c || c.verdict !== 'intact') badComplete.push(`n=${n}:${c ? c.verdict : 'null'}`)
}
check('complete-trail mode false-alarms only where the page is forked near the floor',
  badComplete.length <= 1, badComplete.length ? `flagged ${badComplete.join(', ')}` : 'never')
const full = verifyChain(rows, { complete: true })
check('full window detail', full != null,
  full ? `resolved=${full.resolved} unresolved=${full.unresolved} forks=${full.forks} unreferenced=${full.unreferenced} anomalies=${full.anomalies}` : 'null')
check('single event → no claim', verifyChain(rows.slice(0, 1)) === null)
// A removed event must be caught: drop one from the middle and its child orphans. Needs a
// window big enough to HAVE a middle — a 2-event trail cannot express the scenario.
if (rows.length >= 25) {
  const tampered = [...rows.slice(0, 20), ...rows.slice(21, 60)]
  const tc = verifyChain(tampered, { complete: true })
  check('a removed event is detected', tc != null && tc.verdict !== 'intact',
    tc ? `verdict=${tc.verdict} unresolved=${tc.unresolved}` : 'null')
} else {
  console.log(`  skip  a removed event is detected (only ${rows.length} events in this trail)`)
}

// 5. The console's category catalog must not drift from the backend's AuditAction enum. The enum
// ships no category field, so this taxonomy is a SECOND copy maintained by hand — without this
// check a newly added action silently files itself under "Platform" with a de-underscored label,
// and nobody finds out until an auditor is looking at it.
console.log('\n-- action catalog vs backend enum --')
// Repo-relative so this works in any clone: the documented invocation runs from `apps/console`,
// and RAG_ENUM_PATH overrides it when the script is driven from somewhere else.
const ENUM_PATH = process.env.RAG_ENUM_PATH
  ?? resolve(process.cwd(), '../../libs/schemas/src/rag_schemas/common.py')
let enumSrc = ''
try { enumSrc = readFileSync(ENUM_PATH, 'utf8') } catch { enumSrc = '' }
if (!enumSrc) {
  console.log(`  skip  AuditAction enum not readable at ${ENUM_PATH} (running outside the repo?)`)
} else {
  const body = enumSrc.slice(enumSrc.indexOf('class AuditAction'))
  const enumActions = [...body.matchAll(/^ {4}[A-Z_0-9]+ = "([a-z_0-9]+)"/gm)].map((m) => m[1])
  const known = Object.keys(ACTION_META)
  const uncategorised = enumActions.filter((a) => !known.includes(a))
  const stale = known.filter((k) => !enumActions.includes(k))
  check('every AuditAction has a category + label', uncategorised.length === 0,
    uncategorised.length ? `uncategorised: ${uncategorised.join(', ')}` : `all ${enumActions.length} mapped`)
  check('no category entry for an action the backend no longer emits', stale.length === 0,
    stale.length ? `stale: ${stale.join(', ')}` : 'none')
}

console.log(fails ? `\n${fails} FAILURE(S)\n` : '\nall checks passed\n')
process.exit(fails ? 1 : 0)

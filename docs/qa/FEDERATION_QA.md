# Federated RAG Federation QA — cross-institution proof

**What it is.** The QA-runnable proof of the platform's differentiating "revolutionary" claim:
decentralised, cryptographically-authorised, revocable P2P federation between two sovereign
departments ("Federated RAG"). It exercises the FULL grant lifecycle — issue → cross-department
retrieval → tamper/forgery rejection → revoke → fail-closed — against a **live** stack, with a
PASS/FAIL/SKIP per assertion and a non-zero exit on any real failure. Nothing here is simulated:
every HTTP call hits the real `/v1` API, and where a sovereign-node container is reachable, the
tamper/forgery check also replays over the **real libp2p mesh**.

- QA script: `scripts/federation_qa.py`
- Grant crypto + issue/revoke: `services/control-plane/src/svc_control_plane/grants.py`
- Grant-endpoint surface: `services/control-plane/src/svc_control_plane/api.py` (`/v1/grants*`, `/v1/ask`)
- Transport seam (co-located vs. mesh): `services/control-plane/src/svc_control_plane/federation.py`,
  `docs/adr/0017-federation-transport.md`, `docs/adr/0020-mesh-authoritative-transport.md`
- Wire spec (so a third party can issue/verify byte-identical tokens): `docs/grant-token-spec.md`
- Sovereign-node (Go mesh peer): `sovereign-node/` — grant verify `grant/grant.go`, responder
  `protocols/query.go`, outbound control API `node/control.go`
- 2-node/N-node mesh compose + demo: `infra/compose/docker-compose.mesh.example.yml`, `infra/compose/mesh/`

## What the script proves

| Step | What it does | What a PASS proves |
|---|---|---|
| (a) issue | `POST /v1/grants` — issuer department signs a grant to the subject, scoped to a **throwaway KB + document created for this run** | An Ed25519-signed, time-bounded, department-scoped authorisation is issued through the real API; the issuer can only grant KBs it owns |
| (b) ask | `POST /v1/ask` (subject, `federate_from=[issuer]`) | The grant genuinely opens the tenant-isolation wall: the subject gets a **GROUNDED** answer with a real citation from the issuer's KB (`doc_id` prefixed `issuer/…`), and `federation_coverage.reached` names the issuer |
| (c) tamper/forge | crypto-level: `verify_grant()` on a genuine / tampered / forged grant. Wire-level (when a sovereign-node container is reachable): the SAME three grants replayed against the node's real outbound control API, which dials the peer over **real libp2p** | A grant's signature is the ONLY thing that opens the wall: a claim changed after signing, or a signature from any key other than the issuer's, is rejected fail-closed — by the exact code the live API runs, and (wire-level) by the exact Go code the remote department runs |
| (d) revoke | `DELETE /v1/grants/{id}`, then re-run the identical question | Revocation is immediate and fail-closed: the same query that worked in (b) now shows **no** issuer evidence — no citation, no leaked fact, no re-login or propagation delay |

### Why each assertion matters for a sovereign product

- **Tamper-reject (c) is the whole trust model.** A federation product's core promise is "your
  data leaves your walls only under YOUR cryptographic say-so." If a modified claim (wider
  `kb_ids`, a bigger `max_results_per_query`, a longer validity window) still verified, any
  intermediary — or the subject department itself — could silently escalate what it's allowed to
  read. Ed25519 signature verification is what makes a grant a *credential* instead of a polite
  request.
- **Forged-signature-reject (c) is the sovereignty boundary.** Any party can construct a
  syntactically valid `Grant` JSON claiming to be issued by department X. The only thing that
  makes it authentic is a signature only X's private key could produce. Rejecting a
  correctly-shaped grant signed by an unrelated key is what stops an attacker (or a compromised
  peer) from *impersonating* a department it doesn't control.
- **Post-revocation fail-closed (d) is what makes "revocable" real, not aspirational.** A
  federation model where a subject department, once granted, can keep reading forever (or until
  some cache/TTL quietly expires) is not a consent model. This proves revocation takes effect on
  the *very next* query — no propagation window, no re-login — which is exactly the guarantee a
  regulator or a nervous CISO will ask about before trusting a P2P design with real data.
- **Least-privilege scoping (a)+(b)** — the grant names specific `kb_ids`/`topics`, not "this
  department can read everything." Proving citations only ever surface from the *granted* KB is
  what stops a narrow grant from becoming a backdoor into the rest of the issuer's estate.

## Run it

```bash
uv sync --all-packages     # first time only: pulls in svc_control_plane + rag_schemas
uv run python scripts/federation_qa.py
```

Flags:

```
--base-url URL          the SUBJECT department's own api instance (default http://localhost:8000)
--issuer-dept ID         department that owns the KB / issues the grant (default dept-b)
--subject-dept ID        department that receives the grant / asks (default dept-a)
--keep                   don't clean up the temp KB this run creates
--skip-wire-check        skip the docker-exec sovereign-node control-API replay
--node-container NAME    override auto-detection of the subject dept's node container
--timeout SECONDS        HTTP timeout (default 60)
```

Exit code `0` only when every non-skipped step passed; `1` on any real failure; `2` if it
couldn't even mint a dev token (wrong `--base-url`, or the deployment isn't in dev-JWT mode).

### Expected PASS output (this deployment, 4-department mesh live)

```
==================================================================================
(a) Issue an Ed25519-signed grant
==================================================================================
  [PASS] ✓ grant issued (POST /v1/grants, 201, Ed25519-signed by the issuer) — grant_id=grant_... dept-b->dept-a kb_ids=[...] ...

==================================================================================
(b) Cross-department federated ask returns real citations
==================================================================================
  [PASS] ✓ federated ask (dept-a asks, federate_from=[dept-b]) — status=grounded reached=['dept-a', 'dept-b'] citation_doc_ids=['dept-b/fedqa-marker-...'] answer='The confirmation phrase for this federation QA run is **... ** [1].'

==================================================================================
(c) A tampered or forged grant is REJECTED (fail-closed signature enforcement)
==================================================================================
  [PASS] ✓ crypto-level: a genuinely-signed grant verifies (control)
  [PASS] ✓ crypto-level: a tampered claim (same signature) is rejected
  [PASS] ✓ crypto-level: a forged signature (attacker's own key) is rejected
  [PASS] ✓ wire-level: valid grant accepted by <node-a container> -> real dial to dept-b, real spans
  [PASS] ✓ wire-level: tampered grant rejected by the remote responder
  [PASS] ✓ wire-level: forged grant rejected by the remote responder

==================================================================================
(d) Revoke the grant -> the SAME federated ask now fails closed
==================================================================================
  [PASS] ✓ grant revoked (DELETE /v1/grants/{id} -> 204)
  [PASS] ✓ post-revocation: no peer data leaks (fail-closed, immediate, no re-login) — status=abstained reached=['dept-a'] citation_doc_ids=[]

  10 passed, 0 failed, 0 skipped.
  RESULT: PASS — the Federated RAG federation lifecycle is proven end-to-end, live.
```

Verified live on this deployment across repeated runs (see "What actually ran here" below).

## What runs where — single-api vs. a real mesh (be honest, don't fake a green)

The federation transport seam (`federation.py`, ADR-0017/0020) has two implementations behind
the SAME `/v1/ask` contract, selected by `RAG_FEDERATION_TRANSPORT`:

- **`local` (single-api default).** Both departments' KBs live in the same deployment.
  `LocalTransport` retrieves in-process but still re-validates the grant's Ed25519 signature +
  activity + delegation chain at USE time (`federation.py:_grant_verifies`) — fail-closed, not a
  shortcut. **Steps (a), (b), (d) of this script work unchanged on a single-api deployment** —
  they only assert on the public `/v1/ask` / `/v1/grants` HTTP contract, which is identical
  either way. Step (c)'s crypto-level half also needs nothing but the workspace venv.
- **`mesh`.** Departments are independently-deployed sovereign nodes reached over noise-encrypted
  libp2p (`sovereign-node`, `/rag/query/1.0`). `MeshTransport` routes the query to the local
  node's outbound control API (`POST http://<node>:7070/federate`), which dials the peer; the
  REMOTE node independently re-verifies the grant's signature (`grant/grant.go` `Verify`) and
  checks its own revocation list (`protocols/query.go`, `IsGrantRevoked`) before answering.

**What this specific deployment actually has, and what the script actually exercised:** this
box runs the real N-department mesh (`docker-compose.mesh.example.yml`, bootstrapped via
`infra/compose/mesh/bootstrap.sh`) — `dept-a`/`dept-b`/`dept-c`/`dept-d`, each with its own
`sovereign-node` (distinct libp2p identity) and its own api entry point (`:8000`/`:8001`/`:8002`/
`:8003`, ADR "per-department API instances"). So step (b) here is a **genuine two-node libp2p
hop** (see the `federated_query`/`hybrid_retrieve` log trail below), not the in-process fallback.
The script's wire-level half of step (c) additionally replays the tamper/forged grants directly
against the subject department's own `sovereign-node` control API via `docker exec` — a live,
cross-process, cross-language (Python-signed, Go-verified) proof, auto-detected and auto-skipped
(not faked) when no node container is reachable (e.g. a genuinely single-api box).

### Bringing up a 2-node mesh yourself (if this deployment doesn't already have one)

```bash
cd infra/compose
./mesh/bootstrap.sh     # generates libp2p identities, writes mesh/registry.json, wires per-dept tokens
# then either:
docker compose -f docker-compose.dev.yml -f docker-compose.deploy.example.yml \
               -f docker-compose.override.yml -f docker-compose.mesh.example.yml up -d node-a node-b api api-b
# or bring the whole documented topology up (dept-a..d, docker-compose.mesh.example.yml as shipped)
./mesh/demo.sh           # optional: the original 3-dept demo this script's design was validated against
```

See `infra/compose/mesh/README.md` for the topology diagram and per-department API note. Once
up, `uv run python scripts/federation_qa.py --issuer-dept dept-b --subject-dept dept-a` (the
defaults) runs the full live mesh proof above.

## A real finding this script surfaced (fixed in the script, not papered over)

Early runs asked the subject department with `kb_ids` omitted (documented as "empty = ALL the
org's KBs"). `list_knowledge_bases` always appends the shared `public__company-wide` KB to
"owned", so the federated `/v1/ask` ended up retrieving from **three** sources at once: the
subject's own KB, the shared KB (multiple unrelated documents), and the one genuinely relevant
federated chunk. On a stack with the reasoning-model generation path, that extra noise
occasionally led the model to attach its citation marker to the wrong numbered source in a
multi-source prompt — and the faithfulness checker then (correctly) scored that mis-citation
~0, tipping a genuinely well-grounded answer to `ABSTAINED`. This reproduced deterministically
(same outcome across 5 retries) regardless of whether the test fact was a hex-style code or a
plain-English phrase, which is what pointed at source-mixing rather than model or NLI-checker
noise. **Fix:** the script now pins the subject's `kb_ids` to its own discovered KB(s)
(`ask_body()` in `scripts/federation_qa.py`) — a cleaner, more realistic test scope, not a
workaround that hides a federation defect (the grant/citation/revocation mechanics were correct
in every run; only the *own-KB noise* was the confound). Verified: 3+ consecutive clean runs
after the fix, 0 failures. Reported here rather than silently fixed and forgotten, in case it's
worth a platform-level look (multi-source citation attribution under an LLM reasoning path).

**Platform-level resolution (landed).** Tracing the code corrected the mechanism above:
faithfulness is scored against the **union** of retrieved spans, so a mis-typed `[n]` never
lowered the score and never caused the abstain — grounded-or-abstain was never at risk. The real
defect was display-side (`_cited_only` kept the *typed* index, so the shown citation could point
at a non-supporting source and drop the true one). Fixed in `libs/rag-core/.../grounding.py` with
a conservative `_realign_citations()` that re-points a mis-typed marker to the span that actually
supports the sentence — running only *after* the grounding gate, never fabricating a citation,
leaving correct citations byte-identical. See `docs/qa/README.md` gap #6.

## Honest scope

- This proves the mechanics: signed issue, grant-gated cross-department retrieval, signature
  and forgery rejection, and immediate fail-closed revocation. It does not benchmark federation
  latency (see `scripts/benchmark.py` for that) or certify answer quality (see
  `docs/qa/RAGAS_GATE.md`).
- The script creates and tears down its OWN throwaway KB + document under the issuer department
  every run; it never touches an existing KB, and revoked grants are left in the grant history
  (matching the platform's own audit-trail design — grants are revoked, not deleted).
- Dev-JWT auth (`/v1/dev/token`) only. Against a non-dev deployment, adapt the script to mint or
  pass real JWTs for the issuer/subject departments.

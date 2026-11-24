#!/usr/bin/env bash
# smoke_test.sh — prove a fresh deployment actually works, end to end.
#
# Not a health check. /health returns ok while the embedder 401s, the reranker is unreachable or
# the master key is missing -- every one of which produces a platform that starts cleanly and
# cannot answer a question. This ingests a document and asks about it, which is the only test
# that exercises the whole chain: catalog, object store, embedder, both indexes, reranker,
# generation, the faithfulness gate and the grounded-or-abstain contract.
#
#   scripts/smoke_test.sh                      # against http://localhost:8000
#   scripts/smoke_test.sh https://rag.internal # against a deployed host
set -uo pipefail

API="${1:-http://localhost:8000}"
# KB ids follow the {org}__{slug} convention -- the isolation guard (_assert_owns_kb) requires
# it, so a flat name can be created and then never deleted. The token is minted for the ORG.
ORG="smoke$(date +%s)"
KB="${ORG}__probe"  # created below; ingest does not auto-create on a fresh org
PASS=0; FAIL=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n    %s\n' "$1" "${2:-}"; FAIL=$((FAIL+1)); }

printf '\nRAG as a Service smoke test -> %s\n\n' "$API"

# 1. the process is up
if curl -fsS --max-time 10 "$API/health" >/dev/null 2>&1; then ok "API responds"; else
  bad "API responds" "cannot reach $API/health"; exit 1; fi

# 2. a token. On a prod build /v1/dev/token is DISABLED by design, so fall back to a real login.
TOKEN=$(curl -fsS --max-time 10 -X POST "$API/v1/dev/token" -H 'Content-Type: application/json' \
  -d "{\"subject\":\"smoke@local\",\"tenant_id\":\"$ORG\",\"roles\":[\"admin\",\"ingestor\",\"reader\"]}" \
  2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])' 2>/dev/null)
if [ -n "${TOKEN:-}" ]; then
  ok "dev token minted"
  printf '  \033[33mNOTE\033[0m  /v1/dev/token is OPEN — anyone reaching this API can mint an admin\n'
  printf '        token for any tenant. Expected in dev; on production set RAG_ENV=prod.\n'
else
  # Correct for a production build -- and this script has to work THERE, or it is useless
  # exactly where it matters. So it signs up its own throwaway organisation through the real
  # auth path, which is also a better test: it proves signup, login and org provisioning work.
  printf '  \033[32mPASS\033[0m  /v1/dev/token is closed (RAG_ENV=prod)\n'; PASS=$((PASS+1))
  SIGNUP=$(curl -fsS --max-time 90 -X POST "$API/v1/auth/signup" -H 'Content-Type: application/json' \
    -d "{\"email\":\"smoke-$(date +%s)@local.test\",\"password\":\"Sm0ke-Test-$(date +%s)!\",\"name\":\"Smoke Test\",\"org_name\":\"$ORG\"}" 2>/dev/null)
  TOKEN=$(printf '%s' "$SIGNUP" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("access_token",""))' 2>/dev/null)
  if [ -n "${TOKEN:-}" ]; then ok "signed up a throwaway organisation ($ORG)"; else
    bad "signup" "$(printf '%s' "$SIGNUP" | head -c 200)"
    printf '\n  Cannot continue without a token.\n\n'; exit 2
  fi
fi
AUTH=(-H "Authorization: Bearer $TOKEN")

# 3. the knowledge base. A fresh org has none; ingest does not create one.
curl -fsS --max-time 60 -X POST "$API/v1/knowledge-bases" "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d '{"display_name":"probe"}' >/dev/null 2>&1 \
  && ok "knowledge base created" || printf '  \033[33mNOTE\033[0m  KB may already exist — continuing\n'

# 4. ingest — exercises object store, extraction, chunking, embedder, both indexes, catalog
BODY=$(python3 - <<'PY'
import base64, json
text = ("SMOKE TEST DOCUMENT.\n\n1.0 OBJECTIVE\nTo verify this deployment can ingest and answer.\n"
        "5.0 PROCEDURE\n5.1 The deployment smoke code is ZEBRA-4417.\n"
        "5.2 A deviation must be recorded within 24 hours of detection.\n")
print(json.dumps({"doc_id": "smoke.txt", "mime": "text/plain", "source_kind": "upload",
                  "content_b64": base64.b64encode(text.encode()).decode()}))
PY
)
ING=$(curl -fsS --max-time 180 -X POST "$API/v1/kb/$KB/ingest" "${AUTH[@]}" -H 'Content-Type: application/json' -d "$BODY" 2>&1)
CHUNKS=$(printf '%s' "$ING" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("chunk_count",0))' 2>/dev/null || echo 0)
if [ "${CHUNKS:-0}" -gt 0 ]; then ok "ingest produced $CHUNKS chunk(s)"; else
  bad "ingest" "$(printf '%s' "$ING" | head -c 200)"; fi

# 4. answer — the whole retrieval + generation + faithfulness chain
ASK=$(curl -fsS --max-time 180 -X POST "$API/v1/kb/$KB/query" "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d '{"query":"What is the deployment smoke code?"}' 2>&1)
read -r STATUS CITES ANSWER <<<"$(printf '%s' "$ASK" | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(d.get("status"), len(d.get("citations", [])), (d.get("answer") or "").replace("\n", " ")[:120])
' 2>/dev/null || echo "error 0 -")"
if [ "$STATUS" = "grounded" ] && [ "${CITES:-0}" -gt 0 ]; then ok "grounded answer with $CITES citation(s)"; else
  bad "grounded answer" "status=$STATUS cites=$CITES :: $(printf '%s' "$ASK" | head -c 200)"; fi
case "$ANSWER" in *ZEBRA-4417*) ok "answer contains the planted fact";;
  *) bad "answer contains the planted fact" "got: $ANSWER";; esac

# 5. the gate — a platform that answers everything is the failure this catches
REF=$(curl -fsS --max-time 180 -X POST "$API/v1/kb/$KB/query" "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d '{"query":"What is the annual leave entitlement for employees?"}' 2>&1 \
  | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status"))' 2>/dev/null)
if [ "$REF" = "abstained" ]; then ok "abstains on an unanswerable question"; else
  bad "abstains on an unanswerable question" "status=$REF — the grounding gate is NOT holding"; fi

# 6. clean up after ourselves
curl -fsS --max-time 60 -X DELETE "$API/v1/knowledge-bases/$KB" "${AUTH[@]}" >/dev/null 2>&1 \
  && ok "test knowledge base removed" || printf '  \033[33mNOTE\033[0m  remove the KB "%s" by hand\n' "$KB"

printf '\n  %d passed, %d failed\n\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]

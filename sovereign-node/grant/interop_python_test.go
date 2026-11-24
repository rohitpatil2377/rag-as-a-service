package grant

import (
	"testing"
	"time"
)

// Proves byte-for-byte interop with the Python signer: the canonical form and a
// signature PRODUCED BY PYTHON (services/control-plane/.../grants.py) must validate here.
// Fixture + signature captured from `sign_grant` on 2026-08-03 with the default dev secret.
func TestVerifiesPythonProducedSignature(t *testing.T) {
	nb, _ := time.Parse(time.RFC3339, "2026-08-03T00:00:00+00:00")
	ex, _ := time.Parse(time.RFC3339, "2026-12-31T00:00:00+00:00")
	g := &Grant{
		GrantID:            "grant_fixture01",
		IssuerDeptID:       "hr",
		SubjectDeptID:      "finance",
		Topics:             []string{"leave-policy", "headcount"},
		KBIDs:              []string{"hr__policies"},
		MaxResultsPerQuery: 5,
		NotBefore:          nb,
		ExpiresAt:          ex,
		SignedToken:        "4uh/SJOg8xT4yNPLyY57FYtwMoHen695cIhHdxUTxVoRvVYoABfnu/srID2LvDUOBbGdP+VZJEwEIcBzHPTfDg==",
	}
	const wantCanon = `{"expires_at":"2026-12-31T00:00:00+00:00","grant_id":"grant_fixture01","issuer_dept_id":"hr","kb_ids":["hr__policies"],"max_results_per_query":5,"not_before":"2026-08-03T00:00:00+00:00","subject_dept_id":"finance","topics":["headcount","leave-policy"]}`
	canon, err := g.CanonicalClaims()
	if err != nil {
		t.Fatalf("canonical: %v", err)
	}
	if string(canon) != wantCanon {
		t.Fatalf("canonical mismatch with Python:\n go:     %s\n python: %s", canon, wantCanon)
	}
	if !g.Verify(DevPublicKey(DefaultDevJWTSecret)) {
		t.Fatal("Go failed to verify a signature produced by the Python signer")
	}
}

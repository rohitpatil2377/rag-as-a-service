package grant

import (
	"encoding/base64"
	"encoding/hex"
	"testing"
	"time"
)

// knownGrant is the exact fixture used to generate the golden values below, produced by
// running the real Python code (svc_control_plane.grants._canonical_claims + the dev
// signing key derivation). Keep this identical to the Python fixture — it is the
// byte-level interop contract.
func knownGrant() *Grant {
	return &Grant{
		GrantID:            "grant-001",
		IssuerDeptID:       "dept-finance",
		SubjectDeptID:      "dept-audit",
		Topics:             []string{"taxation", "budget", "audit"}, // intentionally unsorted
		KBIDs:              []string{"kb-2024", "kb-2023"},          // intentionally unsorted
		MaxResultsPerQuery: 5,
		NotBefore:          time.Date(2026, 8, 3, 0, 0, 0, 0, time.UTC),
		ExpiresAt:          time.Date(2026, 9, 3, 12, 30, 45, 0, time.UTC),
	}
}

// Golden values captured from the Python reference implementation.
const (
	pyCanonical = `{"expires_at":"2026-09-03T12:30:45+00:00","grant_id":"grant-001","issuer_dept_id":"dept-finance","kb_ids":["kb-2023","kb-2024"],"max_results_per_query":5,"not_before":"2026-08-03T00:00:00+00:00","subject_dept_id":"dept-audit","topics":["audit","budget","taxation"]}`
	pySeedHex   = "3e15b9449461b51a34ef422f105fa0b4b2943978b1bf49520532e15b9895f3f6"
	pySigB64    = "4+6tExzmoOBTi94AUtwEwIj4/XfHjxpbw+LWi7gb4Tk1gbF/4Br+VwYBPPOkprWh4sRYTHx6k++yaWVCCyTOAA=="
	pyPubHex    = "8b2c9d59ae4c9fcdd79b982924b673d5ea9b06640df5a87e9b6546f9b2a67603"
)

// TestCanonicalClaimsMatchesPython is THE interop test: the Go canonical byte form must
// equal Python's json.dumps(sort_keys=True, separators=(",",":")) output exactly.
func TestCanonicalClaimsMatchesPython(t *testing.T) {
	got, err := knownGrant().CanonicalClaims()
	if err != nil {
		t.Fatalf("CanonicalClaims: %v", err)
	}
	if string(got) != pyCanonical {
		t.Fatalf("canonical bytes differ from Python.\n got: %s\nwant: %s", got, pyCanonical)
	}
	if len(got) != len(pyCanonical) {
		t.Fatalf("length %d != %d", len(got), len(pyCanonical))
	}
}

// TestDevSigningKeySeedMatchesPython checks the Ed25519 seed derivation and the resulting
// public key match Python's sha256("grant-dev::"+secret) -> from_private_bytes(seed).
func TestDevSigningKeySeedMatchesPython(t *testing.T) {
	priv := DevSigningKey(DefaultDevJWTSecret)
	seed := priv.Seed()
	if hex.EncodeToString(seed) != pySeedHex {
		t.Fatalf("seed %s != %s", hex.EncodeToString(seed), pySeedHex)
	}
	pub := DevPublicKey(DefaultDevJWTSecret)
	if hex.EncodeToString(pub) != pyPubHex {
		t.Fatalf("public key %s != %s", hex.EncodeToString(pub), pyPubHex)
	}
}

// TestVerifyPythonSignature proves a signature MINTED BY PYTHON verifies in Go.
func TestVerifyPythonSignature(t *testing.T) {
	g := knownGrant()
	g.SignedToken = pySigB64
	if !g.Verify(DevPublicKey(DefaultDevJWTSecret)) {
		t.Fatal("Python-produced signature failed to verify in Go")
	}
}

// TestSignMatchesPython proves Go signing produces the identical base64 signature as
// Python (Ed25519 is deterministic, so this is exact).
func TestSignMatchesPython(t *testing.T) {
	priv := DevSigningKey(DefaultDevJWTSecret)
	sig, err := knownGrant().Sign(priv)
	if err != nil {
		t.Fatalf("Sign: %v", err)
	}
	if sig != pySigB64 {
		t.Fatalf("Go signature != Python.\n got: %s\nwant: %s", sig, pySigB64)
	}
}

// TestSignVerifyRoundtrip is a standalone sign+verify with the dev key.
func TestSignVerifyRoundtrip(t *testing.T) {
	priv := DevSigningKey("")
	g := knownGrant()
	sig, err := g.Sign(priv)
	if err != nil {
		t.Fatalf("Sign: %v", err)
	}
	g.SignedToken = sig
	if !g.Verify(DevPublicKey("")) {
		t.Fatal("roundtrip verify failed")
	}
}

// TestTamperedGrantFails proves any mutation of the claims invalidates the signature.
func TestTamperedGrantFails(t *testing.T) {
	pub := DevPublicKey(DefaultDevJWTSecret)

	cases := []struct {
		name   string
		mutate func(*Grant)
	}{
		{"grant_id", func(g *Grant) { g.GrantID = "grant-666" }},
		{"issuer", func(g *Grant) { g.IssuerDeptID = "dept-evil" }},
		{"subject", func(g *Grant) { g.SubjectDeptID = "dept-evil" }},
		{"topics", func(g *Grant) { g.Topics = append(g.Topics, "secrets") }},
		{"kb_ids", func(g *Grant) { g.KBIDs = append(g.KBIDs, "kb-secret") }},
		{"max_results", func(g *Grant) { g.MaxResultsPerQuery = 20 }},
		{"not_before", func(g *Grant) { g.NotBefore = g.NotBefore.Add(time.Hour) }},
		{"expires_at", func(g *Grant) { g.ExpiresAt = g.ExpiresAt.Add(time.Hour) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := knownGrant()
			g.SignedToken = pySigB64
			tc.mutate(g)
			if g.Verify(pub) {
				t.Fatalf("tampered %q still verified — signature is not covering that field", tc.name)
			}
		})
	}
}

// TestVerifyRejectsGarbage covers non-base64 tokens, wrong-size keys, empty tokens.
func TestVerifyRejectsGarbage(t *testing.T) {
	pub := DevPublicKey(DefaultDevJWTSecret)
	g := knownGrant()

	g.SignedToken = "not!base64!!"
	if g.Verify(pub) {
		t.Fatal("non-base64 token verified")
	}
	g.SignedToken = ""
	if g.Verify(pub) {
		t.Fatal("empty token verified")
	}
	g.SignedToken = pySigB64
	if g.Verify(nil) {
		t.Fatal("nil public key verified")
	}
	if g.Verify([]byte{1, 2, 3}) {
		t.Fatal("short public key verified")
	}
	// A valid-base64 but wrong signature.
	g.SignedToken = base64.StdEncoding.EncodeToString(make([]byte, 64))
	if g.Verify(pub) {
		t.Fatal("zero signature verified")
	}
}

func TestIsActive(t *testing.T) {
	g := knownGrant()
	nb := g.NotBefore
	exp := g.ExpiresAt

	if g.IsActiveAt(nb.Add(-time.Second)) {
		t.Fatal("active before not_before")
	}
	if !g.IsActiveAt(nb) {
		t.Fatal("should be active at not_before (inclusive)")
	}
	if !g.IsActiveAt(exp.Add(-time.Second)) {
		t.Fatal("should be active just before expiry")
	}
	if g.IsActiveAt(exp) {
		t.Fatal("active at expires_at (exclusive)")
	}
	revoked := nb.Add(time.Hour)
	g.RevokedAt = &revoked
	if g.IsActiveAt(nb.Add(2 * time.Hour)) {
		t.Fatal("revoked grant reported active")
	}
}

func TestAllowsTopicsAndKB(t *testing.T) {
	g := knownGrant()
	if !g.AllowsTopics([]string{"audit"}) {
		t.Fatal("should allow subset")
	}
	if !g.AllowsTopics([]string{"audit", "budget"}) {
		t.Fatal("should allow multi subset")
	}
	if g.AllowsTopics([]string{"audit", "classified"}) {
		t.Fatal("should reject topic outside grant")
	}
	if !g.AllowsTopics(nil) {
		t.Fatal("empty request is trivially a subset")
	}
	if !g.AllowsKB("kb-2023") {
		t.Fatal("kb-2023 should be allowed")
	}
	if g.AllowsKB("kb-9999") {
		t.Fatal("kb-9999 should be rejected")
	}
}

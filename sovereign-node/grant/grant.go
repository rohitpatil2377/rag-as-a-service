// Package grant is the security boundary + Python-interop point of the SovereignNode.
//
// A Grant is a cryptographic authorisation: an ISSUER department signs permission for
// a SUBJECT department to query specific topics from specific KBs, bounded by
// not_before/expires_at and revocable. The token is a detached Ed25519 signature over
// the grant's canonical claims. This package MUST reproduce, byte-for-byte, the
// canonical form produced by the Python control-plane
// (services/control-plane/src/svc_control_plane/grants.py) so a grant signed on the
// Python side verifies here and vice-versa.
//
// This package depends ONLY on the Go standard library (crypto/ed25519, crypto/sha256,
// encoding/json, encoding/base64) so it builds and tests offline with zero external deps.
package grant

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"sort"
	"time"
)

// DefaultDevJWTSecret matches the Python default (rag_common Settings.dev_jwt_secret).
const DefaultDevJWTSecret = "dev-only-insecure-secret-change-me-32b+"

// Grant mirrors rag_schemas.federation.Grant. The json tags cover the wire form used
// on the libp2p stream; the canonical (signed) form is built separately by
// CanonicalClaims so it exactly matches Python's json.dumps(sort_keys, separators).
type Grant struct {
	GrantID            string     `json:"grant_id"`
	IssuerDeptID       string     `json:"issuer_dept_id"`
	SubjectDeptID      string     `json:"subject_dept_id"`
	Topics             []string   `json:"topics"`
	KBIDs              []string   `json:"kb_ids"`
	MaxResultsPerQuery int        `json:"max_results_per_query"`
	NotBefore          time.Time  `json:"not_before"`
	ExpiresAt          time.Time  `json:"expires_at"`
	SignedToken        string     `json:"signed_token"`
	RevokedAt          *time.Time `json:"revoked_at,omitempty"`
	CreatedAt          *time.Time `json:"created_at,omitempty"`
}

// isoUTC formats a time exactly the way Python's datetime.astimezone(UTC).isoformat()
// does for a whole-second, UTC-aware datetime: "2026-08-03T00:00:00+00:00".
//
// CRITICAL interop note: Python emits the offset as "+00:00", NOT "Z" (that is what
// time.RFC3339 would give for UTC), and it does NOT print fractional seconds when the
// microsecond component is zero. All grants issued by the Python side use whole-second
// UTC timestamps, so we replicate that exact shape. (If a sub-second timestamp ever
// appeared, Python would print ".ffffff"; we don't emit that here because the grant
// model only ever carries whole-second instants.)
func isoUTC(t time.Time) string {
	return t.UTC().Format("2006-01-02T15:04:05+00:00")
}

// canonicalClaims is a helper type whose field ORDER does not matter — encoding/json
// sorts map keys, but for structs it preserves declaration order. To guarantee we match
// Python's sort_keys=True we build a map instead, which Go's json.Marshal sorts by key.
//
// Go's json.Marshal uses no spaces between elements and after colons, matching Python's
// separators=(",", ":").

// CanonicalClaims returns the deterministic byte form that is signed by the issuer.
// It MUST be byte-identical to Python's _canonical_claims: sorted keys, no whitespace,
// ISO-8601 UTC timestamps with a "+00:00" offset, and topics/kb_ids sorted ascending.
func (g *Grant) CanonicalClaims() ([]byte, error) {
	topics := append([]string(nil), g.Topics...)
	sort.Strings(topics)
	kbIDs := append([]string(nil), g.KBIDs...)
	sort.Strings(kbIDs)

	// A map[string]any so json.Marshal sorts keys (== Python sort_keys=True). Go's
	// encoder emits compact JSON (no spaces) == Python separators=(",", ":").
	claims := map[string]any{
		"grant_id":              g.GrantID,
		"issuer_dept_id":        g.IssuerDeptID,
		"subject_dept_id":       g.SubjectDeptID,
		"topics":                topics,
		"kb_ids":                kbIDs,
		"max_results_per_query": g.MaxResultsPerQuery,
		"not_before":            isoUTC(g.NotBefore),
		"expires_at":            isoUTC(g.ExpiresAt),
	}
	return json.Marshal(claims)
}

// DevSigningKey derives the deterministic development Ed25519 private key from the dev
// JWT secret, exactly like Python's _load_signing_key dev fallback:
//
//	seed = sha256("grant-dev::" + dev_jwt_secret)   # 32 bytes
//	key  = Ed25519PrivateKey.from_private_bytes(seed)
//
// Ed25519PrivateKey.from_private_bytes(seed) treats the 32 bytes as the *seed*, which is
// exactly what ed25519.NewKeyFromSeed expects.
func DevSigningKey(devSecret string) ed25519.PrivateKey {
	if devSecret == "" {
		devSecret = DefaultDevJWTSecret
	}
	sum := sha256.Sum256([]byte("grant-dev::" + devSecret))
	return ed25519.NewKeyFromSeed(sum[:])
}

// DevPublicKey returns the public half of the dev signing key.
func DevPublicKey(devSecret string) ed25519.PublicKey {
	return DevSigningKey(devSecret).Public().(ed25519.PublicKey)
}

// Sign signs the grant's canonical claims with priv and returns a base64 (std encoding)
// detached signature — the same encoding Python uses (base64.b64encode).
func (g *Grant) Sign(priv ed25519.PrivateKey) (string, error) {
	canon, err := g.CanonicalClaims()
	if err != nil {
		return "", err
	}
	sig := ed25519.Sign(priv, canon)
	return base64.StdEncoding.EncodeToString(sig), nil
}

// Verify reports whether SignedToken is a valid Ed25519 signature over the grant's
// canonical claims for the given issuer public key. It is signature-only — validity in
// time (not_before/expires_at/revoked) is checked separately by IsActive. Any decode or
// verification failure returns false (never panics): the caller fails closed.
func (g *Grant) Verify(pub ed25519.PublicKey) bool {
	if len(pub) != ed25519.PublicKeySize {
		return false
	}
	sig, err := base64.StdEncoding.DecodeString(g.SignedToken)
	if err != nil {
		return false
	}
	canon, err := g.CanonicalClaims()
	if err != nil {
		return false
	}
	return ed25519.Verify(pub, canon, sig)
}

// IsActive mirrors Grant.is_active on the Python side: not revoked, and the current
// instant is within [not_before, expires_at). Uses UTC to match the Python comparison.
func (g *Grant) IsActive() bool {
	return g.IsActiveAt(time.Now().UTC())
}

// IsActiveAt is IsActive at an explicit instant (testable, no wall-clock dependency).
func (g *Grant) IsActiveAt(now time.Time) bool {
	if g.RevokedAt != nil {
		return false
	}
	now = now.UTC()
	return !now.Before(g.NotBefore.UTC()) && now.Before(g.ExpiresAt.UTC())
}

// AllowsTopics reports whether every requested topic is covered by the grant.
func (g *Grant) AllowsTopics(requested []string) bool {
	allowed := make(map[string]struct{}, len(g.Topics))
	for _, t := range g.Topics {
		allowed[t] = struct{}{}
	}
	for _, t := range requested {
		if _, ok := allowed[t]; !ok {
			return false
		}
	}
	return true
}

// AllowsKB reports whether kbID is one of the grant's authorised KBs.
func (g *Grant) AllowsKB(kbID string) bool {
	for _, k := range g.KBIDs {
		if k == kbID {
			return true
		}
	}
	return false
}

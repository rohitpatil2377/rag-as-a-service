package protocols

import (
	"testing"
	"time"
)

func fixedGuard(now time.Time, skew time.Duration) *ReplayGuard {
	g := NewReplayGuard(skew)
	g.now = func() time.Time { return now }
	return g
}

func TestReplayGuard_FreshPasses(t *testing.T) {
	now := time.Date(2026, 8, 4, 12, 0, 0, 0, time.UTC)
	g := fixedGuard(now, time.Minute)
	if err := g.Check("finance", "n1", now.Format(time.RFC3339Nano)); err != nil {
		t.Fatalf("fresh request should pass: %v", err)
	}
}

func TestReplayGuard_ReplayRejected(t *testing.T) {
	now := time.Date(2026, 8, 4, 12, 0, 0, 0, time.UTC)
	g := fixedGuard(now, time.Minute)
	stamp := now.Format(time.RFC3339Nano)
	if err := g.Check("finance", "n1", stamp); err != nil {
		t.Fatalf("first use should pass: %v", err)
	}
	if err := g.Check("finance", "n1", stamp); err == nil {
		t.Fatalf("second use of the same nonce must be rejected")
	}
	// Same nonce from a DIFFERENT subject is a different key — allowed.
	if err := g.Check("legal", "n1", stamp); err != nil {
		t.Fatalf("same nonce, different subject should pass: %v", err)
	}
}

func TestReplayGuard_StaleRejected(t *testing.T) {
	now := time.Date(2026, 8, 4, 12, 0, 0, 0, time.UTC)
	g := fixedGuard(now, time.Minute)
	old := now.Add(-5 * time.Minute).Format(time.RFC3339Nano)
	if err := g.Check("finance", "old", old); err == nil {
		t.Fatalf("a request older than the skew window must be rejected")
	}
	future := now.Add(5 * time.Minute).Format(time.RFC3339Nano)
	if err := g.Check("finance", "future", future); err == nil {
		t.Fatalf("a request too far in the future must be rejected")
	}
}

func TestReplayGuard_MissingFieldsRejected(t *testing.T) {
	g := NewReplayGuard(time.Minute)
	if err := g.Check("finance", "", "2026-08-04T12:00:00Z"); err == nil {
		t.Fatalf("missing nonce must be rejected")
	}
	if err := g.Check("finance", "n", ""); err == nil {
		t.Fatalf("missing issued_at must be rejected")
	}
	if err := g.Check("finance", "n", "not-a-time"); err == nil {
		t.Fatalf("unparseable issued_at must be rejected")
	}
}

func TestReplayGuard_ExpiredNonceEvicted(t *testing.T) {
	now := time.Date(2026, 8, 4, 12, 0, 0, 0, time.UTC)
	g := fixedGuard(now, time.Minute)
	g.Check("finance", "n1", now.Format(time.RFC3339Nano)) //nolint:errcheck // seeding
	// Advance beyond 2×skew: the old nonce is evicted, and a fresh request at the new time
	// with a new nonce passes (the cache stays bounded).
	later := now.Add(3 * time.Minute)
	g.now = func() time.Time { return later }
	if err := g.Check("finance", "n2", later.Format(time.RFC3339Nano)); err != nil {
		t.Fatalf("fresh request after eviction should pass: %v", err)
	}
	if len(g.seen) != 1 {
		t.Fatalf("expired nonce should have been evicted; seen=%d", len(g.seen))
	}
}

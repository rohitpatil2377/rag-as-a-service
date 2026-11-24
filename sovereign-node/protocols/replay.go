package protocols

import (
	"fmt"
	"sync"
	"time"
)

// ReplayGuard rejects stale or replayed federated queries. The GRANT proves "may this
// department read this?"; the guard proves "is THIS request fresh and seen for the first
// time?" — so a captured request cannot be re-sent to re-run a query the subject already made.
//
// It is defence-in-depth on top of the transport: the request rides a noise-encrypted,
// peer-authenticated libp2p stream (ADR-0018) and the subject-dept check binds it to the
// authenticated caller, so the nonce + issued_at it carries are trustworthy. The guard adds
// freshness (a bounded clock-skew window) and single-use (a per-subject seen-nonce cache).
type ReplayGuard struct {
	mu   sync.Mutex
	seen map[string]time.Time // key -> expiry
	skew time.Duration
	now  func() time.Time
}

// NewReplayGuard builds a guard accepting requests whose issued_at is within ±skew of now,
// remembering each nonce for 2×skew (long enough that a replay cannot outlast the freshness
// window). skew <= 0 falls back to 120s.
func NewReplayGuard(skew time.Duration) *ReplayGuard {
	if skew <= 0 {
		skew = 120 * time.Second
	}
	return &ReplayGuard{seen: make(map[string]time.Time), skew: skew, now: time.Now}
}

// Check returns nil if the request is fresh and unseen; otherwise an error naming the reason.
// It records the nonce as seen on success. Callers MUST fail closed on any error.
func (g *ReplayGuard) Check(subjectDept, nonce, issuedAt string) error {
	if nonce == "" || issuedAt == "" {
		return fmt.Errorf("replay: missing nonce or issued_at")
	}
	ts, err := time.Parse(time.RFC3339Nano, issuedAt)
	if err != nil {
		return fmt.Errorf("replay: bad issued_at: %w", err)
	}
	now := g.now()
	if d := now.Sub(ts); d > g.skew || d < -g.skew {
		return fmt.Errorf("replay: stale request (skew %s)", d)
	}

	key := subjectDept + "\x00" + nonce
	g.mu.Lock()
	defer g.mu.Unlock()
	g.gc(now)
	if _, ok := g.seen[key]; ok {
		return fmt.Errorf("replay: nonce already seen")
	}
	g.seen[key] = now.Add(2 * g.skew)
	return nil
}

// gc drops expired nonces so the cache stays bounded. Caller holds the lock.
func (g *ReplayGuard) gc(now time.Time) {
	for k, exp := range g.seen {
		if now.After(exp) {
			delete(g.seen, k)
		}
	}
}

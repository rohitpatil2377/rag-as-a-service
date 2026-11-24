package protocols

import (
	"context"
	"crypto/ed25519"
	"testing"
	"time"

	libp2p "github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/peer"

	"sovereignnode/bridge"
)

// TestReplayGuard_RejectsReplayedQueryOverWire drives two requests with the SAME nonce through
// a real inbound handler that has a ReplayGuard: the first is answered, the replay is denied
// (fail closed), while a fresh nonce still passes. Proves the guard is wired into the handler.
func TestReplayGuard_RejectsReplayedQueryOverWire(t *testing.T) {
	ctx := context.Background()
	hb, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("host B: %v", err)
	}
	defer hb.Close()
	ha, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("host A: %v", err)
	}
	defer ha.Close()

	g, pub := signedGrant(t)
	dir := &fakeDir{
		keys:  map[string]ed25519.PublicKey{"hr": pub},
		peers: map[peer.ID]string{ha.ID(): "finance"},
	}
	RegisterQueryHandler(hb, &QueryService{
		LocalDept: "hr",
		Registry:  dir,
		Bridge:    &fakeBridge{spans: []bridge.Span{{ChunkID: "c1", Text: "hi", Score: 0.9, DocID: "d1"}}},
		Replay:    NewReplayGuard(time.Minute),
	})
	if err := ha.Connect(ctx, peer.AddrInfo{ID: hb.ID(), Addrs: hb.Addrs()}); err != nil {
		t.Fatalf("connect: %v", err)
	}

	stamp := time.Now().UTC().Format(time.RFC3339Nano)
	mkReq := func(nonce string) *FederatedQueryRequest {
		return &FederatedQueryRequest{
			Grant: g, GrantToken: g.SignedToken, QueryText: "q", Topics: []string{"payroll"},
			KBID: "kb-hr", TopK: 5, Nonce: nonce, IssuedAt: stamp,
		}
	}

	// First use of nonce "n1": answered.
	first, err := QueryPeer(ctx, ha, hb.ID(), mkReq("n1"))
	if err != nil {
		t.Fatalf("first query: %v", err)
	}
	if !first.GrantValid || len(first.Spans) != 1 {
		t.Fatalf("first query should be answered: %+v", first)
	}

	// Replay of nonce "n1": denied (fail closed, no spans).
	replay, err := QueryPeer(ctx, ha, hb.ID(), mkReq("n1"))
	if err != nil {
		t.Fatalf("replay query transport: %v", err)
	}
	if replay.GrantValid || len(replay.Spans) != 0 {
		t.Fatalf("replayed nonce must be denied: %+v", replay)
	}

	// A fresh nonce still passes.
	fresh, err := QueryPeer(ctx, ha, hb.ID(), mkReq("n2"))
	if err != nil {
		t.Fatalf("fresh query: %v", err)
	}
	if !fresh.GrantValid || len(fresh.Spans) != 1 {
		t.Fatalf("a fresh nonce should be answered: %+v", fresh)
	}
}

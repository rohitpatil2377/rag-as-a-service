package protocols

import (
	"context"
	"crypto/ed25519"
	"testing"

	libp2p "github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/peer"

	"sovereignnode/bridge"
)

// TestQueryPeer_RoundTrip drives the OUTBOUND client against a REAL inbound handler over two
// in-process libp2p hosts: A (initiator, dept "finance") dials B (responder, dept "hr"), B
// verifies the grant fail-closed and retrieves via a stub bridge, A gets the spans back.
func TestQueryPeer_RoundTrip(t *testing.T) {
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

	g, pub := signedGrant(t) // issuer "hr", subject "finance"
	dir := &fakeDir{
		keys:  map[string]ed25519.PublicKey{"hr": pub},
		peers: map[peer.ID]string{ha.ID(): "finance"}, // the caller (host A) IS the subject
	}
	RegisterQueryHandler(hb, &QueryService{
		LocalDept: "hr",
		Registry:  dir,
		Bridge:    &fakeBridge{spans: []bridge.Span{{ChunkID: "c1", Text: "hi", Score: 0.9, DocID: "d1"}}},
	})

	// A learns B's address and connects.
	if err := ha.Connect(ctx, peer.AddrInfo{ID: hb.ID(), Addrs: hb.Addrs()}); err != nil {
		t.Fatalf("connect: %v", err)
	}

	req := &FederatedQueryRequest{
		Grant: g, GrantToken: g.SignedToken, QueryText: "q",
		Topics: []string{"payroll"}, KBID: "kb-hr", TopK: 5,
	}
	resp, err := QueryPeer(ctx, ha, hb.ID(), req)
	if err != nil {
		t.Fatalf("QueryPeer: %v", err)
	}
	if !resp.GrantValid {
		t.Fatalf("grant should verify over the wire: %+v", resp)
	}
	if len(resp.Spans) != 1 || resp.Spans[0].ChunkID != "c1" {
		t.Fatalf("unexpected spans: %+v", resp.Spans)
	}
	if resp.PeerDeptID != "hr" {
		t.Fatalf("peer dept: got %q want hr", resp.PeerDeptID)
	}
}

// TestQueryPeer_InvalidGrantFailsClosed: a tampered signature comes back as a well-formed
// GrantValid:false (NOT a transport error), and QueryPeer surfaces it as such.
func TestQueryPeer_InvalidGrantFailsClosed(t *testing.T) {
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
	g.SignedToken = "AAAA" // tamper after signing
	dir := &fakeDir{
		keys:  map[string]ed25519.PublicKey{"hr": pub},
		peers: map[peer.ID]string{ha.ID(): "finance"},
	}
	RegisterQueryHandler(hb, &QueryService{LocalDept: "hr", Registry: dir, Bridge: &fakeBridge{}})
	if err := ha.Connect(ctx, peer.AddrInfo{ID: hb.ID(), Addrs: hb.Addrs()}); err != nil {
		t.Fatalf("connect: %v", err)
	}

	req := &FederatedQueryRequest{Grant: g, GrantToken: g.SignedToken, QueryText: "q", KBID: "kb-hr", TopK: 5}
	resp, err := QueryPeer(ctx, ha, hb.ID(), req)
	if err != nil {
		t.Fatalf("QueryPeer should not transport-error on a bad grant: %v", err)
	}
	if resp.GrantValid || len(resp.Spans) != 0 {
		t.Fatalf("bad grant must fail closed with no spans: %+v", resp)
	}
}

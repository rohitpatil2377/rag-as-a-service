package protocols

import (
	"context"
	"crypto/ed25519"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"

	"sovereignnode/bridge"
	"sovereignnode/grant"
)

// fakeDir is an in-test DeptDirectory.
type fakeDir struct {
	keys    map[string]ed25519.PublicKey
	peers   map[peer.ID]string
	revoked map[string]bool
}

func (f *fakeDir) IsGrantRevoked(grantID string) bool { return f.revoked[grantID] }

func (f *fakeDir) PublicKeyFor(dept string) (ed25519.PublicKey, bool) {
	k, ok := f.keys[dept]
	return k, ok
}
func (f *fakeDir) DeptForPeer(p peer.ID) (string, bool) {
	d, ok := f.peers[p]
	return d, ok
}

// fakeBridge returns a fixed set of spans and records how it was called.
type fakeBridge struct {
	calledKB   string
	calledTopK int
	spans      []bridge.Span
}

func (f *fakeBridge) Retrieve(_ context.Context, kbID, _ string, topK int) (*bridge.RetrieveResponse, error) {
	f.calledKB = kbID
	f.calledTopK = topK
	return &bridge.RetrieveResponse{Spans: f.spans}, nil
}

// signedGrant builds and signs a grant with the dev key for issuer "hr".
func signedGrant(t *testing.T) (grant.Grant, ed25519.PublicKey) {
	t.Helper()
	priv := grant.DevSigningKey("")
	pub := priv.Public().(ed25519.PublicKey)
	g := grant.Grant{
		GrantID:            "g-1",
		IssuerDeptID:       "hr",
		SubjectDeptID:      "finance",
		Topics:             []string{"payroll"},
		KBIDs:              []string{"kb-hr"},
		MaxResultsPerQuery: 5,
		NotBefore:          time.Now().Add(-time.Hour).UTC().Truncate(time.Second),
		ExpiresAt:          time.Now().Add(time.Hour).UTC().Truncate(time.Second),
	}
	tok, err := g.Sign(priv)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	g.SignedToken = tok
	return g, pub
}

const callerPeer = peer.ID("peer-finance")

func newService(t *testing.T, br Retriever) (*QueryService, grant.Grant) {
	t.Helper()
	g, pub := signedGrant(t)
	dir := &fakeDir{
		keys:  map[string]ed25519.PublicKey{"hr": pub},
		peers: map[peer.ID]string{callerPeer: "finance"},
	}
	return &QueryService{LocalDept: "hr", Registry: dir, Bridge: br}, g
}

func TestAuthorize_HappyPath(t *testing.T) {
	br := &fakeBridge{spans: []bridge.Span{{ChunkID: "c1", Text: "hi", Score: 0.9, DocID: "d1"}}}
	svc, g := newService(t, br)
	req := &FederatedQueryRequest{Grant: g, GrantToken: g.SignedToken, QueryText: "q", Topics: []string{"payroll"}, KBID: "kb-hr", TopK: 100}

	resp := svc.authorizeAndRetrieve(callerPeer, req)
	if !resp.GrantValid {
		t.Fatalf("expected grant valid")
	}
	if len(resp.Spans) != 1 || resp.Spans[0].ChunkID != "c1" {
		t.Fatalf("unexpected spans: %+v", resp.Spans)
	}
	if resp.Spans[0].SourceDeptID != "hr" || resp.Spans[0].KBID != "kb-hr" {
		t.Fatalf("span not stamped: %+v", resp.Spans[0])
	}
	if br.calledTopK != 5 { // capped at MaxResultsPerQuery
		t.Fatalf("topK not capped: got %d want 5", br.calledTopK)
	}
}

func TestAuthorize_FailClosed(t *testing.T) {
	tamper := func(mut func(r *FederatedQueryRequest, g *grant.Grant)) FederatedQueryResponse {
		br := &fakeBridge{spans: []bridge.Span{{ChunkID: "c1"}}}
		svc, g := newService(t, br)
		req := &FederatedQueryRequest{Grant: g, GrantToken: g.SignedToken, QueryText: "q", Topics: []string{"payroll"}, KBID: "kb-hr", TopK: 5}
		if mut != nil {
			mut(req, &req.Grant)
			req.GrantToken = req.Grant.SignedToken
		}
		return svc.authorizeAndRetrieve(callerPeer, req)
	}

	cases := map[string]func(r *FederatedQueryRequest, g *grant.Grant){
		"unknown_issuer":   func(_ *FederatedQueryRequest, g *grant.Grant) { g.IssuerDeptID = "nope" },
		"bad_signature":    func(_ *FederatedQueryRequest, g *grant.Grant) { g.SignedToken = "AAAA" },
		"tampered_topics":  func(_ *FederatedQueryRequest, g *grant.Grant) { g.Topics = []string{"secrets"} },
		"expired":          func(_ *FederatedQueryRequest, g *grant.Grant) { g.ExpiresAt = time.Now().Add(-time.Minute).UTC() },
		"topic_not_subset": func(r *FederatedQueryRequest, _ *grant.Grant) { r.Topics = []string{"secrets"} },
		"kb_not_allowed":   func(r *FederatedQueryRequest, _ *grant.Grant) { r.KBID = "kb-secret" },
	}
	for name, mut := range cases {
		t.Run(name, func(t *testing.T) {
			resp := tamper(mut)
			if resp.GrantValid {
				t.Fatalf("%s: expected fail-closed (grant_valid=false), got valid", name)
			}
			if len(resp.Spans) != 0 {
				t.Fatalf("%s: expected no spans on denial", name)
			}
		})
	}
}

func TestAuthorize_WrongCallerDept(t *testing.T) {
	br := &fakeBridge{}
	svc, g := newService(t, br)
	req := &FederatedQueryRequest{Grant: g, GrantToken: g.SignedToken, QueryText: "q", KBID: "kb-hr", TopK: 5}
	// Caller peer maps to a department other than the grant's subject.
	resp := svc.authorizeAndRetrieve(peer.ID("intruder"), req)
	if resp.GrantValid {
		t.Fatalf("expected fail-closed for unknown caller peer")
	}
}

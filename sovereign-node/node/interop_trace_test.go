package node_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"

	"sovereignnode/bridge"
	"sovereignnode/grant"
	"sovereignnode/node"
	"sovereignnode/protocols"
)

// TestMeshInterop_TraceparentPropagates proves the initiator's W3C trace context rides the
// libp2p query and lands as a header on the PEER's rag-api retrieval — one distributed trace
// across both departments (T1 distributed tracing).
func TestMeshInterop_TraceparentPropagates(t *testing.T) {
	ctx := context.Background()
	gotTP := make(chan string, 1)
	rag := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotTP <- r.Header.Get("traceparent")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"spans":     []map[string]any{{"chunk_id": "c1", "doc_id": "d1", "text": "x", "score": 0.9}},
			"max_score": 0.9,
		})
	}))
	defer rag.Close()

	hrNode, err := node.New("hr", "/ip4/127.0.0.1/tcp/0", "")
	if err != nil {
		t.Fatalf("hr node: %v", err)
	}
	defer hrNode.Close()
	finNode, err := node.New("finance", "/ip4/127.0.0.1/tcp/0", "")
	if err != nil {
		t.Fatalf("finance node: %v", err)
	}
	defer finNode.Close()

	hrNode.Registry.Register("hr", grant.DevPublicKey(""), hrNode.PeerID(), "")
	hrNode.Registry.Register("finance", nil, finNode.PeerID(), "")
	protocols.RegisterQueryHandler(hrNode.Host, &protocols.QueryService{
		LocalDept: "hr", Registry: hrNode.Registry, Bridge: bridge.New(rag.URL, ""),
	})
	finNode.Registry.Register("hr", grant.DevPublicKey(""), hrNode.PeerID(), "")
	if err := finNode.Host.Connect(ctx, peer.AddrInfo{ID: hrNode.PeerID(), Addrs: hrNode.Host.Addrs()}); err != nil {
		t.Fatalf("connect: %v", err)
	}

	g := grant.Grant{
		GrantID: "g-tp", IssuerDeptID: "hr", SubjectDeptID: "finance",
		Topics: []string{"leave-policy"}, KBIDs: []string{"hr__policies"}, MaxResultsPerQuery: 5,
		NotBefore: time.Now().Add(-time.Hour).UTC().Truncate(time.Second),
		ExpiresAt: time.Now().Add(time.Hour).UTC().Truncate(time.Second),
	}
	tok, err := g.Sign(grant.DevSigningKey(""))
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	g.SignedToken = tok

	tp := "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
	control := node.NewControlServer(finNode, protocols.QueryPeer).Handler()
	body, _ := json.Marshal(map[string]any{
		"peer_dept_id": "hr", "query_text": "q", "top_k": 5, "grant": g, "traceparent": tp,
	})
	rec := httptest.NewRecorder()
	control.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/federate", bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("control status: %d", rec.Code)
	}

	select {
	case seen := <-gotTP:
		if seen != tp {
			t.Fatalf("peer rag-api got traceparent %q, want %q", seen, tp)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("peer rag-api was never called")
	}
}

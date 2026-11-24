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

// TestMeshInterop_EndToEnd is the live two-node interop: department "finance" federates a
// query to department "hr" through the FULL chain — finance's control API → QueryPeer →
// libp2p (noise) → hr's inbound handler → real Ed25519 grant verification → hr's HTTP bridge
// → a stand-in rag-api → spans back. It proves "federate the query, not the vector"
// (ADR-0019) AND that a span's category rides the wire so L1 topic-enforcement can reach
// mesh spans. The only stand-in is the rag-api itself (an httptest server returning the
// exact /retrieve JSON shape); everything between the two nodes is real.
func TestMeshInterop_EndToEnd(t *testing.T) {
	ctx := context.Background()

	// A stand-in rag-api for dept "hr": returns one span with a category tag.
	rag := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"spans": []map[string]any{{
				"chunk_id": "c1", "doc_id": "d1",
				"text": "annual leave is 20 days", "score": 0.88, "category": "leave-policy",
			}},
			"max_score": 0.88,
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

	// hr can verify its own grants and knows the caller (finance) peer -> dept for the
	// subject check; it retrieves via a REAL bridge to the stand-in rag-api.
	hrNode.Registry.Register("hr", grant.DevPublicKey(""), hrNode.PeerID(), "")
	hrNode.Registry.Register("finance", nil, finNode.PeerID(), "")
	protocols.RegisterQueryHandler(hrNode.Host, &protocols.QueryService{
		LocalDept: "hr",
		Registry:  hrNode.Registry,
		Bridge:    bridge.New(rag.URL, ""),
	})

	// finance knows how to reach hr (dept -> peer) and is connected to it.
	finNode.Registry.Register("hr", grant.DevPublicKey(""), hrNode.PeerID(), "")
	if err := finNode.Host.Connect(ctx, peer.AddrInfo{ID: hrNode.PeerID(), Addrs: hrNode.Host.Addrs()}); err != nil {
		t.Fatalf("connect finance->hr: %v", err)
	}

	// A real signed grant: hr authorises finance for topic "leave-policy" on kb "hr__policies".
	g := grant.Grant{
		GrantID:            "g-interop",
		IssuerDeptID:       "hr",
		SubjectDeptID:      "finance",
		Topics:             []string{"leave-policy"},
		KBIDs:              []string{"hr__policies"},
		MaxResultsPerQuery: 5,
		NotBefore:          time.Now().Add(-time.Hour).UTC().Truncate(time.Second),
		ExpiresAt:          time.Now().Add(time.Hour).UTC().Truncate(time.Second),
	}
	tok, err := g.Sign(grant.DevSigningKey(""))
	if err != nil {
		t.Fatalf("sign grant: %v", err)
	}
	g.SignedToken = tok

	// Drive it exactly as the Python MeshTransport would: POST /federate on finance's node.
	control := node.NewControlServer(finNode, protocols.QueryPeer).Handler()
	body, _ := json.Marshal(map[string]any{
		"peer_dept_id": "hr",
		"query_text":   "how many annual leave days",
		"top_k":        5,
		"grant":        g,
	})
	rec := httptest.NewRecorder()
	control.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/federate", bytes.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("control status: %d", rec.Code)
	}
	var resp protocols.FederatedQueryResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !resp.GrantValid {
		t.Fatalf("grant should verify across the mesh: %+v", resp)
	}
	if len(resp.Spans) != 1 || resp.Spans[0].ChunkID != "c1" {
		t.Fatalf("expected the hr span back: %+v", resp.Spans)
	}
	if resp.Spans[0].Category != "leave-policy" {
		t.Fatalf("category must ride the wire for L1 enforcement, got %q", resp.Spans[0].Category)
	}
	if resp.PeerDeptID != "hr" {
		t.Fatalf("peer dept: got %q want hr", resp.PeerDeptID)
	}
}
